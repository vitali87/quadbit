"""Pair-granular SparseGPT: one-shot (no fine-tuning) pruning to PAIR-granular 2:4 with
Hessian-based weight compensation, so the sparse FP4 kernel is usable on a real model at
good accuracy. This is what unlocks the +82%-over-CUTLASS sparse path on real weights --
the pair-granular structure no public checkpoint provides.

SparseGPT (Frantar & Alistarh): H = X^T X from a calibration pass; process columns left to
right, at each group of 4 PAIRS keep the 2 with largest w^2/[H^-1]_jj^2, prune the rest, and
propagate the induced error into the not-yet-processed columns via H^-1. No gradients.
Then FP4-quantize the kept values and run the MLPs through our sparse kernel.

Baselines on Sparse-Llama-3.1-8B: fp16 7.89, dense FP4 8.16, magnitude pair-2:4 93.6, Wanda 59.1.
Run:  uv run modal run harness/sparsegpt_pair.py
"""

import subprocess
from pathlib import Path

import modal

ROOT = Path(__file__).parent.parent
MODEL = "Qwen/Qwen2.5-3B"  # DENSE model (avoid double-sparsifying); MLP dims %256

image = (
    modal.Image.from_registry("nvidia/cuda:12.8.1-devel-ubuntu22.04", add_python="3.12")
    .env({"PATH": "/usr/local/cuda/bin:/usr/local/sbin:/usr/local/bin:/usr/sbin:/usr/bin:/sbin:/bin",
          "LD_LIBRARY_PATH": "/usr/local/cuda/lib64", "HF_HOME": "/cache"})
    .pip_install("torch", index_url="https://download.pytorch.org/whl/nightly/cu128", pre=True)
    .pip_install("transformers", "huggingface_hub", "safetensors", "sentencepiece", "pyarrow")
    .add_local_dir((ROOT / "cuda").as_posix(), "/root/cuda")
)
app = modal.App("quadbit-sparsegpt", image=image)
vol = modal.Volume.from_name("quadbit-hf-cache", create_if_missing=True)


@app.function(gpu="RTX-PRO-6000", timeout=3600, volumes={"/cache": vol})
def run() -> None:
    import ctypes
    import math

    import pyarrow.parquet as pq
    import torch
    import torch.nn as nn
    from huggingface_hub import hf_hub_download
    from transformers import AutoModelForCausalLM, AutoTokenizer

    so = "/root/sparse_fp4.so"
    c = subprocess.run(["nvcc", "-arch=sm_120a", "-O3", "-shared", "-Xcompiler", "-fPIC",
                        "-o", so, "/root/cuda/sparse_fp4_lib.cu", "-lcuda"],
                       capture_output=True, text=True)
    if c.returncode != 0:
        print(c.stderr, flush=True); return
    lib = ctypes.CDLL(so)
    lib.sparse_fp4_mm.argtypes = [ctypes.c_void_p] * 6 + [ctypes.c_int] * 3
    lib.quantize_act_nvfp4.argtypes = [ctypes.c_void_p] * 3 + [ctypes.c_int] * 2
    dev = torch.device("cuda")

    FP4 = torch.tensor([0, .5, 1, 1.5, 2, 3, 4, 6, 0, -.5, -1, -1.5, -2, -3, -4, -6],
                       dtype=torch.float32, device=dev)
    BND = torch.tensor([.25, .75, 1.25, 1.75, 2.5, 3.5, 5.], dtype=torch.float32, device=dev)
    _c = torch.arange(128, device=dev)
    _e, _m = (_c >> 3) & 0xf, _c & 7
    UE4M3 = torch.where(_e == 0, _m.float() * 0.001953125,
                        (1.0 + _m.float() / 8.0) * torch.exp2((_e - 7).float()))
    _MID = (UE4M3[:-1] + UE4M3[1:]) / 2

    def q_fp4(v):
        return torch.bucketize(v.abs(), BND) | ((v < 0).long() << 3)

    def enc(s):
        return torch.bucketize(s, _MID)

    def fp4_only(W):  # per-16-block NVFP4, no pruning (dense FP4 accuracy baseline)
        o, i = W.shape
        b = W.float().view(o, i // 16, 16)
        s = UE4M3[enc(b.abs().amax(-1) / 6.0)]
        return (FP4[q_fp4(b / s[..., None])] * s[..., None]).reshape(o, i)

    class QuadbitLinear(nn.Module):
        """Pack a (pair-2:4-sparse) weight to FP4; magnitude picks the nonzero pairs."""

        def __init__(self, W):
            super().__init__()
            out_f, in_f = W.shape
            ks = in_f // 128
            self.out_f, self.in_f, self.ks = out_f, in_f, ks
            Wg = W.float().to(dev).view(out_f, ks, 16, 4, 2)
            i01, _ = Wg.abs().sum(-1).topk(2, dim=-1).indices.sort(dim=-1)
            keptW = torch.gather(Wg, 3, i01.unsqueeze(-1).expand(-1, -1, -1, -1, 2))
            blk = keptW.reshape(out_f, ks, 4, 8, 2)
            scode = enc(blk.abs().amax(dim=(3, 4)) / 6.0)
            sdeq = UE4M3[scode]
            kc = q_fp4(blk / sdeq[..., None, None])
            Ac = (kc[..., 0] | (kc[..., 1] << 4)).reshape(out_f, ks * 32).to(torch.uint8)
            nib = (i01[..., 0] | (i01[..., 1] << 2)).view(out_f, ks, 2, 8)
            sh = (torch.arange(8, device=dev) * 4).view(1, 1, 1, 8)
            meta = (nib << sh).sum(-1).to(torch.int32).permute(1, 0, 2).contiguous()
            self.register_buffer("Ac", Ac.contiguous())
            self.register_buffer("meta", meta)
            self.register_buffer("scaleA", scode.to(torch.uint8).permute(1, 0, 2).contiguous())

        def forward(self, x):
            lead = x.shape[:-1]
            x2 = x.reshape(-1, self.in_f).to(torch.bfloat16)
            t = x2.shape[0]; pad = (-t) % 128
            if pad:
                x2 = torch.cat([x2, x2.new_zeros(pad, self.in_f)], 0)
            x2 = x2.contiguous(); tp = t + pad
            Bb = torch.empty((tp, self.in_f // 2), dtype=torch.uint8, device=dev)
            sB = torch.empty((self.ks, tp, 4), dtype=torch.uint8, device=dev)
            lib.quantize_act_nvfp4(x2.data_ptr(), Bb.data_ptr(), sB.data_ptr(), tp, self.in_f)
            C = torch.empty((self.out_f, tp), dtype=torch.bfloat16, device=dev)
            lib.sparse_fp4_mm(self.Ac.data_ptr(), Bb.data_ptr(), self.scaleA.data_ptr(),
                              sB.data_ptr(), self.meta.data_ptr(), C.data_ptr(),
                              self.out_f, tp, self.in_f)
            return C.t()[:t].reshape(*lead, self.out_f).to(x.dtype)

    def sparsegpt_pair24(W, H, blocksize=128, percdamp=0.01):
        """One-shot pair-granular 2:4 prune with Hessian compensation. W[out,in], H[in,in]."""
        W = W.float().clone()
        cols = W.shape[1]
        dead = torch.diag(H) == 0
        H[dead, dead] = 1.0; W[:, dead] = 0.0
        H[range(cols), range(cols)] += percdamp * torch.mean(torch.diag(H))
        Hinv = torch.linalg.cholesky(torch.cholesky_inverse(torch.linalg.cholesky(H)), upper=True)
        rows = W.shape[0]
        rowidx = torch.arange(rows, device=dev)
        for i in range(0, cols, blocksize):
            e = min(i + blocksize, cols); B = e - i
            W1 = W[:, i:e].clone()
            Err = torch.zeros_like(W1)
            Hinv1 = Hinv[i:e, i:e]
            dinv = torch.diag(Hinv1)
            curmask = None
            for j in range(B):
                if (j % 8) == 0:                       # new group of 4 pairs (8 cols)
                    tmp = (W1[:, j:j + 8] ** 2) / (dinv[j:j + 8] ** 2)[None, :]
                    pairm = tmp[:, 0::2] + tmp[:, 1::2]           # [rows, 4] pair metric
                    pr = pairm.topk(2, dim=1, largest=False).indices  # 2 pairs to prune
                    curmask = torch.ones(rows, 8, dtype=torch.bool, device=dev)
                    curmask[rowidx[:, None], pr * 2] = False
                    curmask[rowidx[:, None], pr * 2 + 1] = False
                w = W1[:, j]
                q = torch.where(curmask[:, j % 8], w, torch.zeros_like(w))
                err = (w - q) / dinv[j]
                W1[:, j:] -= err[:, None] * Hinv1[j, j:][None, :]
                Err[:, j] = err
                W1[:, j] = q
            W[:, i:e] = W1
            W[:, e:] -= Err @ Hinv[i:e, e:]
        return W

    print(f"loading {MODEL} ...", flush=True)
    tok = AutoTokenizer.from_pretrained(MODEL)

    def load():
        return AutoModelForCausalLM.from_pretrained(MODEL, dtype=torch.bfloat16).to(dev).eval()

    model = load()

    def wikitext(split):
        p = hf_hub_download("Salesforce/wikitext", f"wikitext-2-raw-v1/{split}-00000-of-00001.parquet",
                            repo_type="dataset")
        return tok("\n\n".join(pq.read_table(p).column("text").to_pylist()), return_tensors="pt").input_ids[0]

    test_ids, calib_ids = wikitext("test"), wikitext("train")

    def ppl(m, seq=2048, windows=16):
        nll, n = 0.0, 0
        for i in range(0, min(len(test_ids), windows * seq) - seq, seq):
            w = test_ids[i:i + seq].unsqueeze(0).to(dev)
            with torch.no_grad():
                nll += m(w, labels=w).loss.item() * (seq - 1); n += seq - 1
        return math.exp(nll / n)

    def mlp_lins(m):
        for layer in m.model.layers:
            for name in ("gate_proj", "up_proj", "down_proj"):
                lin = getattr(layer.mlp, name)
                if lin.weight.shape[0] % 256 == 0 and lin.weight.shape[1] % 256 == 0:
                    yield layer.mlp, name, lin

    print(f"PPL fp16 baseline: {ppl(model):.3f}", flush=True)

    # --- accumulate H = (2/n) X^T X per MLP linear over calibration tokens ---
    Hs, ns = {}, {}

    def hook(mod, inp, _out):
        x = inp[0].detach().float().reshape(-1, inp[0].shape[-1])
        k = id(mod); n = x.shape[0]
        if k not in Hs:
            Hs[k] = torch.zeros(x.shape[1], x.shape[1], device=dev); ns[k] = 0
        Hs[k] *= ns[k] / (ns[k] + n)
        Hs[k] += (x * math.sqrt(2.0 / (ns[k] + n))).t() @ (x * math.sqrt(2.0 / (ns[k] + n)))
        ns[k] += n

    handles = [lin.register_forward_hook(hook) for _, _, lin in mlp_lins(model)]
    seq = 2048
    with torch.no_grad():
        for i in range(0, 16 * seq, seq):
            model(calib_ids[i:i + seq].unsqueeze(0).to(dev))
    for h in handles:
        h.remove()
    print(f"calibrated H on {ns[next(iter(ns))]} tokens; pruning...", flush=True)

    # --- SparseGPT pair-2:4 prune each MLP linear, then swap to sparse FP4 kernel ---
    n = 0
    for mlp, name, lin in mlp_lins(model):
        Wp = sparsegpt_pair24(lin.weight.data, Hs.pop(id(lin)))
        setattr(mlp, name, QuadbitLinear(Wp).to(dev)); n += 1
        del Wp; torch.cuda.empty_cache()
    p_sgpt = ppl(model)
    del model; torch.cuda.empty_cache()

    # baselines on the SAME model: dense FP4 (no prune) and magnitude pair-2:4
    m2 = load()
    for mlp, name, lin in mlp_lins(m2):
        lin.weight.data = fp4_only(lin.weight.data).to(torch.bfloat16)
    p_dense = ppl(m2); del m2; torch.cuda.empty_cache()

    m3 = load()
    for mlp, name, lin in mlp_lins(m3):
        setattr(mlp, name, QuadbitLinear(lin.weight.data).to(dev))  # magnitude pair-2:4
    p_mag = ppl(m3)

    print(f"\n=== {MODEL} pair-2:4 FP4 ({n} MLP layers) ===", flush=True)
    print(f"  dense FP4 (no prune) : {p_dense:.3f}", flush=True)
    print(f"  magnitude pair-2:4   : {p_mag:.3f}", flush=True)
    print(f"  SparseGPT pair-2:4   : {p_sgpt:.3f}", flush=True)


@app.local_entrypoint()
def main() -> None:
    run.remote()
