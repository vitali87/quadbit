"""Produce PAIR-GRANULAR 2:4 weights one-shot (no fine-tuning) so the sparse FP4 kernel is
usable on a real model -- the thing no public checkpoint provides (all are element-granular).

Method = Wanda (Sun et al.): importance = |W_ij| * ||X_j||_2 from one calibration pass, keep
the top-2 of every 4 PAIRS (pair-granular, matching Blackwell FP4 mma.sp), no weight updates,
no gradients. Then FP4-quantize the kept values and run the real model's MLPs through our
sparse kernel. Baselines to beat: magnitude pair-2:4 = 93.6 PPL (broken), dense FP4 = 8.16,
fp16 = 7.89. If Wanda-pair lands near dense, the +82%-over-CUTLASS sparse path is real.

Run:  uv run modal run harness/wanda_pair.py
"""

import subprocess
from pathlib import Path

import modal

ROOT = Path(__file__).parent.parent
MODEL = "neuralmagic/Sparse-Llama-3.1-8B-2of4"  # dense-2:4 fp16; we re-prune pair-granular

image = (
    modal.Image.from_registry("nvidia/cuda:12.8.1-devel-ubuntu22.04", add_python="3.12")
    .env({"PATH": "/usr/local/cuda/bin:/usr/local/sbin:/usr/local/bin:/usr/sbin:/usr/bin:/sbin:/bin",
          "LD_LIBRARY_PATH": "/usr/local/cuda/lib64", "HF_HOME": "/cache"})
    .pip_install("torch", index_url="https://download.pytorch.org/whl/nightly/cu128", pre=True)
    .pip_install("transformers", "huggingface_hub", "safetensors", "sentencepiece", "pyarrow")
    .add_local_dir((ROOT / "cuda").as_posix(), "/root/cuda")
)
app = modal.App("quadbit-wanda", image=image)
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

    class QuadbitLinear(nn.Module):
        """Pack W to pair-granular 2:4 FP4; pairs kept = top-2 of each 4 by `score` (None=|W|)."""

        def __init__(self, W, score=None):
            super().__init__()
            out_f, in_f = W.shape
            ks = in_f // 128
            self.out_f, self.in_f, self.ks = out_f, in_f, ks
            Wg = W.float().to(dev).view(out_f, ks, 16, 4, 2)
            ps = Wg.abs().sum(-1) if score is None else score.view(out_f, ks, 16, 4)
            i01, _ = ps.topk(2, dim=-1).indices.sort(dim=-1)          # kept pair indices, ascending
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

    print(f"loading {MODEL} ...", flush=True)
    tok = AutoTokenizer.from_pretrained(MODEL)

    def load():
        return AutoModelForCausalLM.from_pretrained(MODEL, dtype=torch.bfloat16).to(dev).eval()

    def wikitext(split):
        p = hf_hub_download("Salesforce/wikitext", f"wikitext-2-raw-v1/{split}-00000-of-00001.parquet",
                            repo_type="dataset")
        return tok("\n\n".join(pq.read_table(p).column("text").to_pylist()), return_tensors="pt").input_ids[0]

    test_ids = wikitext("test")
    calib_ids = wikitext("train")

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

    model = load()

    # --- Wanda calibration: per-linear input column L2 norm over calibration tokens ---
    sumsq = {}

    def hook(mod, inp, _out):
        x = inp[0].detach().float().reshape(-1, inp[0].shape[-1])
        k = id(mod)
        sumsq[k] = sumsq.get(k, torch.zeros(x.shape[-1], device=dev)) + (x * x).sum(0)

    handles = [lin.register_forward_hook(hook) for _, _, lin in mlp_lins(model)]
    seq = 2048
    with torch.no_grad():
        for i in range(0, 8 * seq, seq):            # 8 calibration windows
            model(calib_ids[i:i + seq].unsqueeze(0).to(dev))
    for h in handles:
        h.remove()

    print(f"\nPPL fp16 baseline: {ppl(model):.3f}  (magnitude pair-2:4 was 93.6, dense FP4 8.16)",
          flush=True)

    # --- swap MLPs to Wanda pair-granular 2:4 FP4 ---
    n = 0
    for mlp, name, lin in mlp_lins(model):
        col = sumsq[id(lin)].sqrt()                 # ||X_j||_2 per input column
        score = lin.weight.data.float().abs() * col[None, :]     # Wanda importance [out,in]
        ps = score.view(lin.weight.shape[0], lin.weight.shape[1] // 128, 16, 4, 2).sum(-1)
        setattr(mlp, name, QuadbitLinear(lin.weight.data, score=ps).to(dev)); n += 1
    torch.cuda.empty_cache()
    print(f"PPL Wanda pair-2:4 FP4 ({n} MLP layers): {ppl(model):.3f}", flush=True)


@app.local_entrypoint()
def main() -> None:
    run.remote()
