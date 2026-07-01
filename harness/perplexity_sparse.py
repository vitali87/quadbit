"""Definitive end-to-end test: run the REAL neuralmagic Sparse-Llama-3.1-8B-2of4 model and
measure WikiText-2 perplexity in three configs -- (1) native fp16 (its designed accuracy),
(2) MLPs swapped to our 2:4-sparse FP4 kernel, (3) MLPs swapped to dense FP4 -- so we see the
actual generation-quality impact, not a weight-reconstruction proxy. Uses their model as-is;
no fine-tuning. The sparse model is element-granular 2:4; our kernel is pair-granular, so this
tells us whether that mismatch actually hurts perplexity or is cosmetic in weight-recon.

Run:  uv run modal run harness/perplexity_sparse.py
"""

import subprocess
from pathlib import Path

import modal

ROOT = Path(__file__).parent.parent
MODEL = "neuralmagic/Sparse-Llama-3.1-8B-2of4"

image = (
    modal.Image.from_registry("nvidia/cuda:12.8.1-devel-ubuntu22.04", add_python="3.12")
    .env({"PATH": "/usr/local/cuda/bin:/usr/local/sbin:/usr/local/bin:/usr/sbin:/usr/bin:/sbin:/bin",
          "LD_LIBRARY_PATH": "/usr/local/cuda/lib64", "HF_HOME": "/cache"})
    .pip_install("torch", index_url="https://download.pytorch.org/whl/nightly/cu128", pre=True)
    .pip_install("transformers", "huggingface_hub", "safetensors", "sentencepiece", "pyarrow")
    .add_local_dir((ROOT / "cuda").as_posix(), "/root/cuda")
)
app = modal.App("quadbit-ppl", image=image)
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
    lib.sparse_fp4_mm.restype = ctypes.c_int
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
        return (torch.bucketize(v.abs(), BND) | ((v < 0).long() << 3))

    def enc(s):
        return torch.bucketize(s, _MID)

    def fp4_only(W):  # per-16-block NVFP4, NO pruning -> isolates FP4-quant error (keeps zeros)
        out_f, in_f = W.shape
        b = W.float().view(out_f, in_f // 16, 16)
        s = UE4M3[enc(b.abs().amax(-1) / 6.0)]
        return (FP4[q_fp4(b / s[..., None])] * s[..., None]).reshape(out_f, in_f)

    class QuadbitLinear(nn.Module):
        """nn.Linear replacement (bias=False): pack W to 2:4-sparse (or dense) FP4, kernel forward."""

        def __init__(self, W, sparse=True):  # W [out, in] (already 2:4-sparse for the NM model)
            super().__init__()
            out_f, in_f = W.shape
            ks = in_f // 128
            self.out_f, self.in_f, self.ks, self.sparse = out_f, in_f, ks, sparse
            Wf = W.float().to(dev)
            if sparse:
                Wg = Wf.view(out_f, ks, 16, 4, 2)
                i01, _ = Wg.abs().sum(-1).topk(2, dim=-1).indices.sort(dim=-1)
                keptW = torch.gather(Wg, 3, i01.unsqueeze(-1).expand(-1, -1, -1, -1, 2))
                blk = keptW.reshape(out_f, ks, 4, 8, 2)
                sdeq = UE4M3[enc(blk.abs().amax(dim=(3, 4)) / 6.0)]
                kc = q_fp4(blk / sdeq[..., None, None])
                Ac = (kc[..., 0] | (kc[..., 1] << 4)).reshape(out_f, ks * 32).to(torch.uint8)
                nib = (i01[..., 0] | (i01[..., 1] << 2)).view(out_f, ks, 2, 8)
                sh = (torch.arange(8, device=dev) * 4).view(1, 1, 1, 8)
                meta = (nib << sh).sum(-1).to(torch.int32).permute(1, 0, 2).contiguous()
                self.register_buffer("Ac", Ac.contiguous())
                self.register_buffer("meta", meta)
                self.register_buffer("scaleA", enc(blk.abs().amax(dim=(3, 4)) / 6.0)
                                     .to(torch.uint8).permute(1, 0, 2).contiguous())
            else:  # dense FP4: emulate via the sparse kernel with an all-kept identity is not
                raise NotImplementedError  # (dense path uses a different kernel; sparse only here)

        def forward(self, x):
            lead = x.shape[:-1]
            x2 = x.reshape(-1, self.in_f).to(torch.bfloat16)
            t = x2.shape[0]
            pad = (-t) % 128
            if pad:
                x2 = torch.cat([x2, x2.new_zeros(pad, self.in_f)], 0)
            x2 = x2.contiguous()
            tp = t + pad
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

    model = load()

    pqf = hf_hub_download("Salesforce/wikitext", "wikitext-2-raw-v1/test-00000-of-00001.parquet",
                          repo_type="dataset")
    text = "\n\n".join(pq.read_table(pqf).column("text").to_pylist())
    ids = tok(text, return_tensors="pt").input_ids[0]

    def ppl(m, seq=2048, windows=16):
        nll, n = 0.0, 0
        for i in range(0, min(len(ids), windows * seq) - seq, seq):
            w = ids[i:i + seq].unsqueeze(0).to(dev)
            with torch.no_grad():
                loss = m(w, labels=w).loss
            nll += loss.item() * (seq - 1); n += seq - 1
        return math.exp(nll / n)

    def mlp_lins(m):
        for layer in m.model.layers:
            for name in ("gate_proj", "up_proj", "down_proj"):
                lin = getattr(layer.mlp, name)
                if lin.weight.shape[0] % 256 == 0 and lin.weight.shape[1] % 256 == 0:
                    yield layer.mlp, name, lin

    p_fp16 = ppl(model)
    print(f"\nPPL fp16 (native NM 2:4 model)            : {p_fp16:.3f}", flush=True)

    # config B: dense FP4 of the same weights, NO re-pruning (torch matmul) -> isolates FP4 error
    n = 0
    for mlp, name, lin in mlp_lins(model):
        lin.weight.data = fp4_only(lin.weight.data).to(torch.bfloat16); n += 1
    p_dense = ppl(model)
    print(f"PPL MLP->dense FP4 (no reprune, {n} layers): {p_dense:.3f}  (delta {p_dense - p_fp16:+.3f})",
          flush=True)
    del model; torch.cuda.empty_cache()

    # config C: our 2:4-sparse FP4 kernel (pair-granular reprune of the element-2:4 weights)
    model = load()
    n = 0
    for mlp, name, lin in mlp_lins(model):
        setattr(mlp, name, QuadbitLinear(lin.weight.data, sparse=True).to(dev)); n += 1
    torch.cuda.empty_cache()
    p_q = ppl(model)
    print(f"PPL MLP->2:4-sparse FP4 ({n} layers)       : {p_q:.3f}  (delta {p_q - p_fp16:+.3f})",
          flush=True)


@app.local_entrypoint()
def main() -> None:
    run.remote()
