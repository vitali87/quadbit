"""Real-weight accuracy of the 2:4-sparse FP4 scheme, measured on a real HuggingFace model
(TinyLlama-1.1B, fully open, FFN dims all %256 so they fit the kernel directly).

Every prior accuracy number was on random gaussian weights = the structural 2:4-pruning floor
(~0.5, since pruning removes half of uncorrelated values). This measures REAL trained weights
and separates the two error sources: FP4 quantization alone (no pruning) vs 2:4-pruning+FP4.
The kernel output equals the dequant path (verified maxrel 0.003 in quadbit_linear.py), so this
runs in pure torch -- no kernel needed for an accuracy study.

Run:  uv run modal run harness/accuracy_hf.py
"""

from pathlib import Path

import modal

ROOT = Path(__file__).parent.parent
MODEL = "TinyLlama/TinyLlama-1.1B-Chat-v1.0"

image = (
    modal.Image.from_registry("nvidia/cuda:12.8.1-devel-ubuntu22.04", add_python="3.12")
    .pip_install("torch", index_url="https://download.pytorch.org/whl/nightly/cu128", pre=True)
    .pip_install("huggingface_hub", "safetensors")
)

app = modal.App("quadbit-accuracy", image=image)


@app.function(gpu="RTX-PRO-6000", timeout=1800)
def run() -> None:
    import torch
    from huggingface_hub import hf_hub_download
    from safetensors.torch import load_file

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

    def fp4_only_dequant(W):  # per-16-block NVFP4, NO pruning -> isolates FP4 quant error
        out_f, in_f = W.shape
        b = W.view(out_f, in_f // 16, 16)
        s = UE4M3[enc(b.abs().amax(-1) / 6.0)]                  # [out, in/16]
        return (FP4[q_fp4(b / s[..., None])] * s[..., None]).reshape(out_f, in_f)

    def sparse_fp4_dequant(W):  # 2:4-by-magnitude + NVFP4 (exactly what QuadbitLinear packs)
        out_f, in_f = W.shape
        ks = in_f // 128
        Wg = W.view(out_f, ks, 16, 4, 2)
        i01, _ = Wg.abs().sum(-1).topk(2, dim=-1).indices.sort(dim=-1)
        kept = i01                                              # [out,ks,16,2] pair indices
        keptW = torch.gather(Wg, 3, kept.unsqueeze(-1).expand(-1, -1, -1, -1, 2))
        blk = keptW.reshape(out_f, ks, 4, 8, 2)
        sdeq = UE4M3[enc(blk.abs().amax(dim=(3, 4)) / 6.0)]
        kc = q_fp4(blk / sdeq[..., None, None])
        kd = (FP4[kc] * sdeq[..., None, None]).reshape(out_f, ks, 16, 2, 2)
        Wd = torch.zeros(out_f, ks, 16, 4, 2, device=dev)
        Wd.scatter_(3, kept.unsqueeze(-1).expand(-1, -1, -1, -1, 2), kd)
        return Wd.reshape(out_f, in_f)

    def rel(a, b):
        return (a - b).norm().item() / b.norm().item()

    print(f"downloading {MODEL} ...", flush=True)
    sd = load_file(hf_hub_download(MODEL, "model.safetensors"))
    torch.manual_seed(0)

    print(f"\n# per-weight reconstruction error (real trained weights vs random gaussian)")
    print(f"{'tensor':<34}{'FP4-only':>10}{'2:4+FP4':>10}{'rand 2:4+FP4':>14}", flush=True)
    projs = ["gate_proj", "up_proj", "down_proj"]
    agg = {p: [] for p in projs}
    for layer in (0, 5, 11, 16, 21):
        for p in projs:
            W = sd[f"model.layers.{layer}.mlp.{p}.weight"].float().to(dev)
            e_fp4 = rel(fp4_only_dequant(W), W)
            e_sp = rel(sparse_fp4_dequant(W), W)
            Wr = torch.randn_like(W) * W.std()
            e_rand = rel(sparse_fp4_dequant(Wr), Wr)
            agg[p].append((e_fp4, e_sp, e_rand))
            print(f"L{layer:<2} {p:<28}{e_fp4:>10.3f}{e_sp:>10.3f}{e_rand:>14.3f}", flush=True)

    print(f"\n# layer-output error on the down_proj (real W, real-input-like acts, incl act-quant)")
    W = sd["model.layers.0.mlp.down_proj.weight"].float().to(dev)   # [2048, 5632]
    Wq = sparse_fp4_dequant(W)
    for batch in (512, 2048):
        x = torch.randn(batch, W.shape[1], device=dev)
        # activations: per-32-block NVFP4 (same scheme as the fused kernel quantizer)
        xb = x.view(batch, W.shape[1] // 32, 32)
        xs = UE4M3[enc(xb.abs().amax(-1) / 6.0)]
        xq = (FP4[q_fp4(xb / xs[..., None])] * xs[..., None]).reshape(batch, W.shape[1])
        print(f"  batch={batch}: rel(x_q @ Wq^T , x @ W^T) = {rel(xq @ Wq.t(), x @ W.t()):.3f}", flush=True)


@app.local_entrypoint()
def main() -> None:
    run.remote()
