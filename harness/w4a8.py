"""W4A8 for decode: fake-quant accuracy check (does 8-bit activations recover the W4A4 gap?).

Decode is memory-bound, so we are NOT compute-limited during decode -> we can afford more precision
on the math for free. Keep weights 4-bit (NVFP4 -> the bandwidth win, the only thing that matters in
decode), but run activations at fp8 (e4m3) instead of fp4. W4A8: 4-bit memory footprint, 8-bit
compute, near-bf16 accuracy; the fp8 compute penalty is hidden because decode is bandwidth-bound.

This is the accuracy half only: does W4A8 recover the accuracy W4A4 gives up? Compare, same model /
eval, dense: bf16 teacher | W4A16 (weight-only) | W4A8 (fp8 acts) | W4A4 (fp4 acts, deployed). If
W4A8 lands near W4A16/bf16, the phase-adaptive idea (W4A4 prefill, W4A8 decode) is worth a kernel.

Run:  uv run modal run harness/w4a8.py --model meta-llama/Meta-Llama-3-8B --all-linears
"""

from pathlib import Path

import modal

MODEL = "meta-llama/Meta-Llama-3-8B"
image = (
    modal.Image.from_registry("nvidia/cuda:12.8.1-devel-ubuntu22.04", add_python="3.12")
    .env({"HF_HOME": "/cache"})
    .pip_install("torch", index_url="https://download.pytorch.org/whl/nightly/cu128", pre=True)
    .pip_install("transformers", "huggingface_hub", "safetensors", "pyarrow")
)
app = modal.App("quadbit-w4a8", image=image)
vol = modal.Volume.from_name("quadbit-hf-cache", create_if_missing=True)


@app.function(gpu="RTX-PRO-6000", timeout=3600, volumes={"/cache": vol},
              secrets=[modal.Secret.from_name("huggingface")])
def run(model: str = MODEL, all_linears: bool = False) -> None:
    import math

    import pyarrow.parquet as pq
    import torch
    import torch.nn as nn
    import torch.nn.functional as F
    from huggingface_hub import hf_hub_download
    from transformers import AutoModelForCausalLM, AutoTokenizer

    dev = torch.device("cuda")
    FP4 = torch.tensor([0, .5, 1, 1.5, 2, 3, 4, 6, 0, -.5, -1, -1.5, -2, -3, -4, -6], device=dev)
    BND = torch.tensor([.25, .75, 1.25, 1.75, 2.5, 3.5, 5.], device=dev)
    _c = torch.arange(128, device=dev); _e, _m = (_c >> 3) & 0xf, _c & 7
    UE4M3 = torch.where(_e == 0, _m.float() * 0.001953125, (1.0 + _m.float() / 8.0) * torch.exp2((_e - 7).float()))

    def q_fp4(v):
        return torch.bucketize(v.abs(), BND) | ((v < 0).long() << 3)

    def enc_ue4m3_t(s):
        mf, e = torch.frexp(s.clamp_min(1e-30)); mant = torch.round((2.0 * mf - 1.0) * 8.0).long()
        biased = (e - 1) + 7; carry = mant == 8
        mant = torch.where(carry, torch.zeros_like(mant), mant); biased = torch.where(carry, biased + 1, biased)
        code = (biased << 3) | mant
        code = torch.where(biased < 1, torch.ones_like(code), code)
        return torch.where((biased > 15) | (s >= 480.0), torch.full_like(code, 0x7f),
                           torch.where(s > 0, code, torch.zeros_like(code)))

    def w_nvfp4(W):  # per-16 two-level NVFP4 weight dequant (deployed 4-bit weights, same for W4A4/W4A8)
        out_f, in_f = W.shape
        gA = (W.abs().amax(-1, keepdim=True) / 2688.0).clamp_min(1e-30)
        b = W.view(out_f, in_f // 16, 16)
        sdeq = UE4M3[enc_ue4m3_t((b.abs().amax(-1) / 6.0) / gA)] * gA
        return (FP4[q_fp4(b / sdeq[..., None])] * sdeq[..., None]).reshape(out_f, in_f)

    def a_fp4(x):  # A4: per-16 two-level NVFP4 activations (the deployed W4A4 path)
        lead = x.shape[:-1]; i = x.shape[-1]
        b = x.to(torch.bfloat16).float().reshape(-1, i)
        gB = (b.abs().amax(-1, keepdim=True) / 2688.0).clamp_min(1e-30)
        bb = b.reshape(b.shape[0], i // 16, 16)
        sdeq = UE4M3[enc_ue4m3_t((bb.abs().amax(-1) / 6.0) / gB)] * gB
        return (FP4[q_fp4(bb / sdeq[..., None])] * sdeq[..., None]).reshape(*lead, i)

    def a_fp8(x):  # A8: fp8 e4m3 activations, per-token scale (near-lossless; the W4A8 math)
        lead = x.shape[:-1]; i = x.shape[-1]
        b = x.to(torch.bfloat16).float().reshape(-1, i)
        scale = (b.abs().amax(-1, keepdim=True) / 448.0).clamp_min(1e-30)  # e4m3 max = 448
        q = (b / scale).to(torch.float8_e4m3fn).float() * scale
        return q.reshape(*lead, i)

    class QLin(nn.Module):  # W4 (nvfp4) + A{4,8,16}
        def __init__(self, weight, aq):
            super().__init__()
            self.register_buffer("Wq", w_nvfp4(weight.float()).to(torch.bfloat16))
            self.aq = aq

        def forward(self, x):
            xq = self.aq(x.float()) if self.aq is not None else x.float()
            return F.linear(xq, self.Wq.float()).to(x.dtype)

    tok = AutoTokenizer.from_pretrained(model)

    def load():
        return AutoModelForCausalLM.from_pretrained(model, dtype=torch.bfloat16).to(dev).eval()

    test_ids = tok("\n\n".join(pq.read_table(hf_hub_download(
        "Salesforce/wikitext", "wikitext-2-raw-v1/test-00000-of-00001.parquet",
        repo_type="dataset")).column("text").to_pylist())).input_ids

    def ppl(m, windows=16, seq=2048):
        nll = n = 0
        for i in range(0, min(len(test_ids), windows * seq) - seq, seq):
            w = torch.tensor(test_ids[i:i + seq], device=dev).unsqueeze(0)
            with torch.no_grad():
                nll += m(w, labels=w).loss.item() * (seq - 1); n += seq - 1
        return math.exp(nll / n)

    def swap(aq):
        m = load()
        for layer in m.model.layers:
            groups = [(layer.mlp, ("gate_proj", "up_proj", "down_proj"))]
            if all_linears:
                groups.append((layer.self_attn, ("q_proj", "k_proj", "v_proj", "o_proj")))
            for parent, names in groups:
                for nm in names:
                    lin = getattr(parent, nm)
                    if lin.weight.shape[0] % 256 == 0 and lin.weight.shape[1] % 256 == 0:
                        setattr(parent, nm, QLin(lin.weight.data, aq).to(dev))
        return m

    teacher = ppl(load())
    w4a16 = ppl(swap(None))
    w4a8 = ppl(swap(a_fp8))
    w4a4 = ppl(swap(a_fp4))
    print(f"\nteacher (bf16)      {teacher:.3f}", flush=True)
    print(f"W4A16 (weight-only) {w4a16:.3f}  (+{w4a16 - teacher:.3f})", flush=True)
    print(f"W4A8  (fp8 acts)    {w4a8:.3f}  (+{w4a8 - teacher:.3f})", flush=True)
    print(f"W4A4  (fp4 acts)    {w4a4:.3f}  (+{w4a4 - teacher:.3f})", flush=True)
    print(f"RESULT teacher={teacher:.3f} w4a16={w4a16:.3f} w4a8={w4a8:.3f} w4a4={w4a4:.3f} "
          f"(W4A8 recovers {(w4a4 - w4a8) / max(w4a4 - w4a16, 1e-6) * 100:.0f}% of the A4->A16 gap)", flush=True)


@app.local_entrypoint()
def main(model: str = MODEL, all_linears: bool = False) -> None:
    run.remote(model=model, all_linears=all_linears)
