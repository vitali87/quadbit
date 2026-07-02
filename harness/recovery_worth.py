"""#4 -- the sparse-net-of-recovery gate: is 2:4-sparse FP4 worth its recovery cost, given
dense FP4 is a zero-training drop-in? Same base model, same wikitext-2 eval, same deploy-matched
activation quant; the ONLY variable is the weight scheme on the same %256 MLP linears:

  teacher (bf16)              -- the ceiling
  dense FP4, zero-train       -- all weights kept, 4-bit (the free alternative, ~3x over bf16)
  one-shot pair-2:4 FP4       -- half the pairs dropped + 4-bit, NO recovery (the starting point)
  [recovered pair-2:4 FP4]    -- after KD+QAT (measured separately by finetune_pair.py; ~4.5x)

If dense-FP4-zero-train already beats recovered sparse, the sparse path only earns its keep on
the extra ~1.5x speed -- and only if recovery at scale closes the accuracy gap. This decides
whether to keep investing in recovery. Fake-quant matched to the kernel (tracks through-kernel
PPL within ~0.04, established by the matched-STE run), so the PPL ordering here is trustworthy.

Run:  uv run modal run harness/recovery_worth.py                         # TinyLlama base
      uv run modal run harness/recovery_worth.py --model meta-llama/Meta-Llama-3-8B
"""

from pathlib import Path

import modal

ROOT = Path(__file__).parent.parent
MODEL = "TinyLlama/TinyLlama-1.1B-intermediate-step-1431k-3T"  # same base as finetune_pair

image = (
    modal.Image.from_registry("nvidia/cuda:12.8.1-devel-ubuntu22.04", add_python="3.12")
    .env({"HF_HOME": "/cache"})
    .pip_install("torch", index_url="https://download.pytorch.org/whl/nightly/cu128", pre=True)
    .pip_install("transformers", "huggingface_hub", "safetensors", "sentencepiece", "pyarrow")
)
app = modal.App("quadbit-recovery-worth", image=image)
vol = modal.Volume.from_name("quadbit-hf-cache", create_if_missing=True)


@app.function(gpu="RTX-PRO-6000", timeout=3600, volumes={"/cache": vol},
              secrets=[modal.Secret.from_name("huggingface")])
def run(model: str = MODEL, all_linears: bool = False) -> None:  # all_linears: +attn q/k/v/o (modelopt scope)
    import math

    import pyarrow.parquet as pq
    import torch
    import torch.nn as nn
    import torch.nn.functional as F
    from huggingface_hub import hf_hub_download
    from transformers import AutoModelForCausalLM, AutoTokenizer

    dev = torch.device("cuda")
    FP4 = torch.tensor([0, .5, 1, 1.5, 2, 3, 4, 6, 0, -.5, -1, -1.5, -2, -3, -4, -6],
                       dtype=torch.float32, device=dev)
    BND = torch.tensor([.25, .75, 1.25, 1.75, 2.5, 3.5, 5.], dtype=torch.float32, device=dev)
    _cc = torch.arange(128, device=dev)
    _e, _m = (_cc >> 3) & 0xf, _cc & 7
    UE4M3 = torch.where(_e == 0, _m.float() * 0.001953125,
                        (1.0 + _m.float() / 8.0) * torch.exp2((_e - 7).float()))
    _MID = (UE4M3[:-1] + UE4M3[1:]) / 2

    def q_fp4(v):
        return torch.bucketize(v.abs(), BND) | ((v < 0).long() << 3)

    def enc(s):
        return torch.bucketize(s, _MID)

    def enc_ue4m3_t(s):  # no-denormal ue4m3, matches the kernel's activation scale encode
        mant_f, e = torch.frexp(s.clamp_min(1e-30))
        mm = 2.0 * mant_f
        biased = (e - 1) + 7
        mant = torch.round((mm - 1.0) * 8.0).long()
        carry = mant == 8
        mant = torch.where(carry, torch.zeros_like(mant), mant)
        biased = torch.where(carry, biased + 1, biased)
        code = (biased.long() << 3) | mant
        code = torch.where(biased < 1, torch.ones_like(code), code)
        code = torch.where(biased > 15, torch.full_like(code, 0x7f), code)
        code = torch.where(s >= 480.0, torch.full_like(code, 0x7f), code)
        return torch.where(s > 0, code, torch.zeros_like(code))

    def act_fp4_dequant(x):  # per-16 TWO-LEVEL NVFP4 (per-row global gB + per-16 local), matches modelopt
        lead = x.shape[:-1]; i = x.shape[-1]
        b = x.to(torch.bfloat16).float().reshape(-1, i)
        gB = (b.abs().amax(-1, keepdim=True) / 2688.0).clamp_min(1e-30)  # 2688 = e4m3max 448 * e2m1max 6
        bb = b.reshape(b.shape[0], i // 16, 16)
        sdeq = UE4M3[enc_ue4m3_t((bb.abs().amax(-1) / 6.0) / gB)] * gB
        return (FP4[q_fp4(bb / sdeq[..., None])] * sdeq[..., None]).reshape(*lead, i)

    def dense_fp4_dequant(W):  # TWO-LEVEL NVFP4 (deployed dense accuracy path): per-row fp32
        out_f, in_f = W.shape   # global gA=rowamax/2688 rescales per-16 ue4m3 local into e4m3 range.
        gA = (W.abs().amax(-1, keepdim=True) / 2688.0).clamp_min(1e-30)  # 2688 = e4m3max 448 * e2m1max 6
        b = W.view(out_f, in_f // 16, 16)
        local = (b.abs().amax(-1) / 6.0) / gA                           # [out,in/16], into e4m3 range
        sdeq = UE4M3[enc_ue4m3_t(local)] * gA                           # effective per-16 scale
        return (FP4[q_fp4(b / sdeq[..., None])] * sdeq[..., None]).reshape(out_f, in_f)

    def sparse_fp4_dequant(W):  # pair-granular 2:4 + NVFP4 (what QuadbitLinear packs)
        out_f, in_f = W.shape; ks = in_f // 128
        Wg = W.view(out_f, ks, 16, 4, 2)
        i01, _ = Wg.abs().sum(-1).topk(2, dim=-1).indices.sort(dim=-1)
        keptW = torch.gather(Wg, 3, i01.unsqueeze(-1).expand(-1, -1, -1, -1, 2))
        blk = keptW.reshape(out_f, ks, 4, 8, 2)
        sdeq = UE4M3[enc(blk.abs().amax(dim=(3, 4)) / 6.0)]
        kd = (FP4[q_fp4(blk / sdeq[..., None, None])] * sdeq[..., None, None]).reshape(out_f, ks, 16, 2, 2)
        Wd = torch.zeros(out_f, ks, 16, 4, 2, device=dev)
        Wd.scatter_(3, i01.unsqueeze(-1).expand(-1, -1, -1, -1, 2), kd)
        return Wd.reshape(out_f, in_f)

    class FakeQuant(nn.Module):  # eval-only: fake-quant weight (per scheme); act FP4 (W4A4) or bf16 (W4A16)
        def __init__(self, weight, wdequant, quant_act=True):
            super().__init__()
            self.register_buffer("Wq", wdequant(weight.float()).to(torch.bfloat16))
            self.quant_act = quant_act

        def forward(self, x):
            xq = act_fp4_dequant(x.float()) if self.quant_act else x.float()
            return F.linear(xq, self.Wq.float()).to(x.dtype)

    tok = AutoTokenizer.from_pretrained(model)
    test_ids = tok("\n\n".join(pq.read_table(
        hf_hub_download("Salesforce/wikitext", "wikitext-2-raw-v1/test-00000-of-00001.parquet",
                        repo_type="dataset")).column("text").to_pylist()),
        return_tensors="pt").input_ids[0]

    def load():
        return AutoModelForCausalLM.from_pretrained(model, dtype=torch.bfloat16).to(dev).eval()

    def ppl(m, windows=16, seq=2048):
        nll, n = 0.0, 0
        for i in range(0, min(len(test_ids), windows * seq) - seq, seq):
            w = test_ids[i:i + seq].unsqueeze(0).to(dev)
            with torch.no_grad():
                nll += m(w, labels=w).loss.item() * (seq - 1); n += seq - 1
        return math.exp(nll / n)

    def targets(m):
        for layer in m.model.layers:
            groups = [(layer.mlp, ("gate_proj", "up_proj", "down_proj"))]
            if all_linears:  # match modelopt: quantize attn projections too (compute stays bf16)
                groups.append((layer.self_attn, ("q_proj", "k_proj", "v_proj", "o_proj")))
            for parent, names in groups:
                for nm in names:
                    lin = getattr(parent, nm)
                    if lin.weight.shape[0] % 256 == 0 and lin.weight.shape[1] % 256 == 0:
                        yield parent, nm, lin

    def swap(dequant, quant_act=True):
        m = load()
        for parent, nm, lin in list(targets(m)):
            setattr(parent, nm, FakeQuant(lin.weight.data, dequant, quant_act).to(dev))
        return m

    teacher = ppl(load())
    print(f"PPL teacher (bf16):                     {teacher:.3f}", flush=True)
    dense_w4a16 = ppl(swap(dense_fp4_dequant, quant_act=False))  # weight-only: what the +0.3 claim is
    print(f"PPL dense FP4 W4A16 (weight-only):      {dense_w4a16:.3f}   (+{dense_w4a16 - teacher:.3f})", flush=True)
    dense = ppl(swap(dense_fp4_dequant))                         # W4A4: what the fp4 tensor core RUNS
    print(f"PPL dense FP4 W4A4 (deployed kernel):   {dense:.3f}   (+{dense - teacher:.3f})", flush=True)
    onesh = ppl(swap(sparse_fp4_dequant))
    print(f"PPL one-shot pair-2:4 FP4 (W4A4):       {onesh:.3f}   (+{onesh - teacher:.3f})", flush=True)
    print(f"\nverdict: FP4 tensor core is W4A4. Weight-only W4A16 (+{dense_w4a16 - teacher:.3f}) is NOT what "
          f"ships; deployed dense W4A4 costs +{dense - teacher:.3f}. Sparse must beat {dense:.3f} after recovery.",
          flush=True)
    print(f"RESULT teacher={teacher:.3f} w4a16={dense_w4a16:.3f} dense_w4a4={dense:.3f} oneshot={onesh:.3f}",
          flush=True)


@app.local_entrypoint()
def main(model: str = MODEL, all_linears: bool = False) -> None:
    run.remote(model=model, all_linears=all_linears)
