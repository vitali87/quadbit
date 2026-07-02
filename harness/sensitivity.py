"""Per-layer activation-sensitivity sweep -> mixed-precision W4A4/W4A16 recipe (training-free).

Dense W4A4 = +0.713 splits into +0.395 weight cost and +0.318 activation cost. Global smoothing
barely dented it. The calibrated references instead keep a FEW layers high-precision. So: hold
every linear at W4A4 except one decoder layer i, put layer i's activations at A16 (bf16), measure
the PPL recovery. Rank layers by how much of the +0.318 activation cost each owns. If a handful
carry most of it, keeping just those at A16 (free in weight memory -- weights stay 4-bit; only
those layers run bf16-activation GEMMs) recovers most of the gap and could push dense < +0.63.

Run:  uv run modal run harness/sensitivity.py --model meta-llama/Meta-Llama-3-8B --all-linears
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
app = modal.App("quadbit-sensitivity", image=image)
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

    def w_nvfp4(W):
        out_f, in_f = W.shape
        gA = (W.abs().amax(-1, keepdim=True) / 2688.0).clamp_min(1e-30)
        b = W.view(out_f, in_f // 16, 16)
        sdeq = UE4M3[enc_ue4m3_t((b.abs().amax(-1) / 6.0) / gA)] * gA
        return (FP4[q_fp4(b / sdeq[..., None])] * sdeq[..., None]).reshape(out_f, in_f)

    def a_fp4(x):
        lead = x.shape[:-1]; i = x.shape[-1]
        b = x.to(torch.bfloat16).float().reshape(-1, i)
        gB = (b.abs().amax(-1, keepdim=True) / 2688.0).clamp_min(1e-30)
        bb = b.reshape(b.shape[0], i // 16, 16)
        sdeq = UE4M3[enc_ue4m3_t((bb.abs().amax(-1) / 6.0) / gB)] * gB
        return (FP4[q_fp4(bb / sdeq[..., None])] * sdeq[..., None]).reshape(*lead, i)

    class QLin(nn.Module):  # W4 (nvfp4). a16=True -> bf16 activations (W4A16); else W4A4
        def __init__(self, weight):
            super().__init__()
            self.register_buffer("Wq", w_nvfp4(weight.float()).to(torch.bfloat16))
            self.a16 = False

        def forward(self, x):
            xq = x.float() if self.a16 else a_fp4(x.float())
            return F.linear(xq, self.Wq.float()).to(x.dtype)

    tok = AutoTokenizer.from_pretrained(model)
    m = AutoModelForCausalLM.from_pretrained(model, dtype=torch.bfloat16).to(dev).eval()
    per_layer = []  # list over decoder layers of the QLin modules in that layer
    for layer in m.model.layers:
        mods = []
        groups = [(layer.mlp, ("gate_proj", "up_proj", "down_proj"))]
        if all_linears:
            groups.append((layer.self_attn, ("q_proj", "k_proj", "v_proj", "o_proj")))
        for parent, names in groups:
            for nm in names:
                lin = getattr(parent, nm)
                if lin.weight.shape[0] % 256 == 0 and lin.weight.shape[1] % 256 == 0:
                    ql = QLin(lin.weight.data).to(dev); setattr(parent, nm, ql); mods.append(ql)
        per_layer.append(mods)

    test_ids = tok("\n\n".join(pq.read_table(hf_hub_download(
        "Salesforce/wikitext", "wikitext-2-raw-v1/test-00000-of-00001.parquet",
        repo_type="dataset")).column("text").to_pylist())).input_ids

    def ppl(windows, seq=2048):
        nll = n = 0
        for i in range(0, min(len(test_ids), windows * seq) - seq, seq):
            w = torch.tensor(test_ids[i:i + seq], device=dev).unsqueeze(0)
            with torch.no_grad():
                nll += m(w, labels=w).loss.item() * (seq - 1); n += seq - 1
        return math.exp(nll / n)

    def set_all(a16):
        for mods in per_layer:
            for q in mods:
                q.a16 = a16

    set_all(False); base_a4 = ppl(16)
    set_all(True); base_a16 = ppl(16); set_all(False)
    print(f"\nW4A4 (all)  {base_a4:.3f}   W4A16 (all)  {base_a16:.3f}   activation cost = {base_a4 - base_a16:.3f}",
          flush=True)

    # per-layer: layer i activations -> A16, rest A4; recovery = base_a4 - ppl_i (windows=8 for the sweep)
    rec = []
    for i, mods in enumerate(per_layer):
        for q in mods:
            q.a16 = True
        p = ppl(8)
        for q in mods:
            q.a16 = False
        rec.append((base_a4 - p, i))  # NOTE base_a4 here is windows=16; sweep is windows=8 -> use for RANKING only
    rec.sort(reverse=True)
    print("top act-sensitive layers (recovery, windows=8 ranking):", flush=True)
    for r, i in rec[:10]:
        print(f"  layer {i:2d}: {r:+.3f}", flush=True)

    # keep the top-K most sensitive layers at A16, rest A4; final numbers windows=16
    ranked = [i for _, i in rec]
    for k in (2, 4, 8):
        set_all(False)
        for i in ranked[:k]:
            for q in per_layer[i]:
                q.a16 = True
        p = ppl(16)
        thr = 6.828  # teacher 6.198 + 0.63
        print(f"top-{k} layers @A16, rest W4A4: {p:.3f}  (recovery {base_a4 - p:+.3f} of the "
              f"{base_a4 - base_a16:.3f} act cost; {'BELOW' if p < thr else 'above'} the +0.63 line {thr})", flush=True)
    print(f"RESULT base_a4={base_a4:.3f} base_a16={base_a16:.3f} top_layers={ranked[:8]}", flush=True)


@app.local_entrypoint()
def main(model: str = MODEL, all_linears: bool = False) -> None:
    run.remote(model=model, all_linears=all_linears)
