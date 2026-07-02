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
    .pip_install("transformers", "huggingface_hub", "safetensors", "pyarrow", "numpy")
)
app = modal.App("quadbit-sensitivity", image=image)
vol = modal.Volume.from_name("quadbit-hf-cache", create_if_missing=True)


@app.function(gpu="RTX-PRO-6000", timeout=3600, volumes={"/cache": vol},
              secrets=[modal.Secret.from_name("huggingface")])
def run(model: str = MODEL, all_linears: bool = False, side: str = "act") -> None:  # side: "act" | "weight"
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

    class QLin(nn.Module):  # a16 -> bf16 acts (W4A16); w16 -> bf16 weight (W16A4). both flags independent
        def __init__(self, weight):
            super().__init__()
            self.register_buffer("Wq", w_nvfp4(weight.float()).to(torch.bfloat16))
            self.register_buffer("Wb", weight.to(torch.bfloat16))  # bf16 weight for the weight-side sweep
            self.a16 = self.w16 = False

        def forward(self, x):
            W = self.Wb.float() if self.w16 else self.Wq.float()
            xq = x.float() if self.a16 else a_fp4(x.float())
            return F.linear(xq, W).to(x.dtype)

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

    def ppl(windows, seq=2048):  # HELD-OUT eval: WikiText-2 test (recipe scored here, ONCE)
        nll = n = 0
        for i in range(0, min(len(test_ids), windows * seq) - seq, seq):
            w = torch.tensor(test_ids[i:i + seq], device=dev).unsqueeze(0)
            with torch.no_grad():
                nll += m(w, labels=w).loss.item() * (seq - 1); n += seq - 1
        return math.exp(nll / n)

    import numpy as np
    c4 = torch.from_numpy(np.load("/cache/corpus_c4_llama3_smoke/shard_0000.npy")[:16]).long().to(dev)

    def ppl_c4():  # SELECTION set: decontaminated C4, DISJOINT from the WT-2 test set (no selection-on-test)
        nll = n = 0
        for w in c4:
            wv = w.unsqueeze(0)
            with torch.no_grad():
                nll += m(wv, labels=wv).loss.item() * (wv.shape[1] - 1); n += wv.shape[1] - 1
        return math.exp(nll / n)

    attr = "a16" if side == "act" else "w16"   # act side: bf16 activations; weight side: bf16 weights
    hp = "W4A16" if side == "act" else "W16A4"

    def set_all(v):
        for mods in per_layer:
            for q in mods:
                setattr(q, attr, v)

    set_all(False); base_lo = ppl(16)          # W4A4 (all)
    set_all(True); base_hi = ppl(16); set_all(False)   # all-high floor: act->W4A16, weight->W16A4
    cost = base_lo - base_hi
    print(f"\nW4A4(all) {base_lo:.3f}   {hp}(all) {base_hi:.3f}   {side} cost = {cost:.3f}", flush=True)

    base_c4 = ppl_c4()                          # rank layers on C4 (selection set), NOT on the WT-2 test set
    rec = []
    for i, mods in enumerate(per_layer):
        for q in mods:
            setattr(q, attr, True)
        p = ppl_c4()
        for q in mods:
            setattr(q, attr, False)
        rec.append((base_c4 - p, i))            # C4-selection recovery (disjoint from the final WT-2 eval)
    rec.sort(reverse=True)
    print(f"top {side}-sensitive layers (C4-selection recovery vs C4 baseline {base_c4:.3f}):", flush=True)
    for r, i in rec[:10]:
        print(f"  layer {i:2d}: {r:+.3f}", flush=True)

    ranked = [i for _, i in rec]
    for k in (2, 4, 8):
        set_all(False)
        for i in ranked[:k]:
            for q in per_layer[i]:
                setattr(q, attr, True)
        p = ppl(16)
        thr = 6.828  # teacher 6.198 + 0.63
        print(f"top-{k} layers @{hp}, rest W4A4: {p:.3f}  (recovery {base_lo - p:+.3f} of {cost:.3f}; "
              f"{'BELOW' if p < thr else 'above'} +0.63 line {thr})", flush=True)
    print(f"RESULT side={side} base_lo={base_lo:.3f} base_hi={base_hi:.3f} top_layers={ranked[:8]}", flush=True)


@app.local_entrypoint()
def main(model: str = MODEL, all_linears: bool = False, side: str = "act") -> None:
    run.remote(model=model, all_linears=all_linears, side=side)
