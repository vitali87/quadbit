"""Training-free activation smoothing (SmoothQuant) for W4A4, co-designed with 2:4 selection.

W4A4's accuracy cost is dominated by activation outliers, not weights. SmoothQuant migrates the
per-input-channel scale from activations into weights before quantizing: x' = x/s, W' = W·s,
s_j = max|x_j|^α / max|W_j|^(1-α). Flattens act outliers into the weights, where per-16 block
scaling already handles them. No training; minutes of calibration on the decontaminated C4 smoke
corpus (clean, no wikitext-2-test leakage into the act stats).

NOVEL co-design: the sparse path keeps 2-of-4 fp4 PAIRS by magnitude. Smoothing changes those
magnitudes, so we ALSO select the kept pairs by ACTIVATION-WEIGHTED magnitude (|W|·max|x|) instead
of raw |W| — pushing activation mass onto the pairs 2:4 will keep. Nobody else has this knob
(nobody else has this pair-granular FP4 packer). We measure selection-aware vs plain smoothing.

Reports dense W4A4 and one-shot sparse W4A4, no-smooth vs α∈{0.5,0.75,0.9} vs selection-aware,
against the current dense 6.91 / one-shot-sparse baseline.

Run:  uv run modal run harness/smooth_quant.py --model meta-llama/Meta-Llama-3-8B --all-linears
"""

from pathlib import Path

import modal

ROOT = Path(__file__).parent.parent
MODEL = "meta-llama/Meta-Llama-3-8B"
SMOKE_CORPUS = "/cache/corpus_c4_llama3_smoke/shard_0000.npy"  # decontaminated C4 (from build_corpus)

image = (
    modal.Image.from_registry("nvidia/cuda:12.8.1-devel-ubuntu22.04", add_python="3.12")
    .env({"HF_HOME": "/cache"})
    .pip_install("torch", index_url="https://download.pytorch.org/whl/nightly/cu128", pre=True)
    .pip_install("transformers", "huggingface_hub", "safetensors", "sentencepiece", "pyarrow", "numpy")
)
app = modal.App("quadbit-smooth", image=image)
vol = modal.Volume.from_name("quadbit-hf-cache", create_if_missing=True)


@app.function(gpu="RTX-PRO-6000", timeout=3600, volumes={"/cache": vol},
              secrets=[modal.Secret.from_name("huggingface")])
def run(model: str = MODEL, all_linears: bool = False) -> None:
    import math
    import os

    import numpy as np
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

    def enc_ue4m3_t(s):  # no-denormal ue4m3, deploy-matched
        mf, e = torch.frexp(s.clamp_min(1e-30)); mant = torch.round((2.0 * mf - 1.0) * 8.0).long()
        biased = (e - 1) + 7; carry = mant == 8
        mant = torch.where(carry, torch.zeros_like(mant), mant); biased = torch.where(carry, biased + 1, biased)
        code = (biased << 3) | mant
        code = torch.where(biased < 1, torch.ones_like(code), code)
        return torch.where((biased > 15) | (s >= 480.0), torch.full_like(code, 0x7f),
                           torch.where(s > 0, code, torch.zeros_like(code)))

    def act_fp4_dequant(x):  # per-16 two-level NVFP4 (deploy-matched)
        lead = x.shape[:-1]; i = x.shape[-1]
        b = x.to(torch.bfloat16).float().reshape(-1, i)
        gB = (b.abs().amax(-1, keepdim=True) / 2688.0).clamp_min(1e-30)
        bb = b.reshape(b.shape[0], i // 16, 16)
        sdeq = UE4M3[enc_ue4m3_t((bb.abs().amax(-1) / 6.0) / gB)] * gB
        return (FP4[q_fp4(bb / sdeq[..., None])] * sdeq[..., None]).reshape(*lead, i)

    def dense_fp4_dequant(W):  # per-16 two-level NVFP4, all weights kept
        out_f, in_f = W.shape
        gA = (W.abs().amax(-1, keepdim=True) / 2688.0).clamp_min(1e-30)
        b = W.view(out_f, in_f // 16, 16)
        sdeq = UE4M3[enc_ue4m3_t((b.abs().amax(-1) / 6.0) / gA)] * gA
        return (FP4[q_fp4(b / sdeq[..., None])] * sdeq[..., None]).reshape(out_f, in_f)

    def sparse_fp4_dequant(W, act=None):  # pair-2:4 + per-16 NVFP4. act!=None -> selection-aware pairs
        out_f, in_f = W.shape; ks = in_f // 128
        Wg = W.view(out_f, ks, 16, 4, 2)
        score = Wg.abs() if act is None else Wg.abs() * act.view(1, ks, 16, 4, 2)  # act-weighted magnitude
        i01, _ = score.sum(-1).topk(2, dim=-1).indices.sort(dim=-1)
        keptW = torch.gather(Wg, 3, i01.unsqueeze(-1).expand(-1, -1, -1, -1, 2))
        gA = (keptW.abs().amax(dim=(1, 2, 3, 4), keepdim=True) / 2688.0).clamp_min(1e-30).reshape(out_f, 1, 1)
        blk = keptW.reshape(out_f, ks, 4, 8, 2)
        sdeq = UE4M3[enc_ue4m3_t((blk.abs().amax(dim=(3, 4)) / 6.0) / gA)] * gA
        kd = (FP4[q_fp4(blk / sdeq[..., None, None])] * sdeq[..., None, None]).reshape(out_f, ks, 16, 2, 2)
        Wd = torch.zeros(out_f, ks, 16, 4, 2, device=dev)
        Wd.scatter_(3, i01.unsqueeze(-1).expand(-1, -1, -1, -1, 2), kd)
        return Wd.reshape(out_f, in_f)

    class SmoothFakeQuant(nn.Module):  # W'=W·s pre-quantized; forward x'=x/s then act-quant
        def __init__(self, weight, s, sparse, sel_aware):
            super().__init__()
            self.register_buffer("s", s)                          # per-input-channel [in]
            Ws = weight.float() * s[None, :]
            act = (s if sel_aware else None)                      # sel-aware: weight pair pick by act (=s proxy)
            Wq = sparse_fp4_dequant(Ws, act) if sparse else dense_fp4_dequant(Ws)
            self.register_buffer("Wq", Wq.to(torch.bfloat16))

        def forward(self, x):
            return F.linear(act_fp4_dequant(x.float() / self.s), self.Wq.float()).to(x.dtype)

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

    def targets(m):
        for layer in m.model.layers:
            groups = [(layer.mlp, ("gate_proj", "up_proj", "down_proj"))]
            if all_linears:
                groups.append((layer.self_attn, ("q_proj", "k_proj", "v_proj", "o_proj")))
            for parent, names in groups:
                for nm in names:
                    lin = getattr(parent, nm)
                    if lin.weight.shape[0] % 256 == 0 and lin.weight.shape[1] % 256 == 0:
                        yield parent, nm, lin

    m = load()
    tgt = list(targets(m))
    origs = {(id(p), nm): lin.weight.data.clone() for p, nm, lin in tgt}

    # --- calibration: per-input-channel act_max on decontaminated C4 (no test leakage) ---
    if os.path.exists(SMOKE_CORPUS):
        calib = torch.tensor(np.load(SMOKE_CORPUS)[:32], device=dev)  # 32 seq-1024 C4 windows
    else:  # fallback: C4 stream not staged yet -> use wikitext-103 TRAIN (disjoint from wt-2 test)
        p = hf_hub_download("Salesforce/wikitext", "wikitext-103-raw-v1/train-00000-of-00002.parquet",
                            repo_type="dataset")
        ci = tok("\n\n".join(pq.read_table(p).column("text").to_pylist()[:2000])).input_ids
        calib = torch.tensor(ci[:32 * 1024], device=dev).reshape(32, 1024)
    amax = {}

    def hook(mod, inp, _o):
        x = inp[0].detach().abs().reshape(-1, inp[0].shape[-1]).amax(0).float()
        k = id(mod)
        amax[k] = torch.maximum(amax[k], x) if k in amax else x

    hs = [lin.register_forward_hook(hook) for _, _, lin in tgt]
    with torch.no_grad():
        for i in range(0, 32, 4):
            m(calib[i:i + 4])
    for h in hs:
        h.remove()
    print(f"calibrated per-channel act_max on {calib.shape[0]} C4 windows", flush=True)

    def smooth_s(lin, alpha):
        a = amax[id(lin)].clamp_min(1e-5)
        wmax = lin.weight.data.abs().amax(0).float().clamp_min(1e-5)  # per-input-channel weight max
        return (a.pow(alpha) / wmax.pow(1 - alpha)).clamp(1e-4, 1e4)

    def measure(sparse, alpha, sel_aware):
        for p, nm, lin in tgt:
            W = origs[(id(p), nm)]
            s = smooth_s(lin, alpha) if alpha > 0 else torch.ones(W.shape[1], device=dev)
            setattr(p, nm, SmoothFakeQuant(W, s, sparse, sel_aware).to(dev))
        r = ppl(m)
        for p, nm, _ in tgt:  # restore originals for the next config
            lin = nn.Linear(origs[(id(p), nm)].shape[1], origs[(id(p), nm)].shape[0], bias=False).to(dev, torch.bfloat16)
            lin.weight.data = origs[(id(p), nm)]; setattr(p, nm, lin)
        return r

    teacher = ppl(m)
    print(f"\nPPL teacher (bf16): {teacher:.3f}", flush=True)
    print(f"{'config':<34}{'dense W4A4':>12}{'sparse W4A4':>13}", flush=True)
    d0, s0 = measure(False, 0.0, False), measure(True, 0.0, False)
    print(f"{'no smoothing':<34}{d0:>12.3f}{s0:>13.3f}", flush=True)
    for a in (0.5, 0.75, 0.9):
        d, s = measure(False, a, False), measure(True, a, False)
        print(f"{'smooth alpha=' + str(a):<34}{d:>12.3f}{s:>13.3f}", flush=True)
    # selection-aware sparse at the best-looking alpha (0.75 default sweet spot)
    ssa = measure(True, 0.75, True)
    print(f"{'smooth a=0.75 + selection-aware':<34}{'':>12}{ssa:>13.3f}", flush=True)
    print(f"RESULT teacher={teacher:.3f} dense_nosmooth={d0:.3f} sparse_nosmooth={s0:.3f} "
          f"sparse_sel_aware_a0.75={ssa:.3f}", flush=True)


@app.local_entrypoint()
def main(model: str = MODEL, all_linears: bool = False) -> None:
    run.remote(model=model, all_linears=all_linears)
