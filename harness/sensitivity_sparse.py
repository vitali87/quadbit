"""Hybrid sparse PLACEMENT sweep -> where can 2:4 sparse NVFP4 go for free? (training-free).

All-sparse recovered 8B is ~1.56 PPL behind dense W4A4 (8.47 vs 6.91). But sparse is ~1.33x
faster. Question: does a HYBRID exist that keeps the accuracy-critical matrices dense and pushes
the insensitive ones to sparse, capturing a meaningful fraction of the sparse speedup while
staying close to dense accuracy? This is the make-or-break systems result.

Method (mirrors sensitivity.py, but the perturbation axis is dense->sparse, not W4A4->W4A16):
  baseline    = dense two-level W4A4 fake-quant on every target linear.
  per candidate (layer i, matrix type m): SparseGPT pair-2:4 prune THAT matrix (Hessian
              compensation, one-shot, no training), two-level fake-quant the kept pairs, measure
              PPL delta on the C4 SELECTION set (disjoint from the WT-2 test set: no selection-on-test).
  rank candidates by delta; sparsify least-damaging-first; report the Pareto: how many matrices
  can be sparse before cumulative delta-PPL (scored ONCE on held-out WT-2 test) crosses
  0.05 / 0.10 / 0.25 / 0.50, plus the all-sparse control. Est prefill speedup at each point from
  the sparse-FLOP fraction and the measured deployed sparse-vs-dense roofline ratio (1.33x).

The sweep is fake-quant (the standard way to RANK placement cheaply). Deploy-verify of the chosen
Pareto points through the real two-level sparse kernel + deploy gap is the follow-on run.

Run:  uv run modal run harness/sensitivity_sparse.py --all-linears
"""

import modal

MODEL = "meta-llama/Meta-Llama-3-8B"
image = (
    modal.Image.from_registry("nvidia/cuda:12.8.1-devel-ubuntu22.04", add_python="3.12")
    .env({"HF_HOME": "/cache"})
    .pip_install("torch", index_url="https://download.pytorch.org/whl/nightly/cu128", pre=True)
    .pip_install("transformers", "huggingface_hub", "safetensors", "pyarrow", "numpy")
)
app = modal.App("quadbit-sensitivity-sparse", image=image)
vol = modal.Volume.from_name("quadbit-hf-cache", create_if_missing=True)

# deployed sparse-vs-own-dense throughput ratio at the roofline (paper Section 5, square 8192).
# est speedup uses this single measured constant; per-shape ratios are the deploy-verify follow-on.
SP_RATIO = 1.33


@app.function(gpu="RTX-PRO-6000", timeout=14400, volumes={"/cache": vol},
              secrets=[modal.Secret.from_name("huggingface")])
def run(model: str = MODEL, all_linears: bool = True) -> None:
    import math

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

    def enc_ue4m3_t(s):
        mf, e = torch.frexp(s.clamp_min(1e-30)); mant = torch.round((2.0 * mf - 1.0) * 8.0).long()
        biased = (e - 1) + 7; carry = mant == 8
        mant = torch.where(carry, torch.zeros_like(mant), mant); biased = torch.where(carry, biased + 1, biased)
        code = (biased << 3) | mant
        code = torch.where(biased < 1, torch.ones_like(code), code)
        return torch.where((biased > 15) | (s >= 480.0), torch.full_like(code, 0x7f),
                           torch.where(s > 0, code, torch.zeros_like(code)))

    def w_nvfp4(W):  # two-level per-16 ue4m3 + per-row fp32 global; zeros stay zeros
        out_f, in_f = W.shape
        gA = (W.abs().amax(-1, keepdim=True) / 2688.0).clamp_min(1e-30)
        b = W.view(out_f, in_f // 16, 16)
        sdeq = UE4M3[enc_ue4m3_t((b.abs().amax(-1) / 6.0) / gA)] * gA
        return (FP4[q_fp4(b / sdeq[..., None])] * sdeq[..., None]).reshape(out_f, in_f)

    def a_fp4(x):  # two-level activation quant (deployed W4A4)
        lead = x.shape[:-1]; i = x.shape[-1]
        b = x.to(torch.bfloat16).float().reshape(-1, i)
        gB = (b.abs().amax(-1, keepdim=True) / 2688.0).clamp_min(1e-30)
        bb = b.reshape(b.shape[0], i // 16, 16)
        sdeq = UE4M3[enc_ue4m3_t((bb.abs().amax(-1) / 6.0) / gB)] * gB
        return (FP4[q_fp4(bb / sdeq[..., None])] * sdeq[..., None]).reshape(*lead, i)

    def enc(s):
        return torch.bucketize(s, (UE4M3[:-1] + UE4M3[1:]) / 2)

    def sparsegpt_pair24(W, H, blocksize=128, percdamp=0.01):  # one-shot pair-2:4, Hessian comp (no training)
        W = W.float().clone(); cols = W.shape[1]
        dead = torch.diag(H) == 0; H[dead, dead] = 1.0; W[:, dead] = 0.0
        H[range(cols), range(cols)] += percdamp * torch.mean(torch.diag(H))
        Hinv = torch.linalg.cholesky(torch.cholesky_inverse(torch.linalg.cholesky(H)), upper=True)
        rows = W.shape[0]; rowidx = torch.arange(rows, device=dev)
        for i in range(0, cols, blocksize):
            e = min(i + blocksize, cols); B = e - i
            W1 = W[:, i:e].clone(); Err = torch.zeros_like(W1)
            Hinv1 = Hinv[i:e, i:e]; dinv = torch.diag(Hinv1); curmask = None
            for j in range(B):
                if (j % 8) == 0:
                    tmp = (W1[:, j:j + 8] ** 2) / (dinv[j:j + 8] ** 2)[None, :]
                    pairm = tmp[:, 0::2] + tmp[:, 1::2]
                    pr = pairm.topk(2, dim=1, largest=False).indices
                    curmask = torch.ones(rows, 8, dtype=torch.bool, device=dev)
                    curmask[rowidx[:, None], pr * 2] = False
                    curmask[rowidx[:, None], pr * 2 + 1] = False
                w = W1[:, j]; q = torch.where(curmask[:, j % 8], w, torch.zeros_like(w))
                err = (w - q) / dinv[j]; W1[:, j:] -= err[:, None] * Hinv1[j, j:][None, :]
                Err[:, j] = err; W1[:, j] = q
            W[:, i:e] = W1; W[:, e:] -= Err @ Hinv[i:e, e:]
        return W

    class QLin(nn.Module):  # holds dense + sparse two-level fake-quant weights; .sp toggles
        def __init__(self, Wdense_fq, Wsparse_fq):
            super().__init__()
            self.register_buffer("Wd", Wdense_fq.to(torch.bfloat16))
            self.register_buffer("Ws", Wsparse_fq.to(torch.bfloat16))
            self.sp = False

        def forward(self, x):
            W = (self.Ws if self.sp else self.Wd).float()
            return F.linear(a_fp4(x.float()), W).to(x.dtype)

    ATTN = ("q_proj", "k_proj", "v_proj", "o_proj")
    MLP = ("gate_proj", "up_proj", "down_proj")

    def targets(mm):
        for li, layer in enumerate(mm.model.layers):
            groups = [(layer.mlp, MLP)]
            if all_linears:
                groups.insert(0, (layer.self_attn, ATTN))
            for parent, names in groups:
                for nm in names:
                    lin = getattr(parent, nm)
                    if lin.weight.shape[0] % 256 == 0 and lin.weight.shape[1] % 256 == 0:
                        yield li, parent, nm, lin

    tok = AutoTokenizer.from_pretrained(model)
    m = AutoModelForCausalLM.from_pretrained(model, dtype=torch.bfloat16).to(dev).eval()

    c4 = torch.from_numpy(np.load("/cache/corpus_c4_llama3_smoke/shard_0000.npy")[:16]).long().to(dev)
    test_ids = tok("\n\n".join(pq.read_table(hf_hub_download(
        "Salesforce/wikitext", "wikitext-2-raw-v1/test-00000-of-00001.parquet",
        repo_type="dataset")).column("text").to_pylist())).input_ids

    def ppl_wt2(windows=16, seq=2048):  # HELD-OUT eval (scored ONCE per Pareto point)
        nll = n = 0
        for i in range(0, min(len(test_ids), windows * seq) - seq, seq):
            w = torch.tensor(test_ids[i:i + seq], device=dev).unsqueeze(0)
            with torch.no_grad():
                nll += m(w, labels=w).loss.item() * (seq - 1); n += seq - 1
        return math.exp(nll / n)

    def ppl_c4():  # SELECTION set (ranking), disjoint from WT-2 test
        nll = n = 0
        for w in c4:
            wv = w.unsqueeze(0)
            with torch.no_grad():
                nll += m(wv, labels=wv).loss.item() * (wv.shape[1] - 1); n += wv.shape[1] - 1
        return math.exp(nll / n)

    # --- calibrate H = (2/n) X^T X per target linear over C4 tokens (one pass) ---
    Hs, ns = {}, {}

    def hook(mod, inp, _out):
        x = inp[0].detach().float().reshape(-1, inp[0].shape[-1]); k = id(mod); nn_ = x.shape[0]
        if k not in Hs:
            Hs[k] = torch.zeros(x.shape[1], x.shape[1], device=dev); ns[k] = 0
        Hs[k] *= ns[k] / (ns[k] + nn_)
        Hs[k] += (x * math.sqrt(2.0 / (ns[k] + nn_))).t() @ (x * math.sqrt(2.0 / (ns[k] + nn_)))
        ns[k] += nn_

    handles = [lin.register_forward_hook(hook) for _, _, _, lin in targets(m)]
    with torch.no_grad():
        for w in c4:
            m(w.unsqueeze(0))
    for h in handles:
        h.remove()
    print(f"calibrated H on {ns[next(iter(ns))]} tokens; pruning + fake-quantizing...", flush=True)

    # --- precompute dense + sparse two-level fake-quant per matrix; swap in QLin (frees H as we go) ---
    cand = []  # (layer, matrix_type, QLin, flops)
    for li, parent, nm, lin in targets(m):
        Wd = w_nvfp4(lin.weight.data.float())
        # ponytail: w_nvfp4 scales per-16 block; the deployed kernel scales per-32 kept-block.
        # faithful two-level fake-quant for RANKING; the winner's exact deploy gap is the follow-on.
        Wp = sparsegpt_pair24(lin.weight.data, Hs.pop(id(lin)))
        Ws = w_nvfp4(Wp)
        out_f, in_f = lin.weight.shape
        ql = QLin(Wd, Ws).to(dev); setattr(parent, nm, ql)
        cand.append((li, nm, ql, out_f * in_f))
        del Wd, Wp, Ws; torch.cuda.empty_cache()

    total_flops = sum(f for _, _, _, f in cand)

    def set_all(v):
        for _, _, ql, _ in cand:
            ql.sp = v

    def est_speedup():  # 1 / ((1-f) + f/SP_RATIO), f = sparse FLOP fraction
        f = sum(fl for _, _, ql, fl in cand if ql.sp) / total_flops
        return 1.0 / ((1.0 - f) + f / SP_RATIO)

    set_all(False); base_c4 = ppl_c4(); base_wt2 = ppl_wt2()
    set_all(True); ctrl_c4 = ppl_c4(); ctrl_wt2 = ppl_wt2(); set_all(False)
    print(f"\ndense W4A4   C4 {base_c4:.3f}  WT-2 {base_wt2:.3f}", flush=True)
    print(f"all-sparse   C4 {ctrl_c4:.3f}  WT-2 {ctrl_wt2:.3f}  (control; {len(cand)} matrices)", flush=True)

    # --- rank each matrix by C4 delta when ONLY it is sparse ---
    rec = []
    for idx, (li, nm, ql, _) in enumerate(cand):
        ql.sp = True; d = ppl_c4() - base_c4; ql.sp = False
        rec.append((d, idx, li, nm))
    rec.sort()  # least-damaging first
    print("\nleast sparse-sensitive matrices (C4 delta-PPL when only it is sparse):", flush=True)
    for d, _, li, nm in rec[:10]:
        print(f"  L{li:2d}.{nm:10s} {d:+.4f}", flush=True)
    print("most sparse-sensitive:", flush=True)
    for d, _, li, nm in rec[-6:]:
        print(f"  L{li:2d}.{nm:10s} {d:+.4f}", flush=True)

    # --- Pareto: sparsify least-damaging-first; WT-2 PPL + est speedup along the curve ---
    order = [idx for _, idx, _, _ in rec]
    thresholds = [0.05, 0.10, 0.25, 0.50]; hit = {t: None for t in thresholds}
    set_all(False)
    print("\nPARETO (sparsify least-damaging-first):", flush=True)
    print(f"{'#sparse':>7} {'frac':>5} {'estx':>5} {'WT-2':>7} {'dPPL':>7}", flush=True)
    step = max(1, len(order) // 24)  # ~24 points along the curve
    for k in range(0, len(order) + 1):
        if k > 0:
            cand[order[k - 1]][2].sp = True
        if k % step == 0 or k == len(order):
            p = ppl_wt2(); frac = sum(fl for _, _, ql, fl in cand if ql.sp) / total_flops
            dppl = p - base_wt2
            print(f"{k:7d} {frac:5.2f} {est_speedup():5.3f} {p:7.3f} {dppl:+7.3f}", flush=True)
            for t in thresholds:
                if hit[t] is None and dppl > t:
                    hit[t] = (k - 1, frac)
    print("\nlargest sparse set under each delta-PPL budget (est prefill speedup on linear FLOPs):", flush=True)
    for t in thresholds:
        if hit[t]:
            ks, _ = hit[t]
            for _, _, ql, _ in cand:
                ql.sp = False
            for idx in order[:ks]:
                cand[idx][2].sp = True
            frac = sum(fl for _, _, ql, fl in cand if ql.sp) / total_flops
            print(f"  dPPL<={t:.2f}: {ks:3d}/{len(cand)} matrices sparse, frac {frac:.2f}, est {est_speedup():.3f}x", flush=True)
        else:
            print(f"  dPPL<={t:.2f}: all {len(cand)} matrices fit (est {est_speedup():.3f}x at all-sparse)", flush=True)
    print(f"RESULT base_wt2={base_wt2:.3f} allsparse_wt2={ctrl_wt2:.3f} n_matrices={len(cand)}", flush=True)


@app.local_entrypoint()
def main(model: str = MODEL, all_linears: bool = True) -> None:
    run.spawn(model=model, all_linears=all_linears)
