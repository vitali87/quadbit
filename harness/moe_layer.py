"""M3.5: real DeepSeek-V4-Flash MoE layer -- segmented sparse operator correctness at full scale.

Loads ONE real MoE layer (256 routed experts + shared) via the M2-verified MXFP4 decode, packs to the
kernel's 2:4-sparse two-level-NVFP4 layout, and runs the full MoE layer three ways on identical routing:
  segmented  -- the M3.2 graph-capturable routed-row kernel + combine + shared
  per-expert -- the per-expert kernel reference (the M1 path)  [tight gate: must match seg to cos~1.0]
  dense-bf16 -- decoded experts run DENSE (no 2:4) in bf16      [gives the real per-layer accuracy tax]
Reports per-expert-output cosine, combined-layer cosine, nonfinite count, per-expert error histogram.
The global MXFP4 nibble-order (unverifiable offline) shows up here as garbage vs dense-bf16 if wrong.
"""

import subprocess
from pathlib import Path

import modal

ROOT = Path(__file__).parent.parent
MODEL = "deepseek-ai/DeepSeek-V4-Flash"
BN = 128
image = (
    modal.Image.from_registry("nvidia/cuda:12.8.1-devel-ubuntu22.04", add_python="3.12")
    .env({"PATH": "/usr/local/cuda/bin:/usr/local/sbin:/usr/local/bin:/usr/sbin:/usr/bin:/sbin:/bin",
          "LD_LIBRARY_PATH": "/usr/local/cuda/lib64", "HF_HOME": "/cache"})
    .pip_install("torch", index_url="https://download.pytorch.org/whl/nightly/cu128", pre=True)
    .pip_install("safetensors", "huggingface_hub", "numpy")
    .add_local_dir((ROOT / "cuda").as_posix(), "/root/cuda")
)
app = modal.App("quadbit-moelayer", image=image)
vol = modal.Volume.from_name("quadbit-hf-cache", create_if_missing=True)


@app.function(gpu="RTX-PRO-6000", timeout=7200, volumes={"/cache": vol},
              secrets=[modal.Secret.from_name("huggingface")])
def run(layer: int = 4, n_experts: int = 256, n_tokens: int = 256, topk: int = 6) -> None:
    import ctypes
    import json

    import torch
    import torch.nn.functional as F
    from huggingface_hub import hf_hub_download
    from safetensors import safe_open

    so = "/root/sparse_fp4.so"
    c = subprocess.run(["nvcc", "-arch=sm_120a", "-O3", "-shared", "-Xcompiler", "-fPIC",
                        "-o", so, "/root/cuda/sparse_fp4_lib.cu", "-lcuda"], capture_output=True, text=True)
    if c.returncode != 0:
        print(c.stderr, flush=True)
        raise SystemExit(1)
    lib = ctypes.CDLL(so)
    lib.quantize_act_nvfp4_2lvl.argtypes = [ctypes.c_void_p] * 4 + [ctypes.c_int] * 2
    lib.sparse_moe_mm_2lvl.argtypes = ([ctypes.c_void_p] * 6 + [ctypes.c_int] * 4 +
                                       [ctypes.c_void_p] * 3 + [ctypes.c_int] + [ctypes.c_void_p])
    lib.sparse_fp4_mm_2lvl_t.argtypes = [ctypes.c_void_p] * 6 + [ctypes.c_int] * 3 + [ctypes.c_void_p] * 2
    lib.qb_init_moe_attrs()
    dev = torch.device("cuda")
    torch.manual_seed(0)

    FP4 = torch.tensor([0, .5, 1, 1.5, 2, 3, 4, 6, 0, -.5, -1, -1.5, -2, -3, -4, -6], device=dev)
    BND = torch.tensor([.25, .75, 1.25, 1.75, 2.5, 3.5, 5.], device=dev)
    _cc = torch.arange(128, device=dev); _e, _m = (_cc >> 3) & 0xf, _cc & 7
    UE4M3 = torch.where(_e == 0, _m.float() * 0.001953125, (1.0 + _m.float() / 8.0) * torch.exp2((_e - 7).float()))

    def q_fp4(v):
        return torch.bucketize(v.abs(), BND) | ((v < 0).long() << 3)

    def enc_ue4m3_t(s):
        mant_f, e = torch.frexp(s.clamp_min(1e-30)); mm = 2.0 * mant_f
        biased = (e - 1) + 7; mant = torch.round((mm - 1.0) * 8.0).long(); carry = mant == 8
        mant = torch.where(carry, torch.zeros_like(mant), mant); biased = torch.where(carry, biased + 1, biased)
        code = (biased.long() << 3) | mant
        code = torch.where(biased < 1, torch.ones_like(code), code)
        code = torch.where(biased > 15, torch.full_like(code, 0x7f), code)
        code = torch.where(s >= 480.0, torch.full_like(code, 0x7f), code)
        return torch.where(s > 0, code, torch.zeros_like(code))

    def sparse_fp4_dequant(W):
        out_f, in_f = W.shape; ks = in_f // 128
        Wg = W.view(out_f, ks, 16, 4, 2)
        i01, _ = Wg.abs().sum(-1).topk(2, dim=-1).indices.sort(dim=-1)
        keptW = torch.gather(Wg, 3, i01.unsqueeze(-1).expand(-1, -1, -1, -1, 2))
        gA = (keptW.abs().amax(dim=(1, 2, 3, 4), keepdim=True) / 2688.0).clamp_min(1e-30).reshape(out_f, 1, 1)
        blk = keptW.reshape(out_f, ks, 4, 8, 2)
        sdeq = UE4M3[enc_ue4m3_t((blk.abs().amax(dim=(3, 4)) / 6.0) / gA)] * gA
        kd = (FP4[q_fp4(blk / sdeq.clamp_min(1e-30)[..., None, None])] * sdeq[..., None, None]).reshape(out_f, ks, 16, 2, 2)
        Wd = torch.zeros(out_f, ks, 16, 4, 2, device=dev)
        Wd.scatter_(3, i01.unsqueeze(-1).expand(-1, -1, -1, -1, 2), kd)
        return Wd.reshape(out_f, in_f)

    def mxfp4_dequant(w_int8, s_e8m0):
        out_f, in_b = w_int8.shape; in_f = in_b * 2
        bb = w_int8.to(dev).to(torch.int32) & 0xFF
        codes = torch.empty(out_f, in_f, dtype=torch.long, device=dev)
        codes[:, 0::2] = bb & 0xF; codes[:, 1::2] = (bb >> 4) & 0xF
        nb = in_f // 32
        s = s_e8m0.to(dev).float().reshape(out_f, nb, 1)
        return (FP4[codes].view(out_f, nb, 32) * s).reshape(out_f, in_f).to(torch.bfloat16)

    def pack(W):
        out_f, in_f = W.shape; ks = in_f // 128
        Wg = W.float().to(dev).view(out_f, ks, 16, 4, 2)
        i01, _ = Wg.abs().sum(-1).topk(2, dim=-1).indices.sort(dim=-1)
        keptW = torch.gather(Wg, 3, i01.unsqueeze(-1).expand(-1, -1, -1, -1, 2))
        gA = (keptW.abs().amax(dim=(1, 2, 3, 4), keepdim=True) / 2688.0).clamp_min(1e-30).reshape(out_f, 1, 1)
        blk = keptW.reshape(out_f, ks, 4, 8, 2)
        scode = enc_ue4m3_t((blk.abs().amax(dim=(3, 4)) / 6.0) / gA)
        sdeq = UE4M3[scode] * gA
        kc = q_fp4(blk / sdeq.clamp_min(1e-30)[..., None, None])
        Ac = (kc[..., 0] | (kc[..., 1] << 4)).reshape(out_f, ks * 32).to(torch.uint8)
        nib = (i01[..., 0] | (i01[..., 1] << 2)).view(out_f, ks, 2, 8)
        sh = (torch.arange(8, device=dev) * 4).view(1, 1, 1, 8)
        meta = (nib << sh).sum(-1).to(torch.int32).permute(1, 0, 2).contiguous()
        return (Ac.contiguous(), meta, scode.to(torch.uint8).permute(1, 0, 2).contiguous(), gA.reshape(out_f).float().contiguous())

    def stack(packs):
        return (torch.cat([p[0] for p in packs], 0).contiguous(), torch.cat([p[1] for p in packs], 1).contiguous(),
                torch.cat([p[2] for p in packs], 1).contiguous(), torch.cat([p[3] for p in packs], 0).contiguous())

    def quant_act(x):
        R, in_f = x.shape; ks = in_f // 128
        x = x.to(torch.bfloat16).contiguous()
        Bb = torch.empty((R, in_f // 2), dtype=torch.uint8, device=dev)
        sB = torch.empty((ks, R, 4), dtype=torch.uint8, device=dev)
        gB = torch.empty((R,), dtype=torch.float32, device=dev)
        lib.quantize_act_nvfp4_2lvl(x.data_ptr(), Bb.data_ptr(), sB.data_ptr(), gB.data_ptr(), R, in_f)
        return Bb, sB, gB

    def seg_gemm(x, W, Mpe, in_f, eblk):
        R = x.shape[0]; Ac, meta, scaleA, gA = W
        Bb, sB, gB = quant_act(x)
        C = torch.empty((R, Mpe), dtype=torch.bfloat16, device=dev)
        lib.sparse_moe_mm_2lvl(Ac.data_ptr(), Bb.data_ptr(), scaleA.data_ptr(), sB.data_ptr(), meta.data_ptr(),
                               C.data_ptr(), Ac.shape[0], Mpe, R, in_f, gA.data_ptr(), gB.data_ptr(),
                               eblk.data_ptr(), 1, 0)
        return C

    def one_lin(x, packe, M, K):
        Ac, meta, scaleA, gA = packe; n = x.shape[0]; padn = (-n) % 128
        xp = torch.cat([x, x.new_zeros(padn, K)], 0).to(torch.bfloat16).contiguous() if padn else x.to(torch.bfloat16).contiguous()
        Bb, sB, gB = quant_act(xp)
        C = torch.empty((n + padn, M), dtype=torch.bfloat16, device=dev)
        lib.sparse_fp4_mm_2lvl_t(Ac.data_ptr(), Bb.data_ptr(), scaleA.data_ptr(), sB.data_ptr(), meta.data_ptr(),
                                 C.data_ptr(), M, n + padn, K, gA.data_ptr(), gB.data_ptr())
        return C[:n]

    def build_routing(assign, E):
        order = torch.argsort(assign, stable=True)
        counts = torch.bincount(assign, minlength=E)
        padc = (counts + BN - 1) // BN * BN
        R_pad = int(padc.sum().item())
        src = torch.full((R_pad,), -1, dtype=torch.long, device=dev)
        eblk = torch.zeros(R_pad // BN, dtype=torch.int32, device=dev)
        off = 0; oi = 0
        for e in range(E):
            ce = int(counts[e].item()); pe = int(padc[e].item())
            if pe == 0:
                continue
            src[off:off + ce] = order[oi:oi + ce]; oi += ce
            eblk[off // BN:(off + pe) // BN] = e
            off += pe
        return src, eblk, R_pad

    # ---- load one real MoE layer ----
    idx = json.load(open(hf_hub_download(MODEL, "model.safetensors.index.json")))["weight_map"]

    def get(name):
        with safe_open(hf_hub_download(MODEL, idx[name]), framework="pt", device="cpu") as f:
            return f.get_tensor(name)

    def load_expert_bf16(e, proj):
        base = f"layers.{layer}.ffn.experts.{e}.{proj}"
        return mxfp4_dequant(get(f"{base}.weight"), get(f"{base}.scale"))

    print(f"# M3.5 real DeepSeek-V4-Flash layer {layer}: {n_experts} experts, {n_tokens} tokens, top-{topk}", flush=True)
    Wg, Wu, Wd = [], [], []
    gu_packs, dn_packs = [], []
    for e in range(n_experts):
        g = load_expert_bf16(e, "w1"); u = load_expert_bf16(e, "w3"); d = load_expert_bf16(e, "w2")
        Wg.append(g); Wu.append(u); Wd.append(d)
        gu_packs.append(pack(torch.cat([g, u], 0))); dn_packs.append(pack(d))
    H = Wg[0].shape[1]; I = Wg[0].shape[0]
    gu_W, dn_W = stack(gu_packs), stack(dn_packs)
    print(f"  loaded {n_experts} experts, H={H} I={I}; sparse-pack footprint "
          f"{(gu_W[0].numel() + dn_W[0].numel()) / 1e6:.0f}MB codes", flush=True)

    # real routing from the layer's router if present, else random
    X = torch.randn(n_tokens, H, device=dev, dtype=torch.bfloat16) * (H ** -0.5) * 4
    gkey = f"layers.{layer}.ffn.gate.weight"
    if gkey in idx:
        Wr = get(gkey).to(dev)
        Wr = mxfp4_dequant(Wr, get(f"layers.{layer}.ffn.gate.scale")) if f"layers.{layer}.ffn.gate.scale" in idx else Wr.float()
        logits = F.linear(X.float(), Wr.float()[:n_experts])
        print("  routing: real gate weight", flush=True)
    else:
        logits = torch.randn(n_tokens, n_experts, device=dev)
        print("  routing: random (gate weight not found in index)", flush=True)
    tw, tidx = logits.softmax(-1).topk(topk, dim=-1)
    tw = (tw / tw.sum(-1, keepdim=True) * 1.5)  # norm_topk_prob + routed_scaling_factor

    # routed rows: one per (token, slot)
    assign = tidx.reshape(-1)                              # [T*topk]
    tok_of = torch.arange(n_tokens, device=dev).repeat_interleave(topk)
    w_of = tw.reshape(-1)
    src, eblk, R_pad = build_routing(assign, n_experts)
    valid = src >= 0; srcc = src.clamp_min(0)
    Xs = X[tok_of[srcc]] * valid[:, None]

    # segmented sparse expert outputs
    GU = seg_gemm(Xs, gu_W, 2 * I, H, eblk)
    Hh = (F.silu(GU[:, :I].float()) * GU[:, I:].float()).to(torch.bfloat16)
    Dseg = seg_gemm(Hh, dn_W, H, I, eblk)                  # [R_pad, H]

    # per-expert kernel reference (tight gate)
    Dker = torch.zeros(R_pad, H, device=dev)
    for e in range(n_experts):
        rows = ((eblk == e).repeat_interleave(BN)) & valid
        if rows.any():
            GUe = one_lin(Xs[rows], gu_packs[e], 2 * I, H)
            he = (F.silu(GUe[:, :I].float()) * GUe[:, I:].float()).to(torch.bfloat16)
            Dker[rows] = one_lin(he, dn_packs[e], H, I).float()

    # dense-bf16 reference (real per-layer tax) + per-expert error histogram
    Dden = torch.zeros(R_pad, H, device=dev); perr = []
    for e in range(n_experts):
        rows = ((eblk == e).repeat_interleave(BN)) & valid
        if rows.any():
            xe = Xs[rows].float()
            he = F.silu(F.linear(xe, Wg[e].float())) * F.linear(xe, Wu[e].float())
            de = F.linear(he, Wd[e].float())
            Dden[rows] = de
            perr.append(F.cosine_similarity(Dseg[rows].float().flatten(), de.flatten(), dim=0).item())

    def cos(a, b):
        return F.cosine_similarity(a.float().flatten(), b.float().flatten(), dim=0).item()

    m = valid
    cK = cos(Dseg[m], Dker[m]); cD = cos(Dseg[m], Dden[m])
    nonfin = int((~torch.isfinite(Dseg[m])).sum().item())
    perr = torch.tensor(perr)
    # combined MoE layer output (with combine): y[token] = sum_slot w * expert_out ; compare seg vs dense
    ycomb = torch.zeros(n_tokens, H, device=dev)
    ycomb.index_add_(0, tok_of[srcc][m], Dseg[m].float() * w_of[srcc][m, None])
    ydense = torch.zeros(n_tokens, H, device=dev)
    ydense.index_add_(0, tok_of[srcc][m], Dden[m].float() * w_of[srcc][m, None])
    cComb = cos(ycomb, ydense)

    print(f"  cos(seg, per-expert-kernel) = {cK:.6f}   -> {'PASS' if cK > 0.9999 else 'FAIL'}  (routing+kernel on real weights)", flush=True)
    print(f"  cos(seg, dense-bf16)        = {cD:.4f}   nonfin={nonfin}   (real per-expert-output 2:4-FP4 tax)", flush=True)
    print(f"  combined-layer cos(seg, dense) = {cComb:.4f}", flush=True)
    print(f"  per-expert cos(seg,dense) histogram: min={perr.min():.3f} p10={perr.quantile(0.1):.3f} "
          f"median={perr.median():.3f} max={perr.max():.3f}", flush=True)
    print(f"# M3.5 {'PASS' if cK > 0.9999 and nonfin == 0 else 'FAIL'} "
          f"(seg==per-expert-kernel on {n_experts} real experts; nibble-order tripwire = M4 PPL)", flush=True)


@app.local_entrypoint()
def main(layer: int = 4, n_experts: int = 256, n_tokens: int = 256, topk: int = 6) -> None:
    run.remote(layer=layer, n_experts=n_experts, n_tokens=n_tokens, topk=topk)
