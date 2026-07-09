"""vLLM general plugin that replaces the model's block-FP8 (ue8m0) dense/attention GEMM -- which has
no consumer-Blackwell/SM120 kernel -- with an SM120-safe path, while leaving the NVFP4 MoE
experts on their native FlashInfer path.

Selected by the QB_DENSE env var (read at install time, inherited by every spawned worker):
  QB_DENSE=bf16  (WS0) dequant each block-fp8 weight to bf16 once at load, run torch F.linear.
  QB_DENSE=nvfp4 (WS1) requantize the dequant weight to NVFP4 and run FlashInfer mm_fp4 (cutlass);
                       per-layer fall back to bf16 if flashinfer rejects a shape.
  QB_DENSE=off   leave the native path (used to confirm the upstream failure reproduces).

vLLM invokes install() once per process via the vllm.general_plugins entry point, BEFORE the model
is built, so the monkeypatch is in place when process_weights_after_loading and the profiling run.
"""

from __future__ import annotations

import os

# Per-process counters; each worker prints its own view so fallback is auditable in the logs.
STATS = {"nvfp4_layers": 0, "bf16_layers": 0, "native_layers": 0, "fp8_calls": 0, "oproj_calls": 0}


def _inv_rope_o(o, positions, cos_sin_cache, nope_dim, rope_dim):
    # bf16 inverse interleaved RoPE on the last rope_dim dims of each head. Mirrors the
    # deepseek_v4 fused_inv_rope_fp8_quant triton kernel elementwise (validated in
    # scratchpad/test_inv_rope_py.py): for pair i, even offset -> a*cos+b*sin, odd -> b*cos-a*sin.
    import torch

    half = rope_dim // 2
    rope = o[..., nope_dim:].float()
    cs = cos_sin_cache.float()[positions]  # [T, rope_dim] = cos||sin
    cos = cs[:, None, :half]
    sin = cs[:, None, half:]
    a = rope[..., 0::2]
    b = rope[..., 1::2]
    rot = torch.empty_like(rope)
    rot[..., 0::2] = a * cos + b * sin
    rot[..., 1::2] = b * cos - a * sin
    out = o.clone().float()
    out[..., nope_dim:] = rot
    return out


def _dequant_block(w, s, bs):
    # w: fp8 [N,K]; s: block scales (multiplier); dequant = w * scale_expanded.
    # The scale grid is [ceil(N/bn), ceil(K/bk)] but some vLLM layers store it transposed; detect
    # that and expand by the per-axis factor derived from the ACTUAL scale shape so se always
    # matches [N,K] and the block->element mapping stays correct.
    import math

    import torch

    n, k = w.shape
    bn, bk = int(bs[0]), int(bs[1])
    exp = (math.ceil(n / bn), math.ceil(k / bk))
    sr, sc = int(s.shape[0]), int(s.shape[1])
    if (sr, sc) != exp and (sc, sr) == exp:
        s = s.t().contiguous()
        sr, sc = sc, sr
    fn, fk = math.ceil(n / sr), math.ceil(k / sc)
    se = s.to(torch.float32).repeat_interleave(fn, 0)[:n].repeat_interleave(fk, 1)[:, :k]
    return (w.to(torch.float32) * se).to(torch.bfloat16)


# e2m1 FP4 value LUT (codes 0-15: sign bit 8): matches moe_layer.py / mxfp4 decode.
_FP4_VALS = [0, .5, 1, 1.5, 2, 3, 4, 6, 0, -.5, -1, -1.5, -2, -3, -4, -6]


def _dequant_nvfp4_expert(w_u8, w_scale_e4m3, w_scale_2, group=16):
    # NVFP4 (modelopt) dequant of one expert weight to bf16:
    #   W[o,k] = FP4_LUT[nibble] * blockscale_e4m3[o, k//group].float() * per_tensor_scale
    # w_u8 [O, K/2] uint8 (2 nibbles/byte; low=even k, high=odd k), w_scale_e4m3 [O, K/group]
    # float8_e4m3fn, w_scale_2 scalar float32.
    import torch

    lut = torch.tensor(_FP4_VALS, dtype=torch.float32, device=w_u8.device)
    o, kh = w_u8.shape
    k = kh * 2
    bb = w_u8.to(torch.int32) & 0xFF
    codes = torch.empty(o, k, dtype=torch.long, device=w_u8.device)
    codes[:, 0::2] = bb & 0xF
    codes[:, 1::2] = (bb >> 4) & 0xF
    vals = lut[codes]
    bs = w_scale_e4m3.float().repeat_interleave(group, dim=1)[:, :k]
    return (vals * bs * float(w_scale_2)).to(torch.bfloat16)


def _mqa_logits_bf16(q, kv, weights, cu_seqlen_ks, cu_seqlen_ke, clean_logits=False):
    # SM120-safe replacement for DeepGEMM fp8_fp4_mqa_logits (prefill). FP8 path only.
    # logits[m,n] = sum_h w[m,h] * sum_d q[m,h,d]*k[n,d]  == (sum_h w[m,h]*q[m,h,d]) . k[n,:]
    import torch

    q_values, _q_scale = q  # [M,H,D] float8_e4m3fn, scale folded into weights
    k_packed, k_scales = kv  # [N,D] float8_e4m3fn, [N] float32
    qf = q_values.to(torch.float32)
    w = weights.to(torch.float32)
    qw = torch.einsum("mhd,mh->md", qf, w)  # [M,D]
    kf = k_packed.to(torch.float32) * k_scales.to(torch.float32).reshape(-1, 1)  # [N,D]
    logits = qw @ kf.t()  # [M,N]
    n = logits.shape[1]
    idx = torch.arange(n, device=logits.device).reshape(1, -1)
    ks = cu_seqlen_ks.to(torch.long).reshape(-1, 1)
    ke = cu_seqlen_ke.to(torch.long).reshape(-1, 1)
    valid = (idx >= ks) & (idx < ke)
    return torch.where(valid, logits, torch.full_like(logits, float("-inf")))


def _paged_mqa_logits_bf16(q, kv_cache, weights, context_lens, block_tables, schedule_metadata,
                           max_model_len, clean_logits=False):
    # SM120-safe replacement for DeepGEMM fp8_fp4_paged_mqa_logits (decode). FP8 cache:
    # kv_cache [num_blocks, block_size, 1, D+4] uint8; last 4 bytes/row = fp32 dequant scale.
    import torch

    q_values, _q_scale = q  # [B, next_n, H, D] float8_e4m3fn
    b, next_n, h, d = q_values.shape
    num_blocks, block_size, _one, width = kv_cache.shape
    kv_u8 = kv_cache.reshape(num_blocks, block_size, width)
    w = weights.to(torch.float32).reshape(b, next_n, h)
    qf = q_values.to(torch.float32)
    qw = torch.einsum("bnhd,bnh->bnd", qf, w)  # [B, next_n, D]
    ctx = context_lens
    out = torch.full((b * next_n, max_model_len), float("-inf"),
                     device=q_values.device, dtype=torch.float32)
    for bi in range(b):
        lens_bn = ctx[bi] if ctx.ndim == 2 else ctx[bi].reshape(1).expand(next_n)
        length = int(lens_bn.max().item())
        if length <= 0:
            continue
        pos = torch.arange(length, device=q_values.device)
        blk = block_tables[bi, torch.div(pos, block_size, rounding_mode="floor").long()].long()
        wpos = (pos % block_size).long()
        rows = kv_u8[blk, wpos]  # [length, width] uint8
        kf = rows[:, :d].contiguous().view(torch.float8_e4m3fn).to(torch.float32)  # [length,D]
        ksc = rows[:, d:d + 4].contiguous().view(torch.float32).reshape(-1)  # [length]
        kf = kf * ksc.reshape(-1, 1)
        for ni in range(next_n):
            ln = int(lens_bn[ni].item()) if ctx.ndim == 2 else length
            if ln <= 0:
                continue
            out[bi * next_n + ni, :ln] = qw[bi, ni] @ kf[:ln].t()
    return out


_SPARSE_SO = "/cache/sparse_fp4.so"
_BN = 128
_SPARSE = None

# --- instrumentation (QB_INSTR=1) + in-situ per-layer tax probe (QB_TAXPROBE=1) ---
# Worker processes accumulate timing/tax here and flush to /cache so the driver can aggregate
# across ranks (vLLM V1 runs the model in worker procs; STATS/timers live per-worker).
_INSTR = os.environ.get("QB_INSTR") == "1"
_TAX = os.environ.get("QB_TAXPROBE") == "1"
_RUNTAG = os.environ.get("QB_RUNTAG", "run")
_TAX_LAYERS = 8  # probe every _TAX_LAYERS-th MoE layer
_T = {"expert": 0.0, "dense": 0.0, "forward": 0.0}  # accumulated ms
_EV = {}  # cat -> list[(start_event, end_event)] pending elapsed_time read
_CNT = {"expert_calls": 0, "dense_calls": 0}
_IMB = []  # per-apply expert imbalance (max/mean tokens-per-expert)
_TAX_COS = []  # per-(layer,expert) cos(sparse, dense) from the probe
_PW_IDX = 0  # MoE layer counter (load order) per worker

# --- WORKSTREAM A: selective sparse placement + real-routed-activation quality map ---
# QB_DENSE_LAYERS: comma list of MoE-layer indices (load order) to keep NVFP4-dense (A1 policy #1,
#   dense-anchor layers). These layers skip packing -> apply falls through to the dense NVFP4 path.
# QB_QMAP=1: on every _TAX_LAYERS-th layer, keep NVFP4 resident alongside the sparse codes and, on
#   the first _QMAP_FWD forward calls, run the SAME real routed rows through both paths to record the
#   per-layer *block* cosine (governs coherence) + per-expert cos/route-freq/weight/norm on REAL
#   activations (A0). Bounded so it doesn't dominate wall-clock.
_DENSE_LAYERS = {int(x) for x in os.environ.get("QB_DENSE_LAYERS", "").split(",") if x.strip()}
_QMAP = os.environ.get("QB_QMAP") == "1"
_QMAP_FWD = int(os.environ.get("QB_QMAP_FWD", "3"))  # probe first N forward calls per layer
_QMAP_ROWS = []  # real-activation A0 map rows (block + per-expert)
_QMAP_SEEN = {}  # layer_idx -> forwards probed so far


def _ev_start():
    import torch

    e = torch.cuda.Event(enable_timing=True)
    e.record()
    return e


def _ev_end(cat, s):
    import torch

    e = torch.cuda.Event(enable_timing=True)
    e.record()
    _EV.setdefault(cat, []).append((s, e))


def _flush_metrics():
    # sum pending CUDA-event elapsed times, merge into _T, and write this worker's metrics to
    # /cache keyed by device so the driver can read them after generation.
    import json

    import torch

    if _EV:
        torch.cuda.synchronize()
        for cat, pairs in _EV.items():
            _T[cat] = _T.get(cat, 0.0) + sum(s.elapsed_time(e) for s, e in pairs)
        _EV.clear()
    dev = torch.cuda.current_device()
    data = {"rank_dev": dev, "t_ms": _T, "counts": _CNT, "stats": STATS,
            "imbalance_mean": (sum(_IMB) / len(_IMB)) if _IMB else 0.0,
            "imbalance_max": (max(_IMB) if _IMB else 0.0),
            "tax_cos": _TAX_COS, "qmap": _QMAP_ROWS}
    try:
        with open(f"/cache/qb_metrics_{_RUNTAG}_dev{dev}.json", "w") as f:
            json.dump(data, f)
    except Exception:  # noqa: BLE001
        pass


def _run_tax_probe(layer, sp, i, e, layer_idx):
    # In-situ per-expert tax: for a sample of experts in this layer, run R=128 realistic random
    # rows through the DENSE path (NVFP4->bf16, no 2:4) and the quadbit 2:4-sparse-FP4 kernel path
    # (the packed weights already on the layer), then cos(sparse_out, dense_out). This is the exact
    # per-expert operator tax (weight+activation quant) that compounds across the 43 layers.
    import torch
    import torch.nn.functional as F

    w13, w13s, w13s2 = layer.w13_weight, layer.w13_weight_scale, layer.w13_weight_scale_2
    w2, w2s, w2s2 = layer.w2_weight, layer.w2_weight_scale, layer.w2_weight_scale_2
    h = w13.shape[2] * 2  # packed uint8 -> hidden dim
    dev = w13.device
    experts = list(range(0, e, max(1, e // 8)))[:8]  # ~8 experts spread across the layer
    for le in experts:
        x = (torch.randn(_BN, h, device=dev, dtype=torch.bfloat16) * (h ** -0.5) * 4)
        # dense reference (NVFP4 -> bf16, dense matmul) -- the S1-validated path
        gu = _dequant_nvfp4_expert(w13[le], w13s[le], w13s2[le, 0]).float()
        dn = _dequant_nvfp4_expert(w2[le], w2s[le], w2s2[le]).float()
        xf = x.float()
        hh_d = F.silu(xf @ gu[:i].t()) * (xf @ gu[i:].t())
        out_d = hh_d @ dn.t()
        # quadbit 2:4-sparse-FP4 path through the real packed weights (eblk selects expert le)
        eblk = torch.full((1,), le, dtype=torch.int32, device=dev)
        gu_s = sp.seg_gemm(x, layer._qb_gu, 2 * i, h, eblk)
        hh_s = (F.silu(gu_s[:, :i].float()) * gu_s[:, i:].float()).to(torch.bfloat16)
        out_s = sp.seg_gemm(hh_s, layer._qb_dn, h, i, eblk)
        c = F.cosine_similarity(out_s.float().flatten(), out_d.flatten(), dim=0).item()
        _TAX_COS.append({"layer": layer_idx, "expert": le, "cos": c})


def _run_qmap_probe(layer, sp, x, topk_ids, topk_weights, y_sparse, on_input):
    # A0 real-routed-activation quality map. On a probe layer (NVFP4 kept resident), rebuild BOTH the
    # dense NVFP4 block output and the per-expert sparse/dense outputs on the SAME real routed rows,
    # then record: per-layer *block* cosine (the number that governs coherence when compounded over
    # 43 layers) + per-expert cos, route frequency, mean route weight, and dense contribution norm
    # (route-weighted). Real activations, not the random ones _run_tax_probe uses.
    import torch
    import torch.nn.functional as F

    li = getattr(layer, "_qb_layer_idx", -1)
    if _QMAP_SEEN.get(li, 0) >= _QMAP_FWD:
        return
    _QMAP_SEEN[li] = _QMAP_SEEN.get(li, 0) + 1

    w13, w13s, w13s2 = layer.w13_weight, layer.w13_weight_scale, layer.w13_weight_scale_2
    w2, w2s, w2s2 = layer.w2_weight, layer.w2_weight_scale, layer.w2_weight_scale_2
    if w13.numel() == 0:  # NVFP4 was freed (not actually a probe layer) -- nothing to compare
        return
    t, h = x.shape
    ii = layer._qb_i
    topk = topk_ids.shape[1]
    flat_ids = topk_ids.reshape(-1)
    flat_w = topk_weights.reshape(-1).to(torch.float32)
    tok = torch.arange(t, device=x.device).repeat_interleave(topk)
    y_dense = torch.zeros(t, h, dtype=torch.float32, device=x.device)
    for le in torch.unique(flat_ids).tolist():
        if le < 0:
            continue
        sel = flat_ids == le
        rows = tok[sel]
        ws = flat_w[sel]
        xe = x[rows].to(torch.bfloat16)
        if on_input:
            xe = xe * ws[:, None].to(torch.bfloat16)
        # dense NVFP4 -> bf16 reference operator
        wg = _dequant_nvfp4_expert(w13[le, :ii], w13s[le, :ii], w13s2[le, 0]).float()
        wu = _dequant_nvfp4_expert(w13[le, ii:], w13s[le, ii:], w13s2[le, 0]).float()
        wd = _dequant_nvfp4_expert(w2[le], w2s[le], w2s2[le]).float()
        xf = xe.float()
        oe_d = (F.silu(xf @ wg.t()) * (xf @ wu.t())) @ wd.t()
        # quadbit 2:4-sparse-FP4 operator on the same rows. The seg kernel tiles N in _BN-row blocks
        # and reads one expert id per block from eblk, so rows MUST be padded up to a _BN multiple
        # (with a matching all-`le` eblk); otherwise blocks past the first read OOB -> NaN.
        n = xe.shape[0]
        npad = ((n + _BN - 1) // _BN) * _BN
        xep = F.pad(xe, (0, 0, 0, npad - n)) if npad != n else xe
        nb = npad // _BN
        eblk = torch.full((nb,), le, dtype=torch.int32, device=x.device)
        gu_s = sp.seg_gemm(xep, layer._qb_gu, 2 * ii, h, eblk)
        hh_s = (F.silu(gu_s[:, :ii].float()) * gu_s[:, ii:].float()).to(torch.bfloat16)
        oe_s = sp.seg_gemm(hh_s, layer._qb_dn, h, ii, eblk)[:n].float()
        c = F.cosine_similarity(oe_s.flatten(), oe_d.flatten(), dim=0).item()
        contrib = oe_d if on_input else oe_d * ws[:, None]
        _QMAP_ROWS.append({"layer": li, "expert": int(le), "cos": round(c, 5),
                           "freq": int(rows.numel()), "mean_w": round(ws.mean().item(), 5),
                           "contrib_norm": round(contrib.norm().item(), 4)})
        y_dense.index_add_(0, rows, (contrib if on_input else oe_d * ws[:, None]))
    block_c = F.cosine_similarity(y_sparse.float().flatten(), y_dense.flatten(), dim=0).item()
    _QMAP_ROWS.append({"layer": li, "expert": -1, "block_cos": round(block_c, 5)})
    _flush_metrics()


def _load_sparse_moe():
    # Lazily ctypes-load the staged sm_120 quadbit 2:4-sparse-FP4 kernel (.so) and build the
    # pack/quant_act/seg_gemm/build_routing helpers (ported verbatim from moe_layer.py / test_so,
    # which validated this .so on the CUDA-13 serve image: finite output, ~0.88 sparse-FP4 tax).
    # Cached per worker process.
    global _SPARSE
    if _SPARSE is not None:
        return _SPARSE
    import ctypes
    from types import SimpleNamespace

    import torch

    lib = ctypes.CDLL(_SPARSE_SO)
    lib.quantize_act_nvfp4_2lvl.argtypes = [ctypes.c_void_p] * 4 + [ctypes.c_int] * 2
    lib.sparse_moe_mm_2lvl.argtypes = ([ctypes.c_void_p] * 6 + [ctypes.c_int] * 4
                                       + [ctypes.c_void_p] * 3 + [ctypes.c_int] + [ctypes.c_void_p])
    lib.qb_init_moe_attrs()
    dev = torch.device("cuda")

    fp4 = torch.tensor(_FP4_VALS, device=dev)
    bnd = torch.tensor([.25, .75, 1.25, 1.75, 2.5, 3.5, 5.], device=dev)
    cc = torch.arange(128, device=dev)
    e_, m_ = (cc >> 3) & 0xf, cc & 7
    ue4m3 = torch.where(e_ == 0, m_.float() * 0.001953125,
                        (1.0 + m_.float() / 8.0) * torch.exp2((e_ - 7).float()))

    def q_fp4(v):
        return torch.bucketize(v.abs(), bnd) | ((v < 0).long() << 3)

    def enc(s):
        mant_f, e = torch.frexp(s.clamp_min(1e-30))
        biased = (e - 1) + 7
        mant = torch.round((2.0 * mant_f - 1.0) * 8.0).long()
        carry = mant == 8
        mant = torch.where(carry, torch.zeros_like(mant), mant)
        biased = torch.where(carry, biased + 1, biased)
        code = (biased.long() << 3) | mant
        code = torch.where(biased < 1, torch.ones_like(code), code)
        code = torch.where(biased > 15, torch.full_like(code, 0x7f), code)
        code = torch.where(s >= 480.0, torch.full_like(code, 0x7f), code)
        return torch.where(s > 0, code, torch.zeros_like(code))

    def pack(w):
        out_f, in_f = w.shape
        ks = in_f // 128
        wg = w.float().view(out_f, ks, 16, 4, 2)
        i01, _ = wg.abs().sum(-1).topk(2, dim=-1).indices.sort(dim=-1)
        kept = torch.gather(wg, 3, i01.unsqueeze(-1).expand(-1, -1, -1, -1, 2))
        ga = (kept.abs().amax(dim=(1, 2, 3, 4), keepdim=True) / 2688.0).clamp_min(1e-30).reshape(out_f, 1, 1)
        blk = kept.reshape(out_f, ks, 4, 8, 2)
        scode = enc((blk.abs().amax(dim=(3, 4)) / 6.0) / ga)
        sdeq = ue4m3[scode] * ga
        kc = q_fp4(blk / sdeq.clamp_min(1e-30)[..., None, None])
        ac = (kc[..., 0] | (kc[..., 1] << 4)).reshape(out_f, ks * 32).to(torch.uint8)
        nib = (i01[..., 0] | (i01[..., 1] << 2)).view(out_f, ks, 2, 8)
        sh = (torch.arange(8, device=dev) * 4).view(1, 1, 1, 8)
        meta = (nib << sh).sum(-1).to(torch.int32).permute(1, 0, 2).contiguous()
        return (ac.contiguous(), meta, scode.to(torch.uint8).permute(1, 0, 2).contiguous(),
                ga.reshape(out_f).float().contiguous())

    def stack(packs):
        return (torch.cat([p[0] for p in packs], 0).contiguous(),
                torch.cat([p[1] for p in packs], 1).contiguous(),
                torch.cat([p[2] for p in packs], 1).contiguous(),
                torch.cat([p[3] for p in packs], 0).contiguous())

    def quant_act(x):
        r, in_f = x.shape
        ks = in_f // 128
        x = x.to(torch.bfloat16).contiguous()
        bb = torch.empty((r, in_f // 2), dtype=torch.uint8, device=dev)
        sb = torch.empty((ks, r, 4), dtype=torch.uint8, device=dev)
        gb = torch.empty((r,), dtype=torch.float32, device=dev)
        lib.quantize_act_nvfp4_2lvl(x.data_ptr(), bb.data_ptr(), sb.data_ptr(), gb.data_ptr(), r, in_f)
        return bb, sb, gb

    def seg_gemm(x, w, mpe, in_f, eblk):
        r = x.shape[0]
        ac, meta, scale_a, ga = w
        bb, sb, gb = quant_act(x)
        c = torch.empty((r, mpe), dtype=torch.bfloat16, device=dev)
        lib.sparse_moe_mm_2lvl(ac.data_ptr(), bb.data_ptr(), scale_a.data_ptr(), sb.data_ptr(),
                               meta.data_ptr(), c.data_ptr(), ac.shape[0], mpe, r, in_f,
                               ga.data_ptr(), gb.data_ptr(), eblk.data_ptr(), 1, 0)
        return c

    def build_routing(assign, e):
        order = torch.argsort(assign, stable=True)
        counts = torch.bincount(assign, minlength=e)
        padc = (counts + _BN - 1) // _BN * _BN
        r_pad = int(padc.sum().item())
        src = torch.full((r_pad,), -1, dtype=torch.long, device=dev)
        eblk = torch.zeros(r_pad // _BN, dtype=torch.int32, device=dev)
        off = 0
        oi = 0
        for ex in range(e):
            ce = int(counts[ex].item())
            pe = int(padc[ex].item())
            if pe == 0:
                continue
            src[off:off + ce] = order[oi:oi + ce]
            oi += ce
            eblk[off // _BN:(off + pe) // _BN] = ex
            off += pe
        return src, eblk, r_pad

    _SPARSE = SimpleNamespace(lib=lib, pack=pack, stack=stack, quant_act=quant_act,
                              seg_gemm=seg_gemm, build_routing=build_routing)
    return _SPARSE


def install() -> None:
    method = os.environ.get("QB_DENSE", "bf16").lower()
    if method == "off":
        print("[qb_sm120] QB_DENSE=off -> native path (expect the SM120 fp8 wall)", flush=True)
        return

    import torch
    import torch.nn.functional as F

    try:
        from vllm.model_executor.layers.quantization.fp8 import Fp8LinearMethod
    except Exception as ex:  # noqa: BLE001
        print(f"[qb_sm120] Fp8LinearMethod import failed: {type(ex).__name__}: {ex}", flush=True)
        return

    nvq = None
    if method == "nvfp4":
        try:
            from flashinfer import SfLayout, mm_fp4, nvfp4_quantize
            nvq = (SfLayout, mm_fp4, nvfp4_quantize)
        except Exception as ex:  # noqa: BLE001
            print(f"[qb_sm120] flashinfer import failed {type(ex).__name__}; bf16", flush=True)
            method = "bf16"

    def _bw(self, layer, scale):
        bs = getattr(self, "weight_block_size", None) or getattr(
            getattr(self, "quant_config", None), "weight_block_size", None)
        if bs is not None:
            return bs
        # infer the block from weight vs scale shapes so we never fall through to the crashing
        # native path just because the attr moved: [N,K] weight, [ceil(N/bn),ceil(K/bk)] scale.
        w = getattr(layer, "weight", None)
        if w is not None and scale is not None and w.dim() == 2 and scale.dim() == 2:
            import math

            return (math.ceil(w.shape[0] / scale.shape[0]), math.ceil(w.shape[1] / scale.shape[1]))
        return None

    def _to_nvfp4(wbf):
        SfLayout, _, nvfp4_quantize = nvq
        gsb = ((448 * 6) / wbf.float().abs().nan_to_num().max().clamp_min(1e-8)).to(torch.float32)
        b_fp4, b_s = nvfp4_quantize(wbf, gsb, sfLayout=SfLayout.layout_128x4, do_shuffle=False)
        return (b_fp4, b_s, gsb, int(wbf.shape[0]), int(wbf.shape[1]))

    orig_pw = Fp8LinearMethod.process_weights_after_loading
    orig_apply = Fp8LinearMethod.apply

    def patched_pw(self, layer):
        scale = getattr(layer, "weight_scale_inv", None)
        bs = _bw(self, layer, scale)
        if bs is None or scale is None:
            STATS["native_layers"] += 1
            return orig_pw(self, layer)  # per-tensor fp8 / non-block: leave native path
        done = STATS["bf16_layers"] + STATS["nvfp4_layers"]
        if done == 0:
            print(f"[qb_sm120] fp8-fallback ACTIVE pid={os.getpid()} method={method}", flush=True)
        if done < 12:
            print(f"[qb_shape] w={tuple(layer.weight.shape)} s={tuple(scale.shape)} bs={bs}",
                  flush=True)
        # Build the bf16/nvfp4 replacement from the RAW load-format weight+scale (validated layout),
        # BEFORE vLLM reprocesses. Do NOT free the fp8 originals: MLA weight absorption reads
        # kv_b_proj/q_b_proj weights at load, and other consumers read weight shapes.
        wbf = _dequant_block(layer.weight.data, scale.data, bs)
        if method == "nvfp4":
            try:
                layer._qb_nv = _to_nvfp4(wbf)
                STATS["nvfp4_layers"] += 1
            except Exception:  # noqa: BLE001 -- flashinfer rejected shape: bf16 for this layer
                layer._qb_bf16 = wbf
                STATS["bf16_layers"] += 1
        else:
            layer._qb_bf16 = wbf
            STATS["bf16_layers"] += 1
        # let vLLM run its normal weight prep so MLA absorption / shape bookkeeping stay intact
        # (safe with VLLM_USE_DEEP_GEMM=0: this is tensor prep, not the unsupported SM120 GEMM).
        try:
            orig_pw(self, layer)
        except Exception as ex:  # noqa: BLE001
            print(f"[qb_sm120] orig process_weights raised {type(ex).__name__}; using ours", flush=True)
        return None

    def patched_apply(self, layer, x, bias=None):
        de = _ev_start() if _INSTR else None
        out = _patched_apply_inner(self, layer, x, bias)
        if _INSTR:
            _ev_end("dense", de)
            _CNT["dense_calls"] += 1
        return out

    def _patched_apply_inner(self, layer, x, bias=None):
        nv = getattr(layer, "_qb_nv", None)
        if nv is not None:
            STATS["fp8_calls"] += 1
            SfLayout, mm_fp4, nvfp4_quantize = nvq
            b_fp4, b_s, gsb, n, k = nv
            xf = x.reshape(-1, k).to(torch.bfloat16)
            amax = xf.float().abs().nan_to_num().max().clamp_min(1e-8)
            gsa = ((448 * 6) / amax).to(torch.float32)
            a_fp4, a_s = nvfp4_quantize(xf, gsa, sfLayout=SfLayout.layout_128x4, do_shuffle=False)
            out = torch.empty((xf.shape[0], n), device=xf.device, dtype=torch.bfloat16)
            mm_fp4(a_fp4, b_fp4.T, a_s, b_s.T, (1.0 / (gsa * gsb)), torch.bfloat16, out,
                   block_size=16, use_8x4_sf_layout=False, backend="cutlass", use_nvfp4=True)
            out = out.reshape(*x.shape[:-1], n)
            return out + bias if bias is not None else out
        wb = getattr(layer, "_qb_bf16", None)
        if wb is None:
            return orig_apply(self, layer, x, bias)
        STATS["fp8_calls"] += 1
        out = F.linear(x.to(wb.dtype), wb)
        return out + bias if bias is not None else out

    Fp8LinearMethod.process_weights_after_loading = patched_pw
    Fp8LinearMethod.apply = patched_apply

    # --- DeepSeek-V4 MLA o_proj: replace the DeepGEMM fp8_einsum path (asserts t.dim()==N on
    # sm_120) with a bf16 inverse-RoPE + torch.einsum. This op calls deep_gemm directly, bypassing
    # Fp8LinearMethod, so the linear patch above never sees it. bf16 is strictly more accurate than
    # the native fp8 einsum, so o_proj output tracks the reference.
    def _wo_a_bf16(wo_a):
        # Return wo_a's weight dequantized to bf16 (2D or 3D preserved). Prefer the plugin's
        # _qb_bf16 (built in patched_pw); else block-dequant the raw fp8 weight+scale.
        wa = getattr(wo_a, "_qb_bf16", None)
        if wa is not None:
            return wa
        scale = getattr(wo_a, "weight_scale_inv", None)
        w = wo_a.weight.data
        if scale is None:
            return w.to(torch.bfloat16)
        if w.dim() == 2:
            return _dequant_block(w, scale.data, (128, 128))
        # 3D bmm weight [G, N, K] with per-batch block scales: dequant each slice.
        return torch.stack([_dequant_block(w[i], scale.data[i], (128, 128))
                            for i in range(w.shape[0])], 0)

    def patched_o_proj(o, positions, cos_sin_cache, wo_a, wo_b, *, n_groups, heads_per_group,
                       nope_dim, rope_dim, o_lora_rank, einsum_recipe, tma_aligned_scales):
        wa = _wo_a_bf16(wo_a).to(torch.bfloat16)
        if STATS["oproj_calls"] == 0:
            sc = getattr(wo_a, "weight_scale_inv", None)
            print(f"[qb_sm120] o_proj bf16 path ACTIVE pid={os.getpid()} o={tuple(o.shape)} "
                  f"n_groups={n_groups} hpg={heads_per_group} o_lora_rank={o_lora_rank} "
                  f"wo_a.weight={tuple(wo_a.weight.shape)} wa={tuple(wa.shape)} "
                  f"qb_bf16={'y' if hasattr(wo_a, '_qb_bf16') else 'n'} "
                  f"scale={tuple(sc.shape) if sc is not None else None}", flush=True)
        STATS["oproj_calls"] += 1
        t = o.shape[0]
        head_dim = o.shape[2]
        o_r = _inv_rope_o(o, positions, cos_sin_cache, nope_dim, rope_dim)
        o_g = o_r.reshape(t, n_groups, heads_per_group * head_dim).to(torch.bfloat16)
        if wa.dim() == 2:
            wa = wa.reshape(n_groups, o_lora_rank, heads_per_group * head_dim)
        z = torch.einsum("bhr,hdr->bhd", o_g, wa)
        return wo_b(z.flatten(1))

    import importlib

    try:
        oc = importlib.import_module("vllm.models.deepseek_v4.nvidia.ops.o_proj")
        orig_op = oc.deep_gemm_fp8_o_proj
        oc.deep_gemm_fp8_o_proj = patched_o_proj
        import sys

        rebound = 1
        for mod in list(sys.modules.values()):
            if mod is not None and getattr(mod, "deep_gemm_fp8_o_proj", None) is orig_op:
                mod.deep_gemm_fp8_o_proj = patched_o_proj
                rebound += 1
        print(f"[qb_sm120] patched deep_gemm_fp8_o_proj in {rebound} module(s)", flush=True)
    except Exception as ex:  # noqa: BLE001
        print(f"[qb_sm120] o_proj patch skipped: {type(ex).__name__}: {ex}", flush=True)

    # --- DeepSeek Sparse Attention (Lightning Indexer): DeepGEMM's paged/dense MQA-logits kernels
    # assert "Unsupported architecture" on sm_120, and the indexer hard-requires DeepGEMM (no native
    # CUDA fallback). Replace the three DeepGEMM entry points with bf16 torch: metadata -> dummy
    # buffer (our path ignores SM scheduling), and the two logits kernels -> exact q.k top-k logits.
    # The FlashInfer sparse-MLA attention core IS sm_120-supported, so only these logits need owning.
    def _cnt_mqa(*a, **k):
        if STATS.get("mqa_prefill", 0) == 0:
            q, kv = a[0], a[1]
            print(f"[qb_sm120] indexer prefill mqa bf16 ACTIVE pid={os.getpid()} "
                  f"q={tuple(q[0].shape)}/{q[0].dtype} k={tuple(kv[0].shape)}/{kv[0].dtype} "
                  f"w={tuple(a[2].shape)}", flush=True)
        STATS["mqa_prefill"] = STATS.get("mqa_prefill", 0) + 1
        return _mqa_logits_bf16(*a, **k)

    def _cnt_paged(*a, **k):
        if STATS.get("mqa_decode", 0) == 0:
            q, kvc = a[0], a[1]
            print(f"[qb_sm120] indexer decode paged-mqa bf16 ACTIVE pid={os.getpid()} "
                  f"q={tuple(q[0].shape)}/{q[0].dtype} kv_cache={tuple(kvc.shape)}/{kvc.dtype} "
                  f"ctx={tuple(a[3].shape)} bt={tuple(a[4].shape)}", flush=True)
        STATS["mqa_decode"] = STATS.get("mqa_decode", 0) + 1
        return _paged_mqa_logits_bf16(*a, **k)

    def _meta_stub(context_lens, block_size, num_sms):
        return torch.zeros((num_sms + 1, 2), dtype=torch.int32, device=context_lens.device)

    try:
        dg = importlib.import_module("vllm.utils.deep_gemm")
        import sys

        repl = {"fp8_fp4_mqa_logits": _cnt_mqa, "fp8_fp4_paged_mqa_logits": _cnt_paged,
                "get_paged_mqa_logits_metadata": _meta_stub}
        origs = {n: getattr(dg, n, None) for n in repl}
        for n, fn in repl.items():
            setattr(dg, n, fn)
        rb = 0
        for mod in list(sys.modules.values()):
            if mod is None or mod is dg:
                continue
            for n, fn in repl.items():
                if getattr(mod, n, None) is origs[n] and origs[n] is not None:
                    setattr(mod, n, fn)
                    rb += 1
        print(f"[qb_sm120] patched indexer mqa-logits (+{rb} rebinds)", flush=True)
    except Exception as ex:  # noqa: BLE001
        print(f"[qb_sm120] indexer patch skipped: {type(ex).__name__}: {ex}", flush=True)

    # --- indexer top-k selection: vLLM's cooperative_topk/persistent_topk CUDA kernels use
    # cooperative CLUSTER launch, which fails on sm_120 ("launch_cooperative_cluster ... invalid
    # argument"). Override both with a pure-torch top-k that fills the same [rows, topk] index buffer
    # (order among the k is irrelevant to the downstream set-attention). Correctness-first.
    def _topk_into(logits, seq_lens, topk_indices, topk_tokens):
        rows, width = logits.shape
        sl = seq_lens.reshape(-1)
        topk_indices[:rows, :topk_tokens] = -1
        for r in range(rows):
            n = int(sl[r].item()) if r < sl.numel() else width
            n = max(0, min(n, width))
            if n <= 0:
                continue
            k = min(topk_tokens, n)
            idx = torch.topk(logits[r, :n], k).indices.to(topk_indices.dtype)
            topk_indices[r, :k] = idx

    def _coop_topk(logits, seq_lens, topk_indices, topk_workspace, topk_tokens, max_seq_len):
        if STATS.get("topk_calls", 0) == 0:
            print(f"[qb_sm120] torch top-k ACTIVE pid={os.getpid()} logits={tuple(logits.shape)} "
                  f"topk={topk_tokens}", flush=True)
        STATS["topk_calls"] = STATS.get("topk_calls", 0) + 1
        _topk_into(logits, seq_lens, topk_indices, topk_tokens)

    try:
        # setattr on the _OpNamespace shadows the cached OpOverloadPacket, so subsequent
        # torch.ops._C.cooperative_topk(...) calls dispatch to our python impl.
        setattr(torch.ops._C, "cooperative_topk", _coop_topk)
        setattr(torch.ops._C, "persistent_topk", _coop_topk)
        print("[qb_sm120] overrode cooperative_topk/persistent_topk with torch top-k", flush=True)
    except Exception as ex:  # noqa: BLE001
        print(f"[qb_sm120] topk override skipped: {type(ex).__name__}: {ex}", flush=True)

    _install_moe()

    if _INSTR or _TAX:
        import atexit

        atexit.register(_flush_metrics)
    print(f"[qb_sm120] installed in pid={os.getpid()} (method={method})", flush=True)


def _install_moe() -> None:
    # QB_MOE selects the MoE expert path (ModelOptNvFp4FusedMoE):
    #   off    (default) native vLLM NVFP4 MoE -- WS0/WS1 unaffected.
    #   dense  dequant NVFP4 experts -> bf16 on-the-fly per routed expert (validates dequant + the
    #          apply-replacement + expert-parallel handling; ~native quality, slow eager loop).
    #   sparse quadbit 2:4 sparse-FP4 experts via the staged sm_120 kernel (.so) -- the thesis path;
    #          sets SPARSE_EXPERT_CALLS>0 (falls back to dense if the .so path is unavailable).
    qb_moe = os.environ.get("QB_MOE", "off").lower()
    if qb_moe not in ("dense", "sparse"):
        return

    import torch
    import torch.nn.functional as F

    try:
        from vllm.model_executor.layers.quantization.modelopt import ModelOptNvFp4FusedMoE
        from vllm.model_executor.layers.fused_moe.runner.shared_experts import (
            SharedExpertsOrder,
        )
    except Exception as ex:  # noqa: BLE001
        print(f"[qb_sm120] MoE patch skipped (import): {type(ex).__name__}: {ex}", flush=True)
        return

    STATS["moe_calls"] = 0
    STATS["sparse_expert_calls"] = 0

    def patched_moe_pw(self, layer):
        # Keep the RAW NVFP4 expert weights on the layer (dequant on the fly in apply); do NOT run
        # the native conversion/kernel build (it would rewrite w13_weight into CUTLASS layout and
        # ~double expert memory). Mark experts so apply knows the raw layout is intact.
        self.moe_kernel = None
        layer._qb_moe = True
        i = layer.w13_weight.shape[1] // 2
        first = STATS["moe_calls"] == 0
        if first:
            print(f"[qb_sm120] MoE {qb_moe} path: raw NVFP4 kept "
                  f"w13={tuple(layer.w13_weight.shape)} w2={tuple(layer.w2_weight.shape)} "
                  f"I={i} experts={layer.w13_weight.shape[0]} "
                  f"emap={'y' if getattr(layer, 'expert_map', None) is not None else 'n'}", flush=True)
        if qb_moe == "sparse":
            global _PW_IDX
            layer_idx = _PW_IDX
            _PW_IDX += 1
            layer._qb_layer_idx = layer_idx
            # A1 policy #1: dense-anchor layers keep raw NVFP4 and skip packing entirely. apply then
            # falls through to the per-expert dense NVFP4 path (cos=1.0 vs native, breaks the
            # per-layer tax compounding at these layers). Costs NVFP4 residency (~1.7GB/layer/rank).
            if layer_idx in _DENSE_LAYERS:
                layer._qb_dense_anchor = True
                if first:
                    print(f"[qb_sm120] dense-anchor layer {layer_idx}: kept NVFP4 (no packing)",
                          flush=True)
                return None
            # A0 map: on probe layers keep NVFP4 resident so apply can build the dense reference on
            # the SAME real routed rows; these layers do NOT free NVFP4.
            is_probe = _QMAP and (layer_idx % _TAX_LAYERS == 0)
            # Pack each NVFP4 expert -> bf16 -> quadbit 2:4-sparse-FP4 codes at load. gate+up = the
            # full w13[e] [2I,H]; down = w2[e] [H,I]. NVFP4 stays resident (dequant is transient,
            # one expert at a time); the packed codes (~1GB/rank) live alongside for apply.
            sp = _load_sparse_moe()
            e = layer.w13_weight.shape[0]
            w13, w13s, w13s2 = layer.w13_weight, layer.w13_weight_scale, layer.w13_weight_scale_2
            w2, w2s, w2s2 = layer.w2_weight, layer.w2_weight_scale, layer.w2_weight_scale_2
            gu_packs, dn_packs = [], []
            for le in range(e):
                gu = _dequant_nvfp4_expert(w13[le], w13s[le], w13s2[le, 0])
                dn = _dequant_nvfp4_expert(w2[le], w2s[le], w2s2[le])
                gu_packs.append(sp.pack(gu))
                dn_packs.append(sp.pack(dn))
            layer._qb_gu = sp.stack(gu_packs)
            layer._qb_dn = sp.stack(dn_packs)
            layer._qb_i = i
            layer._qb_e = e
            del gu_packs, dn_packs
            # in-situ per-layer tax probe on sampled layers -- MUST run before freeing NVFP4
            if _TAX and (layer_idx % _TAX_LAYERS == 0):
                _run_tax_probe(layer, sp, i, e, layer_idx)
                _flush_metrics()
            if is_probe:
                layer._qb_probe = True
                if first:
                    print(f"[qb_sm120] qmap probe layer {layer_idx}: kept NVFP4 for real-activation "
                          "dense reference", flush=True)
            else:
                # Free this layer's raw NVFP4 experts now that they are packed: sparse codes
                # (~1.15GB) are smaller than the NVFP4 experts (~1.7GB) per layer, so freeing keeps
                # peak memory at the already-working dense load (~84GB) and it drops as packing
                # proceeds. Without this the codes accumulate ON TOP of NVFP4 across 43 layers -> OOM.
                for attr in ("w13_weight", "w13_weight_scale", "w13_weight_scale_2",
                             "w2_weight", "w2_weight_scale", "w2_weight_scale_2"):
                    p = getattr(layer, attr, None)
                    if p is not None:
                        p.data = torch.empty(0, dtype=p.dtype, device=p.device)
                torch.cuda.empty_cache()
            if first:
                print(f"[qb_sm120] sparse-packed {e} experts: gu codes "
                      f"{layer._qb_gu[0].shape} dn codes {layer._qb_dn[0].shape}; freed NVFP4",
                      flush=True)
        return None

    def patched_moe_apply(self, layer, x, topk_weights, topk_ids, shared_experts=None,
                          shared_experts_input=None):
        STATS["moe_calls"] += 1
        t, h = x.shape
        y = torch.zeros(t, h, dtype=x.dtype, device=x.device)
        topk = topk_ids.shape[1]
        on_input = bool(getattr(layer, "apply_router_weight_on_input", False))

        if qb_moe == "sparse" and getattr(layer, "_qb_gu", None) is not None:
            # Route through the quadbit 2:4-sparse-FP4 segmented kernel. emap is None here (all
            # experts present per rank, TP-sharded intermediate), so topk_ids are global==local.
            sp = _load_sparse_moe()
            ii, ee = layer._qb_i, layer._qb_e
            assign = topk_ids.reshape(-1).to(torch.long)
            tok_of = torch.arange(t, device=x.device).repeat_interleave(topk)
            w_of = topk_weights.reshape(-1).to(torch.float32)
            src, eblk, _r = sp.build_routing(assign, ee)
            valid = src >= 0
            srcc = src.clamp_min(0)
            xs = x[tok_of[srcc]].to(torch.bfloat16) * valid[:, None]
            if on_input:
                xs = xs * w_of[srcc][:, None].to(torch.bfloat16)
            se = _ev_start() if _INSTR else None
            gu = sp.seg_gemm(xs, layer._qb_gu, 2 * ii, h, eblk)
            hh = (F.silu(gu[:, :ii].float()) * gu[:, ii:].float()).to(torch.bfloat16)
            dseg = sp.seg_gemm(hh, layer._qb_dn, h, ii, eblk)
            if _INSTR:
                _ev_end("expert", se)
                _CNT["expert_calls"] += 1
                # sample imbalance sparsely: the .item() forces a sync, so avoid doing it every apply
                if _CNT["expert_calls"] % 100 == 0:
                    cnt = torch.bincount(assign, minlength=ee).float()
                    nz = cnt[cnt > 0]
                    if nz.numel():
                        _IMB.append((cnt.max() / nz.mean()).item())
                    _flush_metrics()
            rw = valid.float() if on_input else (w_of[srcc] * valid.float())
            y.index_add_(0, tok_of[srcc], (dseg.float() * rw[:, None]).to(x.dtype))
            STATS["sparse_expert_calls"] += int(valid.sum().item())
            if _QMAP and getattr(layer, "_qb_probe", False):
                _run_qmap_probe(layer, sp, x, topk_ids, topk_weights, y, on_input)
            if shared_experts is not None and shared_experts_input is not None:
                shared_experts(shared_experts_input, SharedExpertsOrder.MK_INTERNAL_OVERLAPPED)
            return y

        w13, w13s, w13s2 = layer.w13_weight, layer.w13_weight_scale, layer.w13_weight_scale_2
        w2, w2s, w2s2 = layer.w2_weight, layer.w2_weight_scale, layer.w2_weight_scale_2
        i = w13.shape[1] // 2
        emap = getattr(layer, "expert_map", None)
        flat_ids = topk_ids.reshape(-1)
        flat_w = topk_weights.reshape(-1).to(torch.float32)
        tok = torch.arange(t, device=x.device).repeat_interleave(topk)
        local = emap[flat_ids] if emap is not None else flat_ids
        for le in torch.unique(local).tolist():
            if le < 0:
                continue
            sel = local == le
            rows = tok[sel]
            ws = flat_w[sel]
            xe = x[rows].float()
            if on_input:
                xe = xe * ws[:, None]
            wg = _dequant_nvfp4_expert(w13[le, :i], w13s[le, :i], w13s2[le, 0]).float()
            wu = _dequant_nvfp4_expert(w13[le, i:], w13s[le, i:], w13s2[le, 0]).float()
            wd = _dequant_nvfp4_expert(w2[le], w2s[le], w2s2[le]).float()
            hh = F.silu(xe @ wg.t()) * (xe @ wu.t())
            oe = hh @ wd.t()
            if not on_input:
                oe = oe * ws[:, None]
            y.index_add_(0, rows, oe.to(x.dtype))
            STATS["sparse_expert_calls"] += 1
        # The MoERunner owns shared-expert execution: it calls shared_experts with NO_OVERLAP
        # (before) and MULTI_STREAM_OVERLAPPED (after) itself, then reads shared_experts.output
        # and combines it with our fused_out. We only fire the MK_INTERNAL_OVERLAPPED slot the
        # modular apply is contracted to fire; it is a no-op unless that order was selected (it
        # never is here, since we stubbed maybe_make_prepare_finalize -> no modular kernel). So
        # we must NOT add shared output to y, or it would be double-counted.
        if shared_experts is not None and shared_experts_input is not None:
            shared_experts(shared_experts_input, SharedExpertsOrder.MK_INTERNAL_OVERLAPPED)
        return y

    ModelOptNvFp4FusedMoE.process_weights_after_loading = patched_moe_pw
    ModelOptNvFp4FusedMoE.apply = patched_moe_apply
    # We skip the native modular-kernel build (moe_kernel=None), which flips supports_internal_mk to
    # False so MoERunner.maybe_init_modular_kernel would call maybe_make_prepare_finalize (which the
    # class raises in). Return None instead: no prepare/finalize -> no FusedMoEModularMethod wrapping
    # -> the runner keeps our patched quant_method.apply as the expert path.
    ModelOptNvFp4FusedMoE.maybe_make_prepare_finalize = (
        lambda self, routing_tables=None: None)
    print(f"[qb_sm120] patched ModelOptNvFp4FusedMoE (QB_MOE={qb_moe}) pid={os.getpid()}", flush=True)
