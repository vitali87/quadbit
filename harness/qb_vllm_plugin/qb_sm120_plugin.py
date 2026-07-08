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

    print(f"[qb_sm120] installed in pid={os.getpid()} (method={method})", flush=True)
