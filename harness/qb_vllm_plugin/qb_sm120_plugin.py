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
STATS = {"nvfp4_layers": 0, "bf16_layers": 0, "native_layers": 0, "fp8_calls": 0}


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
    se = s.to(torch.float32).repeat_interleave(fn, 0)[:n].repeat_interleave(fk, 1)[:k]
    return (w.to(torch.float32) * se).to(torch.bfloat16)


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
    print(f"[qb_sm120] installed in pid={os.getpid()} (method={method})", flush=True)
