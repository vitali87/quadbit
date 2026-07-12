# C1 design note: native NVFP4 grouped backend for the dense-anchor branch

**Branch:** `c1-native-dense-anchor` (off `main` @ `a6f545e`). **Goal:** turn the graph-*correct*
deployed D2/down49 path into a graph-*fast* path by replacing the decode-slow dense-anchor projection
(`_dense_seg_gs`, a `range(E)` dequant-to-bf16 loop) with a native fused NVFP4 grouped backend, keeping
sparse experts on `sparse_moe_mm_2lvl`, routing on `route_fixed_cap`, DSA native, no host sync, no
in-capture allocation, graph capture enabled.

## The bottleneck being replaced

`_dense_seg_gs(xs, w, ws, ws2, out_dim, e, cap)` (plugin `qb_sm120_plugin.py:176`) loops `range(e)`
(~64 local experts at DeepSeek EP tp=4), dequantising each expert's NVFP4 weight to bf16 and running a
`[cap,K] @ [K,N]` matmul per expert into a fixed-capacity seg buffer. It captures and is bit-exact, but
touches **all E experts every decode step** regardless of how few rows are routed. This is the precise,
already-attributed decode-speed limiter the P4 verdict named: *"the dense anchored/grouped projection
path, which lacks a fused dense NVFP4 grouped-GEMM."*

## Recon finding: FlashInfer 0.6.14 ships the missing primitive(s)

The P4 assumption ("no fused dense NVFP4 grouped-GEMM") is **false on this stack**. Two native paths
exist (recon logs `docs/audit/logs/c1_recon.log`, `c1_recon2.log`):

### 1. Grouped NVFP4 GEMM — direct drop-in for `_dense_seg_gs` (covers every policy)

```
flashinfer.gemm.group_gemm_nvfp4_nt_groupwise(
    a, b, a_scale, b_scale, m_indptr, alpha=None,
    tile_m=128, tile_n=128, tile_k=128, swap_ab=True, out=None, out_dtype=None)
```

A grouped matmul where row-segments (`m_indptr`) each multiply a different expert weight — exactly the
seg-layout `_dense_seg_gs` already produces. NVFP4-native (no bf16 dequant), so it should be both faster
and lighter. Works for **down49** (single gate_up projection), **gateup49** (single down projection),
and **D2** (both projections) — replace each `_dense_seg_gs` call with one `group_gemm_nvfp4` call.

Exact contract (recon `c1_recon2.log`): `a` = (cum_m, k//2) uint8 packed NVFP4 acts; `b` = (E, n, k//2)
uint8 packed weights; `a_scale` = (cum_m_padded, k//16) uint8 swizzled; `b_scale` = (E, n_padded, k//16)
uint8; `m_indptr` = (E+1,) int32 with **every offset a multiple of 4** (cap=128 satisfies this);
`alpha` = (E,) fp32 = 1/(a_global_sf · b_global_sf[e]); tile_m=128. Activations quantised with
`nvfp4_quantize(x, gsf, sfLayout=layout_128x4, do_shuffle=False)` (the cutlass/sm120 layout). There is
no group_gemm_nvfp4 example in the installed wheel, so the standalone A/B pins the swizzle empirically.

### 2. Native fused MoE over restricted routing — D2-specific

`ModelOptNvFp4FusedMoE.apply(layer, x, topk_weights, topk_ids, ...)` (modelopt.py:2236) **consumes
external `topk_ids`/`topk_weights`** and forwards to `moe_kernel.apply(...)` → CUTLASS SM120 fused MoE
(`flashinfer.fused_moe.cutlass_fused_moe`). So D2's dense group (top-2 slots, both projections + swiglu)
is expressible as one native fused-MoE call over a top-2 routing table; the sparse tail-6 stays on the
seg kernel. This reuses the already-working `_qb_native` delegation machinery.

## Which primitive C1 uses, and why

Lead with **`group_gemm_nvfp4_nt_groupwise`**: it is the literal drop-in for the bottleneck function,
NVFP4-native, and generalises across all three deployed policies rather than being D2-only. The fused-MoE
delegation (#2) is the fallback for D2 both-projection if the grouped GEMM's activation-quant + swiglu
plumbing proves fiddlier than the fused kernel's one-call path.

## Memory contract (D2)

D2 needs both the native NVFP4 weights (for the dense grouped GEMM) and the packed 2:4 codes (for the
sparse tail). Codes are packed transiently from raw NVFP4 at load; the raw NVFP4 then stays resident for
`group_gemm_nvfp4` (no CUTLASS re-layout needed — the grouped GEMM reads the same packed-uint8 + e4m3
block-scale + global-scale layout the checkpoint already has, modulo the swizzle the standalone A/B
pins down). Resident = NVFP4 weights + 2:4 codes = the current D2 profile. Memory-neutral.

## Validation ladder (gates, in order)

- **A. Standalone dense-anchor A/B** (`harness/c1_dense_anchor.py`, 1 GPU): same weights/scales/routing/
  input, `_dense_seg_gs` bf16 reference vs `group_gemm_nvfp4`. Report cos, relL2, max-abs, nonfinite,
  eager + graph-replay latency, capture success. **If the grouped GEMM cannot express the branch here,
  stop and write the failure table (exact blocker); do not proceed to serving.**
- **B. DeepSeek-D2 serving A/B/C** (`serve_dsv4.py::graph_gate4 --dense-anchor-backend native_nvfp4`):
  (1) frozen all-E-dequant D2 graph, (2) graph-safe eager native-delegate, (3) captured native-delegate.
  Metrics: PPL, coherent gen, decode/prefill tok/s, capture status, graph pool mem, route/drop/overflow,
  DSA native status, memory, component timing (sparse / dense-anchor / routing / DSA / EP).

## Success / failure

Success = native-delegate D2 preserves quality vs current D2 **and** materially cuts dense-anchor decode
time (SOTA-relevant: a graph-enabled D2 row beating the best dense/NVFP4 baseline or a strictly better
speed/quality/memory Pareto point). Failure = a precise table (unsupported layout / route-group semantics
/ scales / not graph-capturable / quality mismatch / slower than all-E dequant / inaccessible API), then
the next step is a custom `grouped_dense_nvfp4_moe_mm_2lvl`. **No GLM 8-GPU and no custom CUDA before the
D2 gate resolves.**
