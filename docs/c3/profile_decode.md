# C3 Task 1: DeepSeek-D2 sparse decode profile

Branch `c3-fused-sparse-grouped-decode`. Goal: before writing any CUDA, measure exactly where the 5.972
tok/s (≈167 ms/token) of the C2 D2 native-captured decode goes, and confirm whether a fused sparse grouped
decode primitive is the right lever. Harness `serve_dsv4.py::c3_profile` (config B: graph-safe **eager**,
`QB_GRAPH=1` + `enforce_eager=True`, native anchor, `QB_PROFILE=1`), so `torch.profiler` attributes
per-kernel CUDA time cleanly. Log [c3_profile_decode.log](../audit/logs/c3_profile_decode.log).

## Structural finding (from the code, confirmed before measuring)

Route-slot D2 runs **two** groups per MoE layer, both under fixed-capacity routing padded to **E·cap**:

- **Local experts E = 64** (256+1 experts / 4-GPU EP), **cap = 128** (`_BN` multiple, compile-time const
  for capture) → each group's buffer is **E·cap = 8192 rows**.
- At decode, `max_seqs = 2` and the tail-6 sparse slots per token mean only **~O(B·tail) ≈ a dozen** rows
  are real; the other ~8180 are padding processed as zeros.
- **Dense group** (top-2 slots): `_anchor_gemm` → FlashInfer `group_gemm_nvfp4` over the 8192-row buffer.
- **Sparse group** (tail-6): `_seg_apply_gs` → `sp.seg_into` → the `matmul_sp` kernel
  (`cuda/sparse_fp4_lib.cu`, BM=BN=128, 2-warpgroup pingpong, grid `(N/BN, M/2BM)` with N = 8192 tokens),
  which tiles N in 128-row blocks, **one expert per block**, so it computes the full 8192×out_f grouped
  GEMM every layer regardless of how few rows are real.

So the *fixed-capacity padding* (E·cap = 8192 vs ~12 real) is the suspected root cause, and it inflates
**both** the dense-anchor `group_gemm` and the sparse `matmul_sp`. The measured breakdown below decides
whether the fix is (a) a compact-routing decode path that shrinks the row count for both groups, (b) a
decode-specialized sparse kernel, or (c) neither (if attention/DSA or NCCL dominate, sparse decode is not
the lever).

## Measured MoE-apply breakdown (config B eager, worker-side CUDA events, all 4 workers agree)

Driver-side `torch.profiler` sees nothing under vLLM V1 (model runs in worker subprocesses), so the plugin
times each region with CUDA events **inside the workers**. Cumulative over the run (worker TP0), log
[c3_profile_decode.log](../audit/logs/c3_profile_decode.log):

| MoE-apply region (route-slot D2 layers) | cumulative ms | share |
|---|---:|---:|
| **dense_anchor** (top-2 dense slots: C1 `group_gemm` + per-group quant) | 155,494 | **98%** |
| sparse_group total (tail-6 2:4) | 2,501 | 2% |
| ↳ sparse_seg (`matmul_sp` kernel) | 606 | **0.4%** |
| ↳ sparse_quant (`quant_into`) | 177 | 0.1% |

## Two robust conclusions (survive the eager→capture caveat)

1. **The sparse 2:4 decode kernel is NOT the bottleneck.** `matmul_sp` is **0.4%** of MoE-apply even in
   eager. Capture only removes launch overhead (helping launch-bound paths), so under capture `matmul_sp`
   stays negligible. **The C3 core hypothesis, "sparse decode at tiny M lacks a fused grouped GEMM", is
   refuted.** A `fused_sparse_grouped_decode_nvfp4_2lvl` would optimize a 0.4% path.
2. **The eager 98% dense_anchor is ~97% a Python launch-overhead artifact, not compute.** `_dense_seg_native`
   (the C1 native path) runs a **64-iteration `for le in range(e)` loop** of per-group `amax` +
   `nvfp4_quantize` (~250+ tiny kernel launches per projection, ×2 projections ×21 layers). Decomposing the
   cumulative: sparse_group's non-kernel overhead is ~1,719 ms and its `group_gemm`-analogue `matmul_sp` is
   ~606 ms; subtracting a comparable overhead + matmul from dense_anchor leaves **~153,000 ms ≈ 97% of
   dense_anchor as the per-group Python quant loop**. That is pure eager launch overhead, which
   **CUDA-graph capture unrolls away**. So this eager 98% does **not** represent the captured deployment
   (5.972 tok/s), where the loop is captured.

## What the eager profile can and cannot say

- **Can:** rule the sparse 2:4 kernel out as the lever (robust).
- **Cannot:** localize the *captured* bottleneck. The eager profile is confounded by the launch-overhead of
  the dense-anchor Python quant loop, which capture removes. The genuine *captured* costs are the ones
  capture does NOT remove, chiefly the **E·cap = 8192-row padding compute** through `group_gemm`
  (dense group) and `matmul_sp` (sparse group), and the attention/DSA/EP path. These need a captured-mode
  measurement (differential decode-tok/s, or nsys timeline) that CUDA events cannot provide inside a graph.

## Verdict (Task 1 gate): **do NOT build the fused sparse grouped decode kernel.**

Task 1 **refutes** the premise the kernel was to address (sparse decode is 0.4%, not the bottleneck). The
real lever is the **fixed-capacity E·cap padding** (8192 padded rows vs a real handful) that inflates the
*dense-anchor* `group_gemm` and its per-group quant loop, and equally the sparse group's padded rows, a
**routing/plumbing** fix (compact-route to real token count + batch the per-group quant into one call),
graph-capturable, **no new sparse mma kernel**. Exact captured attribution of that padding vs
attention/DSA/EP needs a captured-mode differential, which is the recommended next measurement. Reported to
the user for a direction decision rather than building the refuted kernel (per the Task 1 gate).
