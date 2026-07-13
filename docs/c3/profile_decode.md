# C3 Task 1: DeepSeek-D2 sparse decode profile

Branch `c3-fused-sparse-grouped-decode`. Goal: before writing any CUDA, measure exactly where the 5.972
tok/s (≈167 ms/token) of the C2 D2 native-captured decode goes, and confirm whether a fused sparse grouped
decode primitive is the right lever. Harness `serve_dsv4.py::c3_profile` (config B: graph-safe **eager**,
`QB_GRAPH=1` + `enforce_eager=True`, native anchor, `QB_PROFILE=1`), so `torch.profiler` attributes
per-kernel CUDA time cleanly. Log `docs/audit/logs/c3_profile_decode.log`.

## Structural finding (from the code, confirmed before measuring)

Route-slot D2 runs **two** groups per MoE layer, both under fixed-capacity routing padded to **E·cap**:

- **Local experts E = 64** (256+1 experts / 4-GPU EP), **cap = 128** (`_BN` multiple, compile-time const
  for capture) → each group's buffer is **E·cap = 8192 rows**.
- At decode, `max_seqs = 2` and the tail-6 sparse slots per token mean only **~O(B·tail) ≈ a dozen** rows
  are real; the other ~8180 are padding processed as zeros.
- **Dense group** (top-2 slots): `_anchor_gemm` → FlashInfer `group_gemm_nvfp4` over the 8192-row buffer.
- **Sparse group** (tail-6): `_seg_apply_gs` → `sp.seg_into` → the `matmul_sp` kernel
  (`cuda/sparse_fp4_lib.cu`, BM=BN=128, 2-warpgroup pingpong, grid `(N/BN, M/2BM)` with N = 8192 tokens),
  which tiles N in 128-row blocks, **one expert per block** — so it computes the full 8192×out_f grouped
  GEMM every layer regardless of how few rows are real.

So the *fixed-capacity padding* (E·cap = 8192 vs ~12 real) is the suspected root cause, and it inflates
**both** the dense-anchor `group_gemm` and the sparse `matmul_sp`. The measured breakdown below decides
whether the fix is (a) a compact-routing decode path that shrinks the row count for both groups, (b) a
decode-specialized sparse kernel, or (c) neither (if attention/DSA or NCCL dominate, sparse decode is not
the lever).

## Measured routing waste (QB_PROFILE, per layer, first decode step)

_PENDING (from `[qb_profile]` log lines)._ Expect `rp(E*cap)=8192`, `real_rows≈O(10)`, `waste≈hundreds×`.

## Measured kernel breakdown (config B eager, torch.profiler, 32-tok decode)

_PENDING._ Categories: sparse-seg (`matmul_sp`), dense-anchor (`group_gemm`/cutlass fp4), attention/DSA-MLA,
NCCL/EP, quant/act, elementwise/other. The bottleneck table + top-30 kernels fill here.

## Verdict (Task 1 gate)

_PENDING._ Confirm sparse grouped decode (or the routing padding feeding it) is the bottleneck → proceed to
Task 2 design. If attention/DSA/NCCL dominate instead, stop and re-scope.
