# C3 Task 1B: compact fixed-capacity routing (active-expert compaction)

Attribution (`captured_attribution.md`): the captured D2 decode step is 89% MoE-apply, and it is the
**E·cap = 64×128 = 8192-row padding** in both groups (dense-anchor 64%, sparse 24%) — per-row overhead
(gather / per-group quant / scatter) + padded compute on rows that are ~99.8% padding at decode. The lever
is to shrink the rows each group processes toward the real decode tokens, **capture-safe**.

## Why not "compact to exactly the real rows"

FlashInfer `group_gemm_nvfp4_nt_groupwise` needs the `a_scale` in a per-group 128-row-aligned swizzle
(`layout_128x4`), and the `matmul_sp` seg kernel tiles N in `_BN=128`-row blocks (one expert per block).
Both force **cap ≥ 128 per expert** and a **static** per-group `a_scale` offset. A fully-compact
variable-group layout would make those offsets data-dependent → dynamic shapes → **capture-illegal**. So we
cannot drop below cap=128 per group without a new kernel/layout (deferred).

## What IS capture-safe: reduce the number of GROUPS (E → A_max active experts)

At decode the captured batch sizes are small (vLLM captures B∈{1,2,4}). The routed (token,expert) slots =
B×top_k, hitting at most B×2 dense experts and B×6 sparse experts. So **almost all of the 64 local experts
get zero tokens** and are pure waste. Process only a fixed `A_max` active experts:

```
route_compact(assign, e=64, cap=128, A_max, valid):
  counts   = scatter_add over valid rows        # [64] device, no host sync
  active   = counts.topk(A_max).indices         # [A_max] device, fixed size (capture-safe)
  slot_of  = full(64, -1); slot_of[active] = arange(A_max)
  a_slot   = slot_of[assign]                     # active slot per row, -1 if not in top-A_max
  src,eblk,drop = route_fixed_cap(a_slot, A_max, cap, valid & (a_slot>=0))   # A_max*cap buffer
  return src, eblk, drop, active                 # `active` = global expert id per active slot
```

Then the group GEMM runs over **A_max groups** with **gathered weights** `b_w[active]`, `b_sf[active]`,
`wgsf[active]` (device index, fixed `[A_max]`), and `m_indptr = arange(A_max+1)*cap` (static, 128-aligned
`a_scale` preserved). Scatter back by the same `tok_of[src]`. Everything is fixed-shape device work → the
whole thing captures.

**Row reduction:** E·cap = 8192 → A_max·cap. For the dense group A_max=8 covers B≤4 (8×128=1024, 8×
fewer rows + 8× fewer per-group quant kernels). For the sparse group A_max=24 covers B≤4 (24×128=3072,
2.7×). Deterministic overflow: experts beyond the top-A_max by token-count are dropped — none at decode
(active ≤ A_max by construction), a graceful degradation only if a batch is unexpectedly wide.

## Also removes the eager quant-loop artifact

`_dense_seg_native`'s per-group `for le in range(e)` quant loop shrinks from 64 to A_max iterations, so the
number of tiny quant kernels drops 8× too (helps both eager and captured).

## Residual (named up front)

The remaining waste is **cap=128 per active expert** (128 rows for ~1 real token). Killing that needs a
decode kernel/layout that processes arbitrary compact rows with per-row expert (the deferred custom path).
So active-expert compaction is expected to reach a **strict-Pareto** improvement (materially faster + same
quality, dual-residency memory unchanged), not necessarily the full 48 tok/s. Measured next.

## Flags (opt-in, default off — deployed path unchanged)

`QB_COMPACT_DECODE=1`, `QB_A_DENSE` (default 8), `QB_A_SPARSE` (default 24). Applies to route-slot D2.
