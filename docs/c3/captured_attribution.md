# C3 Task 1A: captured-mode differential attribution (DeepSeek-D2, 4 GPU)

CUDA events can't time inside a CUDA graph, and the eager profile ([profile_decode.md](profile_decode.md)) is confounded by
the dense-anchor Python quant loop's launch overhead (which capture removes). So we attribute the
**captured** decode cost by differential: run the captured D2 path with one component no-op'd at a time
(env flags `QB_C3_SKIP_{MOE,DENSE,SPARSE}`, plugin-side, graph stays shape-valid) and read decode-only
tok/s. PPL is meaningless when a component is skipped, **timing only**. Harness
`graph_gate4 ... --c3-skip {moe,dense,sparse}`; same D2 config as C2 (route_slot=2, native anchor, cap=128,
max_seqs=2, captured). Logs `docs/audit/logs/c3_diff_*.log`.

## Variants

| id | flag | what runs | isolates |
|---|---|---|---|
| A baseline | (none) | full captured D2 | reference (C2 = 5.972 tok/s) |
| D skip-moe | `QB_C3_SKIP_MOE` | everything but the MoE expert apply (attention/DSA/EP/norms; MoE outputs zeros) | the non-MoE ceiling |
| B skip-sparse | `QB_C3_SKIP_SPARSE` | dense-anchor group + routing, sparse tail no-op | dense-anchor captured cost |
| C skip-dense | `QB_C3_SKIP_DENSE` | sparse tail + routing, dense-anchor no-op | sparse-group captured cost |

## Results (all captured, DeepSeek-D2, 4 GPU, same C2 config)

| variant | decode tok/s | ms/step | isolates |
|---|---:|---:|---|
| A baseline | 5.782 | 172.9 | full captured D2 (≈ C2 5.972) |
| D skip-moe | **51.033** | 19.6 | non-MoE floor (attention/DSA/EP/norms) |
| C skip-dense | 16.067 | 62.3 | sparse group + routing + floor |
| B skip-sparse | 7.642 | 130.9 | dense-anchor + routing + floor |

## Attribution (step budget, ms/token — additive and self-consistent)

| component | ms/step | share | source |
|---|---:|---:|---|
| non-MoE floor (attention/DSA/EP/norms) | 19.6 | 11% | D directly |
| **dense-anchor group** (C1 `group_gemm` + 64 per-group quant kernels, E·cap=8192 rows) | **110.7** | **64%** | A − B |
| **sparse group** (2:4 tail, E·cap=8192 rows) | **42.0** | **24%** | A − C |
| routing/plumbing (shared) | ~0.6 | <1% | residual |
| **total (baseline step)** | **172.9** | 100% | A |

Cross-check: dense 110.7 + sparse 42.0 + floor 19.6 + routing 0.6 = 172.9 ✓. Removing the whole MoE (D)
gives 51 tok/s, **above** the dense NVFP4 fused baseline's 48.248, so the attention/DSA/EP path is not
the wall; the MoE apply is **89% of the captured decode step**.

Note: the sparse group is 24% of the step, yet the sparse `matmul_sp` **kernel** is only 0.4% (eager
profile). So the sparse group's 42 ms is almost entirely **per-row overhead** (gather `x[tok]`, per-group
quant, `index_add` scatter) on the 8192 padded rows, **not** the kernel. Same character for the
dense-anchor group. **The cost is the E·cap padding, in both groups, and it is overhead + padded compute
that capture does not remove, exactly what compact routing attacks.**

## Verdict (Task 1A): compact routing is the right lever; proceed to Task 1B.

- The refuted sparse-decode-kernel premise stays refuted (`matmul_sp` 0.4%).
- The captured bottleneck is the **MoE apply's E·cap=8192-row padding** (dense-anchor 64% + sparse 24%),
  dominated by per-row overhead + padded compute, not by any mma kernel.
- Attention/DSA/EP is a cheap 11% floor (51 tok/s ceiling if MoE were free), large headroom.
- Next (Task 1B): shrink the row count both groups process from E·cap toward the real decode tokens via
  **active-expert compaction** (fixed A_max active experts × cap instead of all E=64), capture-safe, plus a
  single batched quant replacing the 64-iteration Python loop. Attack the dense-anchor group (64%) first.
