# C3 Task 1A: captured-mode differential attribution (DeepSeek-D2, 4 GPU)

CUDA events can't time inside a CUDA graph, and the eager profile (`profile_decode.md`) is confounded by
the dense-anchor Python quant loop's launch overhead (which capture removes). So we attribute the
**captured** decode cost by differential: run the captured D2 path with one component no-op'd at a time
(env flags `QB_C3_SKIP_{MOE,DENSE,SPARSE}`, plugin-side, graph stays shape-valid) and read decode-only
tok/s. PPL is meaningless when a component is skipped — **timing only**. Harness
`graph_gate4 ... --c3-skip {moe,dense,sparse}`; same D2 config as C2 (route_slot=2, native anchor, cap=128,
max_seqs=2, captured). Logs `docs/audit/logs/c3_diff_*.log`.

## Variants

| id | flag | what runs | isolates |
|---|---|---|---|
| A baseline | (none) | full captured D2 | reference (C2 = 5.972 tok/s) |
| D skip-moe | `QB_C3_SKIP_MOE` | everything but the MoE expert apply (attention/DSA/EP/norms; MoE outputs zeros) | the non-MoE ceiling |
| B skip-sparse | `QB_C3_SKIP_SPARSE` | dense-anchor group + routing, sparse tail no-op | dense-anchor captured cost |
| C skip-dense | `QB_C3_SKIP_DENSE` | sparse tail + routing, dense-anchor no-op | sparse-group captured cost |

## Results

_PENDING (4 captured runs in flight)._ 

| variant | decode tok/s | vs baseline | reading |
|---|---:|---|---|
| A baseline | PENDING | — | |
| D skip-moe | PENDING | | if ≈48 → MoE is the whole gap; if ≈6 → attention/DSA/EP is the wall (MoE not the lever) |
| B skip-sparse | PENDING | | dense-anchor's share of the MoE cost |
| C skip-dense | PENDING | | sparse-group's share |

## Attribution logic

- **step budget** ≈ 1/decode_tok_s. Non-MoE floor = 1/D. MoE cost = (1/A − 1/D). Dense-anchor ≈ (1/A − 1/B).
  Sparse ≈ (1/A − 1/C). Routing/plumbing ≈ MoE − dense − sparse.
- **Gate for Task 1B:** only pursue compact routing if the MoE (specifically the E·cap padding on the
  dense-anchor and/or sparse group) is a large share of the captured step. If skip-moe already runs near
  the dense baseline's 48 tok/s, the padding is the lever; if skip-moe is still ~6, the wall is
  attention/DSA/EP and compaction cannot help (stop condition).

## Verdict

_PENDING._
