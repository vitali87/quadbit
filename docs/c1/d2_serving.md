# C1 validation-ladder B: DeepSeek-D2 serving A/B/C

**Harness:** `serve_dsv4.py::graph_gate4` (4 GPU EP, `_graph_gate_body`). Policy: route-slot D2
(`route_slot=2`, `proj=both`, dense_layers 0-21, `cap=128`, `max_seqs=2`, tp=4). Logs `c1_d2_*.log`.

`dense_anchor_backend`: `dequant` = frozen `_dense_seg_gs` range(E) loop; `native_nvfp4` =
`group_gemm_nvfp4_nt_groupwise`. Config: A = frozen eager (QB_GRAPH=0), B = graph-safe eager
(QB_GRAPH=1, enforce_eager), C = captured (QB_GRAPH=1). Decode tok/s = two-run TTFT-subtracted.

## Rows

| # | config | backend | command | capture | PPL | decode tok/s | gen coherent |
|---|---|---|---|---|---|---|---|
| 1 | C-captured | dequant | `graph_gate4 --cap 128 --max-seqs 2 --dense-layers 0..21` | FULL 2/2 | 3.9746 | **0.514** | yes (Paris/fib/RGB/H2O EN+ZH) |
| 2 | B-graphpath-eager | native_nvfp4 | `... --force-graph-path --eager --dense-anchor-backend native_nvfp4` | eager | _tbd_ | _tbd_ | _tbd_ |
| 3 | C-captured | native_nvfp4 | `... --dense-anchor-backend native_nvfp4` | **PIECEWISE 3/3 + FULL 2/2** | **4.0112** | **5.820** | yes (Paris/fib/RGB/H2O EN+ZH) |

All rows same harness/timing (two-run TTFT-subtracted decode). P4 reference (different timing method):
frozen-eager 4.04 tok/s, dequant-captured 0.97 tok/s.

**Headline:** native-captured D2 = PPL 4.0112 (dequant-captured 3.9746; +0.037 from fp4-act on the anchor,
negligible, generation identically coherent) at **5.82 decode tok/s = 11.3× the dequant-captured 0.514**
(same harness) and **1.44× the frozen-eager 4.04**. Under P4/Row-1 the captured path was ~8× *slower*
than eager because the dense-anchor `_dense_seg_gs` loop dominates the decode step (Row 1: 124.6s for 64
tokens); the native grouped GEMM makes the **captured** path *faster than eager*. Row 3: `load+capture ok
in 1116s`, DSA `sparse_mla_sm120_decode_dsv4` native, pool 2.08 GiB/GPU, all route-slot layers `gu=y dn=y`.

## Component timing (dense-anchor branch, isolated)

Under CUDA graph the decode forward replays as one unit, so per-component timing comes from the standalone
isolation (`docs/c1/standalone_ab.md`) for the branch C1 changes; sparse-seg / routing / DSA-MLA / EP
collectives are unchanged from P4.

| component | dequant (`_dense_seg_gs`) | native (`group_gemm_nvfp4`) | note |
|---|---|---|---|
| dense-anchor gate_up (E=64, per layer) | 11.0 ms | 0.48 ms | standalone, graph replay |
| dense-anchor down (E=64, per layer) | 6.55 ms | 0.27 ms | standalone, graph replay |
| sparse-seg tail (both proj) | — | — | C1-invariant (P4 `sparse_moe_mm_2lvl`) |
| routing (`route_fixed_cap`) | — | — | C1-invariant |
| DSA / MLA | — | — | native SM120 `sparse_mla_sm120_decode_dsv4` |
| EP collectives | — | — | C1-invariant (vLLM NCCL) |

## Verdict (gate B): _tbd_
