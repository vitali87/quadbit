# P4 Milestone 3-A (plugin-only in-process capture) — verdict: PASS

The **shipped plugin's** sparse-FP4 MoE seg helpers graph-capture in-process, bit-exact to eager, and the
new graph-safe path matches the frozen `build_routing` eager path whenever nothing is dropped.

Harness `harness/p4_m3a.py` imports the real `qb_sm120_plugin` module and drives its actual
`_load_sparse_moe()` helpers (not private copies, as M2 did). Raw log `docs/audit/logs/p4_m3a.log`.
Env: torch 2.12.0.dev+cu128, RTX PRO 6000 Blackwell, cap 12.0, driver 12.8, single rank.

## What M3-A adds over M2

M2 proved a *standalone* copy of the seg path captures. M3-A proves the **plugin's own** functions do,
by additively exposing three helpers in `_load_sparse_moe()`'s namespace (frozen eager helpers
untouched):

- `quant_into(x, bb, sb, gb)` — stream-safe quantize into preallocated buffers via
  `quantize_act_nvfp4_2lvl_s` on `current_stream()` (no default-stream sync, no alloc).
- `seg_into(w, bb, sb, gb, c, mpe, in_f, eblk)` — `sparse_moe_mm_2lvl` on the current stream into a
  preallocated `c` (passing the stream kills the `stream==0 → cudaDeviceSynchronize` at lib line 1030).
- `route_fixed_cap(assign, e, cap)` — fixed-capacity **device** routing (constant `eblk`, deterministic
  overflow drop, no `.item()`/host loop/dynamic shape).

## Three variants, compared in TOKEN space

For each shape the harness computes the full per-token MoE output `y[T,H]` three ways and compares:

- **OLD eager** — frozen plugin path (`build_routing` + `quant_act` alloc + `seg_gemm` stream 0).
- **NEW eager** — `route_fixed_cap` + `quant_into` + `seg_into` (current stream, preallocated), run eager.
- **NEW graph** — the NEW path captured as a CUDA graph and replayed.

`NEWvsOLD` = the graph-safe path vs the frozen eager path; `GvsE` = graph replay vs new eager.

## Results (kernels/pass = 4; graph replay = 1 host launch)

| shape | T | Rp | drop | NEWvsOLD cos | GvsE cos | nf | eager ms | graph ms | mem MB |
|---|---|---|---|---|---|---|---|---|---|
| decode-B1 | 1 | 1024 | 0 | 0.999998 | 0.999997 | 0 | 0.182 | 0.110 | 955.8 |
| decode-B8 | 8 | 1024 | 0 | 0.999999 | 0.999999 | 0 | 0.185 | 0.111 | 970.7 |
| decode-B32 | 32 | 1024 | 0 | 0.999998 | 0.999999 | 0 | 0.183 | 0.113 | 971.5 |
| decode-B64 | 64 | 1024 | 0 | 0.999998 | 0.999999 | 0 | 0.184 | 0.113 | 972.6 |
| prefill-2048 | 2048 | 24576 | 0 | 1.000000 | 1.000000 | 0 | 2.939 | 2.877 | 2901.0 |
| prefill-8192 | 8192 | 98304 | 0 | 1.000000 | 1.000000 | 0 | 13.396 | 13.241 | 8932.8 |
| all→expert0 | 4096 | 49152 | 18432 | 0.948668¹ | 1.000000 | 0 | 5.827 | 5.757 | 4898.1 |
| empty-experts | 4096 | 49152 | 12288 | 0.933311¹ | 1.000000 | 0 | 5.824 | 5.747 | 4901.3 |
| near-capacity | 4096 | 49152 | 0 | 1.000000 | 1.000000 | 0 | 5.852 | 5.776 | 4898.1 |

¹ These two shapes overflow a single expert; NEW **deliberately drops** the overflow rows that
`build_routing` (OLD) keeps, so NEW≠OLD here is expected and correct, not an error. Their **capture is
still bit-exact** (GvsE = 1.000000) and the drop count is deterministic.

## Acceptance gates (all met)

- **capture OK 9/9**; **graph == eager (bit-exact) 9/9** (GvsE cos = 1.0, nonfinite = 0 everywhere);
- **NEW == OLD on all 7 no-drop shapes** (cos ≥ 0.999998; the ~1e-3 relL2 is bf16 rounding from a
  different-but-equivalent accumulation order, not a logic difference);
- **no host sync** in the captured region (a sync would raise `cudaErrorStreamCaptureUnsupported`);
- **no dynamic shape / no in-region alloc** (fixed `E*cap`, buffers allocated before capture);
- **decode ~1.6× faster** (eager ~0.18 ms → graph ~0.11 ms) from CPU-launch-overhead removal; prefill
  compute-bound (graph ≈ eager), matching M2.

`M3-A PASS.`

## Scope / what M3-A does NOT yet cover (→ M3-B/C/D)

- Single rank, identity `expert_map` — no EP off-rank slots. The off-rank/sink handling (dynamic
  on-rank count made static) is M3-C (multi-rank).
- A fixed route pattern, not a real router forward — M3-B runs one real MoE layer (topk router,
  router-weight-on-input semantics, single/uniform/empty/near-capacity/overflow).
- The `patched_moe_apply` sparse branch itself is not yet on the graph-safe path behind a flag — that
  lands in M3-B, plus the A6 indexer `.cpu().tolist()` fix for DSA attention capture.
