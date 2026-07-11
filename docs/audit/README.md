# Final audit package (Campaign B frozen at `main` 538c7a0)

Self-contained audit of every headline claim in `docs/paper.md`. Nothing here depends on out-of-repo
memory or chat context. Regenerate from a clean checkout with the commands in `command_manifest.md`.

## Contents

| item | file | covers |
|---|---|---|
| Command manifest | `command_manifest.md` | one command per headline table/figure + commit, checkpoint, GPU count, toolchain versions, env vars, eager/graph status |
| Claims checklist | `../claims_checklist.md` | every abstract/conclusion claim -> one backing row (supported / caveated / negative / future) |
| GLM run evidence | `logs/glm_runs.log` | GLM dense + 3 sparse policies: backend selection (`FLASHINFER_MLA_SPARSE_SM120`), generations, PPL, timing, mem, sparse-path anchor count |
| Graph-capture failure | `logs/glm_graphfail.log` | the `torch.unique().tolist()` host-sync traceback (negative result) |
| DeepSeek downstream | `logs/deepseek_downstream.log` | c_down49 / c_down60 / c_gateup49 result blocks, cross-checked vs `deepseek_final.csv` |
| Review fixes | `logs/greptile_fixes.md` | PR #11 Greptile/Gemini findings and fixes (both P1s) |
| Llama serving + leaderboards | committed CSVs (below) | crossover, dense/sparse leaderboards |

Llama serving and leaderboard raw logs are Modal-app logs; their frozen numeric artifacts are the
committed CSVs: `docs/figures/data/crossover_{sparse,nvfp4}.csv` (serving crossover),
`fig5_pareto.csv`, `sparse_serving_sweep.csv`, `dist_scaling.csv`. Reproduce via `command_manifest.md`.

## Decode-timing methodology (audit of the Greptile P1)

The Greptile P1 flagged that decode throughput must exclude prefill. All three serving tables now report
**decode-only** tok/s, via the same prefill-excluding protocol: measure a prefill/TTFT wall with
`generate(max_tokens=1)` and subtract it from the `generate(max_tokens=G)` wall, so only the decode steps
are counted.

- **Llama (crossover) and DeepSeek (serving sweep):** `decode_s = total_wall - ttft`,
  `decode_tps = gen_tokens / decode_s` (equivalently `1 / tpot`), where `ttft` is the
  `generate(max_tokens=1)` wall (paper Section 9). Verifiable in the CSV headers: `crossover_*.csv`
  carries `ttft_s, decode_s, decode_tps`; `sparse_serving_sweep.csv` carries `ttft_s, tpot_ms, decode_tps`
  with `decode_tps ~ 1000/tpot_ms`.
- **GLM (`glm_baseline`):** the same subtraction, `decode_tps = (gtok-1) / (gen64_wall - gen1_wall)`,
  where the `gen=1` wall is the prefill/TTFT proxy.

The protocol is identical across the three; the only difference is that the Llama/DeepSeek harnesses store
the TTFT wall as its own CSV column while GLM subtracts it inline. The earlier (pre-fix) GLM logs in
`logs/glm_runs.log` still print the
**end-to-end** number (e.g. dense `1.58 tok/s`); the committed `glm_results.md` table carries the
**corrected decode-only** number (dense `1.79`), recomputed from those same logs. That discrepancy is the
audit trail of the fix, not an inconsistency.
