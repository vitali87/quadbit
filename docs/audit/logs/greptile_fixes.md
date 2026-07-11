# PR #11 review fixes (Greptile + Gemini) — GLM transfer

PR #11 (`feat/glm-transfer-paper` -> `main`, merged as true merge commit `538c7a0`). Reached Greptile 5/5,
zero unresolved threads, over 3 review iterations. Both P1s were real and are recorded here for audit.

## P1 (Greptile) — decode timing included a second prefill  [commit f28ee63]

The GLM `glm_baseline` timing started the `gen=64` call from the full prompt again, so `dec_s` included
another prefill; the printed "decode tok/s" was end-to-end prompt+gen throughput, mislabeled, and
`glm_results.md` used it as `decode tok/s`. **Fix:** decode-only = `(gtok-1) / (gen64_wall - gen1_wall)`,
subtracting the separate gen=1 call's prefill. `glm_results.md` recomputed from the logs (no re-run):
dense 1.58->1.79, down49 1.52->1.71, route-slot 1.81->2.10, gateup 1.90->2.12 tok/s. End-to-end still
printed alongside. (The DeepSeek and Llama serving tables were already decode-only; see `../README.md`.)

## P1 (Greptile) — default GLM command hit the non-capturable graph path  [commit 11a3e88]

`main` defaults `eager=False`, so the documented `--mode glm_baseline` (without `--eager`) overrode
`glm_baseline`'s own `eager=True` default and started graph capture, which aborts on the plugin's
`torch.unique(...).tolist()` host-sync (see `glm_graphfail.log`). **Fix:** the `main` glm_baseline
dispatch forces `eager=True` (GLM's EP MoE is not graph-capturable today), so the default command works.

## Gemini (medium) findings — all applied in f28ee63

| file | issue | fix |
|---|---|---|
| `harness/serve_dsv4.py` | decode timing includes prefill (same as P1) | decode-only formula |
| `docs/figures/make_figures.py` | `fig_ds_designspace` in-bar labels at x=0.01 sit outside xlim (0.45,0.76) -> invisible | moved to x=0.46 |
| `harness/serve_dsv4.py` | `mem.split(chr(10))` non-idiomatic | `mem.splitlines()` (all 3 sites) |
| `docs/build_paper.sh` | hardcoded `/Library/TeX/texbin` on PATH | guarded with `[ -d ... ]` |

All 5 review threads resolved; final review 5/5 with zero new comments on `11a3e88`.
