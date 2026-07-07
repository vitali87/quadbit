# Accuracy Pareto via reverse hybrid densification (2026-07-07)

**Goal:** attack the +2.3 PPL sparse tax (10.26 all-sparse vs 7.97 dense NVFP4) by reverting selected MLP
projections from sparse-recovered back to the model's stock dense NVFP4 weights, keeping the rest sparse.
Measured through the deployed serving path (`serve_densify`, recovered-Instruct ckpt, WT-2 16x2048).

**Verdict (honest, and negative for the training-free path):** reverse densification recovers accuracy only
by surrendering the sparse speed advantage roughly 1:1. There is no free Pareto point better than the two
endpoints. Closing the tax requires QAT repair of the best hybrid, not placement alone. This is consistent
with the earlier training-free hybrid-placement negative result.

## Accuracy Pareto (PPL; lower is better)
| policy | PPL | ΔPPL vs all-sparse | keeps split-K decode win? |
|--------|-----|--------|------|
| none (all sparse) | 10.256 | 0 | yes (full) |
| dense down only | 10.282 | +0.03 (nothing) | no |
| dense gate_up L22-31 (10 layers) | 10.165 | -0.09 | yes |
| dense gate_up L16-31 (16) | 9.995 | -0.26 | yes |
| dense gate_up L11-31 (21) | 9.859 | -0.40 | yes |
| dense gate_up all (32) | 9.750 | -0.51 | yes |
| dense whole-MLP L0-10 (early) | 9.980 | -0.28 | partial |
| dense whole-MLP L11-21 (mid) | 9.882 | -0.37 | partial |
| dense whole-MLP L22-31 (late) | 9.540 | -0.72 | partial |
| all dense NVFP4 | 7.974 | -2.28 | no |

## Structural findings
1. **down_proj sparsity is accuracy-free when gate_up is sparse** (10.256 -> 10.282, no change) but costs
   1.78 PPL once gate_up is dense (9.750 -> 7.974). The two projections' errors **interact**, they do not add.
2. **gate_up carries the recoverable tax; late layers cost most** (densifying late L22-31 recovers ~2x early).
3. The "keep down sparse" frontier (which preserves the entire split-K decode win) **tops out at 9.750 PPL**
   (all gate_up dense). Going below forces densifying down, forfeiting the decode advantage.

## Speed of the knee (eager serve_densify path; relative comparison only, no graph/split-K)
prefill | decode tok/s at B=1/8/32/64:
| policy | pf B1/8/32/64 | dc B1/8/32/64 |
|--------|---------------|---------------|
| none (all sparse) | 12527/46826/66302/97425 | 30/237/945/1821 |
| gate_up dense | 12163/51552/63560/93567 | 28/219/865/1663 |

Densifying gate_up **hurts** speed (decode -7 to -9%, prefill mixed/-4% at batch): sparse gate_up was already
the fast component (it beats NVFP4 gate_up 1.08-1.14x on SM120), so reverting it surrenders that. Hence the
1:1 accuracy-for-speed trade and no free lunch.

## Best repair candidate (if QAT is pursued)
gate_up-dense + down-sparse (9.750 PPL): keeps the split-K decode-win placement and is the natural
phase-split (dense gate_up for prefill, sparse down for decode). Short QAT on that fixed mask is the way to
push 9.75 toward the 8.x range without touching the decode kernel. This is a fine-tune run, deferred pending
a decision.

## Provenance
Commit `6ea58d7`, branch `track4-crossover`. `serve_densify` policies via `--mode densify --policy <p>`;
`p` in {none, all, down, gateup, L<a>-<b>, gu:<a>-<b>, dn:<a>-<b>}. Recovered-Instruct ckpt as above.
