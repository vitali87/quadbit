# DeepSeek-V4-Flash-NVFP4: frozen final result table (paper reference)

Training-free sparse-FP4 MoE on SM120 (RTX PRO 6000). 43 MoE layers, 256+1 experts, top-6.
Downstream = 400 items/task (ARC-C, HellaSwag, Winogrande, MMLU-5) via loglikelihood MC; PPL held-out
teacher-forced. Source data: `docs/figures/data/deepseek_final.csv`.

| policy | sparse layers % | active sparse FLOP % | realized compute saving %¹ | GPUs | PPL | ARC-C | HellaSwag | Winogrande | MMLU-5 | AVG | Δ dense | gen | mem/GPU | serving |
|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|
| dense (ref) | 0 | 0 | 0 | 2 | 3.537 | .650 | .708 | .768 | .828 | **.7383** | — | coherent | 96.6 GB | reference |
| **c_down49** | 49 | 16 | ~8 | **2** | 3.620 | .645 | .700 | .783 | .814 | **.7354** | −0.29 | coherent | 96.8 GB | path unchanged |
| c_down60 | 60 | 20 | ~10 | 2 | 3.926 | .643 | .678 | .760 | .796 | .7190 | −1.93 | coherent | 96.8 GB | path unchanged |
| D3 route-slot | 49² | 24 | ~12 | 4 | 3.534 | .633 | .705 | .765 | .830 | .7331 | −0.52 | coherent | 95.9 GB | 4-GPU dual-residency |
| **D2 route-slot** | 49² | 33 | ~16 | **4** | 3.528 | .643 | .708 | .768 | .804 | **.7304** | −0.79 | coherent | 95.5 GB | 4-GPU dual-residency |
| D1 route-slot | 49² | 41 | ~20 | 4 | 3.511 | .613 | .703 | .748 | .800 | .7156 | −2.27 | coherent (degraded) | 95.9 GB | 4-GPU dual-residency |
| c_gateup49 | 49 | 33 | ~16 | 2 | 3.541 | .610 | .670 | .753 | .790 | .7056 | −3.27 | coherent | 96.6 GB | misses target |
| a2_49 (both-proj) | 49 | 49 | ~24 | 2 | 3.754 | .608 | .660 | .735 | .784 | .6966 | −4.17 | coherent | 96.5 GB | misses target |

## The two reference results

- **c_down49 is the cleanest 2-GPU capability-preserving row.** Down-only projection anchoring: 49% of MoE layers sparse at −0.29pt from dense, serving path and memory unchanged, 2 GPUs.
- **D2 is the max active sparse-FLOP row, and it requires 4-GPU dual residency.** Route-slot (top-2 highest-weight expert slots dense, low-weight tail 2:4-sparse) reaches ~33% active sparse FLOP at −0.79pt, but keeping dense weights and sparse codes for the same experts co-resident needs 4 GPUs.

## Reading the columns (do not conflate)

- **Sparse layers %** = fraction of MoE layers that have any sparse experts. It is NOT the compute metric.
- **Active sparse FLOP %** = fraction of expert matmul FLOPs that run through the 2:4 kernel. This is the real sparsity lever. For route-slot the layer % (49) and FLOP % (24–41) diverge because only some slots per layer are sparse.
- ¹ **Realized compute saving % (estimate)** = 0.5 × active sparse FLOP %, under an ideal structured-2:4 = 2× assumption on the sparsified matmul. This is an ESTIMATE of expert-compute saved; measured kernel speedups vary by batch/prompt regime (see `sparse_serving_sweep.csv`).
- ² Route-slot D1/D2/D3 all touch 21/43 layers (49%), but sparsify only the low-weight slots within them, so their per-layer sparsity is partial. Layer % overstates their sparsity; use active sparse FLOP %.

## Quality metric caveat

Downstream AVG is the quality metric of record, not PPL. Several rows (D1/D2/D3, c_gateup49) sit at or below dense PPL yet differ sharply in downstream AVG. Do not read D2's near-dense PPL (3.528) as near-dense capability; its AVG is −0.79pt and that is the honest number.

## 8-task breadth check (WS-E)

The 4-task battery above is the frozen paper reference. WS-E ([wse/verdict.md](wse/verdict.md)) widens it
to 8 tasks (restored PIQA at root cause + ARC-Easy + OpenBookQA + BoolQ) at `limit=400` on the two
deployed sparse policies, to confirm the capability claim is not an artifact of the narrow battery. It is
not: **no task collapses**, and both policies stay near dense on the wider battery (dense .7548, down49
−0.57 pt, D2 route-slot +0.03 pt). This is a fresh measurement, not a correction of the frozen rows.
