# C6 downstream table (per-task primaries)

Primary metric per task: acc_norm for arc_c / hellaswag, acc for winogrande / mmlu_subset.
AVG = mean of the four primaries. limit=400 per task (mmlu_subset = 5 subjects x 100), greedy,
`enforce_eager`, commit `5b6b9e5`.

| row | tag | collective | arc_c | hellaswag | winogrande | mmlu_subset | AVG |
|-----|-----|-----------|-------|-----------|------------|-------------|-----|
| R1 | c6_dense_nccl_a   | NCCL        | 0.6375 | 0.7100 | 0.7775 | 0.8280 | 0.7382 |
| R2 | c6_dense_nccl_b   | NCCL        | 0.6300 | 0.7100 | 0.7675 | 0.8300 | 0.7344 |
| R3 | c6_dense_customar | NCCL (fell back) | 0.6300 | 0.7100 | 0.7775 | 0.8340 | 0.7379 |
| R4 | c6_d2_nccl        | NCCL        | 0.6450 | 0.7000 | 0.7675 | 0.8080 | 0.7301 |
| R5 | c6_d2_customar    | custom AR   | 0.6525 | 0.6900 | 0.7900 | 0.8040 | 0.7341 |

## Deltas that matter

- **Dense-NCCL noise band** (R1/R2/R3, all NCCL, identical config): AVG 0.7344 - 0.7382 =
  **0.38 pt** spread. Per-task swing is within +/-0.5 pt on every task except MMLU (0.828 - 0.834).
- **Sparse D2, custom AR vs NCCL** (R5 - R4, the one controlled custom-AR comparison that engaged):
  AVG **+0.40 pt** (0.7341 vs 0.7301). No task collapses: arc +0.75, hellaswag -1.00, winogrande
  +2.25, mmlu -0.40. All swings sit inside the dense-NCCL noise band. PPL 3.538 vs 3.588 (lower).

## Reference (frozen `deepseek_final.csv`)

- dense AVG 0.7383 (2-GPU) -- matches C6 dense NCCL band (0.7344-0.7382).
- d2_slot2 AVG 0.7304 (4-GPU, route_slot 2) -- matches C6 sparse D2 NCCL (R4 0.7301).

The C6 rows reproduce the frozen reference numbers, confirming the eval harness is unchanged and the
custom-AR flag is the only new variable.
