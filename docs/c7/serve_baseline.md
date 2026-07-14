# C7 Task 2: serve-harness parity baseline vs C4 58.126

**Goal.** Measure single-request batch=1 decode tok/s under the DP-attention harness with the same
prompt / quant / checkpoint as C4, so the DP number is compared apples-to-apples against C4's SOTA.

## Metric definition (identical to C4)

- prompt `"The history of the Roman empire spans many centuries and"`, temperature 0.0
- two-run TTFT subtraction: `dtps = 63.0 / (wall64 - wall1)`, batch=1 per rank
- rank 0 only; each rank drives its own batch=1 request in lockstep (single-request latency, not
  aggregate replica QPS — see mode_validation.md guardrail).

## Result

| harness | graph mode | decode tok/s | ppl | source |
|---------|-----------|--------------|-----|--------|
| C4 (tp=4, force_custom_ar) | captured | **58.126** | 4.264 | C4 SOTA |
| C4 (tp=4, NCCL fallback) | captured | 48.248 | 4.264 | C4 |
| **C7 DP-attention (tp=1, dp=4, EP)** | **captured** | **20.450** | 4.264 | `c7_dp_captured.log` |
| C7 DP-attention (tp=1, dp=4, EP) | eager | 4.578 | 4.264 | `c7_dp_eager_smoke.log` |

Captured DP-attention: `wall1=1.558s wall64=4.638s → 63/(4.638-1.558) = 20.450 tok/s`.

## Reading

DP-attention decode is **2.84x slower** than the C4 captured SOTA on the same total 4 GPUs, same
single-request metric, same PPL. Removing the attention all-reduce did not help because it is replaced
by two PCIe-bound EP collectives per layer (allgather + reduce-scatter, see mode_validation.md) that
cost more than the one attention all-reduce they displaced. The serve baseline does not reach parity,
let alone beat, 58.126.

## Raw log

[c7_dp_captured.log](../audit/logs/c7_dp_captured.log) (run b4wh13mbm, exit codes [0,0,0,0]).
