# C7 Task 4: quality guardrail

**Goal.** Confirm DP-attention does not change model quality vs C4 — C7 is a serve/latency change only
(no quant, no recovery, no weights touched).

## Evidence (captured DP-attention run, rank 0)

- **PPL** on the fixed science passage: **4.2640** — identical to C4's 4.264 (same checkpoint, same
  `native_nvfp4` dense path, `QB_MOE=off` dense baseline).
- **Coherence** (greedy, temp 0.0):
  - `"The capital of France is"` → `" Paris."` (correct)
  - `"def fibonacci(n):"` → correct recursive base cases (`if n <= 0: return 0 ... elif n == 1`)
  - `"The three primary colors are"` → `" red, green, and blue. In the RGB color model..."` (correct)

## Reading

PPL matches to 4 decimal places and all three coherence probes are correct, so DP-attention is a pure
execution-mode change with no quality impact — exactly as the C7 spec required ("Do not change model
quality"). The quality guardrail passes; the campaign fails on speed, not quality.

## Raw log

`docs/audit/logs/c7_dp_captured.log`.
