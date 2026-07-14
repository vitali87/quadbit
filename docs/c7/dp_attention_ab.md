# C7 Task 3: DP-attention A/B

**Goal.** A/B the one lever C7 exists to test — data-parallel attention (tp=1, EP experts) against the
C4 tensor-parallel baseline (tp=4) — on the single-request captured decode metric.

## A/B table

| arm | attention | experts | collectives / layer | graph | decode tok/s | ppl |
|-----|-----------|---------|---------------------|-------|--------------|-----|
| **A — C4 baseline** | TP (tp=4) | TP (tp=4) | 1 (attention all-reduce) | captured | **58.126** | 4.264 |
| **B — DP-attention** | replicated (tp=1, dp=4) | EP-sharded | 2 (allgather + reduce-scatter) | captured | **20.450** | 4.264 |

Both arms: same checkpoint, quant path (`QB_DENSE=nvfp4`, `native_nvfp4` backend), prompt, kv_cache
fp8, 4x RTX PRO 6000 (SM120), no NVLink.

## Outcome

Arm B loses by **2.84x**. The A/B isolates exactly the intended variable (where attention's cross-GPU
sync lives) and the answer is unambiguous: moving attention off the TP all-reduce path onto replicated
attention forces the MoE onto an EP allgather+reduce-scatter path whose per-layer collective count is
2 (vs 1) and whose per-call latency is higher (PCIe-bound, 956µs–5.3ms per allgather, no NVLink).

No further A/B knob rescues this: the loss is structural (2 collectives/layer > 1, each costlier), not
a tuning artifact. force_custom_ar does not apply to arm B (no attention all-reduce to accelerate); the
EP collectives are NCCL ring on PCIe. Graph capture already applied (arm B is captured) and only removes
launch overhead, not the network latency that dominates.

## Aggregate-throughput note (the guardrail)

Arm B runs 4 concurrent batch=1 requests (one per rank). Its *aggregate* serving throughput is
`4 × 20.450 = 81.8 tok/s` across 4 requests. Per the C7 semantic guardrail this is **not** a decode
SOTA — it is aggregate serving throughput at 2.84x-worse per-request latency, and is labeled as such.
It does not beat C4 on the single-request decode metric this campaign is scored on.
