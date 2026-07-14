# C7 verdict: **D — C4 ceiling remains**

DP-attention activates and removes the attention all-reduce, but the per-layer cross-GPU collective
floor does not drop; it moves to the EP allgather+reduce-scatter path and doubles (1 → 2 collectives
per layer), and captured DP-attention decode is **2.84x slower** than the C4 SOTA it aimed to beat.

## Verdict against the C7 rubric

- **A (improves SOTA)** — no. Captured 20.450 tok/s << 58.126.
- **B (works, metric becomes aggregate QPS)** — no. Aggregate 4×20.450 = 81.8 tok/s exists but is at
  2.84x-worse per-request latency; not a single-request decode win, labeled aggregate per the guardrail.
- **C (unavailable / not activated)** — no. Mode provably activated: tp=1/dp=4/EP engines constructed,
  attention all-reduce = 0, coherent output, ppl 4.264.
- **D (C4 ceiling remains; AR count drops but another floor dominates)** — **YES.** ✅

## The chain of evidence

1. **Mode activates** (Task 1). tp=1 per rank, env-driven dp=4, EP experts, engines
   `Worker_DP0_EP0..DP3_EP3`. Attention all-reduce eliminated: custom=0, ring=0 in both graph modes.
2. **Floor moved, not removed** (Task 1). EP path fires 1376 allgather + 1376 reduce-scatter kernels
   (identical count eager and captured) = 2 collectives per layer, vs C4's 1 attention all-reduce per
   layer. AllGather alone is 39–76% of decode CUDA time, 956µs–5.3ms per call, PCIe-bound, no NVLink.
3. **Slower on wall-clock** (Task 2/3). Captured DP-attention 20.450 tok/s vs C4 captured 58.126 →
   **2.84x slower**, same single-request metric, same 4 GPUs.
4. **Quality unchanged** (Task 4). ppl 4.2640, all coherence probes correct — pure execution-mode change.
5. **Sparse not run** (Task 5). Gate ("only if dense improves") not met; dense regressed.

## Why (root cause, not symptom)

The bottleneck was never *that* attention synced across GPUs — it was the *number and cost* of
per-layer PCIe collectives with no NVLink. DP-attention trades one attention all-reduce per layer for
two EP collectives per layer, each at least as expensive. On a no-NVLink SM120 box the collective
count per layer is the floor, and DP-attention raises it. The lever to actually attack the floor is
reducing per-layer cross-GPU collectives (or an interconnect with NVLink), not relocating attention's
sync — consistent with the [[quadbit-decode-roofline]] finding that the shared PCIe collective floor,
not the MoE, is what dominates SM120 decode.

## SOTA status

Unchanged. C4's **58.126 tok/s** captured SM120 dense NVFP4 single-request decode remains the SOTA. No
README/paper headline change. No merge without explicit per-PR authorization.

## Deliverables

`mode_validation.md`, `serve_baseline.md`, `dp_attention_ab.md`, `quality_guardrail.md`,
`sparse_d2_transfer.md`, this verdict. Raw logs: `docs/audit/logs/c7_dp_captured.log`,
`docs/audit/logs/c7_dp_eager_smoke.log`.
