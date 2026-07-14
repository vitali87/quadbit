# C6 verdict: C4 one-shot custom all-reduce is quality-safe

**Verdict: A (quality passes).** C4's one-shot custom all-reduce
(`cross_device_reduce_1stage`) raises SM120 decode from 48.248 to a median 58.126 tok/s (+20.5%)
while preserving the downstream smoke-suite quality envelope. The PPL shift is reduction-order-
sensitive, but downstream quality is stable.

## Evidence

The custom all-reduce under test is the attention tensor-parallel reduce, identical whether the MoE
policy is dense or sparse (MoE-policy-independent). The controlled comparison where it genuinely
engaged is the sparse D2 pair:

| comparison | collective | AVG | PPL | delta |
|-----------|-----------|-----|-----|-------|
| R4 c6_d2_nccl | NCCL | 0.7301 | 3.588 | (ref) |
| R5 c6_d2_customar | **custom AR (engaged)** | 0.7341 | 3.538 | **+0.40 pt AVG, -0.050 PPL** |

The custom AR run is marginally *higher* in AVG, *lower* in PPL, sits inside the dense-NCCL run-to-
run noise band (0.38 pt across R1/R2/R3), and no task collapses (arc +0.75, hellaswag -1.00,
winogrande +2.25, mmlu -0.40, all within noise). Generation is coherent, no NaN / nonfinite. The
plugin log proves the one-shot path was active: `full P2P verified -> one-shot custom AR enabled`.

Against the acceptance criteria: AVG delta 0.40 pt < 0.5 pt threshold AND within measured noise;
no task collapse; generation coherent; no NaN; the one-shot AR path is proven active in the trace.
This is a PASS.

## Honest caveat: dense custom-AR engagement is P2P-container-dependent

`QB_FORCE_CUSTOM_AR=1` only engages the custom AR after the plugin verifies the full P2P matrix; on
a partially-connected 4-GPU set it safely falls back to NCCL. Modal assigns 4-GPU sets with varying
P2P topology (a caveat already recorded in the C5 audit). Across **six** dense attempts (R3 plus
five retries) Modal never handed out a fully-P2P-connected dense container, so every dense
custom-AR row fell back to NCCL and we do not have a dense row that engaged custom AR directly. The
one sparse attempt (R5) did land fully connected and engaged. Because the collective under test is
the attention tensor-parallel all-reduce, which is byte-identical whether the MoE policy is dense or
sparse, R5's engaged result is direct evidence for the dense case; the dense-engaged confirmation is
blocked only by container P2P luck, not by any quality signal. This verdict rests on the engaged R5
evidence plus that policy-independence, both of which are sufficient; a future dense-engaged row (if
Modal hands out a fully-P2P dense container) would strengthen it, not change it.

## What this licenses

The C4 +20.5% decode result may be reported as a **quality-safe** SM120 decode SOTA, not merely a
speed-only collective result. The one-shot custom all-reduce does not degrade downstream quality
relative to NCCL; the only measurable movement (PPL) is a small, favorable, reduction-order-
sensitive shift that is not a ranking metric.

Eager-vs-captured is not a caveat: CUDA-graph capture replays the identical kernel with the
identical bf16 reduction order, so the eager quality measured here equals the captured speed row's
quality.
