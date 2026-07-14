# C8 verdict — pipeline/layer-stage serving vs C4 58.126

## Verdict: D (C4 remains the single-request decode ceiling), with a B qualifier

**D — C4 remains the ceiling.** Pipeline/layer-stage execution (tp=1, pp=4, no EP) does **not** improve
single-request SM120 dense NVFP4 decode. C8 captured = **22.930 tok/s vs C4's 58.126 = 2.53x slower**,
on the same 4 GPUs, same checkpoint/quant/prompt/metric.

**B qualifier — stage mode is a real, working mode that trades single-request latency.** C8 is not
verdict C ("not viable"): the mode constructs cleanly, shards the model by layer `[11,11,11,10]`,
generates coherently, keeps quality (ppl 4.19-4.29 vs 4.264 baseline), and removes C4's all-reduce
floor entirely (0/token). It would help **aggregate serving throughput** (the ~75% idle GPUs fill under
microbatching) — but that is QPS, not single-request latency, and is not the C8 metric.

## Why the hypothesis was refuted

The C8 hypothesis: C4's floor is ~1 PCIe sync/layer (the TP all-reduce); removing it via pipeline
execution should raise the decode ceiling. The lever worked mechanically — all-reduce/token went
43.5 → 0 — but the ceiling fell, not rose, because removing the all-reduce also removed TP's benefit:

- **TP=4 parallelizes each layer's compute across 4 GPUs**; the ~43.5 all-reduces/token are the cost of
  that parallelism (reassembling the sharded activation each layer). C4's win was making that
  reassembly cheap (one-shot custom all-reduce), not avoiding it.
- **PP=4 at batch=1 serializes** the same total compute across 4 stages that run one-at-a-time for a
  single request. Per-token wall rose to 43.6 ms (C8) from 17.2 ms (C4) even with the all-reduce gone.
- The 3 remaining stage-boundary send/recv per token are still PCIe-latency-bound (no NVLink), so the
  cross-GPU cost shrank from ~43.5 hops to 3 but now rides on a serial, 25%-utilized compute path.

The C8 physics tension resolved against the bet: **the removed all-reduce time did not exceed the 4-way
weight/compute serialization penalty** that TP hides and PP exposes at batch=1.

## Why no further C8 work

- **Task 3 (bubble removal) cannot help single-request.** Every lever (double-buffered activations,
  send/compute overlap, microbatching, layer rebalance) fills pipeline bubbles by putting more
  concurrent requests in flight = aggregate throughput, which the spec forbids reporting as
  single-request decode. Layer partition is already near-even; rebalancing is a sub-1% effect. A 2.53x
  gap is not bridgeable by scheduling at batch=1.
- **Task 5 (sparse D2) not run.** Gated on "dense stage beats/materially approaches C4." It did not.
- **Task 6 (GLM) not run.** Gated on "DeepSeek dense wins." It did not. No 8-GPU GLM launched.

## Standing conclusion across C4/C7/C8

Three independent attempts to remove C4's per-layer all-reduce floor for single-request decode all lose
to it, because each removal also removes the tensor-parallel compute concurrency the all-reduce pays
for:

- **C7 DP-attention** (0.35x): trades 1 attention all-reduce for 2 EP collectives/layer.
- **C8 PP stage** (0.39x): trades 43.5 all-reduces for 3 PCIe hops but serializes compute at batch=1.

For batch=1 SM120 dense NVFP4 decode with no NVLink, **the per-layer all-reduce is not overhead to be
removed — it is the price of the 4-way concurrency that makes 58.126 possible.** C4's one-shot custom
all-reduce (making that price cheap) remains the SOTA lever. **C4 58.126 stands.**
