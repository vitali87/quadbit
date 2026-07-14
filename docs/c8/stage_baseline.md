# C8 Task 2: pipeline/stage decode baseline vs C4 58.126

**Goal.** Measure single-request batch=1 decode tok/s under pure pipeline/layer-stage execution
(`tensor_parallel_size=1`, `pipeline_parallel_size=4`, no EP), eager and captured, against C4's
58.126 SOTA — same checkpoint / quant / prompt / decode metric as C4/C7.

## Metric (identical to C4)

prompt `"The history of the Roman empire spans many centuries and"`, temperature 0.0, two-run TTFT
subtraction `dtps = 63.0 / (wall64 - wall1)`, batch=1, single request through all 4 stages.

## Result

| harness | graph mode | decode tok/s | ppl | AR/token | stage-xfer/token | source |
|---------|-----------|--------------|-----|----------|------------------|--------|
| C4 (tp=4, force_custom_ar) | captured | **58.126** | 4.264 | ~43.5 | 0 | C4 SOTA |
| C4 (tp=4, NCCL fallback) | captured | 48.248 | 4.264 | ~43.5 | 0 | C4 |
| **C8 PP (tp=1, pp=4, ep=off)** | **captured** | **22.930** | 4.295 | **0** | 3 | `c8_pp_captured.log` |
| C8 PP (tp=1, pp=4, ep=off) | eager | 7.539 | 4.192 | **0** | 3 | `c8_pp_eager.log` |
| C7 DP-attention (tp=1, dp=4, EP) | captured | 20.450 | 4.264 | 0 (2 EP coll./layer) | 0 | C7 |

- Captured: `wall1=0.162s wall64=2.909s → 63/(2.909-0.162) = 22.930 tok/s`, `graph=captured`,
  `cudagraph_mode=FULL_AND_PIECEWISE`, breakable CUDA graph enabled, capture sizes `[1,2,4,8,16]`.
- Graph capture gain: 7.539 → 22.930 = **3.04x** (comparable to C7's 4.5x eager→captured).
- Collectives verified from the trace (`_report_pp_transfers`, profiled pass last): **all-reduce = 0
  per token** (C4's ~43.5 floor removed), only 3 stage-boundary send + 3 recv per token = `pp-1`
  boundaries; allgather / reduce-scatter / all-to-all all 0.
- Coherent generation 3/3 both runs; ppl 4.19 (eager) / 4.29 (captured) vs C4/C7 baseline 4.264 —
  quality unchanged (no quantization/recovery change; PPL movement is decode-path noise, not drift).
- Memory: 41.0 GiB model + 43.3 GiB KV per GPU (model genuinely layer-sharded `[11,11,11,10]`).

## Reading — why removing the all-reduce did not win

C8 captured decode is **2.53x slower** than C4 on the same 4 GPUs / same single-request metric, despite
driving the C4 all-reduce floor (~91% of C4's per-token time, ~43.5 all-reduces/token) to **zero**.
Per-token wall: C8 = 1000/22.930 = **43.6 ms**; C4 = 1000/58.126 = **17.2 ms**. C8 is absolutely
slower even though it deleted C4's dominant kernel.

The reason is structural to batch=1 pipeline execution:

1. **Lost 4-way compute parallelism.** TP=4 splits every layer's GEMM across 4 GPUs that run
   concurrently. PP=4 puts whole layers on one GPU each and runs the stages **serially** for a single
   request — the compute C4 did in parallel is now sequential.
2. **75% idle by construction.** With one request and one microbatch in flight, exactly one stage is
   active at a time; the other 3 GPUs bubble. GPU utilization ceiling for single-request pp=4 is ~25%.
3. **Stage transfers are still PCIe-latency-bound.** The 3 send/recv per token cross PCIe (no NVLink),
   each latency-bound like the all-reduce was — so the collective cost did not vanish, it shrank from
   ~43.5 to 3 hops but rode on top of a now-serial compute path.

Net: the serialization penalty (items 1-2) exceeds the all-reduce time removed. This is the C8 physics
tension resolving against the hypothesis — the removed all-reduce time did **not** exceed the 4-way
weight/compute serialization penalty at batch=1.

## Task 2 acceptance decision

Bar: "continue only if stage mode is faster than C4 OR close enough that graph capture/scheduling could
plausibly beat C4. If much slower, stop and write ceiling verdict." C8 captured (22.930) is **2.53x
slower** than C4 (58.126). Graph capture is already applied (3.04x gain) and the gap is still 2.53x.

**Stop.** Task 3 (bubble removal) cannot close this for single-request decode: at batch=1 there is only
one microbatch, so there is nothing to overlap, double-buffer, or microbatch against — every Task 3
lever (double-buffered activations, send/compute overlap, microbatching) fills bubbles by putting
**more concurrent requests** in flight, which is aggregate serving throughput, explicitly forbidden by
the spec as a single-request-decode claim. Layer rebalancing (`[11,11,11,10]` is already near-even)
is a sub-1% effect and cannot bridge 2.53x. See `verdict.md`.
