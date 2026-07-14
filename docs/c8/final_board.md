# C8 final board — single-request SM120 dense NVFP4 decode

All rows: DeepSeek-V4-Flash-NVFP4, 4x RTX PRO 6000 (SM120), no NVLink, batch=1, temp 0.0, same prompt
and TTFT-subtracted metric `dtps = 63/(wall64-wall1)`. Single request through the whole model — **not**
aggregate replica QPS.

| row | mode | graph | decode tok/s | vs C4 | AR/token | cross-GPU/token | ppl |
|-----|------|-------|--------------|-------|----------|-----------------|-----|
| **C4 TP=4 one-shot AR (SOTA)** | tp=4 | captured | **58.126** | 1.00x | ~43.5 | ~43.5 AR | 4.264 |
| C4 TP=4 NCCL fallback | tp=4 | captured | 48.248 | 0.83x | ~43.5 | ~43.5 AR | 4.264 |
| C7 DP-attention | tp=1 dp=4 EP | captured | 20.450 | 0.35x | 0 | 2 EP coll./layer | 4.264 |
| **C8 PP stage (captured)** | tp=1 pp=4 | captured | **22.930** | **0.39x** | **0** | **3 send/recv** | 4.295 |
| C8 PP stage (eager) | tp=1 pp=4 | eager | 7.539 | 0.13x | 0 | 3 send/recv | 4.192 |
| C8 aggregate serving throughput | — | — | not measured | — | — | — | — |

Aggregate serving throughput (row 6) is deliberately **not** claimed: pp=4 single-request leaves ~75%
GPU idle, so microbatched/multi-request throughput would be higher than the single-request number, but
that is a different metric (QPS, not latency) and the spec forbids reporting it as single-request
decode. Left blank rather than conflated.

## Verdict

C8 removed C4's dominant floor (per-layer TP all-reduce: ~43.5/token → **0**, confirmed by trace) and
the pipeline/stage mode is fully viable (genuine layer-sharding, coherent gen, unchanged quality). But
captured single-request decode is **22.930 tok/s = 2.53x slower than C4's 58.126**, because pipeline
execution serializes at batch=1 the 4-way compute parallelism TP provides and idles 3 of 4 GPUs. C4
remains the single-request SM120 dense NVFP4 decode ceiling. Full reasoning: [verdict.md](verdict.md).
