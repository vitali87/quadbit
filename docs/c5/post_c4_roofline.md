# C5 Task 1: post-C4 decode roofline (is the collective still the floor?)

C4 replaced the NCCL ring all-reduce with vLLM's one-shot custom AR (48.248 -> median 58.126 tok/s). C5 asks
what remains. Measured on the dense NVFP4 baseline (`QB_MOE=off`), 4 GPU, via the vLLM worker profiler
(`floor_profile` in serve_dsv4.py), now instrumented to count all-reduce **invocations** per layer / per
token (topology-independent). Logs `docs/audit/logs/c5_post_c4_roofline*.log`.

## Container P2P topology varies on Modal (new finding, affects C4 reproducibility)

The plugin logs the driver `can_device_access_peer` matrix. Two distinct topologies observed across runs:

| container | matrix (off-diagonal) | custom AR | note |
|---|---|---|---|
| fully connected | `[[.,1,1,1],[1,.,1,1],[1,1,.,1],[1,1,1,.]]` | **enabled** (58.126) | the 4 C4 win runs landed here |
| partial | `[[.,0,0,0],[0,.,1,0],[0,1,.,0],[0,0,0,.]]` (only 1<->2) | **disabled -> NCCL fallback** | this roofline run landed here |

So **C4's 58.126 is conditional on a fully-connected allocation**; on a partial-P2P container the PR#21
safety guard correctly falls back to NCCL (48.248). The guard working here is a good validation; the
variance is an honest caveat on C4's reproducibility (not every Modal 4-GPU allocation is P2P-homogeneous).

## Roofline (NCCL path; the COUNTS are topology-independent and hold for the custom-AR path too)

n_layers=43, ~32 decode forwards profiled. GPU-kernel time by category:

| category | share | ms/tok (eager) |
|---|---:|---:|
| **collective (all-reduce)** | **94.5%** | 157.2 |
| norm/elementwise | 2.6% | 4.35 |
| gemm/moe | 1.4% | 2.30 |
| other | 0.9% | 1.51 |
| attention+DSA | 0.4% | 0.64 |

**All-reduce count: n=1392 over 32 forwards = 43.5 per decode token ~= 1 per layer** (43 layers). The DSA
decode kernel fires once per layer (n=1376 = 43/forward), so **AR-per-layer = 1392/1376 = 1.01**. This is the
attention TP all-reduce; the MoE uses EP all-to-all (not an all-reduce), so there is **one** all-reduce per
layer, not the textbook two. The 43.5 all-reduces are **sequential** (layer N+1's input needs layer N's
all-reduced output) and **non-overlappable** at batch=1 (compute is 0.4% attention + 1.4% gemm, nothing to
hide the collective behind).

**Gate: collective is still 94.5% of the step. Proceed to reduce-count (Task 2) / hierarchical AR (Task 3).**

## Same roofline on the C4 custom-AR path (full-P2P container)

A retry landed a fully-connected container, so custom AR engaged (`custom_1stage=1392, ring=0`). Same count
(43.5 AR/token, 1.01/layer), collective **91.2%** of GPU-busy. Total eager GPU-busy dropped 166 -> 100
ms/tok vs the NCCL path (consistent with the +20% captured win), but **the one-shot AR is still 91% of the
step**, replacing ring with one-shot made each all-reduce cheaper, it did not remove the collective as the
floor. Log [c5_post_c4_roofline_customar.log](../audit/logs/c5_post_c4_roofline_customar.log).

## Estimated remaining floor on the C4 custom-AR path

At 58.126 tok/s the captured step is 17.20 ms; ~94% collective = 16.26 ms across 43.5 sequential all-reduces
= **~374 us per one-shot all-reduce**. For a ~14 KB bf16 payload over PCIe that is almost entirely
**cross-GPU synchronization latency**, not data transfer. The lever, if any, is cutting per-AR sync cost,
since the count (~1/layer) is structural and sequential.

## Reduce-rank lever (TP=2): NEGATIVE

TP=2 fits (81.7 GiB/GPU weights of 96, gpu_mem 0.92) and got a fully-connected 2x2 P2P matrix (custom AR
native at world_size==2, single peer read). Result: **40.565 tok/s (24.65 ms/step), PPL 4.0855, capture
FULL** = **slower** than the TP=4 baseline (48.248 / 20.73 ms) by +3.9 ms. The decode is not purely
AR-latency-bound: halving the shard count **doubles the weight bytes each GPU reads per token**, and that
memory cost outweighs the faster 2-GPU all-reduce. **Fewer ranks is a losing lever.** (Log
[c5_tp2_dense.log](../audit/logs/c5_tp2_dense.log).)

## What this leaves for C5

- All-reduce **count** is structural (~1/layer, sequential, non-overlappable at batch=1). No safe algebraic
  transform removes it at decode (sequence parallelism has no sequence dim to shard at batch=1; the AR after
  attention is required before the next layer's norm).
- The only remaining lever is a **faster 4-GPU collective** than the one-shot (Task 3), i.e. cutting the
  ~374 us per-AR sync. Whether a hierarchical/tree variant beats the one-shot for tiny latency-bound payloads
  is the open question (one-shot is the standard latency-optimal choice; two-shot/ring is bandwidth-optimal
  and expected to be worse here). Measured next in `hierarchical_ar.md`.
