# C5 Task 3: hierarchical / alternative all-reduce for the 4-GPU PCIe collective

Reduce-count is structural (Task 2) and TP=2 is negative, so the only remaining lever is a **faster 4-GPU
all-reduce** than C4's one-shot. Post-C4 roofline: the one-shot AR is still 91% of the step, ~374 us per
all-reduce for a ~14 KB bf16 payload = almost entirely **cross-GPU synchronization latency over PCIe**, not
data transfer. Can a hierarchical variant cut that?

## Measured all-reduce variants (dense baseline, 4 GPU, captured)

| variant | algorithm | decode tok/s | vs 48.248 | trace kernel |
|---|---|---:|---:|---|
| NCCL ring | ring, 2(N-1)=6 serialized hops | 48.248 | 1.00x | `ncclDevKernel_AllReduce_RING_LL` |
| NCCL tree (`allreduce:tree`) | tree, ~2log2(N)=4 hops | 48.983 | +1.5% | `ncclDevKernel_...Tree` |
| **C4 one-shot custom** | 1 stage, each rank reads N-1 peers in parallel | **58.126** | **+20.5%** | `vllm::cross_device_reduce_1stage` |
| two-shot custom | 2 stage (reduce-scatter + all-gather), bandwidth-optimal | not forced | (expected worse) | `cross_device_reduce_2stage` |

## Why a hierarchical variant does NOT beat the one-shot for 4-GPU full-P2P

The one-shot is already **single-sync**: each rank reads the other 3 peers' buffers in parallel (P2P) and
reduces locally, one barrier. A hierarchical/tree splits this into **2 stages** (reduce within pairs, then
across pairs) = **2 barriers**. For a tiny latency-bound payload the cost is dominated by the *number of
synchronizations*, not the data moved, so 2 stages is **worse**, not better. This is exactly why:
- NCCL **tree** (fewer hops than ring) helps only **+1.5%** (still a multi-step NCCL collective with its own
  launch/sync), while the **one-shot** (single parallel-read barrier) gets **+20.5%**.
- vLLM's **two-shot** is its bandwidth-optimal path (chosen for *large* tensors); for our sub-max_size decode
  payloads `should_custom_ar` already selects the one-shot, and forcing two-shot would add a stage.

vLLM's one-shot is a lightweight flag-based P2P reduce (`RankSignals`), so the ~374 us is the PCIe
signal/round-trip floor, not slack a reorganization removes. **For full-P2P 4-GPU, the one-shot is the
latency-optimal collective; no hierarchical variant is expected to beat it, and none was built** (building a
custom capturable AR to chase <the tree's 1.5% is not justified when the one-shot already wins 20%).

## Where a hierarchical AR WOULD matter: partial-P2P robustness (not peak speed)

The real gap is not speed but **coverage**: Modal 4-GPU containers vary (`post_c4_roofline.md`), and on a
partial-P2P topology (e.g. only GPUs 1<->2 connected) the one-shot cannot run and the safety guard falls back
to **full NCCL** (48.248), losing the win. A **topology-aware hierarchical** AR could do P2P *within* each
connected component and NCCL only *across* components, recovering part of the one-shot's advantage on partial
containers. This is a **reliability** lever (make the +20% apply on more allocations), not a way to exceed the
full-P2P 58.126. It is real future work but out of C5's "beat 58.126" scope, and it needs a custom
capturable kernel with per-component P2P + a cross-component NCCL stage.

## Task 3 verdict

For the stated goal (beat 58.126), a hierarchical all-reduce does **not** help: the one-shot is already the
single-sync latency-optimal collective for full-P2P 4-GPU, tree gives only +1.5%, two-shot is bandwidth-
oriented. No variant built. The hierarchical idea's real value is partial-P2P **robustness**, a separate goal.
