# C4: enable a one-shot custom all-reduce on 4 PCIe GPUs -> +20.5% SM120 decode (beats the dense SOTA)

The decode floor is 90.8% `ncclDevKernel_AllReduce_Sum_bf16_RING_LL`
([floor_decomposition.md](floor_decomposition.md)). The fix attacks that kernel directly.

## The wall is a disabled fast-path, not a hardware limit

vLLM ships a one-shot / two-shot **custom all-reduce** (P2P IPC buffers, latency-optimal for small tensors)
but **disables it on more than two PCIe-only GPUs** (`custom_all_reduce.py:150` and `should_custom_ar:241`,
both gated on `is_fully_connected`, i.e. NVLink). Its own comment: "for 4 or more non NVLink-capable GPUs,
custom allreduce provides little performance improvement over NCCL." That conclusion is for general
(bandwidth-bound) workloads. At **decode, batch=1**, the all-reduce payloads are a few KB = **latency-bound**,
where the one-shot AR (1 hop: each rank reads 3 peers' buffers, local sum) crushes RING_LL (6 serialized
hops). The driver confirms full P2P on these boxes: `can_device_access_peer` matrix
`[[0,1,1,1],[1,0,1,1],[1,1,0,1],[1,1,1,0]]`.

## The patch (opt-in, `QB_FORCE_CUSTOM_AR=1`)

In the plugin's `install()` (runs via `vllm.general_plugins` before the TP group builds `CustomAllreduce`):
spoof `current_platform.is_fully_connected -> True` so the NVLink-gated one-shot path activates, and set
`VLLM_SKIP_P2P_CHECK=1` so vLLM trusts the driver's `can_device_access_peer` instead of its cross-process
P2P probe (which fails spuriously on most Modal containers even though the hardware supports P2P). The
one-shot AR is a mathematically correct sum; correctness is verified downstream (coherent generation + PPL),
and a truly P2P-blocked topology would show garbage, not the small FP-reorder drift observed.

## Result (DeepSeek dense baseline, 4 GPU, captured, same harness as C2)

| config | decode tok/s | vs 48.248 SOTA row | PPL (mito80) | capture |
|---|---:|---:|---:|---|
| baseline RING_LL (C2 A1 = prior SOTA row) | 48.248 | 1.000x | 4.1222 | FULL |
| baseline RING_LL (fresh same-session control) | 49.263 | 1.021x | 4.1222 | FULL |
| scoped `NCCL_ALGO=allreduce:tree` | 48.983 | 1.015x | 4.0102 | FULL |
| **custom one-shot AR** (4 runs) | **57.783 / 58.545 / 58.126 / 58.126** (median **58.126**) | **+20.5%** | 4.2514 | FULL |

**Median 58.126 tok/s (4 runs; mean 58.145) vs the 48.248 prior SOTA row = +20.5%.** Against a fresh
same-session RING_LL control (49.263) it is **+18.0%**, so **~+18-20%** depending on baseline container
variance. Reproducible across 4 runs (identical PPL 4.2514 = the deterministic one-shot reduction
signature), CUDA-graph capture FULL. With `VLLM_SKIP_P2P_CHECK=1` (only after the full-P2P check, below)
the AR engages on every container (0 "custom allreduce disabled" warnings across the skip-check runs).

## Mechanism (the kernel swap is visible in the trace)

Re-profiling the floor with custom AR engaged (`floor_profile --force-custom-ar`, log
[c4_floor_profile_customar2.log](../audit/logs/c4_floor_profile_customar2.log), 0 "custom allreduce
disabled" warnings):

| | top collective kernel | eager GPU-busy |
|---|---|---:|
| baseline | `ncclDevKernel_AllReduce_Sum_bf16_RING_LL` | 90.8% |
| custom AR | **`vllm::cross_device_reduce_1stage<bf16, 4>`** (one-shot) | 90.6% |

The RING_LL all-reduce is gone, replaced by the one-stage (one-shot) custom kernel. Note it is **still ~90%
of eager GPU-busy time**: decode stays collective-bound either way (the medium is still PCIe). The +20.5%
is the one-shot's lower **latency per all-reduce** (1 hop vs 6), which shows in **captured wall-clock**
(wall64 1.22 s vs 1.50 s), not in the eager GPU-busy breakdown (which is inflated by launch/sync overhead
and hides the per-op latency). So the win is real but the collective remains the wall (see verdict's next
lever).

## Quality note (speed validated; quality NOT proven neutral)

**What is validated: speed.** The +20.5% decode is measured, reproducible, and capture-FULL.

**What is NOT validated: quality.** mito80 PPL swings with the all-reduce **reduction order**: tree 4.0102,
ring 4.1222, one-shot 4.2514 (an 0.24 spread that goes **both** directions). This is FP non-associativity of
bf16 summation amplified by greedy decode on an 80-token passage: the one-shot AR sums the same values in a
different order, exactly as switching any NCCL algorithm would, so mito80 cannot rank AR algorithms for
quality. We therefore do **not** claim quality-neutral. Quality is not considered regressed **only because
the PPL shift is reduction-order-dependent**, and it must be judged with the downstream eval / fixed quality
protocol, not this serving row. The bit-identical fibonacci completion is a **smoke check** (the AR returns a
correct sum, not garbage), **not a quality proof**. This caution is deliberate: earlier campaigns showed how
easy it is to overread a serving row before the confounds are isolated (the sparse-MLP path first trailed
NVFP4 at batch because the non-MLP linears were still bf16, not because the sparse kernel was bad; the
NVFP4-base + sparse-MLP row only reached parity at B=8/32/64 once the real overhead was isolated).

## Scope

This is a **serving-infra** win (a collective-algorithm swap), independent of the sparse policy: it applies
to the dense fused path AND to sparse D2 (the floor was the all-reduce, not sparse MMA, so the floor is
shared). It is **not** "quadbit sparse beats dense MoE decode" and **not** a sparse-kernel or custom-kernel
speedup. It lifts the whole quadbit SM120 decode stack ~+18-20% past the prior same-harness SOTA row.
Prior sparse-only D2 decode was floor-bound too, so custom AR stacks on the C3 compaction there as well
(measured next if pursued). Verdict: [verdict.md](verdict.md).
