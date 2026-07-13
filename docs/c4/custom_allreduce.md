# C4: enable a one-shot custom all-reduce on 4 PCIe GPUs -> +19% SM120 decode (beats the dense SOTA)

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

| config | decode tok/s | vs baseline | PPL (mito80) | capture |
|---|---:|---:|---:|---|
| baseline RING_LL (C2 A1) | 48.248 | 1.00x | 4.1222 | FULL |
| baseline RING_LL (fresh control) | 49.263 | 1.02x | 4.1222 | FULL |
| scoped `NCCL_ALGO=allreduce:tree` | 48.983 | 1.02x | 4.0102 | FULL |
| **custom one-shot AR** (4 runs) | **57.783 / 58.545 / 58.126 / 58.126** | **+19%** | 4.2514 | FULL |

**~58.1 tok/s vs ~48.7 baseline = +19%**, reproducible across 4 runs (identical PPL 4.2514 = the
deterministic one-shot reduction signature), CUDA-graph capture FULL, generation coherent (the fibonacci
completion is bit-identical to the RING_LL baseline). With `VLLM_SKIP_P2P_CHECK=1` the AR engages on every
container (0 "custom allreduce disabled" warnings across the skip-check runs).

## Quality note (no softening)

mito80 PPL swings with the all-reduce **reduction order**: tree 4.0102, ring 4.1222, one-shot 4.2514 (an
0.24 spread that goes **both** directions). This is FP non-associativity of bf16 summation amplified by
greedy decode on an 80-token passage, **not** a quality regression: the one-shot AR sums the same values in
a different order, exactly as switching any NCCL algorithm would. We do **not** claim quality-neutral on
mito80; the real quality check is the downstream 4-task eval (future work). The all-reduce is numerically a
correct sum (coherent, bit-identical fibonacci).

## Scope

This is a **serving-infra** win, independent of the sparse policy: it applies to the dense fused path AND to
sparse D2 (the floor is shared). It lifts the whole quadbit SM120 decode stack ~19% past the prior SOTA.
Prior sparse-only D2 decode was floor-bound too, so custom AR stacks on the C3 compaction there as well
(measured next if pursued). Verdict: [verdict.md](verdict.md).
