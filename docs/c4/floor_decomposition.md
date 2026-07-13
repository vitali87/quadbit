# C4: the SM120 decode step is 94.5% non-MoE floor, and the floor is 90.8% one NCCL all-reduce

C3 improved quadbit's sparse D2 decode 2.80x but stayed 3x under the dense NVFP4 fused SOTA (48.248 tok/s).
The roofline shows why chasing the MoE was the wrong target, and where the real decode headroom is.

## Roofline (from measured C2/C3 numbers)

| component | ms/tok | share of dense step |
|---|---:|---:|
| dense step (`QB_MOE=off`, native fused NVFP4 = the SOTA) | 20.73 | 100% |
| non-MoE floor (skip-MoE = 51.033 tok/s) | 19.60 | **94.5%** |
| dense MoE apply (CUTLASS fused) | 1.13 | 5.5% |

The entire sparse-vs-dense MoE decode fight is over a **5.5% slice**. Even a perfect fused 2:4-sparse
decode kernel (beat CUTLASS, zero padding, exploit that 2:4 halves the weight bytes for 6 of 8 D2 slots ->
0.625x weight traffic) only takes the MoE apply 1.13 -> 0.71 ms = **+2.1%** (49.3 tok/s). Chasing that
kernel is chasing 2%. The 19.6 ms floor is 94.5% and is shared by dense and sparse.

## What is the 19.6 ms floor? Profile it (vLLM worker profiler, dense baseline, eager, 16 decode steps)

GPU-kernel time by category (rank0, `floor_profile` in serve_dsv4.py, log
[c4_floor_profile.log](../audit/logs/c4_floor_profile.log)):

| category | ms (16 steps) | share |
|---|---:|---:|
| **NCCL AllReduce (`ncclDevKernel_AllReduce_Sum_bf16_RING_LL`)** | **2809.8** | **90.8%** |
| norm/elementwise | 141.1 | 4.6% |
| gemm/moe | 69.1 | 2.2% |
| attention+DSA | 21.0 | 0.7% |

Decode is ~90% blocked inside **one ring all-reduce over PCIe**. Attention, DSA, and GEMM are together
under 3%. This is 4-GPU tensor parallelism with **no NVLink**: every one of 43 layers does a per-layer TP
all-reduce, and at batch=1 the payloads are tiny, so this is **latency-bound**, where a ring (2*(N-1)=6
serialized hops at N=4) is the worst choice.

## Verdict: attack the all-reduce, not the MoE

The decode wall is the PCIe ring all-reduce, 90.8% of the step and 40x the headroom of the MoE slice.
Cutting its latency lifts the whole quadbit SM120 stack (both dense and sparse paths, since the floor is
shared). Next: [custom_allreduce.md](custom_allreduce.md).
