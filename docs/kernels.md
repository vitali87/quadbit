# Kernels and results

Every kernel is a standalone binary in `src/bin/`, run with:

```bash
uv run modal run harness/run_rust.py --bin <name>
```

Each prints its shape, a correctness check, and best-of-N timing in GFLOP/s.

## The ladder

Built bottom up, each rung a separate binary so they stay runnable and comparable. Throughput is `2*M*N*K / time`, measured on `RTX-PRO-6000` (SM120) at 2048x2048x2048 unless noted.

### FP16 path (warm-up: learn CubeCL and the tensor cores)

| Binary | Technique | GFLOP/s |
|--------|-----------|--------:|
| `matmul` | naive CubeCL matmul | ~6,350 |
| `matmul_tiled` | shared-memory tiled | ~8,920 |
| `matmul_regblock` | register-blocked | ~41,900 |
| `matmul_cmma` | tensor-core (cmma) FP16 | ~33,600 |
| `matmul_cmma_fed` | shared staging + 2x2 warp tiles, fragment reuse | ~77,000 |

### FP4 path (the actual goal: block-scaled MXFP4)

| Binary | Technique | GFLOP/s |
|--------|-----------|--------:|
| `matmul_fp4_hello` | one `mma.sync` block-scaled FP4 MMA, m16 n8 k64 | PASS |
| `matmul_fp4` | full matmul, one warp per 16x8 tile, K accumulation | ~82,500 |
| `matmul_fp4_fed` | fed: shared staging, 2x8 warp tile, deep K, scale staging | ~280,000 |
| `matmul_fp4_bench` | interleaved A/B benchmark harness (not a ladder rung) | n/a |

`matmul_fp4_fed` is the current best. It also keeps climbing with problem size as launch and tail quantization fade: about 280,000 GFLOP/s at 2048, around 400,000 at 4096 (interleaved measurement), and roughly 500,000 at 8192 on a clock-boosted run. That is about a quarter to a third of the card's estimated dense FP4 tensor-core peak.

## How the fed FP4 kernel works

A block of 4 warps (a 2x2 grid) owns a 64x128 output tile.

- **Shared staging.** Per block step it cooperatively stages a 64x128 tile of A and a 128x128 tile of B into shared memory, keeping the packed `e2m1x2` data in its native vector layout (no unpacking). It walks the staged tile 64 columns (one MMA-K) at a time with no barrier between sub-steps.
- **Wide warp tile for ILP.** Each warp reads 2 A-fragments and 8 B-fragments and issues 16 `mma.sync...kind::mxf4nvf4.block_scale` MMAs into a 2x8 grid of f32 accumulators, fed back in as C so products accumulate across K. The 16 independent accumulator chains are the instruction-level parallelism that hides the FP4 tensor-core (OMMA) latency.
- **Scale staging.** The per-block `ue8m0` scales are also staged in shared once per step rather than reloaded from global on every k-sub-step.

The only route to the FP4 tensor cores in CubeCL is the low-level `cmma::MmaDefinition` plus `execute_scaled` path. The high-level WMMA `cmma::Matrix` API has no FP4. Per-lane registers and scale indices are managed by hand via `position_of_nth` and `scales_index`.

## What was tried and what the bottleneck is

The fed FP4 kernel is **latency and ILP bound**, not occupancy bound. Findings, each measured:

- **Wider warp tiles win** up to the register limit. The 2x8 tile uses 255 registers per thread (the hardware maximum) with zero spills, giving the most independent accumulator chains per SM. Widening from 2x2 to 2x4 to 2x8 climbed roughly 160k, 230k, 253k.
- **Higher per-thread register use beats more occupancy.** At 255 registers the kernel runs 2 blocks per SM, and that is faster than configurations with fewer registers and more resident blocks. This is the classic "better performance at lower occupancy" regime.
- **Scale staging helped** because it removed memory-pipe global loads (96 to 62 in the SASS) that compete for a contended resource.
- **The index-math hoist is neutral.** Precomputing fragment offsets cuts integer instructions but does not speed the kernel up, because the integer address arithmetic overlaps the OMMA latency for free. Integer issue is not the bottleneck.
- **Software pipelining (cp.async double buffering) regresses.** The kernel is compute bound, so async copy adds barrier and small-transaction overhead with no memory latency left to hide.
- **`ldmatrix` is blocked.** CubeCL 0.10 emits `ldmatrix.m8n16.b8` and a `.trans` variant for FP4 that ptxas (CUDA 12.8) rejects. The hand-managed `position_of_nth` path is the only working route to FP4 fragments in this toolchain.

The remaining headroom sits behind the tensor core's intrinsic MMA latency. Breaking past it needs a structural change (a fixed `ldmatrix` path on a future CubeCL, or a warp-specialized design), not incremental tuning.
