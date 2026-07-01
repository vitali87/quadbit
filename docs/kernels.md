# Kernels and results

> **HISTORICAL — the CubeCL/Rust track (superseded).** This document describes the early
> CubeCL/Rust kernels (`src/bin/`, `run_rust.py`), whose best design topped out at ~505k GFLOP/s
> with "CUTLASS ~3× faster." That track was **abandoned**. The current raw-PTX CUDA kernels
> (`cuda/*.cu`, `harness/run_cuda.py`) reach ~1510k dense (matching CUTLASS) and ~2012k sparse.
> For current results and honest standing see [paper_notes.md](paper_notes.md) and
> [standing.md](standing.md). Kept only as a record of the CubeCL-era negative results.

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
| `matmul_fp4_fed` | fed: shared staging, 2x8 warp tile, deep K, scale staging | ~292,000 |
| `matmul_fp4_bench` | interleaved A/B benchmark harness (not a ladder rung) | n/a |
| `matmul_fp4_ldm` | ldmatrix fragment-load microtest (negative result, see below) | FAIL |
| `matmul_fp4_ws` | warp-specialized producer/consumer (negative result, see below) | ~228,000 |
| `matmul_fp4_tile` | square 128x128 block-tile A/B (negative result, see below) | 0.87x |
| `matmul_fp4_db` | double-buffer prefetch A/B (negative result, see below) | 0.78x |

`matmul_fp4_fed` is our best, and it asymptotes near ~505,000 GFLOP/s:

| size | `matmul_fp4_fed` GFLOP/s |
|------|-------------------------:|
| 2048 | 291,882 |
| 4096 | 408,000 |
| 8192 | 489,052 |
| 16384 | 505,129 |

> **Correction.** This document previously called ~505,000 GFLOP/s the hardware ceiling
> and global maximum. That was wrong: it is the ceiling of *this kernel design*, not the
> silicon. A CUTLASS reference benchmark on the same card (see below) reaches ~3x more, so
> our kernel leaves the tensor cores idle most of the time. The register-file argument in
> "The ceiling is the register file" below is correct *for the 2x8 register-resident
> structure* but does not bound the hardware; read that section as why this particular
> design tops out, not as a hardware law.

## Reference: CUTLASS is ~3x faster (the real headroom)

`harness/cutlass_fp4.py` builds and runs CUTLASS example 79b
(`nv_float4 x nv_float4 -> f32`, ArchTag `Sm120`) on the same `RTX-PRO-6000`. This is the
apples-to-apples reference: the same FP4 `mma.sync` path (consumer Blackwell `sm_120` has no
`tcgen05`), the same `2*M*N*K` flop count, f32 accumulate, with a built-in correctness check
(all sizes report `Disposition: Passed`).

| size | CUTLASS 79b GFLOP/s | `matmul_fp4_fed` | CUTLASS faster by |
|------|--------------------:|-----------------:|:-----------------:|
| 2048 | 637,462 | 291,882 | 2.18x |
| 4096 | 1,240,800 | 408,000 | 3.04x |
| 8192 | 1,503,780 | 489,052 | 3.07x |

CUTLASS's 2048 number already exceeds our 16384 asymptote. The gap is the mainloop
architecture: CUTLASS runs a deep, multi-stage, warp-specialized async pipeline (TMA /
`cp.async` staging, correct FP4 `ldmatrix` operand loads, a persistent tile scheduler) that
keeps OMMA back to back. Our single-technique experiments (warp specialization, async copy,
double buffering) each regressed because they are isolated pieces of that pipeline, not the
whole machine. Build note: CUTLASS 4.6.0 needs CUDA 12.9 (CUDA 12.8.1 nvcc fails on
`__nv_atomic_load_n` in `subbyte_reference.h`).

## How the fed FP4 kernel works

A block of 4 warps (a 2x2 grid) owns a 64x128 output tile.

- **Shared staging.** Per block step it cooperatively stages a 64x128 tile of A and a 128x128 tile of B into shared memory, keeping the packed `e2m1x2` data in its native vector layout (no unpacking). It walks the staged tile 64 columns (one MMA-K) at a time with no barrier between sub-steps.
- **Wide warp tile for ILP.** Each warp reads 2 A-fragments and 8 B-fragments and issues 16 `mma.sync...kind::mxf4nvf4.block_scale` MMAs into a 2x8 grid of f32 accumulators, fed back in as C so products accumulate across K. The 16 independent accumulator chains are the instruction-level parallelism that hides the FP4 tensor-core (OMMA) latency.
- **Scale staging.** The per-block `ue8m0` scales are also staged in shared once per step rather than reloaded from global on every k-sub-step.

The only route to the FP4 tensor cores in CubeCL is the low-level `cmma::MmaDefinition` plus `execute_scaled` path. The high-level WMMA `cmma::Matrix` API has no FP4. Per-lane registers and scale indices are managed by hand via `position_of_nth` and `scales_index`.

## What was tried and what the bottleneck is

The fed FP4 kernel is **latency and ILP bound**, not occupancy bound. The constructive
findings, each measured:

- **Wider warp tiles win** up to the register limit. The 2x8 tile uses 255 registers per thread (the hardware maximum) with zero spills, giving the most independent accumulator chains per SM. Widening from 2x2 to 2x4 to 2x8 climbed roughly 160k, 230k, 253k.
- **Higher per-thread register use beats more occupancy.** At 255 registers the kernel runs 2 blocks per SM, and that is faster than configurations with fewer registers and more resident blocks. This is the classic "better performance at lower occupancy" regime.
- **Scale staging helped** because it removed memory-pipe global loads (96 to 62 in the SASS) that compete for a contended resource.

## The ceiling is the register file (proven, not assumed)

The kernel is latency/ILP bound: more independent accumulator chains in flight would
hide more of the FP4 tensor-core (OMMA) latency. The wall on adding them is the SM
register file, and it is a hardware boundary:

- Registers are capped at **255 per thread** (architectural maximum). The 2x8 tile
  already uses exactly 255 with zero spills, so a warp cannot hold more accumulators.
- SM120 has **65536 registers per SM**. The baseline runs 2 blocks x 4 warps =
  **8 warps x 255 = 65280**, saturating the file. There is no room for more compute warps.

Both structural escapes were built and measured, and both fail:

- **`ldmatrix` (`matmul_fp4_ldm`).** The only sub-byte `ldmatrix` that ptxas accepts on
  `sm_120a` is the format-converting `m8n16`/`m16n16 .b8x16.b4x16_p64`. It **expands**
  each 4-bit value into its own byte (the fragment dump shows `0x02020202` where packed
  all-ones e2m1x2 should be `0x22222222`). CubeCL's `mma.sync.mxf4nvf4` wants packed
  e2m1x2, so it reads the expanded bytes as packed pairs and every other K element is the
  inserted zero, halving every output. A lane's fragment then holds only half of K;
  loading the full operand expanded needs 2x the registers, fatal at the 255 ceiling. The
  CubeCL codegen patch in `vendor/cubecl-cpp-0.10.0` makes the instruction assemble and run
  (the original blocker is genuinely fixed), but the instruction is the wrong tool here.
- **Warp specialization (`matmul_fp4_ws`).** Dedicating warps to staging so consumers spend
  their whole budget on accumulators cannot help, because a specialized block (consumers at
  255 + a producer warp) fits exactly **one** block per SM, giving at most ~7 compute warps
  versus the baseline's 8. Measured: 228,000 vs 291,882 GFLOP/s at 2048 (~22% slower),
  correct. `setmaxnreg` (warpgroup register reallocation) assembles on `sm_120a` but is
  also capped at 255/thread and does not add a second block, so it changes nothing.

Two more staging-side levers were built and measured interleaved (same clock), and both
regress, which pins down *why* the baseline config is optimal:

- **Square block tile (`matmul_fp4_tile`).** An 8-warp 128x128 block has higher staging
  arithmetic intensity (0.50 vs the 4-warp 64x128's 0.33, so ~33% fewer staging loads per
  output) yet runs **0.87x** (351,821 vs 403,200 GFLOP/s at 4096, both PASS). The 8-warp
  block fits only **one** block per SM, and that is the loss: with 2 resident blocks the
  scheduler keeps the tensor cores fed from one block while the other stalls at `sync_cube`,
  and a single big block has no second block to overlap its barrier. The win from 2 blocks
  is **barrier overlap, not occupancy**, and staging is not the wall.
- **Double-buffer prefetch (`matmul_fp4_db`).** Staging the next k-tile into a second shared
  buffer during the current compute (plain loads, halving the barriers per step) runs
  **0.78x** (324,156 vs 413,661, both PASS). The prefetch loads are issued after the compute
  loop, so the barrier waits on their full global-load latency with nothing behind it to
  hide it. Hiding it would need the loads issued early, which means either holding the whole
  next tile in registers (impossible at the 255 ceiling) or `cp.async` (the async pipeline
  that already regresses harder, 0.54x). Every form of prefetch loses because the 2-block
  barrier overlap already hides the loads.

Going wider *in this register-resident accumulator structure* is capped, and `tcgen05`/UMMA
(accumulators in tensor memory) is absent on `sm_120`. But that does not bound the hardware:
CUTLASS reaches ~3x more on the same `mma.sync` path (see the reference section above) by
hiding OMMA latency with a deep async pipeline rather than with more register-resident
accumulator chains. So this section explains why *the 2x8 design* tops out, not why the
hardware does.

Other measured negatives (kept for the record): the index-math hoist is neutral (integer
address arithmetic overlaps OMMA latency for free), and software pipelining (cp.async double
buffering) regresses because the kernel is compute bound and the loads are already hidden.
