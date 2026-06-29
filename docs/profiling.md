# Profiling on Modal

`harness/probe_ncu.py` checks the GPU development environment on Modal. Result on a T4 (2026-06-27):

- `nvcc` 12.8 and `ncu` 2025.1.1 are installed; driver 580, CUDA 13 capable.
- **`ncu` hardware-counter profiling does not work.** It fails at init with `Failed to initialize the profiler: LibraryNotLoaded` (exit 9). Modal runs GPU containers on gVisor and nvproxy, which does not expose the profiling driver interface. The result is the same on every Modal GPU because the runtime is identical.

## Resulting workflow

- **Modal** for the bulk of the work: write, compile, run, and time kernels on real SM120 (`RTX-PRO-6000`) and SM100 (`B200`), validate correctness, and run wide parallel autotune sweeps. These rely on measured wall-clock speed, which needs no `ncu`.
- **A bare-metal box** for occasional deep profiling: a Vast.ai 5090 or a RunPod instance launched with `--cap-add SYS_ADMIN` and profiling counters enabled, used to read `ncu` roofline and memory-throughput metrics for the hottest kernel.

## Static profiling substitute: `harness/dump_sass.py`

Since `ncu` is blocked, this recovers the numbers it would otherwise give. It runs a kernel with `CUBECL_DEBUG_LOG=stdout` to capture the generated CUDA, recompiles that source with `nvcc -arch=sm_120a --ptxas-options=-v`, and prints registers per thread, shared memory, spill stores and loads, plus a SASS opcode histogram (counts of OMMA, LDS, STS, LDG, BAR).

```bash
uv run modal run harness/dump_sass.py --bin matmul_fp4_fed
```

These counts are deterministic and clock independent, so they are the reliable signal for whether a change is a real improvement. The fed FP4 kernel reports 255 registers per thread, 0 spills, and 32 OMMA per step body.

## Clock-drift caveat and the interleaved benchmark

Absolute GFLOP/s on Modal swings with GPU clock boost between invocations. The same 8192 kernel read 488,000 then 510,000 GFLOP/s on back-to-back runs. So cross-run deltas below roughly 10 percent are not trustworthy.

To compare two kernel variants fairly, time them **interleaved in one process** so they share the clock state. `src/bin/matmul_fp4_bench.rs` is a reusable harness for this: it verifies both variants against an f32 reference (random representable FP4 with random per-block scales), then times them interleaved best-of-N and reports the ratio. Use it to vet any candidate change before committing.

```bash
uv run modal run harness/run_rust.py --bin matmul_fp4_bench
```
