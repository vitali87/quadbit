# Profiling on Modal

`harness/probe_ncu.py` checks the GPU development environment on Modal. Result on a T4 (2026-06-27):

- `nvcc` 12.8 and `ncu` 2025.1.1 are installed; driver 580, CUDA 13 capable.
- **`ncu` hardware-counter profiling does not work.** It fails at init with `Failed to initialize the profiler: LibraryNotLoaded` (exit 9). Modal runs GPU containers on gVisor and nvproxy, which does not expose the profiling driver interface. The result is the same on every Modal GPU because the runtime is identical.

## Resulting workflow

- **Modal** for the bulk of the work: write, compile, run, and time kernels on real SM120 (`RTX-PRO-6000`) and SM100 (`B200`), validate correctness, and run wide parallel autotune sweeps. These rely on measured wall-clock speed, which needs no `ncu`.
- **A bare-metal box** for occasional deep profiling: a Vast.ai 5090 or a RunPod instance launched with `--cap-add SYS_ADMIN` and profiling counters enabled, used to read `ncu` roofline and memory-throughput metrics for the hottest kernel.
