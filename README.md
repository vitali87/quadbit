# quadbit

4-bit (FP4) GPU kernels in Rust for the Blackwell gaps NVIDIA's libraries leave open.

Hand-written block-scaled MXFP4 (`e2m1` values with `ue8m0` block scales) tensor-core matmul kernels for SM120, built on CubeCL and developed on Modal cloud.

## Setup

```bash
uv sync --extra dev
modal setup                              # authenticate to your Modal workspace
uv run modal run harness/probe_ncu.py    # check the GPU dev environment
```

## Run a kernel

```bash
uv run modal run harness/run_rust.py --bin matmul_fp4_fed   # current best FP4 matmul
```

## Docs

- [Kernels and results](docs/kernels.md)
- [Hardware and toolchain](docs/hardware.md)
- [Profiling on Modal](docs/profiling.md)
