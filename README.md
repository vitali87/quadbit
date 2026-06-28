# quadbit

Custom GPU kernels for the low-bit Blackwell gaps NVIDIA's libraries leave open, with Rust as the framework shell.

Early scaffold. Developed on Modal cloud.

## Setup

```bash
uv sync --extra dev
modal setup                              # authenticate to your Modal workspace
uv run modal run harness/probe_ncu.py    # check the GPU dev environment
```

## Docs

- [Hardware and toolchain](docs/hardware.md)
- [Profiling on Modal](docs/profiling.md)
