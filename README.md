# quadbit

Custom GPU kernels targeting the gaps NVIDIA's libraries leave open on Blackwell, with Rust as the framework/orchestration shell.

## Goal (re-scoped from "beat PyTorch")

Beating tuned cuBLAS at dense GEMM solo is hopeless. The defensible target is fused / grouped (MoE) low-bit kernels on hardware the vendor libraries have not caught up to:

- **SM120 (consumer/workstation Blackwell, e.g. RTX 5090 / RTX PRO 6000)** is orphaned between datacenter SM100 (`tcgen05`/TMEM) and Hopper SM90 (`wgmma`). It has neither; its native path is warp-level `mma.sync` with `.block_scale`. Grouped block-scaled FP4 GEMM is broken or falls back to slow dequant across CUTLASS / vLLM / SGLang / FlashInfer today.
- **SM100 (datacenter Blackwell, B200)** decode-time / low-batch (M=1-16) NVFP4 grouped MoE GEMM is the top-ranked open target.

Rust enters as the framework layer (CubeCL/Burn already emit the SM120 `mma.sync` block-scaled path), with hand-written kernels for the few cases that need hand-tuning.

## Hardware

Developed on Modal cloud. Both target architectures are available:

| `gpu=` | Arch | $/hr | Use |
|--------|------|------|-----|
| `RTX-PRO-6000` | SM120 (= RTX 5090 silicon, 96GB) | 3.03 | primary SM120 dev |
| `B200` | SM100 datacenter Blackwell | 6.25 | NVFP4 grouped MoE target |
| `T4` | Turing | 0.59 | cheap probes |

Target `sm_120a` (not plain `sm_120`) for block-scaled FP4 MMA; CUDA 12.8+ on `nvidia/cuda:*-devel` images.

## Profiling: confirmed split workflow

`harness/probe_ncu.py` settled the open question on a Modal T4 (2026-06-27):

- `nvcc` 12.8 and `ncu` 2025.1.1 are installed; driver 580 / CUDA 13 capable.
- **`ncu` hardware-counter profiling is BLOCKED.** It fails at init with `Failed to initialize the profiler: LibraryNotLoaded` (exit 9): the gVisor + nvproxy runtime does not expose the profiling driver interface.

So the workflow splits:

- **Modal** (the 90%): write / compile / run / **time** kernels on real SM120 (`RTX-PRO-6000`) and SM100 (`B200`), validate correctness, and run massive parallel autotune / kernel-search sweeps (these use measured wall-clock speedup, which needs no `ncu`).
- **Bare-metal box** (the 10%): a Vast.ai 5090 or RunPod `--cap-add SYS_ADMIN` instance with profiling counters enabled, for occasional deep `ncu` roofline analysis of the hottest kernel.

## Setup

```bash
uv sync --extra dev
modal setup            # interactive: authenticate to your Modal workspace
uv run modal run harness/probe_ncu.py
```
