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

## Open question being probed

Modal runs GPU containers on gVisor + nvproxy, which likely blocks the `ncu` (Nsight Compute) hardware-counter profiling path. `harness/probe_ncu.py` tests this empirically. If `ncu` is blocked, the workflow splits: Modal for dev / run / time / parallel autotune sweeps, plus a cheap bare-metal box (Vast.ai / RunPod `--cap-add SYS_ADMIN`) for occasional deep `ncu` profiling.

## Setup

```bash
uv sync --extra dev
modal setup            # interactive: authenticate to your Modal workspace
uv run modal run harness/probe_ncu.py
```
