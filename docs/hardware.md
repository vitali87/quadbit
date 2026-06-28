# Hardware and toolchain

Development runs on Modal cloud. Both target Blackwell architectures are available there.

| `gpu=` | Arch | VRAM | $/hr | Use |
|--------|------|------|------|-----|
| `RTX-PRO-6000` | SM120 (same silicon as RTX 5090) | 96 GB | 3.03 | primary SM120 development |
| `B200` | SM100 (datacenter Blackwell) | 180 GB | 6.25 | NVFP4 grouped MoE target |
| `T4` | Turing | 16 GB | 0.59 | cheap environment probes |

## Toolchain

- Base image: `nvidia/cuda:12.8.1-devel-ubuntu22.04` (provides `nvcc`).
- Target `sm_120a`, not plain `sm_120` or `sm_120f`. The block-scaled FP4 MMA is rejected by the non-arch-specific targets.
- CUDA 12.8+ is required for SM120. B300 (reachable only via Modal's `B200+` alias) would require CUDA 13.0+.
- SM120 is not binary compatible with SM100, so a kernel built for B200 will not run on RTX-PRO-6000 and vice versa.
