# quadbit

Hand-written 4-bit (FP4) tensor-core kernels for **consumer/pro Blackwell (SM120)**, the RTX PRO 6000 and RTX 5090, where NVIDIA's libraries leave FP4 gaps.

On SM120 the FP4 tensor cores exist but the software mostly does not: cuBLAS and CUTLASS ship dense FP4 only, and **no sparse FP4 path exists at all**. quadbit hand-writes (raw PTX, `nvcc -arch=sm_120a`) both a dense FP4 GEMM that reaches the silicon ceiling and the **only 2:4-sparse FP4 GEMM on SM120**, then builds the full deployment stack around them: a weight packer, a fused activation quantizer, an `nn.Linear` drop-in, fused transformer-block glue kernels, and a one-shot plus QAT recovery pipeline that makes the sparse path usable on real models. Everything is measured on Modal cloud RTX PRO 6000.

There is no tcgen05 on SM120 (that is SM100 only), so the kernels use warp-level `mma.sync` and `mma.sp`. The mma, ldmatrix, scale, and metadata layouts were derived empirically by probe-and-verify (not from docs) and validated to `maxrel 0`.

## Headline results (measured, RTX PRO 6000)

Throughput vs cuBLAS bf16 (what production runs today) and vs real CUTLASS FP4, M=N=K:

| size | cuBLAS bf16 | CUTLASS FP4 | dense FP4 (ours) | 2:4-sparse FP4 (ours) |
|------|-------------|-------------|------------------|------------------------|
| 4096 | 372 TF/s | 1222 | 1136 (3.06× bf16) | 1512 (4.07× bf16) |
| 8192 | 423 TF/s | 1497 | 1556 (3.68× bf16) | 2207 (5.22× bf16) |
| 16384| 405 TF/s | n/a | 1645 (4.06× bf16) | 1782 (4.39× bf16) |

- **Dense FP4 matches or beats CUTLASS at every size** in an apples-to-apples clean measurement (both cudaEvent-timed, no torch dispatch): 758 / 1220 / 1510 TF/s at 2048 / 4096 / 8192 vs CUTLASS 634 / 1222 / 1497.
- **Sparse FP4 is the unique, defensible win**: CUTLASS and cuBLAS have no sparse FP4 on SM120 at all. quadbit beats the best available vendor FP4 (CUTLASS dense) by **+24% at 4096** and **+47% at 8192**, a capability rather than a tuning delta.

Real Llama-3-8B GEMM shapes, best routed kernel per shape vs cuBLAS bf16:

| shape | M/N/K | best dense | 2:4-sparse FP4 |
|-------|-------|-----------|----------------|
| prefill attn qkv/o | 4096³ | 3.20× | 4.14× |
| prefill ffn up | 4096/14336/4096 | 3.38× | 4.49× |
| prefill ffn down | 4096/4096/14336 | 3.64× | 5.09× |
| decode ffn up | 128/14336/4096 | 4.53× | (M<256) |
| decode attn qkv/o | 128/4096/4096 | 1.27× | (M<256) |

Prefill (the bulk of training and long-context compute) runs **3.0 to 3.6× dense, 4.1 to 5.0× sparse** over bf16. Decode is memory-bound: large shapes hit the DRAM ceiling (ffn-up 4.53×), while small square shapes (attn o_proj 4096², GQA fused-QKV) are at the shape's hardware ceiling near 1.3×, because the 4× smaller FP4 weight yields too few output tiles to fill the SM array. See [docs/paper_notes.md](docs/paper_notes.md) for the full roofline analysis.

## Accuracy (real models)

- **Dense FP4: about +0.3 PPL, zero training, any model** (Sparse-Llama-3.1-8B 7.89→8.16; Qwen2.5-3B 7.60→7.91). Production-ready drop-in.
- **Two-level NVFP4** (per-16 `ue4m3` local scale, per-row fp32 global) reaches block reconstruction rel **0.097** on real Qwen3-8B with no training, vs **0.13** for the simpler, faster MXFP4 (`ue8m0`) path. NVFP4 is the accuracy path, MXFP4 the speed path (2.15× over bf16). Both are real-scale and deployable with no fine-tuning.
- **Fused FP4 transformer block** on real Qwen3-8B weights (fused RMSNorm, residual-add, and SwiGLU quantizers around the GEMMs), no training: **2.16 to 2.19× over bf16** end-to-end at block rel 0.13.
- **Sparse FP4** needs recovery, because Blackwell FP4 2:4 is **pair-granular** (the mma selects 2 of every 4 fp4 pairs), unlike existing element-granular 2:4 checkpoints. The pipeline (pair-granular SparseGPT one-shot prune, then knowledge distillation from the dense teacher, then QAT with straight-through fake-quant of both weights and activations) recovers TinyLlama-1.1B to **10.03 PPL through the real sparse FP4 kernel** (dense teacher 7.53). This is monotonic in training data, so production parity is a data-scale question, not a method gap.

## The stack

- `cuda/matmul_sp_*.cu`, `cuda/sparse_fp4_lib.cu`: the 2:4-sparse FP4 GEMM (wide-swizzle TMA path) plus fused NVFP4 activation quantizer and the fused RMSNorm / residual-add / SwiGLU block-glue kernels.
- `cuda/matmul_fp4_*.cu`, `cuda/dense_*_lib.cu`: the dense FP4 GEMM family (prefill, split-K, split-N decode, cached-map async decode) and the real-scale MXFP4/NVFP4 kernels.
- `harness/quadbit_linear.py`: `QuadbitLinear`, an `nn.Linear` drop-in that packs weights and runs through the kernel.
- `harness/finetune_pair.py`: the pair-granular 2:4 recovery pipeline.
- `harness/real_model.py`: the full fused FP4 decoder block on real open-weight models.

## Setup

Requires a Modal workspace (kernels run on cloud RTX PRO 6000) and [uv](https://astral.sh).

```bash
uv sync --extra dev
modal setup                              # authenticate to your Modal workspace
uv run modal run harness/probe_ncu.py    # check the GPU dev environment
```

## Run

```bash
uv run modal run harness/bench_vs_bf16.py    # dense + sparse FP4 throughput vs cuBLAS bf16
uv run modal run harness/bench_llm_shapes.py # real Llama-3-8B GEMM shapes, routed
uv run modal run harness/real_model.py       # fused FP4 decoder block on a real model
uv run modal run harness/accuracy_hf.py      # dense FP4 accuracy (PPL) on real checkpoints
uv run modal run harness/finetune_pair.py    # pair-granular 2:4 sparse recovery
```

## Docs

- [Paper checkpoint (results, contributions, roofline analysis)](docs/paper_notes.md)
- [Kernels and results](docs/kernels.md)
- [Hardware and toolchain](docs/hardware.md)
- [Profiling on Modal](docs/profiling.md)
