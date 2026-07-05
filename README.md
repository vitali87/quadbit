# quadbit

Hand-written 4-bit (FP4) tensor-core kernels for **consumer/pro Blackwell (SM120)**, the RTX PRO 6000 and RTX 5090, where NVIDIA's libraries leave FP4 gaps.

On SM120 the FP4 tensor cores exist but the usable software stack is thin. cuBLAS ships dense FP4 only; CUTLASS ships both dense and a sparse NVFP4 example for GeForce Blackwell ([example 80b](https://github.com/NVIDIA/cutlass/blob/main/examples/80_blackwell_geforce_sparse_gemm/80b_blackwell_geforce_nvfp4_nvfp4_sparse_gemm.cu), since CUTLASS 3.9.0), but the SM120 block-scaled path has documented correctness and autotuner problems in practice ([cutlass#3096](https://github.com/NVIDIA/cutlass/issues/3096)), and nothing wraps a sparse FP4 kernel in a real deployment stack. quadbit hand-writes (raw PTX, `nvcc -arch=sm_120a`) both a dense FP4 GEMM that reaches the silicon ceiling and a 2:4-sparse FP4 GEMM at the SM120 bandwidth roofline, then builds the full deployment stack around them: a weight packer, a fused activation quantizer, an `nn.Linear` drop-in, fused transformer-block glue kernels, and a one-shot plus QAT recovery pipeline that makes the sparse path usable on real models. Everything is measured on Modal cloud RTX PRO 6000. (The head-to-head vs CUTLASS's sparse example 80b is now run, correctness-gated: we win at 4096–8192, CUTLASS wins at 16384 — see below and [docs/paper_notes.md](docs/paper_notes.md).)

There is no tcgen05 on SM120 (that is SM100 only), so the kernels use warp-level `mma.sync` and `mma.sp`. The mma, ldmatrix, scale, and metadata layouts were derived empirically by probe-and-verify (not from docs) and validated to `maxrel 0`.

## Headline results (measured, RTX PRO 6000)

Throughput vs cuBLAS bf16 (what production runs today) and vs real CUTLASS FP4, M=N=K:

| size | cuBLAS bf16 | CUTLASS FP4 | dense FP4 (ours) | 2:4-sparse FP4 (ours) |
|------|-------------|-------------|------------------|------------------------|
| 4096 | 372 TF/s | 1222 | 1136 (3.06× bf16) | 1512 (4.07× bf16) |
| 8192 | 423 TF/s | 1497 | 1556 (3.68× bf16) | 2207 (5.22× bf16) |
| 16384| 405 TF/s | n/a | 1645 (4.06× bf16) | 1782 (4.39× bf16) |

- **Dense FP4 is competitive with CUTLASS on square sizes but loses on real LLM shapes.** Square (both cudaEvent-timed): 758 / 1220 / 1510 TF/s at 2048 / 4096 / 8192 vs CUTLASS 79b 634 / 1222 / 1497 (win / tie / win). But on the *rectangular* Llama-3-8B GEMM shapes that actually run, ours loses to 79b: attn 4096³ **0.89×**, ffn-up (N=14336) **0.93×**, ffn-down (K=14336) **1.01×** (all 79b-verified). The dense win was a square-size artifact; the honest dense story is "competitive, slightly behind CUTLASS." The consistent win is the sparse path below.
- **Sparse FP4 vs CUTLASS 80b** (the gating head-to-head, now run and correctness-gated on 80b's own reference check, which **PASSES** at every size — so #3096's block-scaled correctness bug does not affect this example): we are **1.16× at 4096** and **1.14× at 8192**, but **CUTLASS 80b wins at 16384** (0.96×, 1859 vs 1785 TF/s). So we are the fastest sparse FP4 at the 4–8K tile sizes real LLM GEMMs use, not at every size. On the rectangular Llama-3-8B shapes the sparse win is consistent: attn 4096³ **1.18×**, ffn-up (N=14336) **1.14×**, ffn-down (K=14336) **1.17×** vs 80b (all 80b-verified). (Effective-FLOP metric, 2·M·N·K, both cudaEvent-timed on the same card.)

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

- **Dense FP4, W4A4, +0.63 PPL — matched to the reference, no calibration.** The FP4 tensor core multiplies fp4×fp4, so the deployed kernel is **W4A4** (weights *and* activations 4-bit); the often-quoted +0.3 PPL is weight-only (W4A16), which the hardware never runs. A crude per-32 single-level activation quant costs ~**+2 PPL**, but the **per-16 two-level NVFP4 recipe reaches +0.63** on Llama-3.1-8B (7.27→7.90) — at/below the modelopt-calibrated reference's **+0.71** (vLLM native NVFP4 7.97, same wikitext-2 windows), and **with no calibration data**. The gap was block granularity and two-level activation scaling, not calibration. +0.7 is the real W4A4 floor and our own amax recipe reaches it.
- **Two-level NVFP4** (per-16 `ue4m3` local scale, per-row fp32 global) reaches block reconstruction rel **0.097** on real Qwen3-8B with no training, vs **0.13** for the simpler, faster MXFP4 (`ue8m0`) path. NVFP4 is the accuracy path, MXFP4 the speed path (2.15× over bf16). Both are real-scale and deployable with no fine-tuning.
- **Fused FP4 transformer block** on real Qwen3-8B weights (fused RMSNorm, residual-add, and SwiGLU quantizers around the GEMMs), no training: **2.16 to 2.19× over bf16** end-to-end at block rel 0.13.
- **Sparse FP4** needs recovery, because Blackwell FP4 2:4 is **pair-granular** (the mma selects 2 of every 4 fp4 pairs), unlike existing element-granular 2:4 checkpoints. The pipeline (pair-granular SparseGPT one-shot prune, then knowledge distillation from the dense teacher, then QAT with straight-through fake-quant of both weights and activations) recovers TinyLlama-1.1B to **9.60 PPL through the real sparse FP4 kernel** (dense teacher 7.53). On a **real 8B model, sparse recovery loses to dense on accuracy**: Meta-Llama-3-8B deploys at **8.47 PPL through the two-level sparse kernel** (per-32 matched QAT, 2k warm-restart phase-2), vs dense-W4A4 **6.91 (+0.71)**, so sparse is about **1.56 PPL behind**. Two results settled this. **The deploy gap is closed:** the two-level sparse kernel adds a per-row and per-column fp32 global rescale in the mma epilogue, so through-kernel PPL now **equals the trained fake-quant (8.47 == 8.47)**; the old single-level kernel would deploy the same checkpoint at **11.89** (a same-checkpoint A/B flips 22% of top-1 tokens single-level vs 8% two-level). The rescale costs **2–10%** of throughput and the two-level kernel still beats CUTLASS 80b on every shape (**1.01–1.13×**). **The data lever is negative:** a full-scale diverse-corpus recovery (decontaminated C4, ~196M tokens) flattened at **10.82 on the WikiText-2 test** because C4 is out of distribution for that narrow metric, never approaching the ~7.4 target. Sparse is a **speed play** (~1.33× over our dense) that now deploys at its trained accuracy but does not beat dense; dense FP4 (+0.63, zero-training) is the accuracy result and sparse is an honest speed-only Pareto point.

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
