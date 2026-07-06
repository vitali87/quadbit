# quadbit

Hand-written 4-bit (FP4) tensor-core kernels for **consumer/pro Blackwell (SM120)**, the RTX PRO 6000 and RTX 5090, where NVIDIA's libraries leave FP4 gaps.

On SM120 the FP4 tensor cores exist but the usable software stack is thin. cuBLAS ships dense FP4 only; CUTLASS ships both dense and a sparse NVFP4 example for GeForce Blackwell ([example 80b](https://github.com/NVIDIA/cutlass/blob/main/examples/80_blackwell_geforce_sparse_gemm/80b_blackwell_geforce_nvfp4_nvfp4_sparse_gemm.cu), since CUTLASS 3.9.0), but the SM120 block-scaled path has documented correctness and autotuner problems in practice ([cutlass#3096](https://github.com/NVIDIA/cutlass/issues/3096)), and nothing wraps a sparse FP4 kernel in a real deployment stack. quadbit hand-writes (raw PTX, `nvcc -arch=sm_120a`) both a dense FP4 GEMM that reaches the silicon ceiling and a 2:4-sparse FP4 GEMM at the SM120 bandwidth roofline, then builds the full deployment stack around them: a weight packer, a fused activation quantizer, an `nn.Linear` drop-in, fused transformer-block glue kernels, and a one-shot plus QAT recovery pipeline that makes the sparse path usable on real models. Everything is measured on Modal cloud RTX PRO 6000. The kernels are benchmarked on a correctness-gated SM120 FP4 backend leaderboard against CUTLASS 79b/80b **and** FlashInfer `mm_fp4` (all backends): quadbit does not win dense (FlashInfer `b12x`/`cutlass` lead by 1.35–2.2×), but quadbit sparse is the only deployed 2:4-sparse FP4 GEMM on the platform and beats even the best available dense FP4 in wall-clock on prefill shapes — see the headline below and [docs/paper_notes.md](docs/paper_notes.md).

There is no tcgen05 on SM120 (that is SM100 only), so the kernels use warp-level `mma.sync` and `mma.sp`. The mma, ldmatrix, scale, and metadata layouts were derived empirically by probe-and-verify (not from docs) and validated to `maxrel 0`.

## Headline result (measured, RTX PRO 6000)

**On SM120, the fastest way to run an FP4 GEMM — if the weight can be 2:4-pruned — is quadbit's sparse kernel, and no shipping library provides a sparse FP4 GEMM at all.** quadbit sparse beats the only other sparse FP4 kernel (CUTLASS 80b) on every shape, *and* beats the best available **dense** FP4 kernel (FlashInfer `mm_fp4` `b12x`/`cutlass`) in wall-clock on every Llama-3-8B prefill shape:

| shape (Llama-3-8B) | FlashInfer best **dense** | quadbit **sparse** 2:4 | sparse speedup |
|--------------------|---------------------------|------------------------|----------------|
| prefill attn 4096³ | 0.107 ms (b12x) | 0.100 ms | **1.07×** |
| prefill ffn-up (N=14336) | 0.350 ms (auto) | 0.301 ms | **1.16×** |
| prefill ffn-down (K=14336) | 0.342 ms (b12x) | 0.265 ms | **1.29×** |
| square 8192 | 0.767 ms (cutlass) | 0.557 ms | **1.38×** |

That is the Pareto corner CUTLASS, FlashInfer, SGLang, and vLLM leave empty. The trade is accuracy: sparse needs recovery and stays ~1.56 PPL behind dense on 8B, so it is a speed-only Pareto point.

- **We do NOT win the dense FP4 race.** The SM120 dense baseline moved: FlashInfer `mm_fp4` now ships a CUDA-13 `b12x` NVFP4 kernel plus a `cutlass` path, and on a correctness-gated leaderboard (same card, same shapes, output scored vs fp32 reference) they beat quadbit's deployed two-level dense by **1.35–2.2×** (square-8192 1433 vs 1045 TF/s; prefill attn 1283 vs 838; serving M=65536 ffn-down 1416 vs 639). The old "competitive with CUTLASS 79b" dense claim is stale. Dense stays a zero-training W4A4 accuracy drop-in (+0.63 PPL), not a speed leader.
- **Sparse FP4 vs CUTLASS 80b** (correctness-gated on 80b's own reference, which **PASSES** at every size, so #3096 does not affect it): quadbit's deployed two-level sparse wins every shape — attn 4096³ **1.08×**, ffn-up **1.01×**, ffn-down **1.12×**, square-8192 **1.09×** (1973 vs 1807 TF/s). quadbit is the only *deployed* sparse FP4 path (CUTLASS 80b is an unwrapped example; FlashInfer/SGLang/vLLM ship none).
- **Two SM120 stack findings from the leaderboard:** FlashInfer's `cudnn` FP4 backend fails on *every* shape (shipped cuDNN 9.10 < the 9.14 SM120 FP4 needs), and its fast `b12x` path collapses ~2.2× at large serving batch (M≥65536), where only `cutlass` holds ~1400 TF/s. `b12x` needs CUDA 13; quadbit's `sm_120a` block-scale mma only assembles under CUDA ≤12.8 (ptxas 13 rejects it), so the two cannot share a container. (All effective-FLOP 2·M·N·K, cudaEvent-timed; leaderboard in `harness/leaderboard_fp4.py`.)

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
- **Sparse FP4** needs recovery, because Blackwell FP4 2:4 is **pair-granular** (the mma selects 2 of every 4 fp4 pairs), unlike existing element-granular 2:4 checkpoints. The pipeline (pair-granular SparseGPT one-shot prune, then knowledge distillation from the dense teacher, then QAT with straight-through fake-quant of both weights and activations) recovers TinyLlama-1.1B to **9.60 PPL through the real sparse FP4 kernel** (dense teacher 7.53). On a **real 8B model, sparse recovery loses to dense on accuracy**: Meta-Llama-3-8B deploys at **8.47 PPL through the two-level sparse kernel** (per-32 matched QAT, 2k warm-restart phase-2), vs dense-W4A4 **6.91 (+0.71)**, so sparse is about **1.56 PPL behind**. Two results settled this. **The deploy gap is closed:** the two-level sparse kernel adds a per-row and per-column fp32 global rescale in the mma epilogue, so through-kernel PPL now **equals the trained fake-quant (8.47 == 8.47)**; the old single-level kernel would deploy the same checkpoint at **11.89** (a same-checkpoint A/B flips 22% of top-1 tokens single-level vs 8% two-level). The rescale costs **2–10%** of throughput and the two-level kernel still beats CUTLASS 80b on every shape (**1.01–1.12×**) and beats the best FlashInfer dense in wall-clock on prefill (**1.07–1.38×**). **The data lever is negative:** a full-scale diverse-corpus recovery (decontaminated C4, ~196M tokens) flattened at **10.82 on the WikiText-2 test** because C4 is out of distribution for that narrow metric, never approaching the ~7.4 target. Sparse is a **speed play** (~1.33× over our dense) that now deploys at its trained accuracy but does not beat dense; dense FP4 (+0.63, zero-training) is the accuracy result and sparse is an honest speed-only Pareto point.

## End-to-end serving (RTX PRO 6000, Llama-3.1-8B-Instruct)

Real serving engines on the same card, same model family, same protocol (CUDA graphs on, distinct per-request prompts, S=2048 prefill / GEN=128 decode, WikiText-2 16×2048). Both vLLM and SGLang run **native NVFP4 W4A4** on SM120 for this checkpoint (vLLM `modelopt_fp4` cutlass; SGLang FlashInfer CUTLASS `fp4_gemm`, autotuned), not the Marlin W4A16 fallback.

| engine | quant | weights | WT-2 PPL | prefill B=64 | decode B=64 |
|--------|-------|---------|----------|--------------|-------------|
| vLLM | bf16 | 15.0 GiB | 7.267 | 46880 | 4947 |
| vLLM | NVFP4 | 5.66 GiB | 7.974 | 116831 | 8465 |
| SGLang | NVFP4 | ~5.6 GiB | 7.97 | 109002 | 10145 |

Native NVFP4 is ~1.7× bf16 on both prefill and decode at B=64 for 2.6× smaller weights (+0.71 PPL). **quadbit's deployed W4A4 path matches this accuracy**: full-forward through-kernel PPL **7.90** (vs native NVFP4 7.97) at 3.93 GiB quantized-linear weights, zero calibration.

**quadbit inside vLLM beats native NVFP4 prefill at batch.** The two-level *sparse* MLP is monkeypatched into vLLM's LlamaMLP (V1 engine, paged attention + continuous batching + decode scheduler), NVFP4 for all non-MLP linears. Correct-accuracy fused sparse MLP prefill B=8/32/64 = **63454 / 79116 / 116421** tok/s vs full NVFP4 **61118 / 74916 / 110600** = **+3.8% / +5.6% / +5.3%**, through-serving WT-2 PPL **8.95** (recovered base, all-MLP sparse; vs dense 8.76). Three kernel enablers: a zero-copy transposed epilogue (contiguous MLP output, no transpose+copy), a two-level fused SwiGLU (correct accuracy in the fused path), and a single no-sync `fused_mlp_2lvl` entry that runs the whole MLP in one ctypes call with zero device syncs (removing ~64 `cudaDeviceSynchronize`/forward — the +7.7%/+8.3% at B=32/64 that turned parity into a win). Cleared the ≥5% bar without CUDA graphs. Decode still favors NVFP4 (sparse M=B underfills; no CUDA graphs yet). See `harness/quadbit_serve.py` and [docs/paper.md §9](docs/paper.md).

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
