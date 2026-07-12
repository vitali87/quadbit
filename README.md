# quadbit

A **deployable sparse-FP4 stack** for consumer/pro Blackwell (**SM120**: RTX PRO 6000, RTX 5090).

quadbit is a deployable sparse-FP4 stack for RTX Blackwell / SM120. **Dense NVFP4 is now well served by FlashInfer, vLLM, and SGLang** — on a correctness-gated leaderboard (same card, same shapes, output scored vs an fp32 reference) their `b12x`/`cutlass` kernels beat quadbit's hand-written dense kernel by **1.35–2.2×**, so dense speed is not our headline. quadbit dense stays useful as a **zero-training W4A4 accuracy drop-in** (+0.63 PPL), a reference path, not a speed leader.

quadbit focuses on the remaining **sparse-FP4 Pareto corner**: two-level sparse kernels whose deployed accuracy matches the trained fake-quant, model-level sparse policy, and serving integration. No mainstream stack exposes a 2:4-sparse FP4 deployment path on SM120 — FlashInfer, SGLang, and vLLM ship none, and CUTLASS 80b (the only other sparse FP4 kernel, [example 80b](https://github.com/NVIDIA/cutlass/blob/main/examples/80_blackwell_geforce_sparse_gemm/80b_blackwell_geforce_nvfp4_nvfp4_sparse_gemm.cu) since CUTLASS 3.9.0) is an unwrapped example with documented block-scaled problems ([cutlass#3096](https://github.com/NVIDIA/cutlass/issues/3096)), not a deployment. **The current strongest result is a sparse-MLP + NVFP4 serving path on Llama-3.1-8B-Instruct and a cross-architecture sparse-policy transfer result on DeepSeek-V4-Flash and GLM-5.2**, where GLM route-slot D2 preserves a 4-task downstream smoke-suite average to within about one point of dense (.7508 vs .7603, no task collapsing). Everything is measured on Modal cloud RTX PRO 6000.

There is no tcgen05 on SM120 (that is SM100 only), so the kernels use warp-level `mma.sync` and `mma.sp`. The mma, ldmatrix, scale, and metadata layouts were derived empirically by probe-and-verify (not from docs) and validated to `maxrel 0`.

## Headline result (measured, RTX PRO 6000)

**On SM120, the fastest way to run an FP4 GEMM — if the weight can be 2:4-pruned — is quadbit's sparse kernel, and no shipping library exposes a *deployed* sparse FP4 GEMM.** quadbit sparse beats the only other sparse FP4 kernel (CUTLASS 80b, an unwrapped example) on every shape, *and* beats the best available **dense** FP4 kernel (FlashInfer `mm_fp4` `b12x`/`cutlass`) in wall-clock on every Llama-3-8B prefill shape:

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

**quadbit runs a correct, graph-capturable sparse-FP4 MLP inside vLLM on SM120 and beats production dense NVFP4 on decode.** The two-level *sparse* MLP is registered as a `torch.library` custom op (`quadbit::fused_mlp`) so vLLM's V1 fullgraph compile + CUDA-graph capture include it (NVFP4 for all non-MLP linears, recovered Llama-3.1-8B-Instruct). At decode it dispatches to a **split-K down projection** that fills the GPU; prefill uses the plain kernel. The sparse path provably runs under production graphs (`SPARSE_CALLS=7264`, through-serving PPL **10.2709**, not the 7.97 dense value). **Graph-vs-graph (the production-representative comparison):**

| metric | vLLM NVFP4 (graph, production) | quadbit sparse MLP + split-K down (graph) | Δ |
|--------|-------------------------------|-------------------------------------------|---|
| WT-2 PPL | 7.97 | 10.2709 | +2.30 |
| prefill B=8/32/64 | 66469/80825/119083 | 62914/77605/115069 | **−5.3% / −4.0% / −3.4%** |
| decode B=8/32/64 | 1046/4237/8384 | **1147/4543/8567** | **+9.7% / +7.2% / +2.2%** |

**Decode — the latency-critical, memory-bound serving regime — now beats production NVFP4 at every batch** with correct sparse accuracy. The earlier decode loss (−6 to −12%) was a diagnosed underfill: the sparse **down projection launched only 16 CTAs on ~188 SMs** at decode. A **split-K down kernel** (`matmul_sp_sk`, `gridDim.z` K-splits → 128 CTAs, f32 reduction + two-level-scale epilogue) fills the machine (down 109→56.5 µs, 1.94×) and flips the result. Prefill still trails by ~3–5% (uses the plain non-split-K down, which was never underfilled). Full breakdown, split-factor sweep, proofs, and commands in [docs/graph_serving_result.md](docs/graph_serving_result.md).

**End-to-end, sparse FP4 wins the majority of real request regimes.** A batch × prompt-length × generation-length crossover sweep (graph-vs-graph, prefix-caching off so each request pays a real prefill) shows **sparse wins total request latency outright in 81 of 112 regimes (plus 2 statistical ties)**: **single-stream (B=1) wins everywhere (+3.5–11.6%)**, and any batch wins once the generation length clears a prompt/batch-dependent boundary. NVFP4 keeps only the prefill-bound corner (high batch × long prompt × short generation, ≤3%). So for interactive/low-batch and long-generation serving — the dominant chat and agentic regimes — sparse FP4 is the faster end-to-end path at a fixed +2.3 PPL. The split-K decode advantage is a small-M GPU-underfill fix, so it *shrinks* as effective M grows (multi-token verification does not favor sparse). Heatmap, boundary map, and verification-shape analysis in [docs/crossover_result.md](docs/crossover_result.md); the training-free accuracy Pareto (why the +2.3 tax needs QAT, not placement) in [docs/accuracy_pareto.md](docs/accuracy_pareto.md).

**Accuracy repair reduces the PPL tax but not downstream capability.** A four-family repair tournament (zero-runtime calibration, low-rank adapters, activation-aware mask repair, distillation) found only distillation helps: it cuts the serving tax from **10.27 to 9.10 PPL** and keeps every serving win (speed is weight-value independent, so the 81/112 crossover and split-K decode win carry over unchanged). But the downstream check is decisive: on ARC-Challenge/HellaSwag/PIQA/Winogrande the repaired model is essentially unchanged from the un-repaired all-sparse model, while dense NVFP4 stays within 1–3 points of bf16. The ~20-point ARC-C/HellaSwag loss is **2:4 sparsity, not FP4 quantization**, and WikiText distillation recovers almost none of it (the CE-heavy PPL win is domain overfitting). So sparse FP4 remains a **speed-for-capability** operating point; the open frontier is sparse capability recovery, not serving plumbing. See [docs/crossover_result.md](docs/crossover_result.md), `harness/repair.py`, `harness/downstream_eval.py`.

## Cross-architecture sparse-policy transfer (DeepSeek-V4-Flash, GLM-5.2)

The sparse approach transfers from one dense Llama-8B to large MoE models across GPUs. The experts are **MXFP4** (not the config's FP8); we decode them value-exact (100% round-trip) and prune 2:4 + re-quantize to two-level NVFP4. A **segmented routed-row kernel** (`matmul_sp_moe`, one launch, expert id per column-block) is CUDA-graph-capturable in isolation — validated bit-exact vs the per-expert kernel (**cos 1.000000**, all routing patterns, one real 256-expert layer) with graph capture/replay at cos 1.0. quadbit sparsifies **~91% of parameters / ~80% of active linear FLOPs** (all expert MLPs; attention/router/embeddings stay dense). Expert-parallel scaling is near-linear: **2.17× (2 GPU) / 4.21× (4 GPU)** of expert-kernel time, imbalance 1.04, checksum-identical output; on no-NVLink PCIe the all-reduce is 0.32–0.45 ms vs 1.3–2.5 ms compute.

**These MoE models now run end-to-end in vLLM on SM120.** A vLLM `general_plugins` plugin supplies every missing SM120 path the models need (block-FP8 dense/attention linears, the MLA `o_proj`, the DeepSeek Sparse-Attention Lightning-Indexer logits, and the cooperative-cluster top-k), so **DeepSeek-V4-Flash-NVFP4 generates coherent text end-to-end on 2× RTX PRO 6000**, and **GLM-5.2-NVFP4 (432.9 GiB) loads and generates on 8× RTX PRO 6000** with its Deep Sparse Attention running **natively on SM120** (vLLM selects `FLASHINFER_MLA_SPARSE_SM120`, no fallback). The earlier "the model's FP8 attention has no SM120 kernel" blocker is resolved by building the paths, not waiting for the ecosystem.

**The accuracy cost of sparsity is a placement problem, and it transfers across architectures — training-free.**

- **Projection anchoring.** The downstream tax lives in the gate/up projection; the down projection is nearly free. On DeepSeek, sparsifying only down projections in the later 49% of MoE layers holds **−0.29 pt** downstream from dense (the gate/up-only control at the same coverage falls −3.27 pt). A per-expert weight repair, by contrast, **fails** (−7 pt): structural placement succeeds where local weight repair does not.
- **Route-slot.** Keeping the top-weight routed slots dense and 2:4-sparsifying the low-weight tail (**D2**) is the best quality / sparse-FLOP tradeoff — the dominant experts carry capability, the tail is nearly free.
- **The same rules transfer to GLM-5.2.** Down-only sparsity costs about half of gate/up there too (+0.209 vs +0.432 held-out PPL at matched 49%-layer coverage), and route-slot D2 gives the best quality and highest sparse-FLOP together (+0.065 PPL).

**GLM D2 downstream smoke suite.** To check D2's small PPL cost is not hiding a downstream collapse, we ran the tokenizer-agnostic MC harness on GLM (ARC-C/HellaSwag/Winogrande/MMLU-5, `limit=200`):

| GLM policy | ARC-C | HellaSwag | Winogrande | MMLU-5 | AVG |
|---|---|---|---|---|---|
| dense (ref) | .655 | .780 | .750 | .856 | **.7603** |
| **route-slot D2** | .650 | .780 | .725 | .848 | **.7508** |

D2 holds within **0.95 pt AVG** of dense with **no task collapsing** — the small PPL gap does not mask a downstream regression.

**Graph-enabled and dense-anchor bottleneck removed (P4 + C1).** The deployed sparse MoE policy path is **graph-enabled on SM120** (P4 replaced the plugin host-sync `torch.unique(...).tolist()` with a fixed-capacity device-routing path), and **C1 removes the dense anchored/grouped projection bottleneck by delegating those projections to FlashInfer's native grouped NVFP4 GEMM (`group_gemm_nvfp4_nt_groupwise`)**, with no custom dense grouped-GEMM required. With the native delegate (opt-in `QB_DENSE_BACKEND=native_nvfp4`), the captured DeepSeek-D2 path now decodes **faster than eager**:

| policy (native-delegate, captured) | model | GPUs | PPL | decode tok/s | graph |
|---|---|---|---|---|---|
| DeepSeek-D2 route-slot | DeepSeek-V4-Flash | 4 | 4.0112 | **5.820** | FULL |
| GLM-5.2 route-slot D2 | GLM-5.2 | 8 | 4.0705 | **5.296** | PIECEWISE 3/3 + FULL 2/2 |

DeepSeek-D2 native-captured is **11.3× the dequant-captured baseline (0.514 tok/s, same harness)** and **1.44× the frozen-eager 4.04**, at matched PPL and bounded memory; GLM-D2 native-captured is **2.5× the eager reference 2.10 tok/s**. Both keep native SM120 DSA sparse-MLA, `drop=0`. These are same-model / same-policy speed/quality/memory Pareto results (native-delegate vs our own dequant-loop and eager paths), **not** a production-wide decode-speed claim over other serving stacks; dense FP4 speed still belongs to the ecosystem baselines. Remaining honest limits: the native delegate depends on FlashInfer availability and its swizzled NVFP4 scale layout; the GLM graph run is validated on a short held-out passage (see the PPL-protocol caveat below); **GLM-5.2 requires 8× RTX PRO 6000** (it does not fit on 2 or 4); the **GLM downstream evidence is a 4-task smoke suite on the D2 policy, not an exhaustive benchmark** (no claim of exhaustive GLM downstream preservation; the fullest downstream accounting of record remains DeepSeek's); and MoE per-expert 2:4-FP4 accuracy recovery beyond structural placement is future work. See [docs/paper.md](docs/paper.md) §10, [docs/c1/verdict.md](docs/c1/verdict.md), [docs/glm_results.md](docs/glm_results.md), `harness/serve_dsv4.py`, `harness/qb_vllm_plugin/`, `docs/figures/`.

*Eager-vs-eager (diagnostic ablation, not the production headline):* three kernel enablers — a **zero-copy transposed epilogue**, a **two-level fused SwiGLU**, and a single no-sync **`fused_mlp_2lvl`** (removing ~64 `cudaDeviceSynchronize`/forward) — took the eager path from batch parity to **+3.7/+5.5/+5.6%** prefill and **+23%** decode vs *eager* NVFP4 (PPL 10.27 vs 7.97). That win is **launch-overhead only**; it does not survive once both paths are CUDA-graphed. See [docs/frozen_serving_result.md](docs/frozen_serving_result.md) for the eager table and `harness/quadbit_serve.py`.

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
