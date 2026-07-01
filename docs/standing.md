# Where we actually stand (impartial)

Written for the agent that runs experiments. Purpose: separate what is **measured** from
what is **asserted**, name every baseline we have **not** run, and give a priority-ordered
experiment list. No marketing. Every claim below is tagged `[measured]`, `[asserted]`,
`[unmeasured]`, or `[external]` with a source.

Prior-art sweep date: 2026-07. Card: Modal cloud RTX PRO 6000 (SM120, no tcgen05).

---

## TL;DR standing

- **Dense FP4 GEMM: genuinely strong.** We match/beat CUTLASS *dense* FP4 at 4096/8192 in a
  clean measurement, and the deployed SM120 ecosystem (vLLM) mostly can't even get native FP4
  working, so our 3.0–3.7× over bf16 is real and above what most users can run today. This is
  the most defensible part of the project.
- **Sparse FP4 GEMM: fast, but the one comparison that matters is not run.** We are at the
  SM120 bandwidth roofline (2012k unit / ~1409k deployable). But we have **never** benchmarked
  against CUTLASS example 80b (`Sm120` sparse NVFP4, ships since 3.9.0). Every "sparse win"
  number in the repo is vs CUTLASS *dense*, not vs CUTLASS *sparse*. Until 80b is measured on
  this card, the sparse contribution is unproven as a *speed* claim.
- **Sparse's real ceiling is ~1.33× over our own dense FP4, not 2×.** Sparse is memory-bound
  (2012k vs dense 1510k @8192). The hardware 2× is a datacenter-bandwidth feature we can't
  reach on SM120. So sparse must justify a modest throughput gain against a real accuracy cost
  and an expensive recovery pipeline. That tradeoff is currently unfavorable and undertested.
- **Accuracy: dense FP4 is production-grade; sparse recovery is not yet.** Dense zero-training
  +0.3 PPL is credible and matches the literature. Sparse recovery (TinyLlama 7.53→9.60 through
  the kernel) is a small-model, small-data proof of concept that trails NVIDIA's own NVFP4-QAD
  (>95% FP recovery) and has not been shown on a real target model.
- **Deployment/fusion stack: real and useful, undervalidated end-to-end.** The fused block
  kernels and `nn.Linear` drop-in are genuine engineering. But "2–5× over bf16" is measured on
  isolated blocks/shapes, not a full model forward with correct outputs at scale.

---

## The competitive landscape (what exists on SM120, external)

- **CUTLASS ships dense AND sparse NVFP4 for SM120.** `79b` dense (`nv_float4 → f32`, `Sm120`)
  and `80b` sparse (`OpClassBlockScaledSparseTensorOp`, `Sm120`, same
  `mma...kind::mxf4nvf4.sp::ordered_metadata.block_scale` we use), since CUTLASS 3.9.0
  (2025-04-24). `[external]` These are our true baselines. We benchmark against 79b (dense);
  we do **not** benchmark 80b (sparse).
- **The SM120 block-scaled path is buggy/slow in practice.** CUTLASS issue #3096: grouped GEMM
  produces garbage on SM120, TMA warp-specialized tactics fail to init, autotuner falls back to
  slow tactics; needs `compute_120f` (CUDA 13) not `compute_120a`. `[external]` This is the
  genuine opening: a *working, fast, deployment-integrated* SM120 sparse/dense FP4 path is
  scarce even though a reference kernel exists.
- **vLLM on SM120 largely falls back to Marlin W4A16** (dequant FP4→bf16 in-kernel, forfeits FP4
  FLOPS, ~50 tok/s on a 397B MoE). SGLang gets native FP4: ~1.47× over bf16 at batch 1, ~2.23×
  at batch 128. `[external]` So the *deployed reality* on our card is 1.5–2.2× where it works,
  and often a bf16-dequant fallback. Our 3.0–3.7× dense is above this.
- **Datacenter context (B200, sm_100, NOT our card):** SGLang/FlashInfer/vLLM FP4 MoE reach
  ~1000–1260 TFLOPS, ~3.5× over bf16, using tcgen05/UMMA we don't have. `[external]` Do not
  compare our SM120 numbers to these; different silicon.

---

## Speed — what is measured vs not

| Claim | Status | Evidence |
|-------|--------|----------|
| Dense FP4 matches/beats CUTLASS dense 79b (1220 tie @4096, 1510 vs 1497 @8192) | `[measured]` | `harness/cutlass_fp4.py`, paper_notes headline |
| Dense FP4 3.0–3.7× over cuBLAS bf16 (prefill shapes) | `[measured]` | `bench_vs_bf16.py`, `bench_llm_shapes.py` |
| Sparse FP4 2012k unit / 1409k deployable @8192 (at bandwidth roofline) | `[measured]` | `matmul_sp_bm256v2*`, mem-only probes |
| Sparse FP4 "+24%/+47% vs best vendor FP4" | `[measured but misframed]` | vs CUTLASS **dense**, not sparse |
| **Sparse FP4 vs CUTLASS sparse 80b** | **`[unmeasured]` — GATING** | never run; `cutlass_sparse.py` exists but no head-to-head recorded |
| Decode small-M beats bf16 | `[measured, marginal]` | 1.27× attn-qkv, 4.53× ffn-up; real-scale decode ties bf16 in the small-N corner (router falls back to bf16) |
| Sparse decode is a win | `[measured, negative]` | `sparse_sk_lib.cu` marginal; reverted |

**Honest read on speed:** dense is proven and good. Sparse throughput is proven *in isolation*
but its *competitive* value is unknown because 80b was never run, and its *marginal* value over
our own dense FP4 is only ~1.33× at the roofline.

---

## Accuracy — what is measured vs not

| Claim | Status | Evidence / caveat |
|-------|--------|-------------------|
| Dense FP4 +0.3 PPL, zero training (Sparse-Llama 7.89→8.16; Qwen2.5-3B 7.60→7.91) | `[measured]` | credible, matches FP4 weight+act literature |
| Two-level NVFP4 block rel 0.097; MXFP4 0.13 on Qwen3-8B, no training | `[measured]` | block-level reconstruction, not task accuracy |
| Sparse pair-granular recovery: TinyLlama 7.53 (teacher) → 9.60 through kernel | `[measured]` | **small model, ~1.5M–WikiText-103 data, +2.1 PPL gap** |
| Recovery is "monotonic in data → production parity is a data-scale question" | `[asserted]` | plausible but **not demonstrated** on a real target model at scale |
| element-2:4 checkpoints incompatible w/ pair-granular hw (93.6 PPL naive) | `[measured data point]` | the *hardware constraint* is documented (NVIDIA 4:8 in-pairs); only the measured number is ours |

**External yardstick:** NVIDIA NVFP4-QAD recovers >95% of FP accuracy on real Nemotron/Llama
models `[external, arXiv 2601.20088]`. Our sparse recovery is not in that league yet and hasn't
been run on a real deployment target. Frame sparse recovery as early-stage, not solved.

---

## Deployment / usability — what is measured vs not

- Fused SwiGLU FFN, RMSNorm+quant, add+RMSNorm+quant, concat gate/up: `[measured]` 2–5.8× over
  eager bf16 on isolated blocks; numerically at the FP4/prune floor. Real, valuable.
- `QuadbitLinear` drop-in + packer (maxrel 0.0039 vs kernel): `[measured]`.
- Full fused decoder block on real Qwen3-8B: `[measured]` 2.16–2.19× dense, block rel 0.13.
- **Not shown:** a full model forward (many layers) producing correct logits/PPL end-to-end at
  serving batch sizes, vs a real serving baseline (vLLM Marlin fallback or SGLang FP4). All
  end-to-end numbers are single-block or single-shape. `[unmeasured]`

---

## What we are lacking (the gaps a reviewer or user will hit)

1. **Sparse vs CUTLASS 80b head-to-head.** The single most important missing number. Decides
   whether the sparse kernel is a contribution or a reimplementation.
2. **Dense vs CUTLASS 80/79 across the full shape set**, not just square. We have square; LLM
   shapes are only vs bf16.
3. **A real end-to-end model run** (correct PPL through the full quadbit stack on a real model
   at batch), vs vLLM-Marlin and SGLang-FP4 on the same card. Ties speed + accuracy + usability
   into one defensible number.
4. **Sparse recovery on a real target** (not TinyLlama), with enough data to test the
   "monotonic in data" claim, compared to NVFP4-QAD-style recovery.
5. **The sparse value proposition itself.** Quantify: sparse gives ~1.33× over our dense FP4 but
   costs a recovery pipeline and accuracy. Is there any regime where sparse FP4 beats dense FP4
   *net of recovery*? If not, sparse is a research artifact, not a deployment path — say so.
6. **Stale docs.** `docs/kernels.md` describes the abandoned CubeCL/Rust track (505k ceiling,
   "CUTLASS 3× faster", `src/bin/`, `run_rust.py`) and contradicts the raw-PTX results in
   `paper_notes.md`. Either delete it or mark it clearly as historical. It will confuse anyone.
7. **Reproducibility of the headline table.** README/paper_notes show one dense number as
   "1136 (3.06×)" and another as "1510 (tie with CUTLASS)"; the difference is torch dispatch
   overhead. Pick one measurement protocol and report it consistently.

---

## What we should be doing (priority order for the experiment agent)

1. **Run CUTLASS 80b sparse on this card and benchmark our sparse kernel against it** at
   2048/4096/8192 and the real LLM shapes, same timing protocol (cudaEvent, no torch dispatch).
   Record TFLOPS and whether 80b even runs correctly on SM120 (#3096 suggests it may not).
   → Resolves gap #1, the gating experiment.
2. **Run 80b/79b dense across all LLM shapes** to confirm dense parity beyond square. → Gap #2.
3. **End-to-end: one real model, full forward, correct PPL, at batch**, quadbit vs vLLM-Marlin
   vs SGLang-FP4 on the same RTX PRO 6000. One table: tok/s, PPL, memory. → Gap #3, and it's the
   number that actually earns recognition.
4. **Quantify the sparse-net-of-recovery question** (gap #5) before investing more in sparse
   recovery. If dense FP4 at +0.3 PPL zero-training beats sparse-FP4-after-recovery on the
   accuracy/throughput frontier, pivot the paper's spine to the dense + fusion + roofline story
   and treat sparse as "here is the only place it helps, here is why it usually doesn't."
5. Only if 1 and 5 are favorable: scale sparse recovery on a real target vs NVFP4-QAD (gap #4).

Whatever the experiments show, update `paper_notes.md` and delete/relabel `kernels.md`. The
positioning must follow the measurements, not the other way around.

## Sources

- CUTLASS 80b sparse SM120: https://github.com/NVIDIA/cutlass/blob/main/examples/80_blackwell_geforce_sparse_gemm/80b_blackwell_geforce_nvfp4_nvfp4_sparse_gemm.cu
- CUTLASS SM120 FP4 bugs (#3096): https://github.com/NVIDIA/cutlass/issues/3096
- vLLM SM120 native NVFP4 request (#31085): https://github.com/vllm-project/vllm/issues/31085
- SM120 MoE perf report (Marlin fallback, tok/s): https://discuss.vllm.ai/t/sm120-rtx-pro-6000-nvfp4-moe-performance-report-qwen3-5-397b/2536
- FP4 kernel engineering / throughput (B200 context): https://huggingface.co/blog/apsys/blackwell-nvfp4-comparison
- NVFP4-QAD accuracy recovery: https://arxiv.org/abs/2601.20088
- Pair-wise 4:8 NVFP4 sparsity (hardware spec): https://newsletter.semianalysis.com/p/nvidia-tensor-core-evolution-from-volta-to-blackwell
- FP4 accuracy reality: https://arxiv.org/pdf/2509.23202
