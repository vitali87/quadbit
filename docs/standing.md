# Where we actually stand (impartial)

Written for the agent that runs experiments. Purpose: separate what is **measured** from
what is **asserted**, name every baseline we have **not** run, and give a priority-ordered
experiment list. No marketing. Every claim below is tagged `[measured]`, `[asserted]`,
`[unmeasured]`, or `[external]` with a source.

Prior-art sweep date: 2026-07. Card: Modal cloud RTX PRO 6000 (SM120, no tcgen05).

---

## TL;DR standing

- **Dense FP4 GEMM: competitive but slightly behind CUTLASS on real shapes — NOT the headline.**
  On square sizes we win/tie/win vs 79b (758/1220/1510 @2048/4096/8192). But on the *rectangular*
  Llama-3-8B GEMM shapes that actually run, we **lose** to 79b: attn 4096³ **0.89×**, ffn-up
  (N=14336) **0.93×**, ffn-down (K=14336) **1.01×** (all 79b-verified). The "beats CUTLASS at
  every size" claim was a square-size artifact. Honest dense story: competitive, slightly behind
  CUTLASS. Still 3.0–3.7× over bf16 and above what the deployed SM120 ecosystem (vLLM, mostly no
  native FP4) can run — so it's a fine drop-in, just not a CUTLASS-beating result.
- **Sparse FP4 GEMM: the consistent win, and now the spine of the project.** Gating head-to-head
  vs CUTLASS 80b (`Sm120` sparse NVFP4, ships 3.9.0), correctness-gated on 80b's own reference
  check (**PASSES** every size, so #3096's block-scaled bug does not touch this example). Square:
  **win 4096 (1.16×) / 8192 (1.14×), lose 16384 (0.96×, 1859 vs 1785)**. Rectangular Llama-3-8B
  shapes — where dense loses — sparse **wins consistently**: attn 4096³ **1.18×**, ffn-up **1.14×**,
  ffn-down **1.17×** vs 80b (all 80b-verified). Sparse is the part that beats CUTLASS on the shapes
  that ship.
- **Accuracy headline: dense FP4 is +0.63 PPL, W4A4, zero training, matched to the reference.**
  The FP4 tensor core multiplies fp4×fp4, so the deployed kernel is **W4A4** (weights *and*
  activations 4-bit); there is no weight-only FP4 GEMM in hardware, so the often-quoted +0.3 is a
  **W4A16** number that never ships. With our **per-16 two-level NVFP4 recipe** (amax, **no
  calibration data**): dense W4A4 = **7.90 on Llama-3.1-8B-Instruct (+0.63 over teacher 7.27)**,
  and **6.91 on Meta-Llama-3-8B base (+0.71)** — at/below the modelopt-calibrated reference (vLLM
  native NVFP4 7.97, +0.71). An earlier "+2 PPL" was our *crude* per-32 single-level recipe, NOT
  W4A4's real cost; the gap was block granularity + two-level activation scaling, not calibration.
  Dense FP4 accuracy is a solved, zero-training drop-in. **This is the accuracy story.**
- **Sparse's real ceiling is ~1.33× over our own dense FP4, not 2×** (memory-bound: 2012k vs
  dense 1510k @8192; the hardware 2× is a datacenter-bandwidth feature we can't reach on SM120).
  At only 1.33×, the accuracy cost must be small to justify a recovery pipeline over dense at
  +0.63. On a **real 8B model it currently isn't:** sparse-recovered-8B (Meta-Llama-3-8B, good
  recipe) = **9.01 (+2.81)**, losing to dense-W4A4 (6.91) by **~2.1 PPL**. This is **data-starved**
  — recovery saw ~30M tokens and phase-1 **plateaued** on the 88M-token corpus, ~**400× under**
  NeuralMagic's 13B for element-2:4 — so 9.01 is a lower bound, **not a settled verdict**. Sparse
  is a **speed play with an open, data-limited recovery gap**, not a demonstrated accuracy win.
  (Deploy caveat: 9.01 is fake-quant on the good recipe; the current sparse *kernel* is
  single-level, so through-kernel is 12.55 — realizing 9.01 needs an unbuilt two-level sparse kernel.)
- **Deployment/fusion stack: real and useful, undervalidated end-to-end.** The fused block
  kernels and `nn.Linear` drop-in are genuine engineering. But "2–5× over bf16" is measured on
  isolated blocks/shapes, not a full model forward with correct outputs at scale.

---

## The competitive landscape (what exists on SM120, external)

- **CUTLASS ships dense AND sparse NVFP4 for SM120.** `79b` dense (`nv_float4 → f32`, `Sm120`)
  and `80b` sparse (`OpClassBlockScaledSparseTensorOp`, `Sm120`, same
  `mma...kind::mxf4nvf4.sp::ordered_metadata.block_scale` we use), since CUTLASS 3.9.0
  (2025-04-24). `[external]` These are our true baselines, and we now benchmark against **both**:
  79b (dense, we match/beat) and 80b (sparse, `harness/cutlass_sparse.py` — we win 4–8K, lose 16K).
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
| Dense FP4 vs CUTLASS 79b **square** (win 2048, tie 4096, win 8192) | `[measured]` | `harness/cutlass_fp4.py` |
| Dense FP4 vs CUTLASS 79b **rectangular LLM shapes** (loses: 0.89× attn, 0.93× ffn-up, 1.01× ffn-down) | `[measured]` | `harness/cutlass_shapes.py`; 79b-verified — square win was an artifact |
| Dense FP4 3.0–3.7× over cuBLAS bf16 (prefill shapes) | `[measured]` | `bench_vs_bf16.py`, `bench_llm_shapes.py` |
| Sparse FP4 2012k unit / 1409k deployable @8192 (at bandwidth roofline) | `[measured]` | `matmul_sp_bm256v2*`, mem-only probes |
| **Sparse FP4 vs CUTLASS sparse 80b, square** (win 1.16× @4096, 1.14× @8192; lose 0.96× @16384) | **`[measured]` — GATING RESOLVED** | `harness/cutlass_sparse.py`; 80b ref-verify PASSES every size |
| **Sparse FP4 vs CUTLASS 80b, rectangular LLM shapes** (win 1.18× attn, 1.14× ffn-up, 1.17× ffn-down) | **`[measured]`** | `harness/cutlass_shapes.py`; 80b-verified — the consistent, shipping-shape win |
| Decode small-M beats bf16 | `[measured, marginal]` | 1.27× attn-qkv, 4.53× ffn-up; real-scale decode ties bf16 in the small-N corner (router falls back to bf16) |
| Sparse decode is a win | `[measured, negative]` | `sparse_sk_lib.cu` marginal; reverted |

**Honest read on speed:** dense is competitive but slightly behind CUTLASS on the shapes that
ship (loses on all three rectangular LLM shapes) — it beats bf16, not CUTLASS. Sparse is the
consistent CUTLASS-beating result: it wins on every rectangular LLM shape and at 4–8K square,
losing only at 16K square. Its *marginal* value over our own dense FP4 is ~1.33× at the roofline,
weighed against the accuracy cost + recovery pipeline. Sparse is the CUTLASS-beating **speed**
result; whether it is worth the recovery pipeline over dense (at +0.63) is an **open accuracy
question** — on a real 8B model sparse currently loses by ~2.1 PPL (data-starved, see gap #5).

---

## Accuracy — what is measured vs not

| Claim | Status | Evidence / caveat |
|-------|--------|-------------------|
| **Dense FP4 W4A4 +0.63 PPL, zero training, two-level per-16 recipe** (Llama-3.1-8B-Instruct 7.27→7.90; base 6.20→6.91, +0.71) | `[measured]` | `harness/recovery_worth.py`; matched to modelopt ref (vLLM 7.97, +0.71), **no calibration** — the accuracy headline |
| Dense FP4 crude per-32 single-level = +2 PPL | `[measured, superseded]` | half-finished recipe, NOT W4A4's real cost; fixed by two-level per-16 |
| Dense FP4 **W4A16** (weight-only) +0.3 PPL | `[measured, not deployed]` | no weight-only FP4 GEMM in hardware; do not headline |
| Two-level NVFP4 block rel 0.097; MXFP4 0.13 on Qwen3-8B, no training | `[measured]` | block-level reconstruction, not task accuracy |
| **Sparse-recovered-8B (Meta-Llama-3-8B, good recipe) = 9.01 (+2.81), LOSES to dense-W4A4 6.91 by ~2.1** | `[measured, data-starved]` | ~30M tokens, phase-1 **plateaued** on 88M corpus, ~400× under NM's 13B — lower bound, not verdict |
| Sparse pair-granular recovery: TinyLlama 7.53 → 9.60 through kernel | `[measured, toy]` | small model/data; does NOT generalize to the 8B result above |
| Recovery "monotonic in data → parity is a data-scale question" | `[asserted, untested]` | 8B phase-1 **plateaued** at this budget; needs a diverse-corpus full-scale run to confirm or refute |
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

1. ~~**Sparse vs CUTLASS 80b head-to-head.**~~ **RESOLVED** (`harness/cutlass_sparse.py`): win
   1.16×/1.14× @4096/8192, lose 0.96× @16384; 80b ref-verify PASSES. It is a contribution at the
   sizes that matter, not a reimplementation.
2. ~~**Dense/sparse vs CUTLASS 79b/80b across the full shape set.**~~ **DONE**
   (`harness/cutlass_shapes.py`, committed): rectangular Llama-3 shapes reveal dense **loses** to
   79b (0.89–1.01×) while sparse **wins** vs 80b (1.14–1.18×). The square dense win was an artifact;
   sparse is the consistent win.
3. **A real end-to-end model run** (correct PPL through the full quadbit stack on a real model
   at batch), vs vLLM-Marlin and SGLang-FP4 on the same card. Ties speed + accuracy + usability
   into one defensible number.
4. **Sparse recovery on a real target** (not TinyLlama), with enough data to test the
   "monotonic in data" claim, compared to NVFP4-QAD-style recovery.
5. **The sparse value proposition is OPEN, and on a real 8B model currently negative.** The
   TinyLlama "sparse-recovered 9.60 beats dense-zero-train 9.73" flip was an artifact of the crude
   +2 dense number — with the corrected dense W4A4 (+0.63/+0.71) that flip is dead. On
   Meta-Llama-3-8B, both W4A4 good-recipe: dense-zero-train = **6.91 (+0.71)**, sparse-recovered =
   **9.01 (+2.81)** — sparse **loses by ~2.1 PPL**. It is data-starved (~30M tokens, phase-1
   plateaued on the 88M corpus, ~400× under NM's 13B), so this is a lower bound, not a verdict.
   The payoff is **NOT demonstrated**. Resolving it (gap #4) needs a full-scale diverse-corpus
   recovery run with a **token-vs-PPL curve gating the tail**: target ~+0.5 of dense (~7.4);
   go/no-go at the 5–10× mark — track toward 7.4 → run the tail; flatten above ~7.5 → sparse is
   speed-only and we say so plainly (an honest Pareto point for throughput-bound serving, not a defeat).
6. **Stale docs.** `docs/kernels.md` describes the abandoned CubeCL/Rust track (505k ceiling,
   "CUTLASS 3× faster", `src/bin/`, `run_rust.py`) and contradicts the raw-PTX results in
   `paper_notes.md`. Either delete it or mark it clearly as historical. It will confuse anyone.
7. **Reproducibility of the headline table.** README/paper_notes show one dense number as
   "1136 (3.06×)" and another as "1510 (tie with CUTLASS)"; the difference is torch dispatch
   overhead. Pick one measurement protocol and report it consistently.

---

## What we should be doing (priority order for the experiment agent)

1. ~~**Run CUTLASS 80b sparse and benchmark our sparse kernel against it.**~~ **DONE**
   (`harness/cutlass_sparse.py`): win 4–8K, lose 16K, 80b ref-verify PASSES. Gating experiment
   resolved. Next natural step is #2 (rectangular shapes) to see if the win holds off-square.
2. ~~**Run `harness/cutlass_shapes.py`** beyond square.~~ **DONE** (committed): dense loses to 79b
   on rectangular shapes, sparse wins vs 80b. → Gap #2 resolved; reframed the project spine to sparse.
3. **End-to-end: one real model, full forward, correct PPL, at batch**, quadbit vs vLLM-Marlin
   vs SGLang-FP4 on the same RTX PRO 6000. One table: tok/s, PPL, memory. → Gap #3, and it's the
   number that actually earns recognition. (`harness/vllm_nvfp4.py` is a first SM120-NVFP4 smoke
   test toward the vLLM baseline; the full end-to-end table is still unbuilt.)
4. ~~**Quantify sparse-net-of-recovery on TinyLlama.**~~ **SUPERSEDED** — the TinyLlama flip was a
   crude-dense artifact; on real 8B (`recovery_worth.py` + `finetune_pair.py`) dense-W4A4 6.91
   beats sparse-recovered 9.01 by ~2.1. The question moved to gap #5 (real 8B, data-starved).
5. **Resolve sparse via full-scale diverse-corpus recovery, curve-gated.** In order: (a) confirm
   9.01 is the recipe's real plateau, not a half-finished QAT/LR/KD recipe — cheap, tells us if
   the method has headroom before pouring data; (b) build the diverse-corpus pipeline ONCE (C4 /
   Dolma scale, not 10× the same 88M — some plateau may be diversity starvation); (c) run
   full-scale with the **token-vs-PPL curve as a hard go/no-go**: target ~7.4 (+0.5 of dense);
   at 5–10× tokens, tracking toward 7.4 → run the tail, flattening above ~7.5 → stop, sparse is
   speed-only, write it plainly. Dense at +0.63 stays the accuracy headline regardless of the slope.

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
