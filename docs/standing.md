# Where we actually stand (impartial)

Written for the agent that runs experiments. Purpose: separate what is **measured** from
what is **asserted**, name every baseline we have **not** run, and give a priority-ordered
experiment list. No marketing. Every claim below is tagged `[measured]`, `[asserted]`,
`[unmeasured]`, or `[external]` with a source.

Prior-art sweep date: 2026-07. Card: Modal cloud RTX PRO 6000 (SM120, no tcgen05).

---

## TL;DR standing

- **Dense FP4 GEMM: we LOSE the SM120 dense race to FlashInfer — NOT the headline.** The baseline
  moved: FlashInfer `mm_fp4` now ships a CUDA-13 `b12x` NVFP4 kernel + a `cutlass` path. On a
  correctness-gated leaderboard (same card/shapes/fp32-ref gate, `harness/leaderboard_fp4.py`),
  FlashInfer `b12x`/`cutlass` beat quadbit's deployed two-level dense by **1.35–2.2×**: square-8192
  1433 vs 1045, prefill attn 1283 vs 838, ffn-up 1374 vs 936, ffn-down 1408 vs 1017, serving
  M=65536 ffn-down 1416 vs 639. The old "competitive with CUTLASS 79b" story is stale. Dense stays
  a zero-training W4A4 **accuracy** drop-in (+0.63 PPL), not a speed leader. Leaderboard also found:
  FlashInfer `cudnn` fails on *every* SM120 shape (cuDNN 9.10 < 9.14 needed), `b12x` collapses ~2.2×
  at M≥65536 (only `cutlass` holds), `trtllm`/`cute-dsl` refuse SM120, and `b12x` needs CUDA 13 while
  quadbit's block-scale mma only assembles under CUDA ≤12.8 (ptxas 13 rejects it).
- **Sparse FP4 GEMM: THE headline — the only deployed sparse FP4, and it beats the best dense.** No
  shipping library provides a sparse FP4 GEMM on SM120 (FlashInfer/SGLang/vLLM = dense only; CUTLASS
  80b is an unwrapped example). quadbit is the only *deployed* one. Fresh two-level head-to-head vs
  CUTLASS 80b (correctness-gated on 80b's own reference, **PASSES** every size): quadbit wins every
  shape — attn 4096³ **1.08×**, ffn-up **1.01×**, ffn-down **1.12×**, square-8192 **1.09×** (1973 vs
  1807). **The Pareto point (cross-table):** quadbit two-level sparse beats even the *best FlashInfer
  dense* in wall-clock on every Llama-3-8B prefill shape — attn 0.100 vs 0.107 ms (**1.07×**), ffn-up
  0.301 vs 0.350 (**1.16×**), ffn-down 0.265 vs 0.342 (**1.29×**), square-8192 0.557 vs 0.767
  (**1.38×**). If a weight can be 2:4-pruned, quadbit sparse is the fastest FP4 GEMM on the platform.
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
  +0.63. On a **real 8B model it isn't:** sparse-recovered-8B (Meta-Llama-3-8B) deploys at **8.47
  through the two-level kernel**, about **1.56 PPL above dense-W4A4 (6.91)**. Recipe moved it (9.01
  under-trained → 8.30 per-16 fake-quant via warm-restart phase-2, zero new data), but the honest
  deployable per-32 matched number is 8.47 and **the data lever did not pay: the full-scale C4
  diverse-corpus run flattened at 10.82 on the WT-2 test (C4 is OOD for that narrow metric), never
  approaching the ~7.4 target.** Sparse is a **speed-only Pareto point**, not a demonstrated accuracy
  win. **Deploy gap CLOSED (2026-07-05): the two-level sparse kernel is built** (per-row and per-col
  fp32 global rescale in the epilogue), so through-kernel **== fake-quant (8.47 == 8.47)**; the old
  single-level kernel would deploy the same checkpoint at **11.89** (same-checkpoint A/B: top-1 agree
  2lvl 91.7% / 1lvl 78.3%). The fix costs **2–10%** throughput and still beats CUTLASS 80b on every
  shape (1.01–1.12×). Extending phase-2 2k→5k did not help (8.96, mild overtraining).
- **Deployment/fusion stack: real and useful, undervalidated end-to-end.** The fused block
  kernels and `nn.Linear` drop-in are genuine engineering. But "2–5× over bf16" is measured on
  isolated blocks/shapes, not a full model forward with correct outputs at scale.

---

## The competitive landscape (what exists on SM120, external)

- **FlashInfer `mm_fp4` is now the strongest dense FP4 baseline on SM120, and it beats us.**
  `auto` → `b12x` (CUDA-13-only SM120/121 NVFP4 kernel) → `cutlass` → `cudnn`; `trtllm`/`cute-dsl`
  refuse SM120. On our leaderboard (`harness/leaderboard_fp4.py`, `[measured]`) `b12x`/`cutlass`
  beat quadbit dense 1.35–2.2×. But it's uneven: `cudnn` fails every SM120 shape (cuDNN 9.10 <
  9.14), `b12x` collapses ~2.2× at M≥65536, prebuilt cubins historically had no sm_120 targets
  (#3294) so the fast path needs CUDA-13 JIT. `[external + measured]`
- **CUTLASS ships dense AND sparse NVFP4 for SM120.** `79b` dense (`nv_float4 → f32`, `Sm120`)
  and `80b` sparse (`OpClassBlockScaledSparseTensorOp`, `Sm120`, same
  `mma...kind::mxf4nvf4.sp::ordered_metadata.block_scale` we use), since CUTLASS 3.9.0
  (2025-04-24). `[external]` 80b is the **only other sparse FP4 kernel that exists** and is our
  sparse baseline (`harness/cutlass_sparse.py` — quadbit two-level wins every shape 1.01–1.12×). No
  library wraps *any* sparse FP4 kernel in a deployment stack; quadbit is the only deployed one.
- **The SM120 block-scaled path is buggy/slow in practice.** CUTLASS issue #3096: grouped GEMM
  produces garbage on SM120, TMA warp-specialized tactics fail to init, autotuner falls back to
  slow tactics; needs `compute_120f` (CUDA 13) not `compute_120a`. `[external]` This is the
  genuine opening: a *working, fast, deployment-integrated* SM120 sparse/dense FP4 path is
  scarce even though a reference kernel exists.
- **vLLM AND SGLang both run NATIVE NVFP4 W4A4 on SM120 for the dense Llama-3.1-8B checkpoint**
  (`[measured]`, `harness/vllm_nvfp4.py` + `harness/sglang_fp4.py`, RTX PRO 6000, distinct-prompt,
  CUDA-graphs-on): vLLM binds `modelopt_fp4` cutlass, SGLang `auto` selects FlashInfer CUTLASS
  `fp4_gemm` (autotuned) — **not** Marlin. Native NVFP4 is ~1.7× bf16 prefill and ~1.7× decode at
  B=64 (vLLM 116831 vs 46880 prefill, 8465 vs 4947 decode). SGLang wins decode every batch
  (10145 vs 8465 @B64). The Marlin W4A16 ~50 tok/s fallback figure was a 397B MoE, not this dense
  path. `[external]` See the serving table in `docs/paper.md` §9 and below.
- **Datacenter context (B200, sm_100, NOT our card):** SGLang/FlashInfer/vLLM FP4 MoE reach
  ~1000–1260 TFLOPS, ~3.5× over bf16, using tcgen05/UMMA we don't have. `[external]` Do not
  compare our SM120 numbers to these; different silicon.

---

## Speed — what is measured vs not

| Claim | Status | Evidence |
|-------|--------|----------|
| **Dense FP4 vs FlashInfer `mm_fp4` (leaderboard): we LOSE 1.35–2.2×** (sq8192 1433 vs 1045; prefill attn 1283 vs 838; serving M=65536 1416 vs 639) | **`[measured]`** | `harness/leaderboard_fp4.py`; all backends, fp32-ref gate cos 0.991; `b12x`/`cutlass` win, `cudnn` fails all shapes, `b12x` collapses at M≥65536 |
| Dense FP4 vs CUTLASS 79b **square** (win 2048, tie 4096, win 8192) | `[measured]` | `harness/cutlass_fp4.py` — superseded as a claim: FlashInfer is now the dense baseline that matters |
| Dense FP4 vs CUTLASS 79b **rectangular LLM shapes** (loses: 0.89× attn, 0.93× ffn-up, 1.01× ffn-down) | `[measured]` | `harness/cutlass_shapes.py`; 79b-verified — square win was an artifact |
| Dense FP4 3.0–3.7× over cuBLAS bf16 (prefill shapes) | `[measured]` | `bench_vs_bf16.py`, `bench_llm_shapes.py` |
| **Deployed two-level NVFP4 kernel: async scale prefetch = 1.08–1.22× (maxrel 0)** | `[measured]` | scales double-buffered + `cp.async` 1-ahead hides the ~500cyc scale load; sq8192 865→1055; folded into `dense_nvfp4_fast_lib.cu` |
| Sparse FP4 2012k unit / 1409k deployable @8192 (at bandwidth roofline) | `[measured]` | `matmul_sp_bm256v2*`, mem-only probes |
| **Sparse FP4 vs CUTLASS sparse 80b, square** (win 1.16× @4096, 1.14× @8192; lose 0.96× @16384) | **`[measured]` — GATING RESOLVED** | `harness/cutlass_sparse.py`; 80b ref-verify PASSES every size |
| **Sparse FP4 vs CUTLASS 80b, rectangular LLM shapes** (win 1.18× attn, 1.14× ffn-up, 1.17× ffn-down) | **`[measured]`** | `harness/cutlass_shapes.py`; 80b-verified — the consistent, shipping-shape win |
| **Sparse two-level vs CUTLASS 80b, fresh (win every shape 1.01–1.12×)** | **`[measured]`** | `harness/cutlass_sparse.py`; attn 1.08×, ffn-up 1.01×, ffn-down 1.12×, sq8192 1.09× (1973 vs 1807) |
| **PARETO: sparse two-level beats best FlashInfer DENSE in wall-clock, every prefill shape (1.07–1.38×)** | **`[measured]`** | cross-table: sparse (`cutlass_sparse.py`, CUDA 12.8) vs FI dense best (`leaderboard_fp4.py`, CUDA 13), same card; attn 0.100 vs 0.107 ms, ffn-dn 0.265 vs 0.342, sq8192 0.557 vs 0.767 |
| Decode small-M beats bf16 | `[measured, marginal]` | 1.27× attn-qkv, 4.53× ffn-up; real-scale decode ties bf16 in the small-N corner (router falls back to bf16) |
| Sparse decode is a win | `[measured, negative]` | `sparse_sk_lib.cu` marginal; reverted |
| **End-to-end serving baselines measured** (vLLM bf16 / vLLM NVFP4 / SGLang NVFP4, distinct-prompt, CUDA-graphs) | `[measured]` | `harness/vllm_nvfp4.py`, `harness/sglang_fp4.py`; serving table below + `docs/paper.md` §9 |
| quadbit *inside* a serving engine (paged attn, continuous batching, decode) | `[not built]` | prefill-only prototype (`harness/dense_e2e.py`) trails serving prefill ~10× — eager bf16 attention, not the GEMM |

### Serving table (RTX PRO 6000, Llama-3.1-8B-Instruct family, CUDA graphs on, distinct prompts, S=2048 prefill / GEN=128 decode)

**Table A — real serving engines:**

| engine | quant | weights | WT-2 PPL | prefill B=1/8/32/64 | decode B=1/8/32/64 |
|--------|-------|---------|----------|---------------------|--------------------|
| vLLM 0.21 | bf16 | 15.0 GiB | 7.267 | 10209/26436/31559/46880 | 88/690/2599/4947 |
| vLLM 0.21 | NVFP4 cutlass | 5.66 GiB | 7.974 | 13530/63056/78028/116831 | 131/1049/4259/8465 |
| SGLang 0.5 | NVFP4 FlashInfer-CUTLASS `fp4_gemm` | ~5.6 GiB | 7.97 (ckpt) | 16829/59769/73414/109002 | 186/1491/5424/10145 |

**Table B — quadbit dense FP4 prototype (full-forward, PREFILL-ONLY, no decode engine):** quantized-linear
weights 3.93 GiB (block linears only; embed/lm_head/attn stay bf16, not counted), through-kernel WT-2 PPL
**7.90** (matches vLLM native NVFP4 7.97 within 0.07), full-model prefill **7987 tok/s** (B=8, S=2048, eager
+ HF bf16 attention). NOT a serving-stack number; do not share a tok/s column with Table A.

**Honest read on speed:** dense is competitive but slightly behind CUTLASS on the shapes that
ship (loses on all three rectangular LLM shapes) — it beats bf16, not CUTLASS. Sparse is the
consistent CUTLASS-beating result: it wins on every rectangular LLM shape and at 4–8K square,
losing only at 16K square. Its *marginal* value over our own dense FP4 is ~1.33× at the roofline,
weighed against the accuracy cost + recovery pipeline. Sparse is the CUTLASS-beating **speed**
result; whether it is worth the recovery pipeline over dense (at +0.63) is an **open accuracy
question** — on a real 8B model sparse loses by ~1.56 PPL (deployable 8.47 through-kernel vs dense 6.91; deploy gap closed, data lever negative, see gap #5).

---

## Accuracy — what is measured vs not

| Claim | Status | Evidence / caveat |
|-------|--------|-------------------|
| **Dense FP4 W4A4 +0.63 PPL, zero training, two-level per-16 recipe** (Llama-3.1-8B-Instruct 7.27→7.90; base 6.20→6.91, +0.71) | `[measured]` | `harness/recovery_worth.py`; matched to modelopt ref (vLLM 7.97, +0.71), **no calibration** — the accuracy headline |
| Dense FP4 crude per-32 single-level = +2 PPL | `[measured, superseded]` | half-finished recipe, NOT W4A4's real cost; fixed by two-level per-16 |
| Dense FP4 **W4A16** (weight-only) +0.3 PPL | `[measured, not deployed]` | no weight-only FP4 GEMM in hardware; do not headline |
| Two-level NVFP4 block rel 0.097; MXFP4 0.13 on Qwen3-8B, no training | `[measured]` | block-level reconstruction, not task accuracy |
| **Sparse-recovered-8B (Meta-Llama-3-8B) deployable = 8.47 through two-level kernel == fake-quant; LOSES to dense-W4A4 6.91 by ~1.56** | `[measured]` | per-32 matched STE, 2k warm-restart phase-2; honest deployable number (per-16 8.30 fake-quant ceiling is not deployable through the per-32 sparse mma) |
| Sparse deploy gap CLOSED: two-level kernel through-kernel **8.47 == 8.47** fake-quant; single-level kernel would deploy **11.89** | `[measured]` | `harness/ab_sparse_semantics.py` same-checkpoint A/B (ΔNLL 2lvl −0.002/1lvl +0.282; top-1 2lvl 91.7%/1lvl 78.3%); rescale costs 2–10%, still beats CUTLASS 80b 1.01–1.12× |
| Sparse data lever = NEGATIVE: C4 diverse-corpus phase-1 flattened at **10.82** on WT-2 | `[measured]` | OOD for WT-2 (in-distribution WikiText-103 phase-1 = 8.57); ap-SdSv9zQ9 timed out 192k/300k, never neared 7.4 target — recovery is in-distribution-bound, not data-starved |
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
- **Full model forward + serving baselines: `[measured]`** (serving table above, `docs/paper.md`
  §9). quadbit full-forward through-kernel PPL 7.90 matches native NVFP4 (7.97); serving engines
  (vLLM bf16/NVFP4, SGLang NVFP4) measured on the same card/protocol. **Still not shown:** quadbit
  *inside* a serving stack (paged attn, continuous batching, decode scheduler) — the prefill-only
  prototype trails serving prefill ~10× due to eager bf16 attention, not the GEMM. `[not built]`

---

## What we are lacking (the gaps a reviewer or user will hit)

0. **THESIS REFRAMED (2026-07-06): dense lost, sparse is the Pareto headline.** The SM120 FP4
   backend leaderboard (`harness/leaderboard_fp4.py`) added FlashInfer `mm_fp4` as a competitor:
   `b12x`/`cutlass` beat quadbit dense 1.35–2.2×, so the "competitive with CUTLASS 79b" dense claim
   is dead. What stands: quadbit is the **only deployed sparse FP4 GEMM** on SM120, and it beats the
   best available *dense* FP4 (FlashInfer) in wall-clock on every prefill shape (1.07–1.38×). That
   Pareto corner is the reviewer-obvious systems result. Docs pivoted across README/paper/notes.
1. ~~**Sparse vs CUTLASS 80b head-to-head.**~~ **RESOLVED** (`harness/cutlass_sparse.py`): win
   1.16×/1.14× @4096/8192, lose 0.96× @16384; 80b ref-verify PASSES. It is a contribution at the
   sizes that matter, not a reimplementation.
2. ~~**Dense/sparse vs CUTLASS 79b/80b across the full shape set.**~~ **DONE**
   (`harness/cutlass_shapes.py`, committed): rectangular Llama-3 shapes reveal dense **loses** to
   79b (0.89–1.01×) while sparse **wins** vs 80b (1.14–1.18×). The square dense win was an artifact;
   sparse is the consistent win.
3. ~~**A real end-to-end model run**~~ **DONE for accuracy + baselines** (serving table above):
   quadbit full-forward through-kernel PPL 7.90 == native NVFP4 7.97; vLLM/SGLang NVFP4 measured
   on the same card. **Remaining:** quadbit inside a real serving engine for a true tok/s
   head-to-head (paged attn + continuous batching + decode). The kernel is proven; the serving
   integration is the open engineering step.
4. **Sparse recovery on a real target** (not TinyLlama), with enough data to test the
   "monotonic in data" claim, compared to NVFP4-QAD-style recovery.
5. **The sparse value proposition is RESOLVED: speed-only, loses to dense on accuracy.** The
   TinyLlama "sparse-recovered 9.60 beats dense-zero-train 9.73" flip was an artifact of the crude +2
   dense number — with the corrected dense W4A4 (+0.63/+0.71) that flip is dead. On Meta-Llama-3-8B:
   dense-zero-train W4A4 = **6.91 (+0.71)**; deployable sparse-recovered = **8.47 through the two-level
   kernel** (per-32 matched STE, 2k warm-restart phase-2), so sparse **loses by ~1.56 PPL**. Two
   sub-results closed this: **(a) deploy gap CLOSED** — the two-level sparse kernel (per-row/col fp32
   global rescale in the epilogue) makes through-kernel == fake-quant (8.47 == 8.47); the old
   single-level kernel would deploy the same checkpoint at **11.89** (same-checkpoint A/B: top-1 agree
   2lvl 91.7% / 1lvl 78.3%; rescale costs 2–10%, still beats CUTLASS 80b 1.01–1.12×). **(b) data lever
   NEGATIVE** — the full-scale C4 diverse-corpus run (app ap-SdSv9zQ9) flattened at **10.82 on WT-2**
   (OOD; in-distribution WikiText-103 phase-1 = 8.57), timed out at 192k/300k, never neared the 7.4
   target. Recovery is in-distribution-bound on WT-2, not simply data-starved; extending phase-2 2k→5k
   also did not help (8.96, overtraining). Dense at +0.63 stays the accuracy headline; sparse is an
   honest speed-only Pareto point.
6. **Stale docs.** `docs/kernels.md` describes the abandoned CubeCL/Rust track (505k ceiling,
   "CUTLASS 3× faster", `src/bin/`, `run_rust.py`) and contradicts the raw-PTX results in
   `paper_notes.md`. Either delete it or mark it clearly as historical. It will confuse anyone.
7. ~~**Reproducibility of the headline table (1136 vs 1510).**~~ **RESOLVED (2026-07-06).**
   `paper.md` §5 now labels the 1136/1556/1645 (dense) and 1512/2207/1782 (sparse) figures
   explicitly as the **speed-path ceiling** (MXFP4-fast dense, unit-scale sparse), distinct from
   the **deployed two-level** numbers on the leaderboard (dense sq8192 1045, sparse sq8192 1973).
   Both protocols are reported side by side with which kernel each measures; the leaderboard uses
   one consistent effective-FLOP 2·M·N·K / cudaEvent protocol across all backends.

---

## What we should be doing (priority order for the experiment agent)

0. ~~**Build the true SM120 FP4 backend leaderboard (add FlashInfer `mm_fp4`).**~~ **DONE
   (2026-07-06, `harness/leaderboard_fp4.py`).** Every FlashInfer backend + quadbit, matched
   shapes/card/fp32-gate. Verdict: dense lost (FI 1.35–2.2×), sparse is the Pareto headline
   (beats best FI dense 1.07–1.38× wall-clock, only deployed sparse FP4). `cudnn` broken all
   shapes, `b12x` collapses at M≥65536, CUDA-13-vs-12.8 mma split documented. **Next natural
   step:** quadbit inside a real serving engine (paged attn + continuous batching + decode) — the
   only way to turn the GEMM-level Pareto win into a serving-throughput row.
1. ~~**Run CUTLASS 80b sparse and benchmark our sparse kernel against it.**~~ **DONE**
   (`harness/cutlass_sparse.py`): win 4–8K, lose 16K, 80b ref-verify PASSES. Gating experiment
   resolved. Next natural step is #2 (rectangular shapes) to see if the win holds off-square.
2. ~~**Run `harness/cutlass_shapes.py`** beyond square.~~ **DONE** (committed): dense loses to 79b
   on rectangular shapes, sparse wins vs 80b. → Gap #2 resolved; reframed the project spine to sparse.
3. ~~**End-to-end: one real model, full forward, correct PPL, at batch**~~ **DONE — serving
   table built** (`harness/vllm_nvfp4.py`, `harness/sglang_fp4.py`, `harness/dense_e2e.py`; table
   above + `paper.md` §9). Both vLLM and SGLang run NATIVE NVFP4 (not Marlin) on this card; quadbit
   through-kernel PPL 7.90 matches native NVFP4 7.97. → Gap #3 resolved on accuracy + baselines.
   **Only remaining piece:** quadbit inside a real serving engine for a true tok/s head-to-head.
4. ~~**Quantify sparse-net-of-recovery on TinyLlama.**~~ **SUPERSEDED** — the TinyLlama flip was a
   crude-dense artifact; on real 8B (`recovery_worth.py` + `finetune_pair.py`) dense-W4A4 6.91
   beats deployable sparse-recovered (8.47 through-kernel) by ~1.56. Resolved as gap #5.
5. ~~**Resolve sparse via full-scale diverse-corpus recovery, curve-gated.**~~ **DONE — sparse is
   speed-only.** (a) recipe headroom confirmed (9.01→8.30 warm-restart, zero new data); (b) diverse
   corpus staged (`harness/build_corpus.py`); (c) full-scale C4 run (ap-SdSv9zQ9) flattened at **10.82
   on WT-2** (OOD), never neared 7.4 → **data lever negative**; (d) **two-level sparse kernel built**
   (`harness/verify_sparse_2lvl.py`, `ab_sparse_semantics.py`, `cutlass_sparse.py`) → **deploy gap
   closed** (through-kernel 8.47 == fake-quant, vs single-level 11.89), costs 2–10%, still beats
   CUTLASS 80b. Deployable sparse = 8.47, loses to dense by ~1.56. Dense at +0.63 is the accuracy
   headline; sparse is the speed spine (CUTLASS-beating, ~1.33× over our dense).

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
