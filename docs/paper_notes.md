# quadbit — Paper Checkpoint

Working notes for a paper on hand-written FP4 tensor-core kernels for **consumer/pro
Blackwell (SM120, RTX PRO 6000 / RTX 5090)** — the accessible card where NVIDIA's libraries
leave FP4 gaps. Everything below is measured on Modal RTX PRO 6000 via `harness/run_cuda.py`
(raw CUDA) and `harness/*.py` (PyTorch/model harnesses). Kernels compiled `nvcc -arch=sm_120a`.

## Thesis

On SM120, FP4 tensor cores exist but the usable software stack is thin. cuBLAS gives dense FP4
only. **CORRECTION (prior-art sweep, 2026-07): CUTLASS DOES ship a sparse NVFP4 GEMM for SM120**
(`examples/80_blackwell_geforce_sparse_gemm/80b_blackwell_geforce_nvfp4_nvfp4_sparse_gemm.cu`,
`ArchTag = Sm120`, `OpClassBlockScaledSparseTensorOp`, same `mma...kind::mxf4nvf4.sp::ordered_metadata.block_scale`
instruction we use, shipped in CUTLASS 3.9.0 on 2025-04-24). The earlier claim here of "no sparse FP4
path exists at all" / "the only 2:4-sparse FP4 GEMM on SM120" is FALSE and must not appear in the paper.
What is defensible: the SM120 block-scaled path has documented correctness/autotuner problems in practice
(CUTLASS issue #3096: grouped GEMM garbage output, TMA warp-specialized tactics fail to init on SM120),
and no one wraps a sparse FP4 kernel in a full deployment stack. We hand-write (raw PTX) both a dense FP4
GEMM that is competitive with CUTLASS dense (wins/ties on square, slightly behind on rectangular LLM
shapes) and a 2:4-sparse FP4 GEMM at the SM120 bandwidth roofline that beats CUTLASS 80b on the shapes
that ship, then build the full deployment stack (packer, fused activation quantizer, `nn.Linear`
drop-in) and a one-shot + QAT recovery pipeline that makes the sparse path usable on real models.
**The paper's spine is the sparse path.** UPDATE (2026-07-06, leaderboard `harness/leaderboard_fp4.py`):
the dense baseline moved. FlashInfer `mm_fp4` now ships a CUDA-13 `b12x` NVFP4 kernel + a `cutlass` path,
and they beat quadbit's deployed two-level dense by **1.35–2.2×** — dense is NO LONGER competitive-leading,
it LOSES the SM120 dense race. What makes the sparse spine the headline is now sharper: quadbit is the only
*deployed* 2:4-sparse FP4 GEMM (FlashInfer/SGLang/vLLM ship none; CUTLASS 80b is an unwrapped example), and
its two-level sparse kernel beats even the *best FlashInfer dense* in wall-clock on every prefill shape
(**1.07–1.38×**). That is the Pareto point no library provides. Dense survives only as a zero-training W4A4
accuracy drop-in (+0.63 PPL).
**GATING EXPERIMENT — NOW RUN** (`harness/cutlass_sparse.py`, correctness-gated on 80b's own reference
check, which PASSES at every size, so #3096's block-scaled bug does not affect this example): head-to-head
throughput of our sparse kernel vs CUTLASS 80b on the same RTX PRO 6000 gives **1.16× @4096, 1.14× @8192,
0.96× @16384** (we lose at 16K: 1859 vs 1785 TF/s). Defensible claim: fastest sparse FP4 at the 4–8K tile
sizes real LLM GEMMs use, NOT "at every size" and NOT "the only." Rectangular-shape sweep NOW RUN
(`harness/cutlass_shapes.py`): on real Llama-3-8B shapes sparse **wins consistently** vs 80b (attn
4096³ 1.18×, ffn-up 1.14×, ffn-down 1.17×, all 80b-verified), while dense **loses** to 79b
(0.89×/0.93×/1.01×) — so the dense "beats at every size" claim was a square artifact; sparse is the
consistent CUTLASS-beating result on shipping shapes.

## Headline results (real, measured, RTX PRO 6000)

Throughput vs cuBLAS bf16 (what production runs) and vs real CUTLASS FP4, M=N=K.
bf16/dense/sparse: `harness/bench_vs_bf16.py`; CUTLASS: `harness/cutlass_fp4.py`
(example 79b nvfp4×nvfp4→f32, `-DCUTLASS_NVCC_ARCHS=120a`, verification Passed). Same RTX PRO 6000:

**These are SPEED-PATH CEILING numbers** (MXFP4-fast dense, unit-scale sparse) — the fastest variants,
NOT the accuracy-deployed two-level kernels. The deployed two-level numbers (which carry the accuracy
recipe and are what the leaderboard/head-to-head measure) are lower: dense sq8192 **1045** (not 1556),
sparse sq8192 **1973** (not 2207). Report both; do not conflate the ceiling with the deployed kernel.

| size | cuBLAS bf16 | CUTLASS FP4 | dense FP4 (ours) | 2:4-sparse FP4 (ours) |
|------|-------------|-------------|------------------|------------------------|
| 4096 | 372 TF/s | 1222 | 1136 (3.06× bf16) | 1512 (4.07× bf16) |
| 8192 | 423 TF/s | 1497 | 1556 (3.68× bf16) | 2207 (5.22× bf16) |
| 16384| 405 TF/s | — | 1645 (4.06× bf16) | 1782 (4.39× bf16) |

**SM120 FP4 backend leaderboard (2026-07-06, `harness/leaderboard_fp4.py`, DEPLOYED two-level kernels,
all correctness-gated on fp32 ref cos>0.97; every backend hits cos 0.991 with identical cos/maxrel/mae =
same math, cross-validated).** quadbit dense vs FlashInfer `mm_fp4` best-backend, effective TF/s:

| shape | quadbit dense | FlashInfer best (backend) | FI / quadbit |
|-------|--------------|---------------------------|--------------|
| square 8192 | 1045 | 1433 (cutlass) | 1.37× |
| prefill attn 4096³ | 838 | 1283 (b12x) | 1.53× |
| prefill ffn-up N=14336 | 936 | 1374 (auto) | 1.47× |
| prefill ffn-down K=14336 | 1017 | 1408 (b12x) | 1.38× |
| serving ffn-down M=65536 | 639 | 1416 (cutlass) | 2.22× |

Dense LOSES 1.35–2.2×. Structural findings: FlashInfer `cudnn` fails EVERY shape (`No execution plans
support the graph`; shipped cuDNN 9.10 < 9.14 SM120 FP4 needs); `b12x` collapses ~2.2× at M≥65536 (only
`cutlass` holds ~1400); `trtllm`/`cute-dsl` refuse SM120. Toolchain split: `b12x` needs CUDA 13, quadbit's
`sm_120a` block-scale mma (`kind::mxf4nvf4`/`block_scale`/`scale_vec::4X`) is rejected by ptxas 13 and only
assembles under CUDA ≤12.8 — cannot coexist. **PARETO (the headline): quadbit two-level SPARSE beats the
best FlashInfer DENSE in wall-clock on every prefill shape** — attn 0.100 vs 0.107 ms (1.07×), ffn-up 0.301
vs 0.350 (1.16×), ffn-down 0.265 vs 0.342 (1.29×), sq8192 0.557 vs 0.767 (1.38×). And fresh two-level vs
CUTLASS 80b (`cutlass_sparse.py`): win every shape — attn 1.08×, ffn-up 1.01×, ffn-down 1.12×, sq8192 1.09×
(1973 vs 1807).

- Dense FP4 vs CUTLASS, **square, apples-to-apples clean measurement** (both cudaEvent-timed over 20
  iters, no torch dispatch; ours = `matmul_fp4_pp_bf16` standalone): CUTLASS 634 / 1222 / 1497 @
  2048/4096/8192, ours **758 (+20%) / 1220 (tie) / 1510 (+0.9%)** — win/tie/win on square. BUT on the
  **rectangular Llama-3-8B shapes that actually run, dense LOSES to CUTLASS 79b** (`harness/cutlass_shapes.py`,
  79b-verified): attn 4096³ **0.89×**, ffn-up N=14336 **0.93×**, ffn-down K=14336 **1.01×**. The square
  win does not generalize — CUTLASS's tile/schedule adapts to rectangular shapes better than our fixed
  tiling. vs 79b specifically: competitive, slightly behind on shipping shapes; still 3.0–3.7× over
  bf16. But 79b is NOT the baseline that matters anymore — the leaderboard above shows FlashInfer
  beating dense outright (1.35–2.2×), so this bullet is a scoped 79b sub-result, not the dense verdict.
  (Per-SM square steady state @8192 we are 86% vs CUTLASS 83% of the 1811 TF/s mma peak; the 4096
  tie is wave quantization, 512 tiles / 188 SMs = 2.72 waves.)
- Sparse FP4 beats the CUTLASS **dense** FP4 baseline by **+24% @4096 (1512 vs 1222)** and
  **+47% @8192 (2207 vs 1497)** on the effective-FLOP metric NVIDIA uses for 2:4. And vs CUTLASS's
  **sparse** NVFP4 SM120 example (80b), the gating head-to-head (now run, see Thesis): square **1.16×
  @4096, 1.14× @8192, 0.96× @16384** (win 4–8K, lose 16K), and on **rectangular Llama-3-8B shapes a
  consistent win** — attn 4096³ **1.18×**, ffn-up **1.14×**, ffn-down **1.17×** (all 80b-verified).
  Unlike dense, the sparse win DOES generalize to shipping shapes; this is the headline. Do not claim
  "at every size" (16K square loses) or "the only," but sparse beats CUTLASS 80b on every real LLM shape.
- Unit-scale headline (perf ceiling): sparse **2731k GFLOP/s**, dense **1515k**, both @8192.

Real Llama-3-8B GEMM shapes (`harness/bench_llm_shapes.py`, hidden 4096 / FFN 14336, vs cuBLAS bf16;
dense split-K = `cuda/dense_sk_lib.cu`, auto-tuned split factor):

| shape | M/N/K | plain | split-K | decode-K | best dense | 2:4 FP4 |
|-------|-------|-------|---------|----------|-----------|---------|
| prefill attn qkv/o | 4096³ | 3.20× | 3.04× | 1.61× | **3.20×** | 4.14× |
| prefill ffn up | 4096/14336/4096 | 3.38× | 3.36× | 1.52× | **3.38×** | 4.49× |
| prefill ffn down | 4096/4096/14336 | 3.64× | 3.63× | 1.60× | **3.64×** | 5.09× |
| decode ffn up | 128/14336/4096 | 2.76× | 2.74× | 4.53× cached+async | **4.53×** | (M<256) |
| decode ffn down | 128/4096/14336 | 0.49× | **1.40×** | 1.11× cached | **1.40×** | (M<256) |
| decode attn qkv/o | 128/4096/4096 | 0.49× | 0.68× | **1.27× cached+async** | **1.27×** | (M<256) |

(decode-K = split-N decode kernel; "cached+async" = weight TMA map built once + fire-and-forget launch +
ADAPTIVE config, the deployment path — see decode notes. **Every LLM shape now beats cuBLAS bf16.**)

- **Prefill (bulk of training + long-context compute): 3.0–3.6× dense, 4.1–5.0× sparse over bf16.**
- **Decode small-M underfills the SM array** (grid = N/256 × 1 = 16 blocks for N=4096, 16/188 SMs).
  Two complementary, shape-routed fixes:
  - **split-K** (`dense_sk_lib.cu`, gridDim.z CTAs summing K-subranges into a tiny f32 workspace):
    wins **long-K** decode (ffn-down 128/4096/14336: 0.50×→**1.42×**, s=8). Decode output is tiny so
    the f32 reduction that sank square split-K is negligible.
  - **split-N decode kernel** (`dense_decode_lib.cu`, narrow 128×32 tile, one warpgroup, each warp an
    8-col n-tile over all rows + full K, direct bf16, no reduction): wins **low-K** decode (attn qkv/o
    128/4096/4096: kernel-only **20.5µs = tie with bf16**, up from plain 0.48×). Long-K loses here (A
    re-read from L2 grows) → route to split-K instead.
  - **Cached-map + async launch** (`qb_encode_map` / `dense_fp4_decode_cached_async`): a TMA map
    encodes a buffer's address + tiling, NOT contents, so the weight map is built ONCE at load and
    reused every decode step; the launch is fire-and-forget (no per-call `cudaDeviceSynchronize`,
    matching how torch enqueues). This was the real decode fix — the earlier 0.73× was NOT map-build
    cost (cached-with-sync was still 0.78×) but the per-call device sync exposing ~8µs launch latency
    on a 20µs kernel. With cached+async: **attn-qkv 0.50×→1.11×, ffn-up 2.80×→3.09×**.
  - **Adaptive decode config** (occupancy sweep, `matmul_fp4_dec2.cu`): decode is bound by activation
    re-read (L2) + weight stream (DRAM), NOT compute. More blocks backfires — each N-block re-reads
    all of A from L2, so block count multiplies A traffic. Sweet spot by N: large N → 8 warps/block
    (NWARP=8, more threads since blocks are plentiful); small N → 4 warps + deep pipeline (STAGES=8)
    to hide latency. `dense_decode_lib.cu` dispatches on N. Deployment (cached+async+adaptive):
    ffn-up **3.09×→4.53×**, attn-qkv **1.11×→1.27×**.
  - Routed best per shape: prefill all `plain` (3.0–3.6×); decode attn-qkv **1.27×**, ffn-up **4.53×**,
    ffn-down **1.40×** (split-K). **Every LLM GEMM shape now beats cuBLAS bf16.** Remaining headroom:
    the ffn decode ops are weight-DRAM-bound → 2:4-SPARSE decode would ~halve weight bytes (ffn-up
    ~4.5×→~7×); attn-qkv @4096² is latency-bound (too small to saturate memory), a practical ceiling.

Accuracy (real models, WikiText-2 PPL):
- **Dense FP4 is W4A4, +0.63 PPL, zero training, matched to the reference — the accuracy headline.**
  The FP4 tensor core multiplies fp4×fp4, so the shipped kernel is W4A4 (weights AND activations
  4-bit); there is no weight-only FP4 GEMM in hardware, so the often-quoted +0.3 is a **W4A16**
  number that never ships. With our **per-16 two-level NVFP4 recipe** (amax, **no calibration**),
  measured through `harness/recovery_worth.py`: **Llama-3.1-8B-Instruct 7.27→7.90 (+0.63)**;
  Meta-Llama-3-8B base 6.20→6.91 (+0.71) — at/below the modelopt-calibrated reference (vLLM native
  NVFP4 7.97, +0.71). An earlier "+2 PPL" was our **crude per-32 single-level** recipe, NOT W4A4's
  real cost; the gap was block granularity + two-level activation scaling, not calibration. (W4A16
  weight-only +0.3: Sparse-Llama-3.1-8B 7.89→8.16, Qwen2.5-3B 7.60→7.91 — real but not deployed.)
- Sparse FP4 needs pair-granular 2:4 recovery (see below).

## Core technical contributions

1. **From-scratch SM120 FP4 mma/ldmatrix/scale/metadata layouts, all derived empirically + verified maxrel 0.**
   No tcgen05 on SM120 (that's SM100-only); we use warp-level `mma.sync`/`mma.sp`. Derived by
   probe-and-verify (not docs): the dense block-scaled `mma.sync...m16n8k64...ue8m0` and sparse
   `mma.sp::ordered_metadata...m16n8k128...ue4m3` operand/scale/metadata bit-layouts.
   - Scale layout: `scaleA[row r][kb]→lane (r&7)*4+(r>>3) byte kb`; `scaleB[col c][kb]→lane c*4 byte kb`.
   - 2:4 metadata: `lane L→mma-row (L&1)*8+(L>>2), half (L>>1)&1; nibble=idx0|(idx1<<2)`.

2. **The wide-TMA + swizzle breakthrough (the big kernel win).** A mem-only probe suggested a
   "2012k roofline"; it was FALSE — narrow TMA boxes extracted only 54% of the 7.3 TB/s L2→smem
   ceiling. Loading WK=2 k128-slices per TMA (wider boxes) hits the ceiling, but wide smem rows
   cause ldmatrix bank conflicts → fixed with **swizzled TMA** (A box 64B→`SWIZZLE_64B`, ldmatrix
   XOR `off^=((off>>7)&3)<<4`; B box 128B→`SWIZZLE_128B`, XOR `((off>>7)&7)<<4`). Result: sparse
   2012k→**2731k (+36%)**, deployable 1486k→**2116k (+42%)**. Lesson: never declare a memory
   roofline from a probe whose own load pattern is the limit.

3. **The only DEPLOYED 2:4-sparse FP4 GEMM on SM120** (CUTLASS 80b exists as an unwrapped example;
   FlashInfer/SGLang/vLLM ship no sparse FP4 at all — verified 2026-07): arbitrary per-group 2:4
   metadata + real per-block ue4m3 scales, both staged coalesced through a full/empty async pipeline
   (no CTA-wide `__syncthreads`), shared-B 256×128 traffic-optimal tiling. Defensible framing (now the
   PARETO headline) = "hand-written, roofline-saturating, the only sparse FP4 GEMM wrapped in a real
   packer + fused-quantizer + recovery stack; beats CUTLASS 80b on every shape (1.01–1.12×) AND beats
   the best available *dense* FP4 kernel (FlashInfer `b12x`/`cutlass`) in wall-clock on every prefill
   shape (1.07–1.38×), a Pareto corner no library provides." This is what stands after dense lost the
   race to FlashInfer.

4. **Pair-granular 2:4 handling + its measured accuracy cost (NOT a novel hardware discovery).**
   Blackwell FP4 `mma.sp` metadata selects at b16 = **fp4-pair** granularity: 2 of every 4 *pairs*
   kept, not 2 of every 4 *elements*. **CORRECTION (prior-art sweep, 2026-07): this is NVIDIA's
   documented hardware spec, not something we discovered.** NVIDIA's fifth-gen tensor cores use
   pair-wise 4:8 NVFP4 sparsity (every 8 elements = 4 pairs, 2 pairs kept); this is described publicly
   (e.g. SemiAnalysis, "NVIDIA Tensor Core Evolution"), which also already notes the pair constraint is
   no more relaxed for ML than element-2:4. So drop "novel/paper-worthy" and "we derived it." What may
   still be an original *data point* (no published number found): the measured consequence on a real
   element-2:4 checkpoint. On `neuralmagic/Sparse-Llama-3.1-8B-2of4` (exactly element-2:4, 50% zeros),
   pair-granular selection keeps only **~87% of its nonzero energy** → naive use gives **93.6 PPL**
   (vs 7.9 dense-FP4). Frame this as "we quantify the accuracy cost of the documented pair constraint
   on existing element-2:4 tooling," citing the hardware spec, not as a discovery.

5. **Deployment stack + operator fusion.** `QuadbitLinear` (`nn.Linear` drop-in): torch packer
   reproducing the kernel's exact metadata/compress/scale layout (verified maxrel 0.0039) + a
   **fused 128-bit NVFP4 activation quantizer** (one CUDA pass). End-to-end **4.0–4.2× over torch
   bf16** at 8192; any token count (padding). Real per-block ue4m3 weight scales
   (magnitude-independent: works at wscale=0.02).
   - **Fused SwiGLU FFN** (`swiglu_quant` + concatenated gate/up): the unfused FFN quantizes x twice
     (gate+up) and does silu+mul+casts+a separate down-quant in eager torch (~5 memory passes over
     [batch,hidden]). Two fusions: (a) **fused epilogue** — quant x ONCE (shared gate+up) + one kernel
     that reads g,u, computes silu(g)·u, and emits the FP4-packed down-proj input + scales in a single
     transposing pass (consecutive threads take consecutive batch so the strided g/u reads coalesce);
     (b) **concat gate+up** into one GEMM (out=2·hidden, one launch, one shared xq read, better SM
     fill). SwiGLU FFN block vs torch bf16, cumulative: unfused **2.05×** → +fused-epilogue **4.45×**
     → +concat **4.66×** at batch=2048 (**2.27× over unfused**); **1.74×→2.95×** at batch=512.
     Numerically identical (rel 0.741 = same 2:4 prune floor) — pure memory-traffic win. The kernels
     were already at the silicon ceiling; the remaining end-to-end gain was in fusion, exactly here.
   - **Fused RMSNorm + NVFP4 quant** (`rmsnorm_quant_k`): the block entry (fires twice/block:
     pre-attn, pre-FFN). One CTA per row loads x to smem, block-reduces the sum-of-squares, then
     each thread normalizes (×rms×weight) and quantizes its 32-blocks — a single read of x replaces
     eager rmsnorm (read+reduce+write) + a separate quant pass (read+write), and it's more accurate
     (no bf16 round-trip before quant). **3.7–4.3× over eager rmsnorm+quant** (0.129→0.035ms @2048),
     rel 0.105 vs true = the FP4 activation-quant floor. Feeds straight into the QKV/gate-up GEMM.
   - **Fused residual-add + RMSNorm + quant** (`add_rmsnorm_quant_k`): the full block transition —
     h = inp+residual (written back as the updated residual stream) then rmsnorm(h)·w then quant, one
     kernel. Folds the eager residual add's read2+write into the norm+quant. **5.3–5.8× over eager
     add+rmsnorm+quant** (0.215→0.037ms @2048), rel 0.105, residual maxabs-err 0.016 (bf16 rounding).
   - **Fusion track summary:** the raw GEMMs are at the silicon ceiling, so every end-to-end LLM gain
     came from fusing the glue between GEMMs: fused SwiGLU FFN (2.05→4.66×), fused RMSNorm+quant
     (3.7–4.3×), fused add+RMSNorm+quant block transition (5.3–5.8×), concat gate/up. All at zero
     accuracy cost (FP4/prune floors preserved). Every inter-GEMM memory round-trip in a transformer
     block is now a single fused pass.

6. **Pair-granular recovery pipeline (one-shot + QAT).** (Drop "no NVIDIA equivalent": NVIDIA's
   NVFP4-QAD, arXiv 2601.20088, is the quant-recovery equivalent, and SpenseGPT / SQ-format / SharQ
   are adjacent sparse+quant prior art that must be cited and distinguished. Defensible originality =
   retargeting the *mask* to pair-granular 2:4 for the FP4 sparse path specifically.) SparseGPT retargeted
   to pair-granular masks (keep 2-of-4-pairs by `w²/[H⁻¹]²`, Hessian error compensation) →
   KD from the dense teacher (mask frozen) → QAT with straight-through fake-quant of BOTH weights
   (exact kernel dequant) and activations. Deepest WikiText-103 run (TinyLlama-1.1B, 30k bf16 + 2k QAT
   steps, cosine LR): dense fp16 teacher 7.53; one-shot pair-2:4 FP4 **19.1**; after phase-1 bf16
   recovery **8.95**; after phase-2 matched-STE QAT (FP4 fake-quant) **9.57**; **through the real
   2:4-sparse FP4 kernel 9.60** — within ~2.1 PPL of dense, beating the earlier fp32-STE run's 10.03.
   Phase-2 QAT converges by ~1k steps (measured flat 500→9000 on the 10k run), so the win is in
   phase-1 data scale, not QAT length. **The STE-vs-kernel gap is closed.** It was first **localized**
   (`harness/probe_ste_kernel.py`, real non-uniform weights + activations): it is **100% the
   activation quantizer**, not the weights. The kernel's weight path is arithmetically exact (kernel
   output vs its own dequant, rel **0.0017** — the earlier "wscale=0.02 PASS" used uniform weights
   that hid any scale/layout permutation, this drives real weights and confirms none). The entire
   per-linear divergence (rel **~0.04**) was the fp32 STE activation fake-quant vs the deployed NVFP4
   `quant_act` (bf16 pre-round + no-denormal ue4m3 scales + reciprocal-multiply), compounding over 66
   MLP linears into ~0.9 PPL. Fix landed: the QAT STE now bit-matches the kernel's activation
   quantizer (scale codes verified identical). Result — the matched-STE fake-quant PPL (9.57) now
   tracks the through-kernel PPL (9.60) within **0.04**, versus the old ~0.9 gap, and the deployed
   number dropped 10.03→9.60. Training against exactly what ships is worth ~0.43 PPL. Phase-1 result
   is checkpointed to the volume so phase-2 experiments skip the 30k-step rebuild. **On a REAL 8B model the sparse accuracy case is currently negative and data-limited (2026-07-02).**
   Meta-Llama-3-8B, good recipe (per-16 two-level weight+act NVFP4), 30k phase-1 + 2k QAT on full
   WikiText-103 (88M tok): teacher 6.20; phase-1 bf16-masked 2:4 **8.57** (+2.37 — the sparsity cost,
   *before* any FP4); phase-2 QAT FP4, original under-trained schedule **9.01** (+2.81). RECIPE FIX
   (2026-07-02): re-running phase-2 as a warm-restart (fresh LR + hard-label CE), same corpus, ZERO
   new data, drops it to **8.30** (+2.10, 16-window held-out fake-quant; the in-loop 8-window metric
   read 7.87 — do not quote it as the result). Dense-W4A4 zero-train on the same model is **6.91
   (+0.71)**, so sparse now **loses to dense by ~1.4 PPL, down from ~2.1**. The 0.7-PPL move on recipe
   ALONE confirms 9.01 was **under-trained, not a data wall** (recipe closed ~a third of the gap). The
   TinyLlama "sparse 9.60 beats dense 9.73" flip was an artifact of the old crude-dense (+2) number and
   is retired; with corrected dense W4A4 (+0.63/+0.71) dense wins.
   **DEPLOY GAP CLOSED (2026-07-05): the two-level sparse kernel is built.** The 8.30 was a per-16
   fake-quant ceiling the sparse mma cannot deploy (its B-side activation scale is per-32). Matching
   the QAT activation STE to per-32 and running through the NEW two-level sparse kernel (per-row and
   per-col fp32 global rescale in the epilogue, ported from the dense two-level path), Meta-Llama-3-8B
   recovers to **8.47 through-kernel == 8.47 fake-quant** (2k QAT, warm-restart; deploy gap ~0, was
   +2.67 with the single-level kernel). Same-checkpoint A/B (`harness/ab_sparse_semantics.py`, 5k
   recovered ckpt): single-level kernel **11.89**, two-level kernel **8.95**, fake-quant target **8.96**
   (ΔNLL 2lvl −0.002 / 1lvl +0.282; top-1 agree 2lvl **91.7%** / 1lvl **78.3%**). The rescale costs
   **2–10%** throughput (worst on wide-N) and two-level still beats CUTLASS 80b on every shape
   (**1.01–1.12×**, all correctness-PASS; `harness/cutlass_sparse.py`). Extending phase-2 2k→5k did NOT
   help (regressed to 8.96, mild overtraining vs WT-2 test) — 2k is the sweet spot on this corpus.
   **DATA LEVER = NEGATIVE (2026-07-05):** the full-scale C4 diverse-corpus recovery (app
   ap-SdSv9zQ9, phase-1 to ~196M tokens) flattened at **10.82 on WT-2 test** (vs in-distribution
   WikiText-103 phase-1 8.57) because C4 is OOD for that narrow test; the run timed out at 192k/300k,
   never approaching the ~7.4 target. Diverse-corpus scale did not improve the WT-2 number — the
   recovery is in-distribution-bound on this metric, not simply data-starved.
   **NET: dense FP4 (+0.63, zero-training, matched to the modelopt reference) is the accuracy result;
   sparse is a speed play (~1.33× over dense) that now DEPLOYS at its trained accuracy (deploy gap
   closed) but loses to dense by ~1.56 PPL on recovery, and diverse data did not close it. Sparse is an
   honest speed-only Pareto point, not an accuracy win.**

## Real open-weight models (July 2026), on this hardware

Verified against ACTUAL current model configs (not assumptions), `harness/real_model.py` on RTX PRO 6000:
- **Every projection of the current frontier tiles onto the kernel** (out%256, in%256), because head_dim=128
  and hidden sizes are all multiples of 256. Ran each model's real linear shapes THROUGH the kernel:
  Qwen3.5-397B (Qwen3Moe), GLM-5.2 (Glm4Moe, H=5120), MiniMax-M3 (H=6144), DeepSeek-V3/R1 (H=7168) —
  all ALL-TILE-OK, 24–150µs/linear. Big models deploy via expert/tensor/pipeline sharding; each shard's
  linears ARE the kernel. Small models (Qwen3-8B, Gemma-4, Phi-4) run whole on one card.
- **Full fused FP4 decoder block on REAL Qwen3-8B weights** (fused RMSNorm+quant → concat-QKV → attn(bf16)
  → o-proj → fused add+RMSNorm+quant → fused SwiGLU) vs a fair bf16 tensor-core block: **2.26× @512,
  3.22× @2048 tokens** (sparse path). Block accuracy: sparse 2:4 = 0.55 (needs recovery), dense-FP4 sim = 0.103.
- **DEPLOYABLE capstone — full fused DENSE real-scale FP4 block on real Qwen3-8B, NO training** (all linears
  through `dense_scaled_fast_mm` + MXFP4 fused RMSNorm/add-RMSNorm/SwiGLU ops): **2.16–2.19× over bf16,
  block rel 0.13**. Zero fine-tuning, real frontier weights, real scales, through the actual CUDA kernels —
  the universal drop-in. (Block rel 0.13 < per-linear 0.165 because attention stays bf16 + residuals dilute.)
- **Deployable dense real-scale FP4 THROUGH the kernel** (MXFP4 e2m1+ue8m0 both operands, from the
  proven verify_scaled mma; fused MXFP4 act quantizer): real Qwen3-8B linears **rel 0.165, NO training**
  — the drop-in that works on any model. Two kernels: `dense_scaled_lib.cu` (BM=64 verify_scaled tiling,
  ~bf16 speed) and `dense_scaled_fast_lib.cu` (the fast 2-warpgroup pingpong tiling + real scales:
  the mma is identical so the scale lane layout ports directly). Optimized via step-major scale layout
  + coalesced synchronous smem staging (STAGES=3 tile pipeline, single-buffer scales): real Qwen3-8B
  linears **2.28–2.62× over bf16, rel 0.165, no training** (q/o 2.29×, gate 2.40–2.58×, down 2.57–2.62×).
  That is **75–85% of the unit-scale speed ceiling** (measured 2.9–3.5× at the same shapes). The residual
  gap is FUNDAMENTAL: with real scales the scale operands are per-k-step smem LDS in the mma hot loop
  (unit-scale has them as free compile-time zeros), and that LDS-vs-OMMA issue contention caps block-scaled
  FP4 — CUTLASS pays it too. Tried: per-mma global loads (1.3×), smem-staged STAGES=2 (2.0×), cp.async
  prefetch STAGES=2 (1.9×, the extra bulk ops + lost stage hurt), coalesced sync STAGES=3 (**2.6×, best**).
  So ~2.6× is the real-scale ceiling; 3× is unit-scale-only.
- **NVFP4 (ue4m3, per-16) dense mma — DERIVED + VERIFIED** (`verify_nvfp4_dense.cu`): the dense
  `scale_vec::4X.m16n8k64...ue4m3` scale lane layout was unknown; probed and confirmed it's the SAME
  A/B row→lane mapping as the ue8m0 2X case (fixed by m16n8k64), just a 4-byte (4 per-16) scale reg —
  **PASS, maxrel 0.0000 with BOTH uniform AND wide-range (2^-4..2^2) ue4m3 scales** — the wide-range
  probe is the strong test (it confirms the 4 per-16 scale bytes map to the right k-sub-blocks; a
  uniform-scale probe can't). Built deployable NVFP4 kernel (`dense_nvfp4_fast_lib.cu`, SFA/SFB[step][·][8]
  per-16). The mma is DEFINITIVELY correct (wide-scale maxrel 0).
  - **TWO-LEVEL NVFP4 (per-16 ue4m3 local × per-row fp32 global) — SOLVED the block.** The single-level
    NVFP4 block regressed to 0.38: the ue4m3 scale carries a mantissa, so per-block scale-rounding bias
    accumulates across the 3-matmul chain (MXFP4's power-of-2 ue8m0 has no scale-rounding, stays robust).
    Fix = the standard NVFP4 two-level recipe: per-row global gA=rowamax/2688 (2688=e4m3max·e2m1max)
    rescales the local ue4m3 scales into e4m3's precise range; the mma applies the locals, the fp32
    globals multiply the accumulator in the epilogue (`dmatmul_nvf` + `quant_act_nv2_k`). Result on real
    Qwen3-8B, NO training, through the kernel: **per-linear 0.165→0.134, full block 0.13→0.097** — below
    the ~0.10 target. Speed 1.2–2.3× over bf16 (STAGES=2 + per-16 scales); NVFP4 = the accuracy path
    (0.097), MXFP4 = the speed path (2.15× @ 0.13). The ue4m3 dense mma accuracy follow-up is DONE.
  - **ASYNC SCALE PREFETCH — dense two-level NVFP4 kernel sped up 1.08–1.22×, maxrel 0**
    (`cuda/dense_nvfp4_fast_lib.cu`, one-time same-card A/B vs the prior synchronous kernel, 2026-07-06).
    The deployed kernel loaded each step's per-16 scales SYNCHRONOUSLY between the tile TMA
    try_wait and the mma — a ~500-cycle global-load latency fully EXPOSED every step (STAGES=2 hides the
    tile TMA but not the scales). Fix: double-buffer the scales and prefetch step s+1 via `cp.async` (16B
    chunks, commit_group / `wait_group 1`) during step s's mma, so the scale latency overlaps compute.
    Same math (maxrel **0.00000** vs the old kernel at every shape), STAGES=2 and DBN=128 unchanged, +4KB
    smem (72KB total, under the 99KB cap). A/B same-card TF/s (deployed → async): sq2048 390→424 (1.09×),
    sq4096 763→824 (1.08×), sq8192 865→1055 (1.22×), attn-4096³ 762→826, ffn-up N=14336 813→922 (1.13×),
    ffn-down K=14336 905→984 (1.09×). Biggest win at 8192 (more k-steps → more exposed scale loads to
    hide). This narrows the real-scale-vs-unit-scale gap (the ~1510 unit-scale ceiling was scale-free);
    it does NOT touch the 1136/1510 headline (those are the MXFP4-fast / unit-scale pp kernels, a
    different path). Folded into `dense_nvfp4_fast_lib.cu` (same public names → all callers inherit it).
    End-to-end full-model prefill moved 7815→7987 tok/s (+2.2%: the GEMM is a small fraction of the eager
    forward, so the kernel win dilutes; the prefill bottleneck is eager bf16 attention). PPL unchanged 7.899.

## End-to-end serving comparison (RTX PRO 6000, Llama-3.1-8B-Instruct family)

Real serving engines on the same card/protocol (CUDA graphs on, distinct per-request prompts — identical
prompts collapse under prefix caching and inflate prefill; prefill = B×S=2048 with 1 out tok, decode =
GEN=128 ignore_eos; WT-2 16×2048). `harness/vllm_nvfp4.py serve`, `harness/sglang_fp4.py --mode bench`,
`harness/dense_e2e.py`. Weights = loader footprint; both engines also pre-allocate a KV pool to their util
fraction, so total device reservation is ~86–89 GiB regardless of quant (not the weight number).

**Table A — real serving engines:**

| engine | quant | weights | WT-2 PPL | prefill B=1/8/32/64 | decode B=1/8/32/64 |
|--------|-------|---------|----------|---------------------|--------------------|
| vLLM 0.21 | bf16 | 15.0 GiB | 7.267 | 10209/26436/31559/46880 | 88/690/2599/4947 |
| vLLM 0.21 | NVFP4 modelopt-cutlass | 5.66 GiB | 7.974 | 13530/63056/78028/116831 | 131/1049/4259/8465 |
| SGLang 0.5 | NVFP4 FlashInfer-CUTLASS `fp4_gemm` | ~5.6 GiB | 7.97 (ckpt) | 16829/59769/73414/109002 | 186/1491/5424/10145 |

- **Both engines run NATIVE NVFP4 W4A4, not Marlin**, on this card for the dense checkpoint: vLLM binds
  `modelopt_fp4` cutlass; SGLang `auto` selects the FlashInfer CUTLASS `fp4_gemm` (autotuned at startup,
  in-log). The ~50 tok/s Marlin fallback figure was a 397B MoE, not this path.
- NVFP4 vs bf16 (vLLM, B=64): **~1.7× prefill** (116831 vs 46880), **~1.7× decode** (8465 vs 4947), for
  **2.6× smaller weights** and +0.71 PPL. SGLang wins decode every batch (10145 vs 8465 @B64) and low-batch
  prefill; vLLM wins high-batch prefill. NVFP4 KV is fp8_e4m3 (checkpoint config); bf16 row uses bf16 KV.

**Table B — quadbit dense FP4 prototype (full-forward, PREFILL-ONLY, no decode engine):** quantized-linear
weights **3.93 GiB** (block linears only; embeddings/lm_head/attention stay bf16, not counted → not a
full-model on-device footprint), through-kernel WT-2 PPL **7.899**, full-model prefill **7987 tok/s**
(B=8, S=2048, eager + HF bf16 attention + fused act-quantizer + per-linear kernel call). NOT a serving
number — no paged attention, no continuous batching, no CUDA graphs, no decode scheduler; it trails
serving prefill ~10× because of the eager bf16-attention forward, not the GEMM. What it proves: the
deployed W4A4 path is accuracy-correct end to end — **7.90 matches native NVFP4 7.97 within 0.07 on the
same windows**, so the zero-calibration two-level recipe reaches the calibrated reference. Getting quadbit
into a serving stack for a true tok/s head-to-head is the open engineering step.

## Measured hardware ceilings (SM120 / RTX PRO 6000)

- mma peaks (register-only, DCE-defeated): sparse `mma.sp` m16n8k128 = **3626k**, dense
  `mma.sync` m16n8k64 = **1811k**, ratio 2.002× (2:4 is a real 2× FLOP feature).
- L2→smem TMA bandwidth ceiling = **7.3 TB/s** (`tma_bw.cu`).
- smem cap = **99 KB/block** (101376 B), 100 KB/SM. L2 = 128 MB.
- Dense FP4 is compute-bound (~84% of 1811k); sparse FP4 is load-bound (swizzle floor 6.0 TB/s).
- **Peak DRAM BW = 1.46 TB/s** (measured, 1 GiB d2d copy). Decode is DRAM-bound on the weight
  stream + output write (`bench_decode_bw.py`, achieved = (N·K/2 + M·N·2)/time): **ffn-up = 84.6%
  of peak (4.75×), big-N = 91.8% (4.12×)** → the memory-bound 4× is realized once the op fills the
  SMs. **attn-qkv/o (128×4096×4096) = 39% (1.38×, 16.5 µs vs 6.5 ideal)** is latency/fill-bound: at
  N=4096, TN=32 gives only 128 blocks on 188 SMs, and it's the cheapest decode op. Both ways to add
  parallelism lose — smaller TN blows up activation L2 re-reads (swept), and split-K's f32 workspace
  (memset + atomic + convert ≈ doubles DRAM traffic) exceeds the fill gain even on a clean
  cached-map async path (correctness maxrel 0; s1 1.23× but s≥2 = 0.79–0.86×). FUSED shapes:
  fused QKV MHA (N=12288) = 72% (5.47×, fusion fills SMs), but fused QKV **GQA** (N=6144) = 30%
  (worse than isolated 4096 — 192 blocks re-read the 256 KB activation → ~48 MB L2, the
  co-bottleneck grows with block count). Root cause: fp4's 8 MB weight is 4× smaller → too few bytes
  to fill 188 SMs + amortize the 32-step pipeline, so it tips latency/L2-bound while bf16 (32 MB
  weight) stays bandwidth-bound at 96% of peak. **Every lever to add parallelism was built and
  loses:** smaller TN blows up the activation re-read; split-K's f32-partial reduction costs more
  than the fill gain in *both* forms tried — global-atomic (0.86×) and a Marlin-style
  per-tile-semaphore in-kernel reduction (no memset, no separate convert, plain non-atomic writes;
  0.64–0.77×, reverted). The shape yields only 128 output tiles (8 MB weight) → 68% max SM fill +
  exposed per-block latency at 1 block/SM; you cannot manufacture parallelism, and Marlin-style
  weight pre-permutation wouldn't change the tile count. bf16 escapes only because its 4×-larger
  weight gives 4× more tiles. **fp4's memory advantage is exactly what starves small decode.**
  Batched-M sweep (measured) settles the remedy: batching this split-N decode kernel does NOT
  recover the ceiling — %peak *falls* (o_proj 39→11% over M 128→2048) and speedup plateaus ~1.45×,
  because BM=128 blocks re-read the weight per 128-row tile (weight is not stationary across M), so
  batching multiplies weight traffic. The real remedy is a **kernel switch**: large M is prefill,
  served by the weight-stationary prefill kernel (3.6–5×). Decode kernel owns only the small-M
  regime, where it is at the shape's hardware ceiling. No middle-ground batch size to chase.

  **Decode/prefill router, measured (`harness/bench_router.py`).** The split-N decode kernel is
  unit-scale (no real scales), so a *deployable* router can only dispatch between real-scale kernels;
  the only real-scale one is the weight-stationary MXFP4 prefill kernel `dense_scaled_fast_mm`. Swept
  its speedup vs cuBLAS bf16 over token counts 256–4096 on real linear shapes. It does **not**
  uniformly beat bf16 at decode sizes: it *loses* (0.68–0.77×) in exactly one regime — small tokens
  (256) AND small output-N (4096: o_proj, ffn-down) — and wins everywhere else (large-N even at
  M256: qkv 2.75×, ffn-up 2.53×; all shapes at M≥1024: 2.0–2.8×). The loss is the same fill deficit:
  grid = ⌈M/256⌉·⌈N/128⌉, and below ~48 blocks the 188-SM array underfills. So the router is a
  **never-regress fill rule**: use FP4 iff ⌈M/256⌉·⌈N/128⌉ ≥ 48 blocks, else fall back to bf16
  (`route_dense()`; the bench asserts FP4 is never selected in a losing cell, and it held — routed
  speedup ≥ bf16 in every cell). This makes the go-to call regression-free today.

  **Real-scale decode kernel: built, measured, negative (settles the corner).** To try to turn the
  0.68–0.77× corner into an FP4 win, we built the split-N decode kernel *with* real per-32-block
  ue8m0 scales fed to the mma (the layout ported from the prefill kernel). Correctness verified
  (maxrel 0.003 vs a torch dequant-matmul reference). But it does **not** beat the bf16 fallback:
  smem scale-staging + an extra barrier tanked it to 0.44–0.59× (latency-bound kernel, staging
  serialized the pipeline); reading scales directly from L2-resident global lifted it to 0.68–0.88×;
  the fair cached-map + async path (no per-call sync) reached only **1.01× at o_proj M128 and
  0.73–0.99× elsewhere**. The unit-scale decode kernel hit 1.27× here, but carrying the *mandatory*
  real scales erases the win. So the small-batch + small-N corner has no real-scale FP4 kernel that
  beats bf16, and the router's **bf16 fallback is provably optimal there**, not a stopgap. Kernel
  reverted from the deployable lib (finding kept), matching the split-K reversion. **Router
  question closed:** FP4 where the grid fills, bf16 in the one corner it can't.

## Dead-ends (what didn't work — for the paper's honesty + "we tried X")

- **Cluster TMA B-multicast**: ptxas advisory "reduced performance on sm_120a" (datacenter-gated),
  and cross-CTA pipeline deadlocked. FP4 multicast is not for consumer Blackwell.
- **Bigger accumulator tile**: register spill (128 acc regs/thread already pins occupancy).
- **STAGES-3 deployable / scatter scales**: smem-capped (1946k < 2116k staged STAGES-2).
- **A-without-swizzle** (2577k), **asymmetric-A WK=4** (2463k): both < 2731k symmetric wide-swz.
- **Dense wide-TMA** (1515k, neutral) and **dense all-ldmatrix-then-all-mma ILP reorder** (1523k,
  neutral): dense is compute-bound; ptxas already schedules the mma stream optimally.
- **Dense B-operand ldmatrix.x4** (`matmul_fp4_bx4.cu`, cut per-k-step LDSM 12→8): REGRESSED
  1510k→1400k @8192. LDSM cost scales with matrices moved, not instruction count (x4 ≈ 2× x2
  cycles), and batching 2 n-tiles/load kills the ILP that overlaps LDSM with OMMA. LDSM *count*
  is not the lever; the ~14% gap to peak is LDSM *bandwidth* competing with OMMA (needs bigger
  register-tile reuse, which spills past ~48 mma/warp).
- **Dense split-K via f32 global + convert pass** (`matmul_fp4_splitk.cu`): REGRESSED hard
  (1034k @8192 even at splits=1). A full-tensor f32 write+read+convert adds ~0.22ms @8192 (30%
  of the matmul) — larger than the ~10% wave-quantization tail it was meant to recover. Split-K
  only pays if the reduction is partial-tile-only (stream-K), not a whole-tensor roundtrip.
- **Dense stream-K** (`matmul_fp4_streamk.cu`, 188 persistent CTAs, even (tile×kstep) work split,
  bf16-direct for sole-owner tiles + f32 atomic reduction for only the ~180 boundary-split tiles):
  CORRECT (PASS) but REGRESSED — 1380k @8192 (vs 1510 DP), 924k @4096 (vs 1220 DP). The reduction
  was cheap (partial-tile-only, as designed), but the persistent per-CTA tile loop must DRAIN+REFILL
  the async pipeline at every tile boundary (mbarrier reinit + 3-stage TMA prologue with no mma to
  hide it). That kills the inter-tile overlap the HARDWARE block scheduler gives data-parallel for
  free (it prefetches the next block while the current one's mma drains). Beating DP would require
  cross-tile SOFTWARE pipelining (CUTLASS's true persistent design) — the wave-quant win (~10% @4096)
  is smaller than the overlap DP already gets.
- **Dense persistent cross-tile pipeline** (`matmul_fp4_persist.cu`): the true CUTLASS persistent
  design — 188 CTAs walk ONE continuous (tile×kstep) unit stream, TMA issues STAGES ahead across
  tile boundaries (mbarriers inited once, never drained), tile change just flushes+resets the
  register accumulator, partial-tile-only atomic reduction. CORRECT (PASS) but REGRESSED: 1207k
  @8192, 853k @4096, and 3× slower at 2048. Root cause is REGISTER PRESSURE — the 128-register FP4
  accumulator tile + the persistent producer/consumer cursor state pins the kernel at 254 registers
  (vs DP's 187), throttling the mma stream; and small sizes turn every tile into an all-atomic tiny
  segment. Incremental-cursor rewrite (no hot-loop division) did not drop below 254. VERDICT (3
  persistent/stream-K variants, all regressed): on SM120 FP4 the huge accumulator tile leaves no
  register headroom for a software scheduler, and the HW block scheduler already overlaps consecutive
  tiles for free — data-parallel is at the practical ceiling and beats explicit persistence here.
  Clean-measured, DP already matches/beats CUTLASS at all three square sizes (rectangular loses, see headline).
- **Small-tile (single-WG 128×128)**: only +12% @2048, loses ≥4096 (shared-B B-traffic dominates);
  mid-shape is fixed-overhead-bound, not tile-bound.
- **Thin-M split-K**: worse (atomicAdd contention); thin-M is latency/overhead-bound (~0.037ms floor).
- **Recovery without weight update**: magnitude pair-2:4 = 93.6/16129 PPL, Wanda (importance-only)
  = 59.1 — the Hessian compensation (SparseGPT) is essential for the constrained pair mask.
- **One-shot SparseGPT-pair** (independent per-layer) = 20.6–21.8 PPL: needs recovery fine-tuning.
- **Sparse weight-stationary DECODE** (`sparse_sk_lib.cu`, orient C[out,tok]=W@Xᵀ so the 2:4 weight is
  the compressed mma-A, M=out large; + split-K to fill SMs): WORKS (correct) but net MARGINAL.
  (Was "novel — first 2:4 FP4 decode on SM120"; drop the "first" claim, unverified against CUTLASS 80b
  used in a decode orientation.) Wins only long-K ffn-down (128/4096/14336: 1.42×→**1.54×**, s=6);
  loses ffn-up (3.40× < dense 4.43×) and attn (0.80×). The half-weight-DRAM benefit is real but the
  thin [out,tok] output forces either split-K (f32 atomic + convert overhead eats the savings) or
  too-few blocks (s=1 = 56, underfill). Dense adaptive decode (direct-bf16 split-N) is already too
  efficient to beat on ffn-up. Confirms sparse decode is a real capability but not a big lever; the
  half-weight win is capped by the reduction overhead the thin output forces. Prefill sparse remains
  the big sparse win (4–5×).

## Reproducibility

- Kernels: `cuda/matmul_sp_wide_swz2.cu` (unit sparse 2731k), `cuda/matmul_sp_full_wide.cu`
  (deployable sparse 2116k), `cuda/matmul_fp4_pp_bf16.cu` (dense 1503k), `cuda/dense_fp4_lib.cu`,
  `cuda/sparse_fp4_lib.cu` (PyTorch-callable + fused quantizer).
- Probes: `mma_peak`, `tma_bw`, `smemq`, `sp_*_probe`, `verify_*`, `pack_verify`, `pack_accuracy`.
- Harnesses: `leaderboard_fp4.py` (SM120 FP4 backend leaderboard: quadbit vs FlashInfer `mm_fp4`
  all backends, correctness-gated), `cutlass_sparse.py` (sparse vs CUTLASS 80b),
  `bench_vs_bf16.py` (throughput), `cutlass_fp4.py` (real CUTLASS FP4 baseline +
  SASS dissect), `quadbit_linear.py` (drop-in), `accuracy_hf.py` /
  `accuracy_sparse.py` (weight-recon), `perplexity_sparse.py` (end-to-end PPL),
  `sparsegpt_pair.py` (one-shot), `finetune_pair.py` (recovery).
- Full chronological build log (breakthroughs + tedium): memory `quadbit-raw-ptx.md`.

## Paper narrative arc (draft)

1. SM120 is the accessible Blackwell; FP4 hardware present, dense software now strong (FlashInfer
   `b12x`/`cutlass`), but **no library ships a sparse FP4 GEMM** and the block-scaled path has documented
   issues (#3096) → the gap is sparse + deployment, not dense. (NOT "software absent.")
2. Deriving the FP4 mma/ldmatrix/scale/metadata layouts from scratch on SM120 (no docs, probe-verify).
3. Dense FP4 to the silicon ceiling; the false-roofline lesson → wide-TMA+swizzle → sparse +36/42%.
4. The backend leaderboard: honest loss on dense (FlashInfer `b12x`/`cutlass` beat us 1.35–2.2×), and
   the reveal that sparse is the only deployed FP4 sparse path and beats the best dense in wall-clock →
   the Pareto point that is the paper's systems result.
5. Handling the documented pair-wise NVFP4 sparsity end-to-end, and quantifying its accuracy cost on
   existing element-2:4 checkpoints (cite the hardware spec; this is not a discovery).
6. Deployment: packer + fused quantizer + drop-in; 3.7–5.2× over bf16.
7. Making sparse usable: pair-granular SparseGPT + QAT recovery; the data-scale reality of 2:4.
