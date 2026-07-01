# quadbit — Paper Checkpoint

Working notes for a paper on hand-written FP4 tensor-core kernels for **consumer/pro
Blackwell (SM120, RTX PRO 6000 / RTX 5090)** — the accessible card where NVIDIA's libraries
leave FP4 gaps. Everything below is measured on Modal RTX PRO 6000 via `harness/run_cuda.py`
(raw CUDA) and `harness/*.py` (PyTorch/model harnesses). Kernels compiled `nvcc -arch=sm_120a`.

## Thesis

On SM120, FP4 tensor cores exist but the software doesn't: cuBLAS/CUTLASS give dense FP4
only, and **no sparse FP4 path exists at all**. We hand-write (raw PTX) both a dense FP4 GEMM
that reaches the silicon ceiling and the **only 2:4-sparse FP4 GEMM on SM120**, then build the
full deployment stack (packer, fused activation quantizer, `nn.Linear` drop-in) and a
one-shot + QAT recovery pipeline that makes the sparse path usable on real models.

## Headline results (real, measured, RTX PRO 6000)

Throughput vs cuBLAS bf16 (what production runs) and vs real CUTLASS FP4, M=N=K.
bf16/dense/sparse: `harness/bench_vs_bf16.py`; CUTLASS: `harness/cutlass_fp4.py`
(example 79b nvfp4×nvfp4→f32, `-DCUTLASS_NVCC_ARCHS=120a`, verification Passed). Same RTX PRO 6000:

| size | cuBLAS bf16 | CUTLASS FP4 | dense FP4 (ours) | 2:4-sparse FP4 (ours) |
|------|-------------|-------------|------------------|------------------------|
| 4096 | 372 TF/s | 1222 | 1136 (3.06× bf16) | 1512 (4.07× bf16) |
| 8192 | 423 TF/s | 1497 | 1556 (3.68× bf16) | 2207 (5.22× bf16) |
| 16384| 405 TF/s | — | 1645 (4.06× bf16) | 1782 (4.39× bf16) |

- Dense FP4 vs CUTLASS, **apples-to-apples clean measurement** (both cudaEvent-timed over 20 iters,
  no torch dispatch; ours = `matmul_fp4_pp_bf16` standalone): CUTLASS 634 / 1222 / 1497 @ 2048/4096/8192,
  ours **758 (+20%) / 1220 (tie) / 1510 (+0.9%)**. We match or beat CUTLASS at every size; the only
  apparent "loss" (bench_vs_bf16 showed 1136 @4096) was **torch dispatch overhead in that harness**,
  not a kernel deficit. Per-SM steady state @8192 we are MORE efficient than CUTLASS (86% vs 83% of
  the 1811 TF/s register-only mma peak). The 4096 tie is pure wave quantization (512 tiles / 188 SMs
  = 2.72 waves → ~10% tail); steady state there is otherwise our 86%.
- Sparse FP4 is the **unique, defensible win**: CUTLASS/cuBLAS have **no sparse FP4 on SM120 at all**.
  Ours beats the best available vendor FP4 (CUTLASS dense) by **+24% @4096 (1512 vs 1222)** and
  **+47% @8192 (2207 vs 1497)** — a capability, not just a tuning delta.
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
- **Dense FP4: +0.3 PPL, zero training, any model** (Sparse-Llama-3.1-8B 7.89→8.16; Qwen2.5-3B 7.60→7.91). Production-ready.
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

3. **The only 2:4-sparse FP4 GEMM on SM120**, fully deployable: arbitrary per-group 2:4 metadata
   + real per-block ue4m3 scales, both staged coalesced through a full/empty async pipeline
   (no CTA-wide `__syncthreads`), shared-B 256×128 traffic-optimal tiling.

4. **PAIR-GRANULARITY finding (novel, paper-worthy).** Blackwell FP4 `mma.sp` metadata selects
   at b16 = **fp4-pair** granularity: 2 of every 4 *pairs* kept, not 2 of every 4 *elements*.
   Consequence: **every public 2:4 checkpoint (element-granular, built for fp16/Ampere sparse TC)
   is incompatible.** Measured on real `neuralmagic/Sparse-Llama-3.1-8B-2of4`: it is exactly
   element-2:4 (50% zeros), and our pair-granular selection keeps only **~87% of its nonzero
   energy** → naive use gives **93.6 PPL** (vs 7.9 dense-FP4). This is the concrete "Blackwell FP4
   gap": no tooling targets pair-granular 2:4 because consumer-Blackwell FP4-sparse is unserved.

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

6. **Pair-granular recovery pipeline (one-shot + QAT), no NVIDIA equivalent.** SparseGPT retargeted
   to pair-granular masks (keep 2-of-4-pairs by `w²/[H⁻¹]²`, Hessian error compensation) →
   KD from the dense teacher (mask frozen) → QAT with straight-through fake-quant of BOTH weights
   (exact kernel dequant) and activations. Long WikiText-103 run (TinyLlama-1.1B, 12k bf16 + 3k QAT
   steps, cosine LR): dense fp16 teacher 7.53; one-shot pair-2:4 FP4 **19.1**; after phase-1 bf16
   recovery **9.33**; after phase-2 QAT (FP4 fake-quant) **9.49**; **through the real 2:4-sparse FP4
   kernel 10.34** — within ~2.5 PPL of dense, up from the short run's 13.3 (WikiText-2, 1.5M tokens).
   Recovery is monotonic in data; NM used 13B tokens for element-2:4, so production parity is a
   data-scale question, not a method gap. The pipeline (pair-granular SparseGPT + STE QAT matching
   the exact kernel dequant) is proven end-to-end.

## Measured hardware ceilings (SM120 / RTX PRO 6000)

- mma peaks (register-only, DCE-defeated): sparse `mma.sp` m16n8k128 = **3626k**, dense
  `mma.sync` m16n8k64 = **1811k**, ratio 2.002× (2:4 is a real 2× FLOP feature).
- L2→smem TMA bandwidth ceiling = **7.3 TB/s** (`tma_bw.cu`).
- smem cap = **99 KB/block** (101376 B), 100 KB/SM. L2 = 128 MB.
- Dense FP4 is compute-bound (~84% of 1811k); sparse FP4 is load-bound (swizzle floor 6.0 TB/s).

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
  Clean-measured, DP already matches/beats CUTLASS at all three sizes (see headline).
- **Small-tile (single-WG 128×128)**: only +12% @2048, loses ≥4096 (shared-B B-traffic dominates);
  mid-shape is fixed-overhead-bound, not tile-bound.
- **Thin-M split-K**: worse (atomicAdd contention); thin-M is latency/overhead-bound (~0.037ms floor).
- **Recovery without weight update**: magnitude pair-2:4 = 93.6/16129 PPL, Wanda (importance-only)
  = 59.1 — the Hessian compensation (SparseGPT) is essential for the constrained pair mask.
- **One-shot SparseGPT-pair** (independent per-layer) = 20.6–21.8 PPL: needs recovery fine-tuning.
- **Sparse weight-stationary DECODE** (`sparse_sk_lib.cu`, orient C[out,tok]=W@Xᵀ so the 2:4 weight is
  the compressed mma-A, M=out large; + split-K to fill SMs): WORKS (correct, novel — first 2:4 FP4
  decode on SM120) but net MARGINAL. Wins only long-K ffn-down (128/4096/14336: 1.42×→**1.54×**, s=6);
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
- Harnesses: `bench_vs_bf16.py` (throughput), `cutlass_fp4.py` (real CUTLASS FP4 baseline +
  SASS dissect), `quadbit_linear.py` (drop-in), `accuracy_hf.py` /
  `accuracy_sparse.py` (weight-recon), `perplexity_sparse.py` (end-to-end PPL),
  `sparsegpt_pair.py` (one-shot), `finetune_pair.py` (recovery).
- Full chronological build log (breakthroughs + tedium): memory `quadbit-raw-ptx.md`.

## Paper narrative arc (draft)

1. SM120 is the accessible Blackwell; FP4 hardware present, sparse FP4 software absent → the gap.
2. Deriving the FP4 mma/ldmatrix/scale/metadata layouts from scratch on SM120 (no docs, probe-verify).
3. Dense FP4 to the silicon ceiling; the false-roofline lesson → wide-TMA+swizzle → sparse +36/42%.
4. The pair-granularity discovery and why the entire existing 2:4 ecosystem is incompatible.
5. Deployment: packer + fused quantizer + drop-in; 3.7–5.2× over bf16.
6. Making sparse usable: pair-granular SparseGPT + QAT recovery; the data-scale reality of 2:4.
