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

Throughput vs cuBLAS bf16 (what production runs), `harness/bench_vs_bf16.py`, M=N=K:

| size | cuBLAS bf16 | dense FP4 (ours) | 2:4-sparse FP4 (ours) |
|------|-------------|------------------|------------------------|
| 4096 | 372 TF/s | 1136 (3.06×) | 1512 (4.07×) |
| 8192 | 423 TF/s | 1556 (3.68×) | 2207 (5.22×) |
| 16384| 405 TF/s | 1645 (4.06×) | 1782 (4.39×) |

- Dense FP4 = **84–91% of the 1811 TF/s hardware mma peak**; matches/edges CUTLASS FP4 (~1504).
  Both sit at the instruction ceiling — dense FP4 is a solved, ceiling-bound problem.
- Sparse FP4 = **+42% over the dense ceiling**, and a capability CUTLASS/cuBLAS do not have on SM120.
- Unit-scale headline (perf ceiling): sparse **2731k GFLOP/s**, dense **1515k**, both @8192.

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

5. **Deployment stack.** `QuadbitLinear` (`nn.Linear` drop-in): torch packer reproducing the
   kernel's exact metadata/compress/scale layout (verified maxrel 0.0039) + a **fused 128-bit
   NVFP4 activation quantizer** (one CUDA pass). End-to-end **4.0–4.2× over torch bf16** at 8192;
   SwiGLU FFN block 2×; any token count (padding). Real per-block ue4m3 weight scales
   (magnitude-independent: works at wscale=0.02).

6. **Pair-granular recovery pipeline (one-shot + QAT), no NVIDIA equivalent.** SparseGPT retargeted
   to pair-granular masks (keep 2-of-4-pairs by `w²/[H⁻¹]²`, Hessian error compensation) →
   KD from the dense teacher (mask frozen) → QAT with straight-through fake-quant of BOTH weights
   (exact kernel dequant) and activations. Trajectory (TinyLlama-1.1B): one-shot pair-2:4 FP4
   24.3 → QAT recovery 11.6 (fake-quant) / **13.3 through the real kernel**, on only **1.5M tokens**
   (NM used 13B for element-2:4). Recovery is monotonic/converging; production parity is a
   data-scale question, not a method gap. (Long WikiText-103 run in progress.)

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
- **Small-tile (single-WG 128×128)**: only +12% @2048, loses ≥4096 (shared-B B-traffic dominates);
  mid-shape is fixed-overhead-bound, not tile-bound.
- **Thin-M split-K**: worse (atomicAdd contention); thin-M is latency/overhead-bound (~0.037ms floor).
- **Recovery without weight update**: magnitude pair-2:4 = 93.6/16129 PPL, Wanda (importance-only)
  = 59.1 — the Hessian compensation (SparseGPT) is essential for the constrained pair mask.
- **One-shot SparseGPT-pair** (independent per-layer) = 20.6–21.8 PPL: needs recovery fine-tuning.

## Reproducibility

- Kernels: `cuda/matmul_sp_wide_swz2.cu` (unit sparse 2731k), `cuda/matmul_sp_full_wide.cu`
  (deployable sparse 2116k), `cuda/matmul_fp4_pp_bf16.cu` (dense 1503k), `cuda/dense_fp4_lib.cu`,
  `cuda/sparse_fp4_lib.cu` (PyTorch-callable + fused quantizer).
- Probes: `mma_peak`, `tma_bw`, `smemq`, `sp_*_probe`, `verify_*`, `pack_verify`, `pack_accuracy`.
- Harnesses: `bench_vs_bf16.py` (throughput), `quadbit_linear.py` (drop-in), `accuracy_hf.py` /
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
