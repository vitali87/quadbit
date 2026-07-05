# quadbit: Hand-Written FP4 Tensor-Core Kernels and a Deployment Stack for Consumer Blackwell (SM120)

**Draft v0.1.** Working paper. Every quantitative claim is measured on a Modal cloud
RTX PRO 6000 (SM120, no tcgen05) unless tagged otherwise, and traces to a harness in
`harness/` or a kernel in `cuda/`. Numbers here are copied from `docs/paper_notes.md` and
`docs/standing.md`; if they disagree, those two files are the source of truth and this draft
is stale.

---

## Abstract

Consumer and pro Blackwell cards (SM120: RTX PRO 6000, RTX 5090) ship FP4 tensor cores, but
the software that reaches them is thin. cuBLAS exposes dense FP4 only. CUTLASS ships a dense
(example 79b) and a sparse (example 80b) NVFP4 GEMM for SM120, but the block-scaled path has
documented correctness and autotuner problems in practice, and no library wraps a sparse FP4
kernel in a real deployment stack. SM120 also lacks the `tcgen05`/UMMA tensor-memory path that
the datacenter B200 (SM100) uses, so every FP4 GEMM here must run through warp-level
`mma.sync` and `mma.sp`.

We hand-write, in raw PTX compiled with `nvcc -arch=sm_120a`, a dense FP4 GEMM that reaches
the SM120 compute ceiling and a 2:4-sparse FP4 GEMM that reaches the SM120 bandwidth roofline,
having first derived the mma, `ldmatrix`, scale, and metadata bit-layouts empirically by
probe-and-verify (validated to relative error 0). We then build the deployment stack around
them: a weight packer, a fused NVFP4 activation quantizer, an `nn.Linear` drop-in, fused
transformer-block glue kernels, and a pair-granular one-shot-plus-QAT recovery pipeline.

Three results stand out. First, on the square sizes commonly benchmarked our dense kernel
wins or ties CUTLASS 79b (758/1220/1510 TF/s at 2048/4096/8192 vs 634/1222/1497), but on the
rectangular Llama-3-8B shapes that actually run it loses (0.89 to 1.01x); the square win does
not generalize. Second, our sparse kernel beats CUTLASS 80b on every rectangular LLM shape
(1.14 to 1.18x) and at 4K to 8K square, losing only at 16K square; the sparse win does
generalize, so the sparse path is the project's spine. Third, on accuracy the deployed dense
kernel is W4A4 (weights and activations both 4-bit, because the tensor core multiplies
fp4 by fp4), and with a per-16 two-level NVFP4 recipe using no calibration data it costs
+0.63 PPL on Llama-3.1-8B-Instruct, at or below the modelopt-calibrated reference. Sparse
recovery on a real 8B model now deploys at its trained accuracy, because we close the
fake-quant-to-kernel deploy gap with a two-level sparse kernel that still beats CUTLASS 80b, but
it still loses to dense by about 1.56 PPL and diverse-corpus data did not close that gap; sparse
is an honest speed-only Pareto point, not an accuracy win.

---

## 1. Introduction

FP4 is the smallest numeric format with tensor-core support on Blackwell, and on paper it
promises roughly 4x the throughput of bf16 and 4x smaller weights. On the datacenter B200
that promise is largely delivered by NVIDIA's own libraries through the `tcgen05` UMMA path.
On the consumer and pro cards that most people can actually buy, SM120, the situation is
different: the tensor cores are present, the UMMA path is absent, and the library coverage is
partial and in places buggy. That gap is the subject of this paper.

We set out to answer a concrete question: on SM120, can a hand-written kernel plus a real
deployment stack turn FP4 into a usable speedup on real language models, and what does 4-bit
actually cost in accuracy once you account for the fact that the hardware quantizes
activations too? Answering it required deriving the low-level layouts ourselves (the SM120 FP4
operand, scale, and 2:4 metadata layouts are not documented at the bit level), building both a
dense and a sparse kernel to their respective hardware ceilings, and then quantifying accuracy
on real checkpoints rather than asserting it.

Our contributions:

1. **Empirically derived SM120 FP4 layouts.** The dense block-scaled `mma.sync` and sparse
   `mma.sp::ordered_metadata` operand, scale, and metadata bit-layouts, recovered by
   probe-and-verify and validated to relative error 0 (Section 3).
2. **A dense FP4 GEMM at the compute ceiling** and the false-roofline lesson that a probe whose
   own load pattern is the limit will lie about the hardware ceiling (Section 4).
3. **A 2:4-sparse FP4 GEMM at the bandwidth roofline**, via a wide-TMA-plus-swizzle design that
   lifted sparse throughput +36% over a false intermediate ceiling, and that beats CUTLASS 80b
   on the shapes that ship (Section 5).
4. **A deployment stack**: packer, fused activation quantizer, `nn.Linear` drop-in, and fused
   transformer-block glue kernels that convert the raw GEMM wins into end-to-end block speedups
   of 2 to 5.8x over eager bf16 (Section 6).
5. **An honest accuracy accounting**: dense FP4 is W4A4 and costs +0.63 PPL with no
   calibration, matched to the modelopt reference; the widely quoted +0.3 is a weight-only
   W4A16 number that the hardware never runs (Section 7).
6. **A pair-granular recovery pipeline** and a clear-eyed report that on a real 8B model sparse
   recovery is currently data-limited and does not yet beat dense (Section 8).

---

## 2. Background: FP4 on SM120

**The formats.** NVFP4 encodes values as E2M1 (the 4-bit `{0, .5, 1, 1.5, 2, 3, 4, 6}` grid
with sign) in blocks of 16, with a two-level scale: a per-16 local scale in `ue4m3` and a
per-row fp32 global scale, where the global rescales the local codes into `ue4m3`'s precise
range (global = rowamax / 2688, and 2688 = 448 x 6 is the product of the `ue4m3` and E2M1
maxima). MXFP4 uses blocks of 32 with a single power-of-two `ue8m0` scale; because that scale
carries no mantissa, it has no per-block rounding bias, which matters for accuracy (Section 7).

**What the tensor core runs.** The FP4 mma multiplies fp4 by fp4. There is no weight-only FP4
GEMM in silicon. So the deployed kernel is always W4A4: both operands are 4-bit. Any accuracy
number that keeps activations at 16-bit (W4A16) describes a kernel that does not exist on this
hardware; it is a dequant-then-bf16-matmul path (Marlin-class), not an FP4 GEMM. This
distinction drives the entire accuracy discussion.

**No tcgen05.** SM100 (B200) accumulates in tensor memory via UMMA. SM120 has none of that;
the only route to the FP4 tensor cores is warp-level `mma.sync` (dense) and `mma.sp`
(2:4-sparse), with accumulators living in the register file. This is the structural constraint
behind every kernel decision below.

**Measured hardware ceilings (RTX PRO 6000).** Register-only mma peaks (dead-code-elimination
defeated): sparse `mma.sp` m16n8k128 = 3626k GFLOP/s, dense `mma.sync` m16n8k64 = 1811k, a
ratio of 2.002x (2:4 is a real 2x FLOP feature). L2-to-smem TMA bandwidth ceiling = 7.3 TB/s.
Peak DRAM bandwidth = 1.46 TB/s. Shared memory = 99 KB per block, 100 KB per SM; L2 = 128 MB.
Dense FP4 is compute-bound at roughly 84% of the mma peak; sparse FP4 is load-bound against a
swizzle floor near 6.0 TB/s.

---

## 3. Deriving the FP4 layouts

None of the SM120 FP4 bit-layouts we needed are documented, so we recovered them by
constructing inputs with known structure, running the mma, and checking which arrangement
reproduced the reference to relative error 0.

- **Scale layout.** For the dense block-scaled mma, `scaleA[row r][kb]` maps to lane
  `(r & 7) * 4 + (r >> 3)`, byte `kb`; `scaleB[col c][kb]` maps to lane `c * 4`, byte `kb`.
- **2:4 metadata.** For `mma.sp`, lane `L` maps to mma-row `(L & 1) * 8 + (L >> 2)`, half
  `(L >> 1) & 1`, with the kept-pair nibble encoded `idx0 | (idx1 << 2)`.
- **NVFP4 dense scale.** The `scale_vec::4X` `ue4m3` case uses the same A/B row-to-lane mapping
  as the `ue8m0` `2X` case, just with a 4-byte (four per-16) scale register. We confirmed this
  with a wide-range scale probe (2^-4 to 2^2), which is the strong test: a uniform-scale probe
  cannot tell whether the four scale bytes map to the correct k-sub-blocks; the wide-range one
  can. Result: relative error 0.

The `ldmatrix` story is a negative result worth recording. The only sub-byte `ldmatrix` ptxas
accepts on `sm_120a` is the format-converting `m8n16`/`m16n16 .b8x16.b4x16_p64`, which expands
each 4-bit value into its own byte. Our mma wants packed `e2m1x2`, so the expanded fragment is
the wrong shape and would need 2x the registers to hold a full operand, which is fatal at the
255-register ceiling. We patched CubeCL to emit the instruction and confirmed it assembles and
runs; it is simply the wrong tool for this kernel.

---

## 4. Dense FP4 to the compute ceiling

The dense kernel stages packed `e2m1x2` tiles through shared memory and issues a wide 2x8 grid
of `mma.sync` instructions per warp, feeding 16 independent f32 accumulator chains back in as C
so products accumulate across K. Those 16 chains are the instruction-level parallelism that
hides the FP4 mma (OMMA) latency; the kernel is latency-and-ILP-bound, not occupancy-bound.
Widening the warp tile from 2x2 to 2x4 to 2x8 climbed roughly 160k, 230k, 253k GFLOP/s, and
2x8 uses exactly 255 registers per thread with zero spills, saturating the register file (8
warps x 255 = 65280 of 65536 registers per SM). This is the classic "better performance at
lower occupancy" regime.

**The false-roofline lesson.** An early mem-only probe suggested a ceiling that turned out to
be an artifact of the probe's own narrow load pattern, which extracted only 54% of the
L2-to-smem bandwidth. The lesson, which recurs throughout the project: never declare a memory
roofline from a probe whose own load pattern is the limit.

**Versus CUTLASS 79b.** On square sizes, apples-to-apples (both cudaEvent-timed over 20
iterations, no torch dispatch), our dense kernel reaches 758 / 1220 / 1510 TF/s at 2048 / 4096
/ 8192 versus CUTLASS 79b's 634 / 1222 / 1497: a win, a tie, and a win. But the square win does
not generalize. On the rectangular Llama-3-8B GEMM shapes that actually run, we lose to 79b:
attention 4096-cubed at 0.89x, ffn-up (N=14336) at 0.93x, ffn-down (K=14336) at 1.01x, all
79b-verified. CUTLASS's tile and schedule adapt to rectangular shapes better than our fixed
tiling. The honest dense story is therefore "competitive, slightly behind CUTLASS on shipping
shapes," while still delivering 3.0 to 3.7x over cuBLAS bf16.

We built and measured three persistent/stream-K variants to try to beat data-parallel on the
rectangular shapes (stream-K, a true CUTLASS-style persistent cross-tile pipeline, and split-K
via an f32 global reduction). All three regressed. The root cause is register pressure: the
128-register FP4 accumulator tile plus a software scheduler's cursor state pins the kernel near
254 registers and throttles the mma stream, while the hardware block scheduler already overlaps
consecutive tiles for free. On SM120 FP4, the huge accumulator tile leaves no register headroom
for a software scheduler, so data-parallel is at the practical ceiling.

---

## 5. Sparse FP4 to the bandwidth roofline

Sparse FP4 is load-bound, so the kernel win is a memory-traffic win. The design stages
arbitrary per-group 2:4 metadata and real per-block `ue4m3` scales coalesced through a
full/empty async pipeline (no CTA-wide `__syncthreads`), on a shared-B 256x128 traffic-optimal
tiling.

**The wide-TMA-plus-swizzle breakthrough.** A mem-only probe suggested a "2012k roofline"; it
was false for the same reason as Section 4, narrow TMA boxes extracted only 54% of the 7.3 TB/s
L2-to-smem ceiling. Loading WK=2 k128-slices per TMA (wider boxes) hits the ceiling, but wide
smem rows cause `ldmatrix` bank conflicts. Swizzled TMA fixes that (A box 64B with
`SWIZZLE_64B` and an `ldmatrix` XOR `off ^= ((off >> 7) & 3) << 4`; B box 128B with
`SWIZZLE_128B` and XOR `((off >> 7) & 7) << 4`). Result: unit-scale sparse went from 2012k to
2731k (+36%), and the deployable kernel from 1486k to 2116k (+42%).

**Versus CUTLASS 80b.** Correctness-gated on 80b's own reference check (which passes at every
size, so CUTLASS issue #3096's block-scaled bug does not affect this comparison). On square
sizes we win at 4096 (1.16x) and 8192 (1.14x) and lose at 16384 (0.96x, 1859 vs 1785 TF/s). On
the rectangular Llama-3-8B shapes where dense loses, sparse wins consistently: attention
4096-cubed at 1.18x, ffn-up at 1.14x, ffn-down at 1.17x, all 80b-verified. Unlike dense, the
sparse win generalizes to shipping shapes, which is why the sparse path is the project's spine.

**Ceiling honesty.** Sparse's real advantage over our own dense FP4 is roughly 1.33x at the
roofline (2012k vs 1510k deployable at 8192), not the 2x the mma FLOP ratio suggests. The
hardware 2x is a datacenter-bandwidth feature we cannot reach on SM120. At 1.33x, the accuracy
cost of sparsity has to be small to justify the recovery pipeline over dense, which is exactly
the tension Section 8 confronts.

**Throughput summary (M=N=K, vs cuBLAS bf16):**

| size | cuBLAS bf16 | CUTLASS FP4 | dense FP4 (ours) | 2:4-sparse FP4 (ours) |
|------|-------------|-------------|------------------|------------------------|
| 4096 | 372 TF/s | 1222 | 1136 (3.06x bf16) | 1512 (4.07x bf16) |
| 8192 | 423 TF/s | 1497 | 1556 (3.68x bf16) | 2207 (5.22x bf16) |
| 16384| 405 TF/s | n/a | 1645 (4.06x bf16) | 1782 (4.39x bf16) |

---

## 6. Deployment stack and operator fusion

The raw GEMMs are at the silicon ceiling, so every remaining end-to-end gain came from fusing
the glue between GEMMs, removing the memory round-trips a transformer block otherwise pays in
eager mode.

- **`QuadbitLinear`** is an `nn.Linear` drop-in: a torch packer reproduces the kernel's exact
  metadata, compression, and scale layout (verified maxrel 0.0039), fed by a fused 128-bit
  NVFP4 activation quantizer in a single CUDA pass. End-to-end it is 4.0 to 4.2x over torch
  bf16 at 8192, for any token count.
- **Fused SwiGLU FFN.** Quantize the activation once (shared across gate and up), concatenate
  gate and up into one GEMM, and fuse the epilogue (read gate and up, compute silu(gate)*up,
  emit the FP4-packed down-proj input and scales in one transposing pass). Cumulative versus
  torch bf16: unfused 2.05x, plus fused epilogue 4.45x, plus concat 4.66x at batch 2048 (2.27x
  over unfused), and 1.74x to 2.95x at batch 512. Numerically identical to the unfused path; a
  pure memory-traffic win.
- **Fused RMSNorm-plus-quant** (`rmsnorm_quant_k`): one read of x replaces eager rmsnorm's
  read-reduce-write plus a separate quant pass, and it is more accurate (no bf16 round-trip
  before quant). 3.7 to 4.3x over eager rmsnorm-plus-quant.
- **Fused residual-add-plus-RMSNorm-plus-quant** (`add_rmsnorm_quant_k`): the full block
  transition in one kernel. 5.3 to 5.8x over the eager sequence.

Every inter-GEMM memory round-trip in a transformer block is now a single fused pass, at zero
accuracy cost (the FP4 and prune floors are preserved).

**On real models.** Every projection of the current frontier models tiles onto the kernel
(all hidden sizes are multiples of 256 and head_dim is 128): we ran the real linear shapes of
Qwen3.5-397B, GLM-5.2, MiniMax-M3, and DeepSeek-V3/R1 through the kernel, all all-tile-ok. A
full fused dense FP4 decoder block on real Qwen3-8B weights, with no training, runs 2.16 to
2.19x over bf16 at block reconstruction rel 0.13.

---

## 7. Accuracy: dense FP4 is W4A4

Because the tensor core multiplies fp4 by fp4, the accuracy question is W4A4, not W4A16. With a
per-16 two-level NVFP4 recipe using amax and no calibration data, measured through
`harness/recovery_worth.py` on WikiText-2:

- Llama-3.1-8B-Instruct: 7.27 to 7.90, **+0.63 PPL**.
- Meta-Llama-3-8B base: 6.20 to 6.91, **+0.71 PPL**.
- Reference: vLLM native NVFP4 (modelopt-calibrated) is 7.97, +0.71 on the same windows.

So our own amax recipe, with no calibration, lands at or below the calibrated reference. The
earlier "+2 PPL" figure was a crude per-32 single-level recipe, not W4A4's real cost; the gap
was block granularity and two-level activation scaling, not calibration. The often-quoted +0.3
PPL is a weight-only W4A16 number that the hardware never runs, and we do not headline it.

At the block-reconstruction level, two-level NVFP4 reaches rel 0.097 on real Qwen3-8B with no
training, versus 0.13 for the simpler and faster MXFP4 (`ue8m0`) path. NVFP4 is the accuracy
path; MXFP4 is the speed path (2.15x over bf16). The single-level NVFP4 block regressed to 0.38
because the `ue4m3` scale carries a mantissa and its per-block rounding bias accumulates across
the three-matmul chain; the two-level recipe (per-row fp32 global rescaling the per-16 `ue4m3`
locals) is what fixes it, since MXFP4's power-of-two scale has no such bias.

> **In progress (not yet in the committed docs).** A training-free, memory-free mixed-precision
> refinement keeps the most activation-sensitive layers at W4A16 and the rest at W4A4. Selected
> cleanly (ranked on decontaminated C4, scored on held-out WikiText-2 to avoid selection-on-test
> overfit), keeping the top-8 of 32 layers at W4A16 brings the base-model cost from +0.71 down to
> roughly +0.60. This is pending the minimal-K sweep (deploy the smallest K that crosses the
> target, since each bf16-activation layer is prefill compute you pay for) and will land in the
> headline once the concurrent recovery ablation converges and the docs re-sync.

---

## 8. Sparse recovery and the two-level deploy fix

Blackwell FP4 2:4 is pair-granular: the `mma.sp` metadata selects at fp4-pair granularity (2
of every 4 pairs kept, not 2 of every 4 elements). This is NVIDIA's documented hardware spec,
not a discovery of ours. The consequence we do contribute as a data point: on an existing
element-2:4 checkpoint (`neuralmagic/Sparse-Llama-3.1-8B-2of4`), pair-granular selection keeps
only about 87% of the nonzero energy, so naive reuse gives 93.6 PPL versus 7.9 for dense FP4.
Element-2:4 tooling and pair-granular hardware are not interchangeable.

Our recovery pipeline retargets SparseGPT to pair-granular masks (keep the two pairs of four
by `w^2 / [H^-1]^2` with Hessian error compensation), then distills from the dense teacher with
the mask frozen, then runs QAT with straight-through fake-quant of both weights (exact kernel
dequant) and activations. On TinyLlama-1.1B this recovers to 9.60 PPL through the real sparse
FP4 kernel (dense teacher 7.53), and closing the STE-versus-kernel gap (matching the QAT
activation fake-quant to the deployed quantizer bit-for-bit) was worth about 0.43 PPL.

**The deploy gap and its fix.** Recovery trains against a two-level fake-quant: a per-row and
per-column fp32 global rescaling the per-16 `ue4m3` local scale, the same move that took dense
NVFP4 from 0.38 to 0.097 block-reconstruction error. The originally deployed sparse kernel
applied only the local scale, so its outputs diverged from what QAT trained against. We measured
that deploy gap directly, running three matmuls of one recovered Meta-Llama-3-8B checkpoint
through each path: the single-level kernel deploys at 11.89 PPL, the two-level kernel at 8.95,
against a fake-quant target of 8.96. The single-level path flips 22% of top-1 next-token
predictions relative to the target; the two-level path reproduces it (mean NLL delta 0.002, 92%
top-1 agreement). We built the missing piece, a per-row and per-column fp32 global rescale in
the sparse mma epilogue ported from the working dense two-level path. Through the two-level
sparse kernel the recovered checkpoint tracks its fake-quant within 0.02 PPL: the deploy gap is
closed. The rescale costs 2 to 10% of throughput (worst on wide-N shapes) and the two-level
sparse kernel still beats CUTLASS 80b on every shape (1.01 to 1.13x, all correctness-verified).

**The accuracy number, honestly deployable.** The deployed sparse mma constrains activations to
per-32 granularity on the B operand, so the honest recipe matches the QAT activation fake-quant
to per-32 rather than the per-16 the fake-quant ceiling used. With that match, Meta-Llama-3-8B
recovers to 8.47 PPL through the two-level kernel, equal to its fake-quant (deploy gap closed
end to end). Dense W4A4 zero-training on the same model is 6.91, so deployable sparse loses to
dense by about 1.56 PPL. Extending phase-2 QAT from 2k to 5k steps did not help (it regressed to
8.96, mild overtraining against the WikiText-2 test), so 2k is the recovery sweet spot on this
corpus.

**The data lever did not pay.** To test whether the residual gap was data-limited, we ran a
full-scale diverse-corpus recovery (decontaminated C4, phase-1 to about 196M tokens). Phase-1
flattened at 10.82 PPL on the WikiText-2 test, far above the in-distribution WikiText-103
phase-1 (8.57), because C4 is out of distribution for that narrow test. Diverse-corpus scale did
not improve the WikiText-2 number, so the recovery is in-distribution-bound on this metric, not
simply data-starved. This is a negative result for the diverse-data hypothesis.

The honest position: dense FP4 (+0.63, zero-training) is the accuracy result. Sparse is a speed
play (about 1.33x over our own dense) that now deploys at its trained accuracy, since the
two-level kernel closes the deploy gap, but it still loses to dense by about 1.56 PPL on
recovery and diverse data did not close that gap on the WikiText-2 metric. Sparse is an honest
Pareto point for throughput-bound serving, not an accuracy win.

---

## 9. End-to-end serving comparison (RTX PRO 6000)

We ran real serving engines on the same card and model family, same protocol, to place quadbit
against what a practitioner would actually deploy. All rows: Llama-3.1-8B-Instruct family, CUDA
graphs on (`enforce_eager=False`), distinct per-request prompts (identical prompts collapse under
prefix caching and inflate prefill), prefill = B prompts of S=2048 with 1 output token, decode =
B short prompts with GEN=128 (`ignore_eos`), WikiText-2 PPL on the same 16x2048 windows. Memory
is the loader-reported weight footprint; both engines additionally pre-allocate a KV pool to
their utilization fraction, so total device reservation is about 86 to 89 GiB regardless of quant.

**Table A: real serving engines** (paged attention, continuous batching, decode scheduler).

| engine | quant | weights | WT-2 PPL | prefill tok/s (B=1/8/32/64) | decode tok/s (B=1/8/32/64) |
|--------|-------|---------|----------|-----------------------------|----------------------------|
| vLLM 0.21 | bf16 | 15.0 GiB | 7.267 | 10209 / 26436 / 31559 / 46880 | 88 / 690 / 2599 / 4947 |
| vLLM 0.21 | NVFP4 (modelopt, cutlass) | 5.66 GiB | 7.974 | 13530 / 63056 / 78028 / 116831 | 131 / 1049 / 4259 / 8465 |
| SGLang 0.5 | NVFP4 (auto → FlashInfer CUTLASS `fp4_gemm`) | ~5.6 GiB | 7.97 (same ckpt) | 16829 / 59769 / 73414 / 109002 | 186 / 1491 / 5424 / 10145 |

Native NVFP4 W4A4 is real on SM120 through both engines: vLLM binds the `modelopt_fp4` cutlass
method (not a Marlin W4A16 fallback), and SGLang's `auto` backend selects the FlashInfer CUTLASS
`fp4_gemm` path (autotuned at startup, visible in its log). Against bf16, NVFP4 is 1.7x prefill
and 1.7x decode at B=64 (vLLM), for 2.6x smaller weights and +0.71 PPL. SGLang and vLLM NVFP4
trade the lead: SGLang wins decode at every batch (10145 vs 8465 at B=64) and low-batch prefill,
vLLM wins high-batch prefill (116831 vs 109002 at B=64).

**Table B: quadbit dense FP4 prototype** (full-forward, PREFILL-ONLY, no decode engine).

| build | quant | quantized-linear weights | through-kernel WT-2 PPL | full-model prefill tok/s |
|-------|-------|--------------------------|-------------------------|--------------------------|
| quadbit dense two-level | W4A4 (per-16 ue4m3 + fp32 global) | 3.93 GiB* | 7.899 | 7815 (B=8, S=2048) |

*3.93 GiB counts only the transformer-block linears swapped to the FP4 kernel; embeddings,
lm_head, and attention stay bf16 and are not counted, so this is not a full-model on-device
footprint comparable to Table A's loader figures.

Table B is deliberately not in Table A. quadbit ships a kernel plus an `nn.Linear` drop-in, not a
serving stack: no paged attention, no continuous batching, no CUDA graphs, no decode scheduler,
and attention runs in HuggingFace eager bf16. The prefill number is GEMM-plus-activation-quantizer
bound over an eager Python forward, so it trails the serving engines' prefill by roughly an order
of magnitude and must not be read as a serving-throughput result. What Table B does establish is
that the deployed W4A4 path is accuracy-correct end to end: through-kernel PPL 7.90 matches vLLM's
native NVFP4 7.97 to within 0.07 on the same windows, confirming quadbit's zero-calibration
two-level recipe reaches the calibrated reference's accuracy. A real serving-engine integration
(the number that would put quadbit in Table A) remains future work.

---

## 10. Related work

**CUTLASS** ships dense (79b) and sparse (80b) NVFP4 GEMMs for SM120 since 3.9.0. These are our
true baselines and we benchmark against both. The opening is not that a reference kernel is
absent but that the SM120 block-scaled path has documented correctness and autotuner problems
(CUTLASS issue #3096: grouped GEMM garbage output, TMA warp-specialized tactics failing to
init) and that no library wraps a sparse FP4 kernel in a deployment stack.

**Deployed inference on SM120.** The Marlin W4A16 fallback (dequant FP4 to bf16 in-kernel,
forfeiting the FP4 FLOPS) was reported around 50 tok/s on a 397B MoE, but for the dense
Llama-3.1-8B NVFP4 checkpoint we measure both vLLM 0.21 and SGLang 0.5 binding the *native* NVFP4
W4A4 cutlass path on this card (Section 9: vLLM `modelopt_fp4`, SGLang FlashInfer CUTLASS
`fp4_gemm`), not Marlin. Against those real engines our through-kernel W4A4 accuracy matches
(PPL 7.90 vs 7.97); the remaining gap is serving-stack engineering, not the kernel.

**Datacenter FP4 (B200, SM100, not our card).** SGLang/FlashInfer/vLLM FP4 MoE reach roughly
1000 to 1260 TFLOPS using `tcgen05`/UMMA that SM120 does not have. We do not compare our SM120
numbers to these; the silicon differs.

**Quantization and sparsity recovery.** NVIDIA's NVFP4-QAD (arXiv 2601.20088) is the
quant-recovery equivalent and recovers over 95% of FP accuracy on real Nemotron/Llama models;
our sparse recovery is not yet in that league. SparseGPT, and the adjacent SQ-format and SharQ
lines, are the sparse-plus-quant prior art we build on and must be distinguished from; our
specific move is retargeting the mask to pair-granular 2:4 for the FP4 sparse path.

---

## 11. Limitations

- **No quadbit serving-engine integration yet.** Section 9 now measures the serving baselines
  (vLLM bf16, vLLM NVFP4, SGLang NVFP4) and quadbit's full-forward prefill against them, and
  confirms quadbit's through-kernel accuracy (PPL 7.90) matches native NVFP4 (7.97). What is still
  missing is quadbit *inside* a real serving stack (paged attention, continuous batching, decode
  scheduler); the prefill-only prototype trails serving prefill by about an order of magnitude
  because it runs an eager forward with bf16 attention, not because of the GEMM. This is the
  single most valuable remaining engineering step.
- **Sparse recovery loses to dense on accuracy** (Section 8). The deployable sparse-recovered 8B
  is 8.47 PPL through the two-level kernel, about 1.56 above dense W4A4 (6.91), and diverse-corpus
  data (C4) did not close it on the WikiText-2 test. Sparse is speed-only on accuracy grounds.
- **The deploy gap is now closed.** The two-level sparse kernel (per-row and per-column fp32
  global rescale in the epilogue) makes the deployed accuracy equal the trained fake-quant
  (8.47 == 8.47; the old single-level kernel would deploy the same checkpoint at 11.89). The fix
  costs 2 to 10% of throughput and still beats CUTLASS 80b.
- **Dense loses to CUTLASS on rectangular shapes.** We win square, but the shapes that ship
  favor CUTLASS's adaptive tiling.

---

## 12. Conclusion

On SM120, FP4 is a real speedup but only through kernels and a stack that the ecosystem does not
yet provide. We hand-wrote both, took the dense kernel to the compute ceiling and the sparse
kernel to the bandwidth roofline, and wrapped them in a packer, quantizer, drop-in, and fused
block. The sparse kernel beats CUTLASS 80b on the shapes that ship and is the project's spine;
dense is a competitive zero-training drop-in whose real accuracy cost, correctly measured as
W4A4, is +0.63 PPL with no calibration. Sparsity does not earn its recovery pipeline over dense
on accuracy: the deployable sparse-recovered 8B is 8.47 PPL, about 1.56 above dense W4A4, and
diverse-corpus data did not close that gap on our metric. We closed the sparse deploy gap with a
two-level kernel so the deployed accuracy equals the trained accuracy, and we report sparse as an
honest speed-only Pareto point rather than an accuracy win.

---

## Appendix: reproducibility

- **Kernels.** `cuda/matmul_sp_wide_swz2.cu` (unit sparse 2731k), `cuda/matmul_sp_full_wide.cu`
  (deployable sparse 2116k), `cuda/matmul_fp4_pp_bf16.cu` (dense 1503k), `cuda/dense_fp4_lib.cu`,
  `cuda/sparse_fp4_lib.cu` (PyTorch-callable plus fused quantizer).
- **Probes.** `mma_peak`, `tma_bw`, `smemq`, `sp_*_probe`, `verify_*`, `pack_verify`,
  `pack_accuracy`.
- **Harnesses.** `bench_vs_bf16.py` (throughput), `cutlass_fp4.py` / `cutlass_sparse.py` /
  `cutlass_shapes.py` (CUTLASS baselines), `quadbit_linear.py` (drop-in), `recovery_worth.py`
  (dense W4A4 accuracy), `sparsegpt_pair.py` (one-shot), `finetune_pair.py` (recovery),
  `vllm_nvfp4.py` (vLLM SM120 NVFP4 smoke test).
- **Full chronological build log.** Memory `quadbit-raw-ptx.md`.
</content>
</invoke>
