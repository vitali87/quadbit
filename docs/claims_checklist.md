# Claims checklist

Every load-bearing claim in `docs/paper.md`, the evidence file or number that backs it, and its status.
Status is one of: **backed** (a doc/CSV/harness in-repo supports the exact number), **needs-rerun** (the
result exists but the artifact is stale, on-volume-only, or not re-verified at the committed commit), or
**reserved** (in progress, no number claimed). No claim below may be promoted to backed without the cited
source reproducing the number.

Card: Modal RTX PRO 6000 (SM120) throughout. Recovered checkpoint:
`/cache/recovered_Llama-3.1-8B-Instruct_P30000_p25000_2sh_lr3e-05.pt`.

---

## 1. Dense FP4 is table stakes, not the headline

| Claim | Evidence | Status |
|-------|----------|--------|
| FlashInfer `b12x`/`cutlass` beat quadbit deployed dense by 1.35 to 2.2x on SM120 | `harness/leaderboard_fp4.py`; `docs/paper.md` Section 4 table (sq8192 1045 vs 1433; attn 838 vs 1283; ffn-up 936 vs 1374; ffn-down 1017 vs 1408; M=65536 639 vs 1416) | backed |
| FlashInfer `cudnn` fails every SM120 shape (cuDNN 9.10 < 9.14) | `harness/leaderboard_fp4.py`; `docs/standing.md` | backed |
| `b12x` collapses ~2.2x at M>=65536, only `cutlass` holds ~1400 | `harness/leaderboard_fp4.py`; `docs/paper.md` Section 4 | backed |
| quadbit dense wins/ties/wins CUTLASS 79b on square, loses on rectangular LLM shapes | `harness/cutlass_fp4.py`, `harness/cutlass_shapes.py`; `docs/paper.md` Section 4 | backed |
| Dense FP4 is 3.0 to 3.7x over cuBLAS bf16 on prefill shapes | `harness/bench_vs_bf16.py`, `harness/bench_llm_shapes.py` | backed |
| Async scale prefetch lifts deployed two-level dense 1.08 to 1.22x, maxrel 0 | `cuda/dense_nvfp4_fast_lib.cu`; `docs/paper_notes.md` (sq8192 865->1055) | backed |

## 2. Sparse FP4 GEMM is the kernel Pareto result

| Claim | Evidence | Status |
|-------|----------|--------|
| quadbit is the only deployed 2:4-sparse FP4 GEMM on SM120 (FI/SGLang/vLLM ship none; CUTLASS 80b is an unwrapped example) | prior-art sweep in `docs/standing.md`, `docs/paper_notes.md`; sources listed there | backed |
| Sparse two-level beats CUTLASS 80b every shape 1.01 to 1.12x | `harness/cutlass_sparse.py`; `docs/paper.md` Section 5 (attn 1.08x, ffn-up 1.01x, ffn-down 1.12x, sq8192 1.09x = 1973 vs 1807) | backed |
| Sparse two-level beats best FlashInfer dense in wall-clock on every prefill shape 1.07 to 1.38x | cross-table `harness/cutlass_sparse.py` (CUDA 12.8) vs `harness/leaderboard_fp4.py` (CUDA 13), same card; `docs/paper.md` Section 5 (attn 0.100 vs 0.107; ffn-up 0.301 vs 0.350; ffn-down 0.265 vs 0.342; sq8192 0.557 vs 0.767) | backed |
| Wide-TMA-plus-swizzle lifted unit sparse 2012k->2731k (+36%), deployable 1486k->2116k (+42%) | `cuda/matmul_sp_wide_swz2.cu`, `cuda/matmul_sp_full_wide.cu`; `docs/paper.md` Section 5 | backed |
| Sparse advantage over quadbit's own dense is ~1.33x at roofline, not 2x | `docs/paper.md` Section 5 (2012k vs 1510k @8192) | backed |
| Speed-path ceiling table (bf16/CUTLASS/dense/sparse at 4096/8192/16384) | `harness/bench_vs_bf16.py`, `harness/cutlass_fp4.py`; `docs/paper.md` Section 5 | backed |

## 3. Graph-captured split-K decode win

| Claim | Evidence | Status |
|-------|----------|--------|
| Sparse MLP is a `torch.library` custom op inside vLLM fullgraph + CUDA-graph capture | `harness/quadbit_serve.py` `_install_graph_customop`; `docs/graph_serving_result.md` | backed |
| Sparse path provably ran under graphs: SPARSE_CALLS=7264, PPL 10.2709 (not 7.97) | `docs/graph_serving_result.md` | backed |
| Decode win +9.7/+7.2/+2.2% at B=8/32/64 vs production NVFP4 | `harness/quadbit_serve.py --graph --splits 8`; `docs/graph_serving_result.md` (1147/4543/8567 vs 1046/4237/8384) | backed |
| Prefill trails -5.3/-4.0/-3.4% at B=8/32/64 | `docs/graph_serving_result.md` (62914/77605/115069 vs 66469/80825/119083) | backed |
| Split-K down: 16->128 CTAs, 109->56.5 us (1.93x) at split=8, cos 1.0000 | `harness/quadbit_serve.py --mode profile_decode`; `docs/graph_serving_result.md` | backed |
| splits=8 beats 4 and 16 end-to-end | `docs/graph_serving_result.md` split-factor sweep | backed |

## 4. End-to-end serving crossover

| Claim | Evidence | Status |
|-------|----------|--------|
| Sparse wins 81/112 total-latency regimes outright plus 2 ties vs production dense NVFP4 | `harness/quadbit_serve.py --graph --crossover`; `docs/crossover_result.md`; `/cache/crossover_{nvfp4,sparse}.csv` | backed |
| B=1 single-stream wins every cell (+3.5% to +11.6%) | `docs/crossover_result.md` B=1 heatmap | backed |
| NVFP4 keeps only the batch-prefill corner (high B, long prompt, short gen, <=3%) | `docs/crossover_result.md` B=64 heatmap and boundary table | backed |
| Prefix caching must be OFF or the result is corrupted (spurious 112/112) | `docs/crossover_result.md` method note | backed |
| Accuracy tax constant at +2.3 PPL (10.27 vs 7.97) across the whole map | `docs/crossover_result.md`, `docs/graph_serving_result.md` | backed |
| Real serving engine baselines (vLLM bf16/NVFP4, SGLang NVFP4) | `harness/vllm_nvfp4.py`, `harness/sglang_fp4.py`; `docs/paper.md` Section 9 Table A | backed |
| quadbit dense prototype through-kernel PPL 7.90 matches native NVFP4 7.97 within 0.07 | `harness/dense_e2e.py`, `harness/recovery_worth.py`; `docs/paper.md` Section 9 Table B | backed |

## 5. Track 4C (phase-adaptive same-weight) refuted; other negatives

| Claim | Evidence | Status |
|-------|----------|--------|
| Track 4C semantics hold (dense NVFP4 over recovered weights PPL 10.30 == all-sparse 10.27) | `harness/quadbit_serve.py --phase-adaptive`; `docs/crossover_result.md` Section 4C | backed |
| Track 4C loses: 39 win / 66 loss of 105 cells vs all-sparse 81/29 | `docs/crossover_result.md` Section 4C; `/cache/crossover_phaseadaptive.csv` | backed |
| Hand-rolled dense MLP ~2x native NVFP4 (unfused activation quant, 517 us down-input quant at M=2048) | `--mode phase_bench`; `docs/crossover_result.md` Section 4C (1387/5779/11530 vs 661/2741/5456 us) | backed |
| Sparse fused MLP already 7 to 10% faster per layer than native NVFP4 (618 vs 661 us at M=2048) | `--mode phase_bench`; `docs/crossover_result.md` Section 4C | backed |
| Reverse densification trades speed for accuracy ~1:1, no free knee (all-sparse 10.256, all-dense 7.974, gate_up-dense 9.750) | `harness/quadbit_serve.py --mode densify`; `docs/accuracy_pareto.md` | backed |
| Training-free hybrid placement negative (+0.05 PPL buys 3% MLP FLOPs, ~1.008x) | `harness/sensitivity_sparse.py`; `docs/paper.md` Section 8 | backed |
| Verification / multi-token decode does not favor sparse (margin shrinks with M) | `harness/quadbit_serve.py --versweep`; `docs/crossover_result.md` Section 4B; `/cache/versweep_{nvfp4,sparse}.csv` | backed |

## 6. Accuracy accounting

| Claim | Evidence | Status |
|-------|----------|--------|
| Dense FP4 W4A4 +0.63 PPL zero-calibration (Llama-3.1-8B-Instruct 7.27->7.90) | `harness/recovery_worth.py`; `docs/paper.md` Section 7 | backed |
| Base Meta-Llama-3-8B dense W4A4 6.20->6.91 (+0.71), at/below modelopt ref 7.97 | `harness/recovery_worth.py`; `docs/paper.md` Section 7 | backed |
| Two-level NVFP4 block rel 0.097; MXFP4 0.13 on Qwen3-8B, no training | `harness/real_model.py`; `docs/paper.md` Section 7 | backed |
| Deployable sparse-recovered 8B = 8.47 through two-level kernel, loses to dense 6.91 by ~1.56 | `harness/finetune_pair.py`, `harness/ab_sparse_semantics.py`; `docs/paper.md` Section 8 | backed |
| Deploy gap closed: two-level kernel through-kernel 8.47 == fake-quant; single-level would deploy 11.89 | `harness/ab_sparse_semantics.py`; `docs/paper.md` Section 8 | backed |
| Data lever negative: C4 diverse-corpus phase-1 flattened at 10.82 on WT-2 | app ap-SdSv9zQ9, `docs/paper_notes.md`, `docs/standing.md` | needs-rerun (on-volume run, timed out 192k/300k; not re-verified at committed commit) |
| element-2:4 checkpoint incompatible with pair-granular hardware (93.6 PPL naive) | `harness/perplexity_sparse.py`; `docs/paper.md` Section 8 | backed |
| TinyLlama recovery 7.53 -> 9.60 through kernel | `harness/finetune_pair.py`; `docs/paper.md` Section 8 | backed (toy, does not generalize) |

## Open axis (reserved, no number claimed)

| Claim | Evidence | Status |
|-------|----------|--------|
| Distillation reduces the sparse serving tax from 10.27 to 9.10 PPL and keeps all serving wins | `docs/crossover_result.md`, `harness/repair.py`, serving confirmation SPARSE_CALLS=7264 | backed |
| Distillation does NOT recover downstream capability (ARC-C/HellaSwag ~20pt below dense, ~unchanged from un-repaired) | `docs/crossover_result.md`, `harness/downstream_eval.py` | backed |
| Calibration / low-rank adapters / Wanda-pair mask repair are all negative | `harness/repair.py` run logs | backed |
| Dense mixed-precision W4A16/W4A4 refinement | in progress; `docs/paper.md` Section 7 reserved slot | reserved |
| Prefill parity for the sparse serving path (needs stream-K / persistent tiling) | not built; `docs/paper.md` Section 11 | needs-rerun |

## Deployment / fusion

| Claim | Evidence | Status |
|-------|----------|--------|
| QuadbitLinear drop-in, packer maxrel 0.0039, 4.0 to 4.2x over torch bf16 at 8192 | `harness/quadbit_linear.py`; `docs/paper.md` Section 6 | backed |
| Fused SwiGLU FFN cumulative 4.66x at batch 2048, numerically identical | `docs/paper.md` Section 6, `docs/paper_notes.md` | backed |
| Fused RMSNorm+quant 3.7 to 4.3x; fused add+RMSNorm+quant 5.3 to 5.8x | `docs/paper.md` Section 6 | backed |
| Full fused dense FP4 block on real Qwen3-8B 2.16 to 2.19x over bf16, block rel 0.13 | `harness/real_model.py`; `docs/paper.md` Section 6 | backed |
| All frontier-model projections tile onto the kernel | `harness/real_model.py`; `docs/paper.md` Section 6 | backed |
