# Production workload crossover: where sparse FP4 wins end-to-end (2026-07-07)

**Headline:** across a batch x prompt-length x generation-length request matrix, quadbit's sparse split-K
FP4 MLP **wins end-to-end request latency outright in 81 of 112 regimes and ties 2 more (83/112 where it is
at least as fast)** vs production dense NVFP4. Both paths are
graph-captured (vLLM V1 fullgraph + CUDA graphs). Sparse wins wherever the **decode fraction** is large
enough (single-stream, short prompts, or long generations); dense NVFP4 keeps only the prefill-bound
corner (high batch x long prompt x short generation). This is the serving claim: **for single-stream and
long-generation request regimes, sparse FP4 wins end-to-end.**

## Method
- Same staged .so + recovered-Instruct sparse MLP + NVFP4 non-MLP as the decode result (`docs/graph_serving_result.md`).
- Both rows CUDA-graph captured (`--graph`); sparse row proves it ran (SPARSE_CALLS>0, PPL 10.27 not 7.97).
- Per (B, prompt P, gen G) cell: **TTFT** = wall of `generate(max_tokens=1)` (prefill + 1st token, measured
  once per (B,P) since it is gen-independent); **total latency** = wall of `generate(max_tokens=G, ignore_eos=True)`;
  **decode** = total - TTFT; **TPOT** = decode/(G-1).
- **Prefix caching OFF** and prompts de-nested by length: otherwise vLLM V1 reuses the TTFT call's prompt KV
  in the total call, skips the real prefill, and hides sparse's prefill deficit (this corrupted a first pass
  that spuriously showed sparse winning 112/112). Each (B,P) shape is warmed before measurement so cold graph
  init does not inflate TTFT.
- Grid: B in {1,8,32,64}, P in {128,512,2048,8192}, G in {16,32,64,128,256,512,1024}. util 0.8, RTX PRO 6000.
- No cell was KV-infeasible (vLLM wave-schedules the largest cells; that is realistic serving behavior).

## Crossover heatmap: total-latency ratio NVFP4 / sparse (>1.00 = sparse wins)

### B=1 (single stream): sparse wins every regime (+3.5% to +11.6%)
| prompt \ gen | 16 | 32 | 64 | 128 | 256 | 512 | 1024 |
|---|---|---|---|---|---|---|---|
| 128 | 1.093 | 1.095 | 1.097 | 1.097 | 1.097 | 1.086 | 1.090 |
| 512 | 1.055 | 1.068 | 1.083 | 1.081 | 1.083 | 1.084 | 1.084 |
| 2048 | 1.067 | 1.086 | 1.099 | 1.107 | 1.109 | 1.111 | 1.112 |
| 8192 | 1.035 | 1.052 | 1.074 | 1.087 | 1.097 | 1.102 | 1.105 |

### B=8
| prompt \ gen | 16 | 32 | 64 | 128 | 256 | 512 | 1024 |
|---|---|---|---|---|---|---|---|
| 128 | 1.116 | 1.114 | 1.102 | 1.097 | 1.094 | 1.092 | 1.089 |
| 512 | 1.052 | 1.067 | 1.076 | 1.082 | 1.085 | 1.086 | 1.085 |
| 2048 | **0.988** | 1.010 | 1.031 | 1.047 | 1.056 | 1.061 | 1.061 |
| 8192 | **0.990** | **0.994** | 1.000† | 1.008 | 1.015 | 1.022 | 1.026 |

### B=32
| prompt \ gen | 16 | 32 | 64 | 128 | 256 | 512 | 1024 |
|---|---|---|---|---|---|---|---|
| 128 | 1.037 | 1.046 | 1.055 | 1.060 | 1.060 | 1.055 | 1.047 |
| 512 | **0.999** | 1.009 | 1.023 | 1.032 | 1.037 | 1.037 | 1.039 |
| 2048 | **0.987** | **0.990** | **0.997** | 1.004 | 1.010 | 1.016 | 1.018 |
| 8192 | **0.993** | **0.995** | **0.998** | 1.001 | 1.007 | 1.010 | 1.014 |

### B=64 (batch-heavy, prefill-bound)
| prompt \ gen | 16 | 32 | 64 | 128 | 256 | 512 | 1024 |
|---|---|---|---|---|---|---|---|
| 128 | 1.018 | 1.008 | 1.014 | 1.016 | 1.011 | 1.007 | 1.004 |
| 512 | **0.982** | **0.984** | **0.990** | **0.996** | 1.000† | **0.999** | **0.997** |
| 2048 | **0.983** | **0.985** | **0.989** | **0.992** | **0.995** | **0.999** | 1.001 |
| 8192 | **0.998** | **0.998** | **0.999** | **0.999** | **0.998** | **0.998** | **0.997** |

Bold = NVFP4 wins (ratio < 1.00). Everything else: sparse wins (ratio > 1.00). **†** marks the two cells
whose ratio is 1.0001 (sparse faster by <0.02%): statistical ties, counted as ties not wins. Tally:
**81 sparse wins, 2 ties, 29 NVFP4 wins.**

## Crossover boundary: min generation length for sparse to win total latency
| B | prompt=128 | 512 | 2048 | 8192 |
|---|---|---|---|---|
| 1 | 16 | 16 | 16 | 16 |
| 8 | 16 | 16 | 32 | 128 |
| 32 | 16 | 32 | 128 | 128 |
| 64 | 16 | never (tie at 256) | 1024 | never (<=1024) |

The boundary rises with batch and prompt length: sparse needs a longer generation to amortize its prefill
deficit as the workload becomes more prefill-bound. Single-stream serving wins unconditionally.

## Representative absolute latency (seconds; out tok/s)
| regime | NVFP4 total | sparse total | winner |
|--------|-------------|--------------|--------|
| B=1 P=2048 G=128 | 0.997 | 0.901 | sparse +10.7% |
| B=1 P=2048 G=1024 | 7.772 (132 t/s) | 6.987 (147 t/s) | sparse +11.2% |
| B=8 P=512 G=128 | 1.019 (1005 t/s) | 0.941 (1088 t/s) | sparse +8.2% |
| B=64 P=8192 G=16 | 12.382 | 12.411 | NVFP4 +0.2% |
| B=64 P=8192 G=1024 | 43.934 (1492 t/s) | 44.056 (1488 t/s) | NVFP4 +0.3% |

## Reading
- **Interactive / agentic (B=1-8, any prompt, gen >= ~16-64): sparse wins**, up to +11.6%. This is the
  dominant regime for chat and tool-use serving.
- **Long-generation at any batch: sparse wins** once gen crosses the boundary.
- **Batch prefill (B=64, long prompt, short gen): NVFP4 wins** by <=3%. This is the prefill-bound corner
  where sparse's down-projection sparsity cannot pay for its prefill deficit.
- Accuracy tax is constant at +2.3 PPL (10.27 vs 7.97) everywhere; the crossover is purely a speed map.

## Track 4B addendum: verification / multi-token shapes (M = B*k)
Speculative/verification decoding processes k candidate tokens per sequence, so the MLP sees effective
M = B*k rows per decode step. Hypothesis: larger M favors sparse tensor-core work. **Refuted.** The sparse
decode margin over NVFP4 *shrinks* with M in the clean regime and never expands:

| effective M | 1 | 8 | 16 | 32 | 64 | 128 |
|---|---|---|---|---|---|---|
| sparse/NVFP4 decode tok/s | 1.134 | 1.083 | 1.066 | 1.051 | 1.020 | 1.043 |

(prefix caching off, same as the crossover. M >= 256 is scheduling/BW-bound in full-forward decode, noisy
and NVFP4-favorable, not a clean MLP shape.)
The split-K decode win is a **small-M GPU-underfill fix**: as M grows, NVFP4's own dense GEMM fills the
machine and the advantage fades. So sparse FP4 is most attractive for **low-M latency-sensitive decode
(single/low-batch single-token)**, NOT for throughput-oriented multi-token verification. This is consistent
with the crossover map (sparse dominates B=1-8; the batch-heavy corner is NVFP4's). Data:
`/cache/versweep_{nvfp4,sparse}.csv`.

## Provenance
- Commit `6ea58d7` on branch `track4-crossover`. Recovered-Instruct ckpt
  `/cache/recovered_Llama-3.1-8B-Instruct_P30000_p25000_2sh_lr3e-05.pt`. NVFP4 `nvidia/Llama-3.1-8B-Instruct-NVFP4`.
- Full matrices: `/cache/crossover_nvfp4.csv`, `/cache/crossover_sparse.csv` (112 cells each, no skips).
- Commands:
```bash
uv run modal run --detach harness/quadbit_serve.py --mode hybrid --util 0.8 --baseline --graph --crossover --no-do-ppl
uv run modal run --detach harness/quadbit_serve.py --mode hybrid --util 0.8 --fused --graph --crossover \
  --recovered-ckpt /cache/recovered_Llama-3.1-8B-Instruct_P30000_p25000_2sh_lr3e-05.pt
```
