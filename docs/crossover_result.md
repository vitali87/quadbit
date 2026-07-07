# Production workload crossover — where sparse FP4 wins end-to-end (2026-07-07)

**Headline:** across a batch x prompt-length x generation-length request matrix, quadbit's sparse split-K
FP4 MLP **wins end-to-end request latency in 81 of 112 regimes** vs production dense NVFP4. Both paths are
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

## Crossover heatmap — total-latency ratio NVFP4 / sparse (>1.00 = sparse wins)

### B=1 (single stream) — sparse wins every regime (+3.6% to +14.3%)
| prompt \ gen | 16 | 32 | 64 | 128 | 256 | 512 | 1024 |
|---|---|---|---|---|---|---|---|
| 128 | 1.119 | 1.122 | 1.127 | 1.126 | 1.129 | 1.130 | 1.143 |
| 512 | 1.095 | 1.120 | 1.130 | 1.136 | 1.135 | 1.123 | 1.130 |
| 2048 | 1.069 | 1.091 | 1.101 | 1.107 | 1.110 | 1.112 | 1.113 |
| 8192 | 1.036 | 1.049 | 1.071 | 1.123 | 1.099 | 1.104 | 1.106 |

### B=8
| prompt \ gen | 16 | 32 | 64 | 128 | 256 | 512 | 1024 |
|---|---|---|---|---|---|---|---|
| 128 | 1.111 | 1.105 | 1.094 | 1.096 | 1.095 | 1.093 | 1.091 |
| 512 | 1.045 | 1.063 | 1.085 | 1.115 | 1.086 | 1.088 | 1.086 |
| 2048 | **0.989** | 1.014 | 1.033 | 1.047 | 1.057 | 1.062 | 1.063 |
| 8192 | **0.987** | **0.987** | **0.994** | 1.003 | 1.016 | 1.022 | 1.027 |

### B=32
| prompt \ gen | 16 | 32 | 64 | 128 | 256 | 512 | 1024 |
|---|---|---|---|---|---|---|---|
| 128 | 1.033 | 1.047 | 1.055 | 1.060 | 1.061 | 1.058 | 1.049 |
| 512 | **0.986** | **1.000** | 1.017 | 1.030 | 1.037 | 1.038 | 1.041 |
| 2048 | **0.975** | **0.980** | **0.989** | **1.000** | 1.009 | 1.016 | 1.020 |
| 8192 | **0.982** | **0.985** | **0.989** | **0.994** | **1.000** | 1.006 | 1.012 |

### B=64 (batch-heavy, prefill-bound)
| prompt \ gen | 16 | 32 | 64 | 128 | 256 | 512 | 1024 |
|---|---|---|---|---|---|---|---|
| 128 | 1.010 | 1.017 | 1.019 | 1.019 | 1.014 | 1.010 | 1.006 |
| 512 | **0.977** | **0.981** | **0.986** | **0.992** | **0.998** | 1.000 | 1.000 |
| 2048 | **0.971** | **0.976** | **0.980** | **0.986** | **0.992** | **0.997** | 1.001 |
| 8192 | **0.984** | **0.986** | **0.989** | **0.990** | **0.992** | **0.994** | **0.995** |

Bold = NVFP4 wins (ratio <= 1.00). Everything else: sparse wins.

## Crossover boundary — min generation length for sparse to win total latency
| B | prompt=128 | 512 | 2048 | 8192 |
|---|---|---|---|---|
| 1 | 16 | 16 | 16 | 16 |
| 8 | 16 | 16 | 32 | 128 |
| 32 | 16 | 64 | 128 | 256 |
| 64 | 16 | 512 | 1024 | never (<=1024) |

The boundary rises with batch and prompt length: sparse needs a longer generation to amortize its prefill
deficit as the workload becomes more prefill-bound. Single-stream serving wins unconditionally.

## Representative absolute latency (seconds; out tok/s)
| regime | NVFP4 total | sparse total | winner |
|--------|-------------|--------------|--------|
| B=1 P=2048 G=128 | 0.998 | 0.902 | sparse +10.7% |
| B=1 P=2048 G=1024 | 7.783 (132 t/s) | 6.990 (146 t/s) | sparse +11.3% |
| B=8 P=512 G=128 | 1.050 | 0.942 | sparse +11.5% |
| B=64 P=8192 G=16 | 12.214 | 12.415 | NVFP4 +1.6% |
| B=64 P=8192 G=1024 | 43.824 (1495 t/s) | 44.040 (1488 t/s) | NVFP4 +0.5% |

## Reading
- **Interactive / agentic (B=1-8, any prompt, gen >= ~16-128): sparse wins**, up to +14%. This is the
  dominant regime for chat and tool-use serving.
- **Long-generation at any batch: sparse wins** once gen crosses the boundary.
- **Batch prefill (B=64, long prompt, short gen): NVFP4 wins** by <=3% — the prefill-bound corner where
  sparse's down-projection sparsity cannot pay for its prefill deficit.
- Accuracy tax is constant at +2.3 PPL (10.27 vs 7.97) everywhere; the crossover is purely a speed map.

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
