# C10: does the sparse-FP4 MoE win serving throughput in the compute-bound (prefill/large-M) regime?

**Branch:** `c10-sparse-prefill`. **Verdict:** [verdict.md](verdict.md) (KILL). **Model:** DeepSeek-V4-Flash-NVFP4,
4-GPU EP. **Harness:** `serve_dsv4.py::graph_gate4` + the new `prefill_p` metric.

## Why this question

C2 settled that the dense NVFP4 fused MoE is the SM120 **decode** SOTA (48.248 vs sparse D2 5.972 =
8.1x) — at decode M=1-2 the MoE is memory/launch-bound, so halving expert FLOPs buys nothing and the
fused kernel wins. C2 explicitly did **not** measure serving prefill throughput and flagged the 2:4
advantage as "a prefill/large-M bandwidth effect, absent at decode." C10 measures exactly that gap:
at prefill (or high concurrency) the expert GEMMs become **compute-bound**, so 2:4 sparsity halving
expert FLOPs should translate into a throughput win. This is the "win where sparse wins" front — the
same logic as the dense large-M collapse where FlashInfer b12x weakens.

## The crux (settled by recon)

Per-local-expert row count is `M ~ tokens * top_k(6) / 256` (EP over 4 GPUs, 64 local experts/GPU).
At the default `max_num_batched_tokens=2048`, M ~ 48/expert — far below the sparse kernel's large-M
winning regime (~512+). To lift M we raise `max_len` (→ `max_num_batched_tokens = max(2048, max_len)`
in lockstep): P=16384 → M~384, P=32768 → M~768.

**The captured graph path CAPS M per expert at `cap` and DROPS the overflow** (`route_fixed_cap`,
`qb_sm120_plugin.py:1207`), so large-M is unreachable captured without raising `cap` (which grows the
`E*cap` scratch and capture/OOM cost). The **eager** path (`build_routing`) is **uncapped** = true
routed count. Prefill is compute-bound, so CUDA-graph capture (a decode/launch-overhead lever) is not
needed. **Therefore C10 runs eager.**

## Method

New metric in `_graph_gate_body` (`--prefill-p P`): build a P-token prompt, time TTFT
(`generate(max_tokens=1)`), `prefill_tps = P / TTFT`. Prefix caching is on, so each timed call uses a
UNIQUE first token to force a cold prefill (else call 2 reuses call 1's KV and reads ~0s); a warm call
first compiles the TileLang/FlashInfer-autotune kernels. Best of two timed calls.

Two arms, same harness / eager / `max_len`:
- **dense** = `--baseline dense_nvfp4` → `QB_MOE=off` = vLLM native FlashInfer-CUTLASS fused NVFP4 MoE
  (the C2 decode SOTA path).
- **sparse (all-sparse)** = `--route-slot 0 --proj both` (empty baseline) → `QB_MOE=sparse`, every MoE
  layer runs `sparse_moe_mm_2lvl` = max FLOP reduction = cleanest mechanism signal. If all-sparse does
  not win prefill throughput, no partial policy (down49/D2) will.

`--max-len = P + margin` so the whole prompt is ONE prefill chunk (avoids the known P=8192 chunked-
prefill scheduler `KeyError`, an unfixed blocker on this MoE path).

## Sweep

| P (prefill_p) | max_len | M/expert (~P*6/256) | expectation |
|---|---|---|---|
| 2048 | 2560 | ~48 | dense wins (decode-like, small M) |
| 16384 | 16896 | ~384 | approaching crossover |
| 32768 | 33280 | ~768 | thesis test — sparse should win if 2:4 pays |

First launch: the two ends (P=2048, P=32768), both arms. If sparse wins at 32768, fill in the middle
to locate the crossover P; if it loses even at 32768, push higher / diagnose whether the sparse serving
path (routing overhead, dual residency, un-fused per-expert seg) eats the FLOP win.

## What to report (honesty guards)

- prefill tok/s per (P, arm); the crossover P where sparse ≥ dense (if any).
- Since the sparse path carries +26-27% weight memory (dual residency, C2 V3), any high-concurrency
  follow-up must report the max sustained concurrency each path allows (sparse sustains fewer seqs).
- If the win exists only at all-sparse but the deployed quality policy is down49/D2 (partial), state
  the coverage-vs-throughput tradeoff explicitly — the throughput win scales with sparse coverage.
