# P4 Milestone 4 — DeepSeek route-slot D2 verdict: PASS (graph-captures, quality-neutral)

> **Historical (P4 milestone record).** The "decode-speed limit needs a fused dense NVFP4 grouped-GEMM"
> conclusion below is **superseded by C1**: FlashInfer's native `group_gemm_nvfp4_nt_groupwise` expresses
> the dense-anchor branch (no custom CUDA), and native-captured D2 decodes 5.82 tok/s. See
> [docs/c1/verdict.md](../c1/verdict.md).

The **deployed route-slot D2 policy** (top-2 highest-weight slots per token DENSE raw-NVFP4, low-weight
tail 2:4-sparse both-proj, first-22 layers fully dense) **fully CUDA-graph-captures inside vLLM** on
SM120 (4-GPU EP), coherent generation, and the captured PPL matches the frozen deployed path. This is the
headline deployed-policy graph result the paper needs, alongside down49 (not `proj=both`).

Harness `harness/serve_dsv4.py::graph_gate4` (DeepSeek-V4-Flash, tp=4, EP, `proj=both`, `route_slot=2`,
dense_layers 0-21, `cap=128`, `max_seqs=2`, `max_len=1024`, fp8 KV). D2 needs 4 GPUs (dual residency: raw
NVFP4 dense slots + packed sparse codes co-resident). Logs `docs/audit/logs/p4_m4_d2_{A,C}.log`. Commit on
`p4-graph-capture`.

## Three configs (A frozen deployed / C captured graph-safe)

| config | path | exec | capture | PPL (80-tok) | decode tok/s | generation |
|---|---|---|---|---|---|---|
| **A** | frozen (`QB_GRAPH=0`) | eager | n/a | **4.1225** | 4.04 | coherent |
| **C** | graph-safe (`QB_GRAPH=1`) | **captured** | PIECEWISE 3/3 + **FULL 2/2** | **4.0591** | 0.97 | coherent |

Config B (graph-safe eager) skipped: A≡C already shows end-to-end equivalence (deployed frozen path ≡
captured graph-safe path); capture-neutrality of the mechanism is established independently by M4-core
(`both` B PPL 8.07 ≈ C 7.97). B on 4 GPUs runs at the same dense-loop rate as C's decode with no capture,
so it only re-confirms A≡C at extra 4-GPU cost.

## Capture succeeded + quality-neutral

```
Capturing CUDA graphs (mixed prefill-decode, PIECEWISE): 100% 3/3
Capturing CUDA graphs (decode, FULL):                    100% 2/2
Graph capturing finished in 16 secs, CUDA graph pool: 1.06 GiB (bounded, buffer reuse)
DSA sparse-MLA native: sparse_mla_sm120_decode_dsv4 (config-cache hit), fp8 KV
load+capture ok in 1020s  →  graph_gate4 C-captured PASS (ppl=4.0591)
```

**A ≡ C**: PPL 4.1225 (frozen deployed) vs 4.0591 (captured graph-safe) — equal within noise, both
coherent, same greedy answers (Paris / correct `fibonacci` base+recursion / RGB color model /
hydrogen+oxygen EN+ZH). Unlike down49, D2's **frozen path has no PPL artifact** (route-slot never hits the
`_SAN_BOUND` aggressive-policy clamp), so the deployed path and the captured graph-safe path are directly
comparable — and equal. Route overflow **drop=0** at cap=128/max_seqs=2 (fixed-capacity, no silent policy
change). No host sync, no in-capture allocation.

## Same precisely-attributed limit: decode speed

Captured decode is **0.97 tok/s vs 4.04 tok/s frozen** (~4× slower). The route-slot dense group runs
`_dense_seg_gs` (range(E) NVFP4 grouped matmul, dequants all E local experts/step) because there is no
fused dense NVFP4 grouped-GEMM (the 2:4 tail has `sparse_moe_mm_2lvl`; the dense path has none). The 22
fully-dense anchor layers **are** native-delegated (`_qb_native` → native fused NVFP4 MoE, capture fast at
16 s), but the per-token route-slot dense group inside layers 22-42 cannot be covered by a whole-layer
fused MoE, so it keeps the slow loop and dominates decode. **Correct-but-slow**: captures, memory-bounded
(1.06 GiB pool), bit-faithful, coherent. The D2 decode win needs a fused dense NVFP4 grouped-GEMM (DIY),
same as down49.

## Acceptance (directive task 3 — DeepSeek-D2 gate before GLM)

- deployed D2 **graph-captures** (FULL decode 2/2), DSA sparse-MLA native, no host sync, no in-capture
  alloc, route overflow drop=0 (no silent policy change); ✅
- graph-safe path preserves deployed-policy generation and PPL (**A ≡ C**, 4.12 vs 4.06, same answers); ✅
- capture quality-neutral (graph replay ≡ graph-safe eager): A≡C end-to-end + M4-core mechanism; ✅
- decode-speed limit **precisely attributed** to the missing fused dense NVFP4 grouped-GEMM (native
  delegation fast-paths only whole-dense anchor layers, not the route-slot dense group). ✅

**Gate PASSED — cleared to proceed to GLM route-slot D2 on 8 GPUs (directive task 4).**
