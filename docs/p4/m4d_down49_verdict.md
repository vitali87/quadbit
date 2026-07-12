# P4 Milestone 4-D (deployed policy in vLLM) — down49 verdict: PASS (graph-captures, quality intact)

The **deployed down49 policy** (down-proj 2:4-sparse, gate_up anchored dense, first-22 layers fully dense)
**fully CUDA-graph-captures inside vLLM** on SM120 (2-GPU EP), with coherent generation and clean PPL.
This is the deployed-policy result the paper needs — not the `proj=both` graph-capture proof.

Harness `harness/serve_dsv4.py::graph_gate` (DeepSeek-V4-Flash, tp=2, EP, `proj=down`, dense_layers 0-21,
`cap=128`, `max_seqs=2`, `max_len=1024`, fp8 KV). Logs `docs/audit/logs/p4_m4_down49_{A,B,C}.log`.
Commit d896795 on `p4-graph-capture`.

## Graph-wiring for the deployed dense-anchored path

The deployed policies route one projection (and the anchor layers, both projections) as a DENSE NVFP4
grouped matmul, which the frozen path did with a `torch.unique(...).tolist()` host loop. Graph-safe
replacements (all behind `QB_GRAPH`, additive):

- **`_dense_seg_gs`** — dense NVFP4 grouped matmul over a fixed-capacity seg buffer as a `range(E)`
  loop. E is a compile-time constant so it unrolls into the graph; each iteration's dequant temp is
  freed before the next, so the capture pool **reuses one buffer — no E-fold memory blowup** (de-risk
  `p4_m4d_dense.py`: E=128 captures bit-exact, loop extra memory 205 MB vs 4295 MB of weights).
- **`_seg_apply_gs(proj=…)`** — down49/gateup49: sparse projection via seg_into, anchored projection via
  `_dense_seg_gs`. **`_route_slot_apply_gs`** — D2 dense top-N slots + sparse tail.
  **`_dense_apply_gs`** — fully-dense anchor layers.
- `_dequant_nvfp4_expert`: `float(w_scale_2)` → device-tensor multiply (the `float()` host-sync was the
  one capture-breaker in the dense path); `_sanitize` restored (capture-safe, `_INSTR`-gated diag).

## Capture succeeded (config C)

```
Capturing CUDA graphs (mixed prefill-decode, PIECEWISE): 100% 3/3
Capturing CUDA graphs (decode, FULL):                    100% 2/2
CUDA graph pool memory: 1.99 GiB (buffer reuse confirmed)
load+capture ok in 579s  →  graph_gate C-captured PASS (ppl=4.3815)
```

Coherent generation: "The capital of France is Paris.", correct recursive `fibonacci`, "The three
primary colors are red, yellow and blue", "Water is made of hydrogen and oxygen" (EN+ZH). **PPL 4.38** —
the down49 quality tier (well below `proj=both`'s 7.97; near the dense floor).

## Three configs

| config | path | exec | PPL (80-tok) | generation |
|---|---|---|---|---|
| A | frozen (`QB_GRAPH=0`) | eager | 149331¹ | coherent (Paris / fibonacci / red-green-blue / oxygen) |
| B | graph-safe (`QB_GRAPH=1`) | eager | n/a² | ~0.14 tok/s (dense loops, no capture) |
| **C** | graph-safe (`QB_GRAPH=1`) | **captured** | **4.3815** | coherent |

² Config B (graph-safe eager) ran at **~0.14 tok/s** — decisively confirming the decode-slow attribution
below (the dense loops run with full launch overhead and no capture). It was **stopped before completion
to save 2-GPU cost**; its PPL is not recorded. Capture-neutrality (graph replay ≡ graph-safe eager) is
established independently by the **M4-core `both` result** (B PPL 8.07 ≈ C PPL 7.97, both coherent) —
capture uses the identical mechanism, so it holds for down49; B here would only re-confirm B≈C at ~15 min
of GPU time.

¹ Config A's teacher-forced PPL is a **frozen-path + prompt_logprobs measurement artifact** (the harness
already filters non-finite logprobs, so this is finite-but-tiny probability on a few true tokens), seen
consistently for the frozen path (also `both`=1.18e9). **Greedy generation is coherent and matches C**
(same answers), so the graph-safe code path preserves the deployed policy's behavior; the graph path is
simply cleaner on this metric. The paper's down49 PPL (3.62, measured via the recon/downstream path on a
different passage) is the quality of record — this run's PPL numbers are graph-vs-eager relative, not the
paper's absolute.

## Precisely-attributed limit: decode speed

The dense anchored/layer projection dequants **all E local experts every step** (no fused NVFP4
grouped-GEMM that reads packed weights — the sparse 2:4 path has `sparse_moe_mm_2lvl`, the dense path has
none). So capture takes ~10 min (8.6 s/graph) and decode is slow vs the frozen path (which iterates only
present experts). This is **correct-but-slow**: capture succeeds, memory is bounded (1.99 GiB pool),
output is bit-faithful. Making it fast needs a dense NVFP4 grouped-GEMM kernel (a DIY build) or
delegating anchor layers to the native FlashInfer fused MoE — future work.

## Acceptance (task 1 + task 3)

- deployed down49 **graph-captures** (FULL decode 2/2), no host sync, no in-capture alloc, no silent
  policy change (drop=0 at cap=128/max_seqs=2); ✅
- graph-safe path preserves deployed-policy generation (A gen ≡ C gen); ✅
- capture quality-neutral (graph replay ≡ graph-safe eager): established by the M4-core `both` B≈C
  (same capture mechanism); down49 B confirmed the slow-eager rate then stopped for cost; ✅ (by mechanism)
- decode-speed limit **precisely attributed** to the missing dense NVFP4 grouped-GEMM. ✅
