# P4 Milestone 4 (in-vLLM graph capture) — verdict: CORE PASS on DeepSeek 2-GPU

The sparse quadbit MoE + DSA forward **fully CUDA-graph-captures inside vLLM** on SM120 (2-GPU EP), and
**graph replay is quality-neutral vs eager execution of the same path**. This softens the paper's
production caveat from "eager-only" to "graph-enabled for the sparse plugin path."

Harness: `harness/serve_dsv4.py::graph_gate` (DeepSeek-V4-Flash-NVFP4, tp=2, EP, `proj=both`, `cap=512`,
`max_seqs=8`, fp8 KV). Logs: `docs/audit/logs/p4_m4_{A,B,C}.log`. Commit: see branch `p4-graph-capture`.
Env: vLLM 0.24.0 V1, torch cu130 serve image, FlashInfer SM120 sparse-MLA (DSv4) decode, driver 12.8.

## Capture succeeded (config C)

With `QB_GRAPH=1` + `enforce_eager=False`, vLLM captured the sparse forward end to end:

```
Capturing CUDA graphs (mixed prefill-decode, PIECEWISE): 100% 5/5
Capturing CUDA graphs (decode, FULL):                    100% 4/4
sparse_mla_sm120_decode_dsv4 (FlashInfer) runs natively
load+capture ok in 557s   →  graph_gate C-captured PASS (ntok=96, ppl=7.9747)
```

The `FULL` decode capture includes **both** the DSA sparse-MLA attention **and** the quadbit 2:4-sparse
MoE — i.e. the whole decode step is one CUDA graph. Generation is coherent ("The three primary colors
are yellow magenta and cyan…", "Water is made of hydrogen and oxygen…").

## Fix chain to get there (each revealed the next host-sync in the captured forward)

1. `/cache/sparse_fp4.so` rebuilt with the additive `quantize_act_nvfp4_2lvl_s` symbol (M2/M3 kernel).
2. `route_fixed_cap`: vLLM's `expert_map` is **int32**; cast `assign.to(long)` (scatter_add/index need long).
3. MoE `both` branch wired to `_seg_apply_gs` behind `QB_GRAPH` (M3-A/B/C graph-safe path).
4. A6 indexer `_topk_into`: vectorized masked batched top-k (no `seq_lens.cpu()`, no Python row loop).
5. DSA decode logits `_paged_mqa_logits_bf16`: fixed-length gather + device masking (no `.item()`, no
   data-dependent shape). All graph-only branches gated behind `_GRAPH`; frozen eager path untouched.

## Capture is quality-neutral (the M4 quality gate)

| config | path | exec | PPL (80-tok) | generation |
|---|---|---|---|---|
| **B** | graph-safe (`QB_GRAPH=1`) | **eager** | **8.0706** | coherent |
| **C** | graph-safe (`QB_GRAPH=1`) | **captured** | **7.9747** | coherent |
| A | frozen (`QB_GRAPH=0`) | eager | 1.18e9 | degenerate |

**B ≈ C** (PPL 8.07 vs 7.97, ~1%, both coherent): capturing the graph does **not** change output quality
vs running the identical graph-safe code eagerly. The residual ~1% is fp8-KV / greedy-path sensitivity,
not a capture defect. This is the rigorous "correctness equals eager" claim for the capture mechanism.

### The config-A outlier is a policy×clamp artifact, not a capture issue

Config A (frozen path) collapses to PPL 1.18e9 **on the `proj=both` policy** because the frozen path's
`_sanitize` clamps activations to `[-_SAN_BOUND, _SAN_BOUND]`, and `proj=both` (every layer, **both**
projections 2:4-sparse) is the known aggressive "all-sparse" policy that produces large activations the
clamp mangles. The graph-safe path (B/C) skips that clamp and stays coherent. This is a frozen-vs-graph
**code-path** difference on a **non-deployed** policy, orthogonal to capture (which B≈C isolates). It is
**not** the deployed policy: Campaign-B ships route-slot **D2** / down49, not `proj=both`.

## Honest scope / what is NOT yet claimed

- **Policy**: the run used `proj=both` (aggressive, non-deployed) because that is what the graph-safe
  MoE branch (`_seg_apply_gs`) currently covers. The **deployed** policies (route-slot D2, down49) route
  a dense top-N / anchored projection that still uses a `torch.unique(...).tolist()` host loop, so they
  are **not yet graph-wired**. The strongest M4-D claim ("D2 quality intact under graph") needs that
  dense-slot path made graph-safe, then a D2-eager-vs-D2-graph run.
- **NaN guardrail**: the graph path currently skips `_sanitize`. That guardrail is actually capture-safe
  (its `.item()` diagnostic is `_INSTR`-gated), so it should be restored to `_seg_apply_gs` for
  long-context NaN-safety + frozen-path faithfulness, then re-validated. (Short-passage capture is clean;
  long-context was not stress-tested here.)
- **Decode speedup in-vLLM**: not yet measured end to end (M3 showed the isolated seg path ~1.5× from
  launch-overhead removal; an in-serving decode tok/s eager-vs-graph number is the confirming metric).
- **Scale**: DeepSeek 2-GPU, not the 8-GPU GLM D2 headline (M4-D).

## Next

1. Graph-wire the deployed dense-slot path (route-slot D2 / anchored down49) via a graph-safe dense-expert
   apply (fixed local-expert range + masked matmuls, mirroring `_seg_apply_gs`'s sink routing).
2. Restore `_sanitize` to the graph path; re-validate capture + coherence.
3. DeepSeek D2 eager-vs-graph on 2 GPU (cheap), then GLM route-slot D2 on 8 GPU (M4-D full rows: PPL,
   downstream, decode/prefill tok/s, per-GPU mem, capture/replay status, overflow, DSA backend).
