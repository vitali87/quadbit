# C1 verdict: native NVFP4 grouped-GEMM removes the D2 dense-anchor decode bottleneck

**Branch:** `c1-native-dense-anchor` (off `main` @ `a6f545e`). **Model:** DeepSeek-V4-Flash-NVFP4,
route-slot D2 (`route_slot=2`, `proj=both`, dense_layers 0-21, `cap=128`, `max_seqs=2`, tp=4, 4 GPU EP).

## Verdict: **native delegation WORKS — proceed to GLM-D2.**

P4 left one precise limitation: the deployed sparse MoE policies graph-*capture* but decode-*slow*, because
the dense anchored/grouped projection runs `_dense_seg_gs` (a `range(E)` dequant-to-bf16 loop over all
local experts) and "lacks a fused dense NVFP4 grouped-GEMM." C1 shows that premise is false on this stack
and removes the bottleneck by delegation, no custom CUDA:

1. **Recon:** FlashInfer 0.6.14 ships `group_gemm_nvfp4_nt_groupwise` (a grouped NVFP4 GEMM) and
   `ModelOptNvFp4FusedMoE.apply` consumes external `topk_ids` (a restricted-routing fused MoE).
2. **Standalone gate (A):** `group_gemm_nvfp4` vs `_dense_seg_gs` at DeepSeek shapes — cos 0.991 vs bf16,
   `nf=0`, **graph-captures** (bit-identical replay), **~18-25× faster**.
3. **Serving gate (B), DeepSeek-D2, native captured:** PPL **4.0112** (≈ dequant-captured 4.0591, quality
   preserved; coherent Paris/fibonacci/RGB/H2O EN+ZH, drop=0), graph **PIECEWISE 3/3 + FULL 2/2**, DSA
   `sparse_mla_sm120_decode_dsv4` native, pool **2.08 GiB/GPU**, decode **5.82 tok/s**.

## Speed / quality / memory Pareto (the SOTA-relevant point)

| D2 config | PPL | decode tok/s | graph | GPU |
|---|---|---|---|---|
| captured, dequant `_dense_seg_gs` (Row 1, same harness) | 3.9746 | 0.514 | FULL | 4 |
| frozen eager, dequant (P4 ref) | 4.1225 | 4.04 | none | 4 |
| **captured, native `group_gemm_nvfp4` (Row 3)** | **4.0112** | **5.82** | **FULL** | 4 |

Native-captured D2 is **11.3× the dequant-captured 0.514** (same harness) and **1.44× the frozen-eager
4.04**, at equal PPL (+0.037 from fp4 activations on the anchor, negligible) and bounded memory (2.08
GiB/GPU pool). Under the dequant backend the captured path was ~8× *slower* than eager (the dense-anchor
loop dominates the decode step, 124.6s for 64 tokens); the native grouped GEMM makes the **captured** path
*faster than eager* — a strictly better speed/quality point on the same 4 GPUs. This is the graph-fast D2
the campaign targeted.

## What was kept intact (task invariants)

Sparse selected experts still run `sparse_moe_mm_2lvl` (2:4 tail); routing still `route_fixed_cap`
(graph-safe fixed-capacity device routing); `_sanitize` graph-safe; DSA native SM120; no host sync in the
captured region (static `cap` makes the `a_scale` offsets compile-time constants); no policy semantics
changed (opt-in `QB_DENSE_BACKEND=native_nvfp4`, default `dequant` untouched). Only the dense-anchor
projection's backend changed.

## GLM route-slot D2 transfer (8 GPU) — CONFIRMED

The D2 win transfers to the second model. GLM-5.2-NVFP4 route-slot D2 (dense_layers 0-37, `cap=128`,
`max_seqs=2`, tp=8, `native_nvfp4`; log `c1_glm_d2_native_C.log`): graph **PIECEWISE 3/3 + FULL 2/2**, DSA
`sparse_mla_sm120_decode_dsv3_2` native, pool **1.21 GiB/GPU**, PPL **4.0705** (P4 band: frozen-eager 4.004
/ dequant-captured 4.157 — quality preserved), coherent (Paris+distances/fibonacci/red-blue-yellow/H2O),
decode **5.296 tok/s = 2.5× the eager D2 reference 2.10**. `load+capture ok in 2038s`. So native delegation
makes the deployed sparse MoE decode graph-fast on **both** DeepSeek (4 GPU) and GLM (8 GPU), quality-neutral,
no custom CUDA.

## Bottom line

`grouped_dense_nvfp4_moe_mm_2lvl` (custom CUDA) is **not needed** — FlashInfer's `group_gemm_nvfp4_nt_
groupwise` is the fused dense NVFP4 grouped-GEMM P4 assumed absent. The P4 canonical claim ("the remaining
speed limitation is the dense anchored/grouped projection path, which lacks a fused dense NVFP4 grouped-
GEMM") is **overturned**: the deployed sparse MoE policy path now graph-captures *and* decodes fast on
SM120 for both DeepSeek-D2 and GLM route-slot D2. Updating the user-mandated P4 wording in
`paper.md` / `command_manifest.md` / `glm_results.md` is flagged for user review (headline claim change),
not done unilaterally on this branch.

## Deliverables

1. Branch `c1-native-dense-anchor`; commits `bda69ae` (gate), `6dd2d55` (wiring).
2. Design note `docs/c1/design_note.md`; standalone A/B `docs/c1/standalone_ab.md`; serving A/B/C
   `docs/c1/d2_serving.md`; this verdict.
3. Raw logs `docs/audit/logs/c1_{recon,recon2,dense_anchor,d2_native_C,d2_dequant_C,d2_native_B}.log`.
4. Commands: `uv run modal run --detach harness/serve_dsv4.py::c1_dense_anchor` (standalone);
   `uv run modal run --detach harness/serve_dsv4.py::graph_gate4 --cap 128 --max-seqs 2
   --dense-layers 0,1,...,21 --dense-anchor-backend {native_nvfp4|dequant} [--eager --force-graph-path]`.
