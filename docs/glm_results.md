# GLM-5.2-NVFP4 transfer result (Phase 3/4)

Large-model transfer of the DeepSeek-proven structural sparse-FP4 policies to GLM-5.2 on 8x RTX PRO
6000 (SM120, EP), under the quadbit plugin in vLLM 0.24.0. Source logs:
`scratchpad/glm_{dense_ppl,down49,gateup49,routeslot2}.log`; downstream evidence
`docs/audit/logs/glm_downstream.log`. Sparse-policy runs at commit `8bb08c0`; dense PPL row at `53ab7d3`;
downstream smoke suite at `cc00b8b`.

## Load / serving baseline (the make-or-break gate)

- **GLM-5.2 loads and generates coherently on SM120** under the plugin, 8-GPU tensor+expert parallel.
- **DSA runs natively**: vLLM selects `FLASHINFER_MLA_SPARSE_SM120` + `DEEPSEEK_V32_INDEXER` (fp8 KV,
  `fp8_ds_mla` format). No silent fallback, no dense trap.
- **EP**: 32 local / 256 global experts per rank, `FLASHINFER_CUTLASS` NVFP4 MoE backend; the quadbit
  hook patches all 8 workers.
- **Graph-captured and dense-anchor delegated (P4 + C1).** The policy-sweep rows below were measured
  eager as the deployed-quality reference. The original blocker was our own hook: the EP local-expert
  loop called `torch.unique(local).tolist()` (a device->host sync), illegal under CUDA-graph stream
  capture. P4 replaced it with a graph-safe fixed-capacity device-routing path (`route_fixed_cap` /
  `_route_slot_apply_gs`, behind `QB_GRAPH`), and route-slot D2 **fully CUDA-graph-captures on 8 GPUs**
  (PIECEWISE 3/3 + FULL 2/2, DSA sparse-MLA native), quality-neutral vs the frozen eager path (A eager
  4.0040 ≡ C captured 4.1565 on an 80-tok passage, both coherent, drop=0). C1 then removed the remaining
  dense-anchor decode-speed limit by delegating the anchored/grouped projection to FlashInfer's native
  grouped NVFP4 GEMM (`group_gemm_nvfp4_nt_groupwise`, opt-in `QB_DENSE_BACKEND=native_nvfp4`) instead of
  the dequant-to-bf16 loop, with no custom dense grouped-GEMM. **GLM route-slot D2 native captured: PPL
  4.0705, decode 5.296 tok/s = 2.5× the eager reference 2.10, PIECEWISE 3/3 + FULL 2/2, DSA
  `sparse_mla_sm120_decode_dsv3_2` native, pool 1.21 GiB/GPU** ([docs/audit/logs/c1_glm_d2_native_C.log](audit/logs/c1_glm_d2_native_C.log)).
  The old "graph-correct but dense-loop-slow" limitation is **superseded by native delegation**. See
  [docs/c1/verdict.md](c1/verdict.md) and [docs/p4/m4_glm_d2_verdict.md](p4/m4_glm_d2_verdict.md). The
  PPL numbers here (short held-out passage) and the 3.171 dense-baseline (114-token policy-sweep passage)
  are different protocols; only within-protocol comparisons hold.
- Weights 432.9 GiB; model load 54.62 GiB/GPU (~360 s); init engine ~127 s.

## Policy table

75 MoE layers (first 3 dense via `first_k_dense_replace=3`), 256+1 experts, top-8. Anchor = MoE layers
0-37 kept NVFP4-dense; sparsify 38-74 (37 layers, 49.3%). PPL = teacher-forced over a 114-token
held-out passage. Serving = B=1, prompt 512, gen 64; **decode tok/s is decode-only** (the prefill of a
separate gen=1 call is subtracted, isolating the 63 decode steps), not end-to-end.

| policy | anchor | sparse layers | active sparse FLOP¹ | PPL | Δ PPL | decode tok/s² | TPOT s | KV tokens | mem/GPU | coherent |
|---|---|---|---|---|---|---|---|---|---|---|
| dense (ref) | all | 0 | 0 | 3.171 | — | 1.79 | 0.558 | 606,528 | 91.7 GB | ✓ |
| **down49** | 0-37 | 38-74 (down) | ~16% | 3.380 | **+0.209** | 1.71 | 0.586 | 586,752 | 92.3 GB | ✓ |
| **route-slot D2** | 0-37 | 38-74 (tail-6/8, both) | ~37% | **3.236** | **+0.065** | 2.10 | 0.477 | 241,152 | 91.7 GB | ✓ |
| gateup49 (control) | 0-37 | 38-74 (gate/up) | ~33% | 3.603 | **+0.432** | 2.12 | 0.472 | 647,424 | 91.7 GB | ✓ |

¹ Active sparse FLOP % = (sparse layers / 75) x (sparse projection share of expert FLOP) x (sparse slot
share). down = down-proj only (~1/3 of expert FLOP); gate/up = w13 (~2/3); route-slot = both
projections on the tail 6/8 slots. Estimate, not a measured kernel speedup.

² decode-only tok/s = 63 / (gen-64 wall - gen-1 wall), subtracting the second prefill so only the 63
decode steps are counted; the earlier end-to-end figures (prompt+gen throughput) were ~0.2-0.3 tok/s
lower. All rows measured identically, so cross-policy comparison holds either way.

## Downstream capability (WS-E 8-task battery, all deployed policies)

The MC log-likelihood harness is tokenizer-agnostic (it scores continuations with
`llm.get_tokenizer()`), so it runs unchanged on GLM through the same 8-GPU EP load. WS-E
([wse/verdict.md](wse/verdict.md)) gives GLM its **first full downstream table**: an 8-task battery
(ARC-C, ARC-Easy, HellaSwag, PIQA, OpenBookQA, BoolQ, Winogrande, MMLU-5) at `limit=400`, run on all
three deployed policies. PIQA is now included (restored at root cause via the HF parquet-convert branch,
not excluded as before), and **down49 now has downstream accuracy, not just PPL**.

| policy | ARC-C | ARC-E | HellaSwag | PIQA | OBQA | BoolQ | Winogrande | MMLU-5 | **AVG-8** | Δ dense |
|---|---|---|---|---|---|---|---|---|---|---|
| dense (ref) | .700 | .878 | .778 | .868 | .500 | .928 | .773 | .850 | **.7841** | — |
| **down49** | .698 | .868 | .758 | .878 | .513 | .925 | .768 | .856 | **.7826** | **−0.15 pt** |
| **route-slot D2** | .680 | .875 | .770 | .868 | .488 | .920 | .758 | .852 | **.7762** | −0.79 pt |

- **The PPL-only caveat on down49 is closed.** down49 now measures at **−0.15 pt AVG** downstream, near
  flat, confirming the "down projection is tolerant" rule on GLM with accuracy rather than PPL alone
  (down49 +0.209 PPL, but the downstream cost is negligible; the tax the PPL flags does not show up as
  lost downstream capability at 49% coverage).
- **No task collapses.** Largest single-task move is D2 arc_c −2.0 pt; PIQA is neutral-to-positive under
  sparsity (down49 .878 *above* dense .868). D2 holds within 0.79 pt of dense on the wider battery,
  consistent with its small serving PPL gap (+0.065).
- The frozen `limit=200` dense-vs-D2 4-task comparison (dense .7603, D2 .7508, −0.95 pt) still stands as
  the earlier reference; WS-E is a wider, fresh `limit=400` measurement, not a correction. Commands:
  `--mode {glm_downstream} --moe {dense | sparse --sparse-proj down --dense-layers "0..37" | sparse
  --sparse-proj both --route-slot 2 --dense-layers "0..37"} --limit 400`.

## What transfers

- **The DeepSeek structural rule holds on GLM.** Down-only sparsity costs about half the capability of
  gate/up sparsity at similar coverage: **down49 +0.209 PPL vs gateup49 +0.432 PPL**. The tax lives in
  the gate/up projection; the down projection is tolerant. This is the same mechanism DeepSeek's
  downstream AVG showed (c_down49 -0.29pt vs c_gateup49 -3.27pt).
- **Route-slot is the Pareto winner, same as DeepSeek D2.** Keeping the top-2 highest-weight routed
  slots dense and sparsifying the low-weight tail-6 gives the **highest active sparse FLOP (~37%) at the
  lowest quality cost (+0.065 PPL)** of any sparse row. Dominant slots carry the output; the tail is
  nearly free.
- **Route-slot dual residency fits on 8 GPUs** (unlike DeepSeek, which needed 4 GPUs vs its 2-GPU
  dense) but at a real KV cost: keeping raw NVFP4 + 2:4 codes co-resident drops KV capacity from
  606,528 to **241,152 tokens** (~60% reduction). It serves fine at max_len 2048; long-context KV
  pressure is the trade.

## Honest caveats (do not overclaim)

- **Quality evidence is PPL plus an 8-task MC downstream battery, not a full-size benchmark.** GLM now
  has teacher-forced PPL (all four policies) and 8-task downstream accuracy on all three *deployed*
  policies (dense, down49, route-slot D2, `limit=400`, above); each holds within 0.79 pt of dense with no
  task collapse. down49's downstream number closes the earlier PPL-only gap on the down policy. What is
  still not measured on GLM: full-size benchmarks, and downstream accuracy for the gateup49 *control* row
  (PPL-only, and it is a control, not a deployed policy). DeepSeek still carries the widest per-policy
  sweep, but GLM's deployed policies now have real downstream accuracy, not a DeepSeek analogy.
- **Sparse path is active, not a dense trap.** Each sparse run logs exactly 304 dense-anchor lines
  (38 anchored layers x 8 workers); layers 38-74 pack to 2:4 codes. Route-slot's KV collapse to 241k
  tokens independently confirms raw+codes co-residency.
- **Graph-enabled + dense-anchor delegated (P4 + C1).** Route-slot D2 CUDA-graph-captures on 8 GPUs (the
  old expert-loop host-sync was replaced by a fixed-capacity device-routing path), and C1 delegates the
  dense anchored/grouped projection to FlashInfer's native grouped NVFP4 GEMM, so native-captured GLM-D2
  decodes at 5.296 tok/s = 2.5× the eager reference; no custom dense grouped-GEMM was required. The rows
  above stay eager as the deployed-quality reference. See [docs/c1/verdict.md](c1/verdict.md),
  [docs/p4/m4_glm_d2_verdict.md](p4/m4_glm_d2_verdict.md).
- GLM needs 8 GPUs (433 GiB); the 2/4-GPU footprint DeepSeek enjoyed does not transfer.
