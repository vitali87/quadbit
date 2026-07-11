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
- **Eager only.** The run launched with graph capture on first fails: the plugin's EP local-expert
  loop calls `torch.unique(local).tolist()` (a device->host sync), illegal under CUDA-graph stream
  capture (`cudaErrorStreamCaptureUnsupported`, `qb_sm120_plugin.py:1255`). This is our own hook, not
  a DSA/attention/memory/loader blocker; it is the same limitation as DeepSeek's graph-capture status.
  Graph-capturable EP MoE is future work; all rows below are eager, matching the DeepSeek evals.
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

## Downstream capability (P1 smoke suite)

The MC log-likelihood harness (ARC-C, HellaSwag, Winogrande, and a 5-subject MMLU subset; PIQA is
excluded because `ybisk/piqa` is not loadable on the serve image, same as the DeepSeek 4-task table) is
tokenizer-agnostic (it scores continuations with `llm.get_tokenizer()`), so it runs unchanged on GLM
through the same 8-GPU EP load. Two runs, `limit=200` per task: dense NVFP4 reference vs the route-slot
D2 policy. This directly tests whether D2's small PPL gap hides a downstream collapse. It does not.

| policy | ARC-C | HellaSwag | Winogrande | MMLU-5 | **AVG** | PPL |
|---|---|---|---|---|---|---|
| dense (ref) | 0.655 | 0.780 | 0.750 | 0.856 | **0.7603** | 3.171 |
| **route-slot D2** | 0.650 | 0.780 | 0.725 | 0.848 | **0.7508** | 3.216 |
| Δ | -0.005 | 0.000 | -0.025 | -0.008 | **-0.0095** | +0.045 |

D2 holds within **0.95 pt AVG** of dense with no task collapsing (HellaSwag exactly flat; ARC-C and MMLU
within the n=200 / n=5 sampling band; Winogrande -2.5 pt is the largest move and near Winogrande's
per-200 noise). The downstream PPL (3.216) tracks the serving PPL (3.236). This is a small smoke suite,
not a full benchmark, but it removes the specific reviewer objection that D2's low PPL cost could mask a
downstream regression. Commands: `--mode glm_downstream --moe {dense | sparse --sparse-proj both
--route-slot 2 --dense-layers "0..37"} --limit 200`. Logs: `docs/audit/logs/glm_downstream.log`.

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

- **Quality evidence is PPL plus a small downstream smoke suite, not a full benchmark.** GLM now has
  both teacher-forced PPL (all four policies) and a 4-task MC downstream comparison (dense vs route-slot
  D2, `limit=200`, above); the D2 downstream AVG holds within 0.95 pt of dense, matching the small PPL
  gap. This backs D2's capability preservation directly on GLM, not by DeepSeek analogy. What is still
  not measured on GLM: full-size benchmarks, and downstream numbers for the down49/gateup49 rows (only
  their PPL is measured). The **most thorough downstream evidence of record is still DeepSeek's** (full
  AVG across every policy); GLM's is a confirmation smoke suite on the Pareto winner.
- **Sparse path is active, not a dense trap.** Each sparse run logs exactly 304 dense-anchor lines
  (38 anchored layers x 8 workers); layers 38-74 pack to 2:4 codes. Route-slot's KV collapse to 241k
  tokens independently confirms raw+codes co-residency.
- **Eager only.** Graph-captured EP MoE is future work (plugin host-sync in the expert loop).
- GLM needs 8 GPUs (433 GiB); the 2/4-GPU footprint DeepSeek enjoyed does not transfer.
