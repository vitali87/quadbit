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
| mma / ldmatrix / scale / 2:4-metadata bit-layouts derived empirically by probe-and-verify, validated to relative error 0 | `harness/verify_sparse_2lvl.py` (maxrel 0), `harness/probe_ldmatrix.py`; `docs/paper.md` Section 3 | backed |
| SM120 lacks the `tcgen05`/UMMA tensor-memory path (B200/SM100 only), so every FP4 GEMM runs through warp-level `mma.sync`/`mma.sp` | `docs/hardware.md`; `docs/paper.md` Section 2 | backed |

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

## Distributed sparse-FP4 MoE (DeepSeek-V4-Flash) — next-phase

| Claim | Evidence | Status |
|-------|----------|--------|
| DeepSeek-V4-Flash experts are MXFP4 (int8-packed E2M1 + e8m0 per-32 scale), not the config's FP8 | `harness/moe_prep.py` (raw dtypes: int8 weight + float8_e8m0fnu scale) | backed |
| MXFP4->bf16 decode is value-exact (100% round-trip; 88.7% raw-byte = harmless +0/-0 aliasing) | `harness/moe_prep.py::run` validate (`docs` scratch m2e log) | backed |
| Segmented routed-row kernel == per-expert kernel, cos 1.000000, all routing patterns | `harness/moe_seg.py` (uniform/all-to-one/one-per-expert/imbalanced, tiny + deepseek shapes) | backed |
| Segmented MoE op is CUDA-graph capturable+replayable, cos 1.000000 | `harness/moe_seg.py` graph-capture block | backed |
| On one real layer (256 experts, real gate), seg == per-expert kernel, cos 1.000000, 0 non-finite | `harness/moe_layer.py` | backed |
| quadbit sparsifies ~91% of params / ~80% of active linear FLOPs (all expert MLPs; attn/router/embed dense) | `docs/figures/coverage.py`, `docs/figures/data/coverage.csv` | backed |
| Expert-parallel scaling 2.17x (2 GPU) / 4.21x (4 GPU), imbalance 1.04, checksum-identical | `harness/moe_dist.py`; `docs/figures/data/dist_scaling.csv` | backed |
| PCIe all-reduce comm 0.32-0.45 ms vs 1.3-2.5 ms expert compute (no NVLink caveat) | `harness/moe_dist.py` | backed |
| Real-weight per-expert-output 2:4-FP4 tax ~cos 0.70 (random-act worst case); MoE accuracy recovery is future work | `harness/moe_layer.py` | backed |
| DeepSeek-V4-Flash NVFP4 weights load in vLLM 0.24 on 2x RTX PRO 6000 (FlashInfer CUTLASS MoE selected) | `harness/serve_dsv4.py` (serve3 log) | backed |
| Vanilla vLLM cannot init the model's FP8 (ue8m0 W8A8) attention on SM120 (DeepGEMM SF-transform asserts, CUTLASS c3x scaled_mm no-dispatch) -- an ecosystem gap, not quadbit | `harness/serve_dsv4.py` (serve3/serve4 logs) | backed |
| That ecosystem gap is OVERTURNED by the SM120-unblock plugin (block-FP8 dequant, MLA o_proj reimpl, DSA Lightning-Indexer logits, cooperative-topk override) -> DeepSeek-V4-Flash serves end-to-end on SM120 | `harness/qb_vllm_plugin/qb_sm120_plugin.py`; `docs/paper.md` Section 10.1 | backed |
| In-vLLM EAGER sparse-FP4 MoE serving (this model, end-to-end): downstream c_down49 (2 GPU) / D2 (4 GPU) / GLM (8 GPU) all ran through the quadbit sparse op | `harness/serve_dsv4.py`; `docs/deepseek_final_table.md`, `docs/glm_results.md` | backed |
| In-vLLM GRAPH-CAPTURED sparse MoE serving (this model) | P4: plugin host-sync replaced by fixed-capacity device routing; DeepSeek-D2 graph-captures (FULL decode 2/2), A frozen 4.12 == C captured 4.06, DSA native, drop=0; [docs/p4/m4_d2_verdict.md](p4/m4_d2_verdict.md) | backed (see Section 9 P4) |

## 8. Training-free capability-preserving structural sparsity (DeepSeek + GLM transfer)

The paper's large-model claim. Downstream = 400 items/task (frozen 4-task ARC-C/HellaSwag/Winogrande/
MMLU-5, dense ref AVG .7383; widened to an 8-task battery in WS-E, [docs/wse/verdict.md](wse/verdict.md)). All rows below
run in-vLLM on SM120 with the quadbit 2:4 sparse-FP4 expert op.

| Claim | Evidence | Status |
|-------|----------|--------|
| Single-GPU sparse-FP4 kernel + accuracy proven | Sections 2/6/7 above (kernel Pareto, fused blocks, W4A4 accounting) | backed |
| Multi-GPU (EP) sparse-FP4 MoE proven | EP 2.17x/4.21x scaling (Section 7); DeepSeek downstream ran on 2-GPU (c_down49) and 4-GPU (D2) | backed |
| Projection anchoring recovers capability training-free: down-only 49% layers at -0.29pt (c_down49) | `docs/figures/data/deepseek_final.csv`, `docs/deepseek_final_table.md`; merged PR #9 | backed |
| Max down-only coverage clearing .718 is 60% layers (.7190); cliff at 65% (.7150) | `wsa_downstream.csv` c_down60/c_down65; `docs/paper_notes.md` WS-C | backed |
| Route-slot extends the Pareto: top-2-dense D2 = ~33% active sparse FLOP at -0.79pt, needs 4-GPU dual residency | `deepseek_final.csv` d2_slot2; merged PR #10; `docs/paper_notes.md` WS-D | backed |
| Mechanism: tax lives in gate_up projection + dominant route slots; down + low-weight tail are safe | c_gateup49 (-3.27) vs c_down49 (-0.29); D1 (-2.27) vs D2/D3; `fig_ds_designspace` | backed |
| Failure controls: magnitude/Wanda-alone/A3-repair/all-sparse all fail (>= -4.2pt) | `wsa_downstream.csv` (magnitude_100 .5096, a2_100_wanda .5092, a3_*, a2_49 .6966); `fig_ds_designspace` | backed |
| Do NOT conflate sparse layer % with active sparse FLOP %; PPL is not the quality metric | `docs/deepseek_final_table.md` column notes | backed |
| GLM-5.2 architecturally compatible: glm_moe_dsa supported in vLLM 0.24.0, NVFP4 routed experts identical scheme | `harness/serve_dsv4.py --mode glm_inspect`; `docs/glm_feasibility.md` | backed |
| GLM-5.2-NVFP4 = 432.9 GiB; does NOT fit on 2 or 4 RTX PRO 6000, needs 8 (EP) | `glm_inspect` safetensors-index probe; `docs/glm_feasibility.md` | backed |
| GLM-5.2 loads + generates coherently on SM120 under the quadbit plugin; DSA runs natively (`FLASHINFER_MLA_SPARSE_SM120` + `DEEPSEEK_V32_INDEXER`), 8-GPU EP | `harness/serve_dsv4.py --mode glm_baseline`; `docs/glm_results.md`; `docs/audit/logs/glm_runs.log` | backed |
| GLM structural transfer: down-only (+0.209 PPL) costs ~half of gate/up (+0.432 PPL) at matched 49%-layer coverage; same mechanism as DeepSeek | `docs/glm_results.md` down49/gateup49; `docs/audit/logs/glm_runs.log` | backed |
| GLM route-slot D2 (top-2 dense, tail-6 sparse) = highest sparse FLOP (~37%) at lowest cost (+0.065 PPL); dual residency fits 8 GPUs at 241k-vs-607k KV | `docs/glm_results.md` routeslot2; `docs/audit/logs/glm_runs.log` | backed |
| GLM route-slot D2 downstream AVG holds within -0.95pt of dense (.7508 vs .7603, frozen 4-task limit=200 MC harness on 8-GPU EP) | `docs/glm_results.md` downstream table; `docs/audit/logs/glm_downstream.log` | backed |
| WS-E 8-task breadth (arc_c/arc_e/hellaswag/piqa/obqa/boolq/winogrande/mmlu-5, limit=400): DeepSeek dense .7548 / down49 -0.57pt / D2 +0.03pt; GLM dense .7841 / down49 -0.15pt / D2 -0.79pt; no task collapse, PIQA restored at root cause | [docs/wse/verdict.md](wse/verdict.md); `/cache/qb_downstream_wse_*.csv` | backed |
| GLM deployed policies (dense, down49, D2) all have 8-task downstream accuracy; down49 PPL-only caveat closed (-0.15pt); only GLM's gateup49 CONTROL stays PPL-only | [docs/wse/verdict.md](wse/verdict.md); [docs/glm_results.md](glm_results.md) downstream table | backed |
| gate_up STE-QAT layerwise repair does NOT recover the gate_up tax (DeepSeek, 8-task limit=400): QAT gateup49 .7332 vs one-shot .7363 = -0.31pt (within noise); tight per-layer rel 0.02-0.07 buys no downstream; 2:4-FP4 gate_up floor is real, avoid via down49 anchor not repair | [docs/qat/gateup_moe_verdict.md](qat/gateup_moe_verdict.md); `/cache/qb_downstream_qat_gu49_reio2.csv`, `qb_downstream_gateup49_oneshot8.csv` | backed |
| Global MoE QAT (router-weighted combine fit to routed aggregate) does NOT recover either, and is WORSE than both weaker variants: global QAT gateup49 .7259 vs one-shot .7363 (-1.04pt) vs per-expert QAT .7332 (-0.73pt); tightest reconstruction of all three (agg rel 0.004-0.017) yields worst downstream; closes the last layerwise-repair lever | [docs/qat/global_moe_verdict.md](qat/global_moe_verdict.md); `/cache/qb_downstream_qat_gu49_gs.csv` | backed (negative) |
| No claim of exhaustive GLM downstream preservation (WS-E is 8 MC tasks, not full-size benchmarks) | paper §10 / §12 limitations | limitation (explicit) |
| GLM sparse path graph-capturable end-to-end | P4: route-slot D2 graph-captures on 8 GPUs (PIECEWISE 3/3 + FULL 2/2, pool 1.01 GiB/GPU), A frozen 4.0040 == C captured 4.1565, DSA `sparse_mla_sm120_decode_dsv3_2` native, drop=0; [docs/p4/m4_glm_d2_verdict.md](p4/m4_glm_d2_verdict.md) | backed (see Section 9 P4) |
| Full-coverage sparsity (>60% down-only, both-proj, top-1 slot) needs QAT/KD | c_down74/100, a2_49, D1 all miss .718 | backed (negative) |

## 9. P4 graph-enablement of the deployed sparse MoE path (SM120)

Overturns the prior "graph capture is future work" caveat (old Section 8/10 graph rows). All rows on
branch `p4-graph-capture` / PR #16; main frozen at `campaign-b-freeze-a91c5d9`.

| Claim | Evidence | Status |
|-------|----------|--------|
| A. Deployed sparse MoE policies graph-capture on SM120 | DeepSeek-D2 (FULL decode 2/2) + GLM route-slot D2 (PIECEWISE 3/3 + FULL 2/2), DSA native SM120, drop=0, no host sync / no in-capture alloc; `docs/p4/m4_d2_verdict.md`, `docs/p4/m4_glm_d2_verdict.md`, logs `p4_m4_d2_{A,C}.log` / `p4_m4_glm_d2_{A,C}.log` | backed |
| B. Graph capture is quality-neutral for deployed policies | DeepSeek-D2 A 4.12 vs C 4.06; GLM-D2 A 4.0040 vs C 4.1565 (same short-passage harness), both coherent. Caveat: GLM dense baseline 3.171 uses a different 114-token policy-sweep passage, so only A-vs-C is the valid capture-neutrality comparison | backed (caveat) |
| C. Graph-enabled sparse MoE decode bottleneck removed | C1: the dense anchored/grouped path is delegated to FlashInfer's native grouped NVFP4 GEMM; captured DeepSeek-D2 decodes 5.82 tok/s (Section 10 C1 table); superseded, see Section 10 rows below | backed (native delegate) |
| D. GLM quality (PPL + 8-task downstream battery on all deployed policies, not exhaustive) | WS-E dense .7841 / down49 -0.15pt / D2 -0.79pt (Section 8); no full-benchmark claim | backed (scope limit stated) |
| E. CUTLASS / dense-baseline positioning | CUTLASS 80b is sparse-kernel prior art (Section 5 rows 29-30); FlashInfer/vLLM/SGLang are dense NVFP4 ecosystem baselines (Section 1 rows 18-22); quadbit does NOT claim dense FP4 speed leadership | guarded |

## 10. C1 native dense-anchor delegation (SM120)

Removes the P4 dense-anchor decode bottleneck by delegation, no custom CUDA. Branch
`c1-native-dense-anchor`, commits `bda69ae` -> `1333dc4`. Full result `docs/c1/verdict.md`; logs
`docs/audit/logs/c1_*.log`.

| Claim | Evidence | Status |
|-------|----------|--------|
| Dense anchored/grouped projection bottleneck removed by native FlashInfer grouped NVFP4 delegation | FlashInfer 0.6.14 `group_gemm_nvfp4_nt_groupwise`; standalone A/B cos 0.991 vs bf16, nf=0, graph-captures, ~18-25x faster than the dequant loop; `docs/c1/standalone_ab.md`, `docs/audit/logs/c1_dense_anchor.log` | backed |
| DeepSeek-D2 native-delegated graph path is quality-neutral and faster | PPL 4.0112 (dequant-captured 3.9746, +0.037), decode 5.820 tok/s = 11.3x the dequant-captured 0.514 (same harness), 1.44x frozen-eager 4.04; FULL capture, DSA native; `docs/c1/d2_serving.md`, `docs/audit/logs/c1_d2_native_C.log` | backed |
| GLM-5.2 route-slot D2 native-delegated graph path transfers | PPL 4.0705 (P4 band), decode 5.296 tok/s = 2.5x eager reference 2.10, PIECEWISE 3/3 + FULL 2/2, DSA `sparse_mla_sm120_decode_dsv3_2` native, coherent; `docs/audit/logs/c1_glm_d2_native_C.log` | backed |
| Custom dense grouped-GEMM required | Disproven for this stack: FlashInfer's `group_gemm_nvfp4_nt_groupwise` expresses the dense-anchor branch; `grouped_dense_nvfp4_moe_mm_2lvl` not built | not claimed (disproven) |
| Production-wide decode SOTA | NOT claimed: C1 measures native-delegate vs our own dequant-loop and eager paths (same model/policy), no production-serving-stack baseline table; phrased as a model/policy speed/quality/memory Pareto result | guarded (not claimed) |

## 11. C2 SOTA board — dense NVFP4 fused MoE is the SM120 MoE decode SOTA (measured)

Direct same-harness board (branch `c2-sota-board`) vs the strongest dense/NVFP4 baseline that runs on
SM120. Full docs [sota_board](../c2/sota_board.md) / [deepseek_sota](../c2/deepseek_sota.md) / [glm_sota](../c2/glm_sota.md) / [verdict](../c2/verdict.md); logs `docs/audit/logs/c2_*.log`.

| Claim | Evidence | Status |
|-------|----------|--------|
| Dense NVFP4 fused MoE (vLLM native FlashInfer-CUTLASS, QB_MOE=off) beats quadbit sparse D2 on decode | DeepSeek captured 48.248 vs 5.972 tok/s (8.1x); GLM captured 33.810 vs 5.367 tok/s (6.3x); same harness/passage/graph; `c2_ds_*`/`c2_glm_*` logs | backed |
| quadbit sparse D2 is NOT a decode-speed SOTA on SM120 MoE | above board; C1's 11.3x was vs the dequant loop (0.514), not the dense path | backed (negative) |
| Route-slot D2 dual residency INCREASES memory vs dense | weights +27% DeepSeek (51.7 vs 40.83) / +26% GLM (68.98 vs 54.62); GLM KV 236,672 vs 629,760 tok (-62%); pool 1.10/0.80 vs 0.18/0.10 GiB | backed (negative) |
| C2 quality: mito80 PPL is protocol-noise, excluded from ranking; downstream is the metric | D2 vs dense mito80 wash; downstream -0.79pt DeepSeek / -0.95pt GLM (paper §10, glm_results) | backed |
| Vanilla vLLM (no plugin) runs DeepSeek-V4-Flash NVFP4 on SM120 | fails to init (ue8m0 FP8 attention); the plugin's attention/DSA unblock is what lets the dense MoE baseline run at all | not claimed (disproven) |
| SGLang NVFP4 MoE baseline for these models on SM120 | unavailable (not in serve image; no SM120 DSA path); reported, not hidden | n/a (recorded) |
| quadbit MoE value = decode SOTA | NOT claimed. Value = only deployed 2:4-sparse FP4 MoE + training-free quality-preserving structural sparsity + graph-enabled cross-arch transfer (DeepSeek->GLM, DSA native) + prefill/large-M kernel Pareto (§5). Decode SOTA belongs to the dense fused MoE. | guarded (not claimed) |

## 12. C3 compact-routing decode — active-expert compaction speeds D2, dense fused MoE stays the decode SOTA

Branch `c3-compact-routing-decode`. Full docs [verdict](../c3/verdict.md) / [compact_routing_ab](../c3/compact_routing_ab.md) / [deepseek_compact_decode](../c3/deepseek_compact_decode.md) / [captured_attribution](../c3/captured_attribution.md); logs `docs/audit/logs/c3_*.log`.

| Claim | Evidence | Status |
|-------|----------|--------|
| The sparse 2:4 decode kernel is NOT the bottleneck (sparse-kernel premise refuted) | `matmul_sp` = 0.4% of the step (`profile_decode.md`); sparse *group* 24% is per-row overhead + E·cap padding, not the MMA; `fused_sparse_grouped_decode` not built | backed (negative) |
| The captured decode bottleneck is the E·cap=8192-row padding (MoE = 89% of step) | differential attribution: dense-anchor 64% (A−B), sparse 24% (A−C), non-MoE floor 11%, additive to 172.9 ms; `captured_attribution.md` | backed |
| Active-expert compaction is capture-safe and bit-correct | A_max=E correctness runs reproduce baseline PPL in the mito80 noise band, dense gather 4.096, sparse 4-tuple gather 4.045 (vs baseline 4.001, band 3.95–4.09); FULL capture; `compact_routing_ab.md` | backed |
| Compaction makes quadbit D2 decode 2.80× faster | compact-both 16.203 vs same-build non-compact 5.782 tok/s (2.80×); closes D2→dense gap 8.1×→3.0×; cross-run tok/s variance noted, direction monotone in rows removed | backed |
| Compact D2 beats the dense NVFP4 fused decode SOTA | NOT claimed (disproven): 16.203 < 48.248, still 3.0× slower | not claimed (disproven) |
| Compact D2 creates a strict Pareto point vs dense | NOT claimed: memory unchanged (D2 weights +27% dual residency), downstream quality −0.95pt (P1); dense dominates speed+memory+quality | not claimed (disproven) |
| GLM compact decode run | skipped: structurally identical to DeepSeek (own dense fused baseline far faster, same dual-residency memory + quality tax), cannot flip the verdict; 8-GPU cost avoided | n/a (skipped, reasoned) |

## 13. C4 floor-decode — one-shot all-reduce beats the SM120 decode SOTA (+20.5%)

Branch `c4-floor-decode`. Full docs [verdict](c4/verdict.md) / [floor_decomposition](c4/floor_decomposition.md) / [custom_allreduce](c4/custom_allreduce.md); logs `docs/audit/logs/c4_*.log`.

| Claim | Evidence | Status |
|-------|----------|--------|
| SM120 decode is 94.5% non-MoE floor / 5.5% MoE apply | dense step 20.73 ms/tok, skip-MoE floor 19.60 ms/tok (51.033 tok/s); MoE apply 1.13 ms; roofline from C2/C3 | backed |
| The floor is 90.8% one NCCL RING_LL all-reduce (not attention/DSA/EP-a2a) | vLLM worker profiler, dense baseline: AllReduce_RING_LL 90.8%, norm 4.6%, gemm 2.2%, attn+DSA 0.7%; `c4_floor_profile.log` | backed |
| Custom one-shot all-reduce beats the dense NVFP4 decode SOTA row | median 58.126 tok/s (4 runs; mean 58.145) vs the 48.248 prior SOTA row = +20.5% (+18.0% vs a 49.263 fresh control), FULL capture; `c4_custom_ar*.log` | backed (speed) |
| C4 is quality-neutral | NOW BACKED by C6 downstream eval (see section 15): custom AR engaged (sparse D2 R5) vs its NCCL twin = +0.40 pt AVG, lower PPL, no task collapse, inside the 0.38 pt dense-NCCL noise band; the AR is the MoE-policy-independent attention-TP reduce so it transfers to dense; mito80 PPL movement is reduction-order noise, not a ranking metric | backed (C6, scoped: engaged row is sparse; dense-engaged direct row blocked by Modal P2P-container luck) |
| The win is a disabled fast-path, not new hardware capability | vLLM disables custom AR on >2 PCIe GPUs; driver `can_device_access_peer` all-connected; `VLLM_SKIP_P2P_CHECK=1` makes it engage (0 disable warnings across skip-check runs) | backed |
| NCCL tree all-reduce alone matches the win | NO (disproven): scoped `allreduce:tree` = 48.983 tok/s (+1.5% only); the one-shot AR is the lever | not claimed (disproven) |
| C4 is a sparse-MoE or custom-kernel contribution | NOT claimed: it is a serving-infra collective-algorithm swap, applies to dense and sparse alike (shared floor) | guarded (not claimed) |
| C4 is a general multi-GPU all-reduce improvement | NOT claimed: targets small latency-bound decode all-reduces on PCIe-only topologies with driver P2P; vLLM's disable is correct for bandwidth-bound large-tensor cases | guarded (not claimed) |
| mito80 PPL 4.2514 (vs 4.1222) is a quality regression | NOT claimed: reduction-order noise (tree 4.01 / ring 4.12 / one-shot 4.25, both directions); AR is a correct sum; the downstream 4-task eval that is the real check was RUN in C6 and shows no regression (engaged custom AR +0.40 pt AVG, lower PPL) | not claimed (disproven by C6 downstream) |

## 14. C5 collective-floor — ceiling reached (C4 one-shot AR remains the SM120 decode SOTA)

Branch `c5-collective-floor`. Full docs [verdict](c5/verdict.md) / [post_c4_roofline](c5/post_c4_roofline.md) / [reduce_count_ab](c5/reduce_count_ab.md) / [hierarchical_ar](c5/hierarchical_ar.md) / [final_board](c5/final_board.md); logs `docs/audit/logs/c5_*.log`.

| Claim | Evidence | Status |
|-------|----------|--------|
| Decode floor is 91-94% one collective, ~1 all-reduce per layer | `floor_profile` count: 1392 AR / 32 forwards = 43.5/tok, DSA 1376 = 43/forward, AR/layer=1.01; NCCL 94.5% + custom-AR 91.2% | backed |
| Each one-shot all-reduce is ~374 us = PCIe sync latency, not transfer | 58.126 captured step 17.20 ms x 94% / 43.5 = 374 us for a ~14 KB payload | backed |
| All-reduce count is reducible by a safe algebraic transform at batch=1 | NOT true (disproven): the ~1 AR/layer is structural (RMSNorm needs the reduce; no slice/delay/fuse at batch=1) | not claimed (disproven) |
| Reduce ranks (TP=2) improves decode | NO (disproven): 40.565 < 48.248; 2x weight bytes/GPU outweighs the faster 2-GPU AR | not claimed (disproven) |
| A hierarchical all-reduce beats the one-shot for full-P2P 4-GPU | NO: one-shot is single-sync (optimal for tiny payloads); tree adds a stage (+1.5% only); no variant built | not claimed |
| DP attention removes the attention all-reduce (the lever) | correct structurally but BLOCKED: offline `LLM` rejects data_parallel_size>1 in all modes; needs vllm serve/AsyncLLM | n/a (blocked, next lever) |
| C5 beats C4's 58.126 tok/s | NO: no C5 row beats it; C4 +20.5% stands; success condition not met | not claimed (honest negative) |
| Modal 4-GPU P2P topology is uniform | NOT true: full-P2P and partial-P2P (`[1<->2]` only) containers both observed; C4 win conditional on full-P2P, safety guard falls back otherwise | backed (caveat) |

## 15. C6 collective-quality validation — C4 one-shot custom AR is quality-safe

Branch `c6-c4-quality-validation`, commit `5b6b9e5`. Full docs [c4_quality_eval](c6/c4_quality_eval.md) / [downstream_table](c6/downstream_table.md) / [verdict](c6/verdict.md); logs `docs/audit/logs/c6_*.log`. 4x RTX PRO 6000, `enforce_eager`, greedy, limit 400, repo 4-task MC smoke suite.

| Claim | Evidence | Status |
|-------|----------|--------|
| The C4 custom AR preserves downstream quality | engaged custom-AR row (sparse D2 R5 0.7341) vs its NCCL twin (R4 0.7301) = +0.40 pt AVG, PPL 3.538 vs 3.588, no task collapse, inside the 0.38 pt dense-NCCL noise band; plugin log `full P2P verified -> one-shot custom AR enabled` | backed (scoped to engaged sparse row) |
| Eager eval validates the captured speed row | CUDA-graph capture replays identical kernels with identical bf16 reduction order; only latency differs, not logits | backed (argument) |
| The dense custom-AR row engaged custom AR | NO: 6 dense attempts (R3 + 5 retries) all hit partial-P2P Modal containers and fell back to NCCL; dense-engaged direct point not obtained | not obtained (Modal P2P-container luck; R5 + policy-independence carry it) |
| C4 is a quality-safe decode SOTA (not speed-only) | verdict A: +20.5% decode with a stable downstream envelope; PPL shift is reduction-order-sensitive, not a regression | backed (verdict) |
| C6 reproduces the frozen reference AVGs | dense NCCL band 0.7344-0.7382 vs `deepseek_final.csv` dense 0.7383; sparse D2 NCCL 0.7301 vs d2_slot2 0.7304 | backed |

## 16. C7 DP-attention serve-latency lever — verdict D, C4 ceiling stands

Branch `c7-dp-attention-serve`. Full docs [verdict](c7/verdict.md) / [mode_validation](c7/mode_validation.md) / [serve_baseline](c7/serve_baseline.md) / [dp_attention_ab](c7/dp_attention_ab.md) / [quality_guardrail](c7/quality_guardrail.md) / [sparse_d2_transfer](c7/sparse_d2_transfer.md); logs `docs/audit/logs/c7_dp_*.log`. 4x RTX PRO 6000 (SM120), no NVLink.

| Claim | Evidence | Status |
|-------|----------|--------|
| DP-attention activates offline (breaks the C5 blocker) | env-driven SPMD (VLLM_DP_* per rank, no data_parallel_* kwargs) constructs tp=1/dp=4/EP engines `Worker_DP0_EP0..DP3_EP3`; both graph modes exit [0,0,0,0] | backed |
| DP-attention removes the attention TP all-reduce | trace custom=0 ring=0 per-token in both eager and captured | backed |
| The collective floor drops toward 0 (the C7 hope) | NO (disproven): floor moves to EP path — 1376 allgather + 1376 reduce-scatter = 2 collectives/layer vs C4's 1; AllGather 39-76% of decode CUDA time, 956us-5.3ms/call, PCIe-bound | not claimed (disproven) |
| Captured DP-attention beats C4 58.126 | NO (honest negative): 20.450 tok/s = 2.84x slower, same single-request metric, same 4 GPUs | not claimed |
| DP-attention changes model quality | NO: ppl 4.2640 (= C4 4.264), coherence probes correct (Paris / fibonacci / RGB) — pure execution-mode change | backed (quality-safe) |
| Aggregate serving throughput of the 4 DP replicas is a decode SOTA | NO: 4x20.450 = 81.8 tok/s exists but at 2.84x-worse per-request latency; labeled aggregate per the guardrail, not a single-request decode win | not claimed (guardrail honored) |
| Sparse D2 transfer under DP-attention | not run: spec gate "only if dense improves" not met (dense regressed 2.84x) | not obtained (gate not met) |
| C7 verdict | D — AR count drops to 0 but a worse EP allgather+reduce-scatter floor dominates; C4 58.126 SOTA unchanged | verdict |

## 17. C8 pipeline/layer-stage serve-latency lever — verdict D, C4 ceiling stands

Branch `c8-pipeline-stage-decode`. Full docs [verdict](c8/verdict.md) / [mode_feasibility](c8/mode_feasibility.md) / [stage_baseline](c8/stage_baseline.md) / [final_board](c8/final_board.md); logs `docs/audit/logs/c8_pp_*.log`. 4x RTX PRO 6000 (SM120), no NVLink. Same checkpoint/quant/prompt/metric as C4/C7.

| Claim | Evidence | Status |
|-------|----------|--------|
| Offline PP is expressible (unlike C5/C7 DP, which raised) | plain `LLM(tensor_parallel_size=1, pipeline_parallel_size=4)` constructs; no single-process guard; `mp` executor; deepseek_v2 base class is PP-capable | backed |
| PP is genuine layer-staging, not replicas/aggregate | one model sharded by distinct layer range `[11,11,11,10]`, 41.0 GiB/GPU (~1/4, not 142GB whole); single batch=1 request flows through all 4 stages | backed (guardrail honored) |
| PP removes C4's per-layer TP all-reduce floor | trace all-reduce = **0 per token** (C4 ~43.5), replaced by 3 stage-boundary send/recv (pp-1); no EP collectives | backed |
| Captured PP stage beats C4 58.126 | NO (honest negative): 22.930 tok/s = **2.53x slower**; capture gave 3.04x (7.539→22.930) and still fell short | not claimed |
| Why it lost | batch=1 pp=4 serializes the 4-way compute TP parallelizes and idles 3/4 GPUs; per-token wall 43.6ms (C8) vs 17.2ms (C4) despite all-reduce gone — serialization penalty > removed all-reduce time | backed |
| PP stage changes model quality | NO: ppl 4.19 (eager) / 4.29 (captured) vs 4.264 baseline, coherence probes correct — pure execution-mode change | backed (quality-safe) |
| Aggregate serving throughput of pp=4 is a decode SOTA | NO: pp=4 leaves ~75% GPU idle so microbatched QPS would be higher, but that is throughput not single-request latency; left unmeasured/unclaimed per guardrail | not claimed (guardrail honored) |
| Task 3 bubble removal could close the gap for single-request | NO: at batch=1 there is one microbatch — every bubble-fill lever (double-buffer/overlap/microbatch) adds concurrent requests = aggregate throughput, forbidden as single-request claim | not obtained (structurally aggregate-only) |
| Sparse D2 (Task 5) / GLM (Task 6) | not run: gates "only if dense improves/wins" not met (dense regressed 2.53x); no 8-GPU GLM launched | not obtained (gate not met) |
| C8 verdict | D — all-reduce floor removed to 0 but batch=1 pipeline serialization dominates; C4 58.126 SOTA unchanged. Across C4/C7/C8 the per-layer all-reduce is the price of 4-way concurrency, not removable overhead | verdict |

## 18. Gap C full-stack QAT capability recovery (single dense model) — KILL

Controlled test of whether the single-dense-model 2:4-FP4 downstream residual is a recovery artifact or a
real capacity floor. Branch `qat-fullstack-capability`, harness `harness/finetune_fullstack.py`, full note
[docs/qat/design.md](qat/design.md) Result section. Llama-3.1-8B-Instruct, single RTX PRO 6000. Downstream
= in-loop 3-task MC (ARC-Challenge/HellaSwag/Winogrande, 200 each = 600); NOT the 4-task MoE AVG, do not
cross-compare absolute values.

| Claim | Evidence | Status |
|-------|----------|--------|
| Full-stack QAT (widen sparsified+trainable+STE to attn q/k/v/o + MLP; capability corpus from downstream TRAIN splits; downstream-in-loop selection) does NOT recover capability | `harness/finetune_fullstack.py`; `docs/qat/design.md` Result | backed (negative) |
| Final through-kernel downstream 0.3967 is BELOW the one-shot 0.4333 bar and far below dense teacher 0.6150 | `docs/qat/design.md` Result core table; run app `ap-Ggm58ogbWn2eNNs0cQQNBW` logs | backed (negative) |
| Deploy gap is 0.005 (fake-quant 0.4017 -> kernel 0.3967): limiting factor is capacity/recovery, not kernel/STE fidelity | `docs/qat/design.md` Result; `finetune_fullstack.py::QuadbitLinear` pack | backed |
| Phase-2 downstream oscillated 0.378-0.402 across all 6 evals with no upward drift | `docs/qat/design.md` Result trajectory table | backed |
| Capability corpus has zero eval leakage (positive-control PASS, 13-gram decontam vs all eval splits) | `harness/build_capability_corpus.py` manifest (`leak_in_sample` 0, `positive_control_hits` > 0) | backed |
| Per-task through-kernel breakdown (broad vs concentrated failure) | `harness/finetune_fullstack.py::evalpertask`, read-only on the saved best-cap weights; `docs/qat/design.md` Result per-task table | backed |
| Recovery beats one-shot / closes the residual | NOT claimed (disproven): recovery worsened both PPL (149->203) and downstream (0.4333->0.3967) vs one-shot | not claimed (disproven) |
| This is a kernel-fidelity, OOM, or attention-packing/STE failure | NOT claimed (disproven): deploy gap 0.005, run completed with ~14.4 GB free at phase-2 peak, STE bit-matches the kernel | not claimed (disproven) |
| The run used the pre-registered recipe (no knob tuning) | infra-only fixes `2ff580a` (resume GPU-duplicate) + `90be3e2` (durable phase-2 resume); LR/steps/alpha/attn/optimizer/selection/corpus/hardware/model/eval unchanged | backed |
