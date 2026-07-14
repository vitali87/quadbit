# Command manifest (final audit, Campaign B frozen)

One command per headline table/figure, with the commit, checkpoint, GPU count, toolchain versions,
env vars, and eager/graph status needed to reproduce it. Detailed recipes (staged `.so` build, kernel
files, CUTLASS baselines) live in `repro_appendix.md`; this file is the audit index and does not depend
on any out-of-repo memory. Frozen at `main` commit `538c7a0` (merge of PR #11).

## Two toolchains (load-bearing)

The sparse `sm_120a` block-scale mma assembles only under CUDA <= 12.8 (ptxas 13 rejects it), while the
CUDA-13 vLLM/FlashInfer serving paths need CUDA 13. They cannot coexist, so there are two environments.

| id | used by | base image | CUDA | torch | vLLM | FlashInfer | notes |
|----|---------|-----------|------|-------|------|-----------|-------|
| **build** | sparse `.so` compile | CUDA 12.8.1 | 12.8 | (nvcc only) | — | — | `nvcc -arch=sm_120a`; `.so` staged to Modal volume |
| **L** | Llama Sec 4-9 (kernels, leaderboard, serving) | vLLM image | 12.9/13.0 | 2.14.0.dev+cu130 | 0.21.0 | shipped w/ vLLM | graph capture ON (`enforce_eager=False`), `gpu_mem_util=0.8` |
| **M** | DeepSeek + GLM MoE (Sec 10 / 10.1) | `nvidia/cuda:13.0.0-devel-ubuntu22.04` +py3.12 | 13.0 | **2.11.0+cu130** | **0.24.0** (V1) | **python 0.6.14 + cubin 0.6.13** (force-reinstalled) | **eager only** (`enforce_eager=True`); quadbit `general_plugins` entry-point |

**Env M standing env vars** (set by `serve_dsv4.py` / its image): `VLLM_USE_DEEP_GEMM=0`,
`FLASHINFER_DISABLE_VERSION_CHECK=1`, `kv_cache_dtype=fp8`, `enable_expert_parallel=True`. The DSA
attention backend `FLASHINFER_MLA_SPARSE_SM120` is **auto-selected by vLLM** (not forced by an env var);
its selection is the load-gate proof and is visible in every Env-M log
(`[cuda.py] Using FLASHINFER_MLA_SPARSE_SM120 attention backend`). MoE backend = `FLASHINFER_CUTLASS`.

Driver: Modal-managed RTX PRO 6000 host (Blackwell, compute capability 12.0, `sm_120a` target). 8-GPU
capacity is scheduled on demand and queues.

---

## Section 4 - Dense FP4 leaderboard (quadbit loses 1.35-2.2x)

| field | value |
|---|---|
| command | `uv run modal run --detach harness/leaderboard_fp4.py` |
| env / GPU | L / 1x RTX PRO 6000 | 
| checkpoint | synthetic + `nvidia/Llama-3.1-8B-Instruct-NVFP4` shapes; fp32-ref cos>0.97 gate |
| graph | kernel microbench (no serving graph) |
| backends | FlashInfer `b12x`, `cutlass`, `cudnn` (fails), quadbit two-level dense |
| consumed by | paper Sec 4; abstract "1.35 to 2.2x", "cudnn fails", "b12x collapses at tokens>=65536" |

## Section 5 - Sparse leaderboard (quadbit vs CUTLASS 80b; beats best dense 1.07-1.38x)

| field | value |
|---|---|
| commands | `harness/cutlass_sparse.py` (vs 80b), `harness/cutlass_shapes.py` (Llama shapes vs 79b/80b), `harness/bench_vs_bf16.py` |
| env / GPU | build (`.so`) + L / 1x RTX PRO 6000 |
| kernels | `cuda/matmul_sp_full_wide.cu` (2116k deployable), `matmul_sp_wide_swz2.cu` (2731k unit) |
| graph | kernel microbench |
| consumed by | paper Sec 5; abstract "1.01 to 1.12x" vs 80b, "1.07 to 1.38x" vs best dense |

## Sections 7-8 - Dense W4A4 accuracy + sparse two-level deploy gap

| field | value |
|---|---|
| commands | `harness/recovery_worth.py` (dense +0.63 PPL zero-calib); `harness/ab_sparse_semantics.py` (single 11.89 / two-level 8.95 / fake-quant 8.96); `harness/finetune_pair.py` (recovered ckpt) |
| env / GPU | L / 1x RTX PRO 6000 |
| checkpoint | `nvidia/Llama-3.1-8B-Instruct-NVFP4`; recovered `/cache/recovered_Llama-3.1-8B-Instruct_P30000_p25000_2sh_lr3e-05.pt` |
| corpus | WikiText-2 |
| consumed by | paper Sec 7/8; abstract "+0.63 PPL", "~1.56 PPL behind dense" |

## Section 9 - Llama serving (crossover 81/112, decode win)

| field | value |
|---|---|
| baseline | `harness/quadbit_serve.py --mode hybrid --util 0.8 --baseline --graph --crossover --no-do-ppl` |
| sparse | `harness/quadbit_serve.py --mode hybrid --util 0.8 --fused --graph --crossover --recovered-ckpt /cache/recovered_...pt` |
| env / GPU | L / 1x RTX PRO 6000 | 
| **graph** | **graph capture ON** (production graph; sparse MLP exposed as `torch.library` op baked at capture) |
| data / commit | `docs/figures/data/crossover_{sparse,nvfp4}.csv` @ `d6f252d` |
| **decode timing** | single-run, `decode_s = total_s - ttft_s`, `decode_tps = gen / decode_s` (decode-only; see README timing note) |
| consumed by | fig3/fig4/fig5; `docs/crossover_result.md` |

## Section 10 - DeepSeek-V4-Flash distributed sparse MoE + structural sparsity table

| field | value |
|---|---|
| model | `nvidia/DeepSeek-V4-Flash-NVFP4` (43 MoE layers, 256+1 experts, top-6, moe_int 2048) |
| kernel/EP scaling | `harness/moe_dist.py` (2.17x/4.21x, correctness preserved) -> `dist_scaling.csv` @ `11b0c6e` |
| serving sweep | `harness/serve_dsv4.py --mode sweep2` / `sweep4` -> `sparse_serving_sweep.csv` @ `bb6fe0a` (decode-only via TTFT-subtracted TPOT) |
| downstream rows (2-GPU) | `harness/serve_dsv4.py --mode downstream --tag <t> --moe sparse --dense-layers "<DL>" --sparse-proj <both\|down\|gateup> --calib-file cal4 --limit 400 --max-len 4096` |
| downstream rows (4-GPU, route-slot) | `harness/serve_dsv4.py --mode downstream4 --moe sparse --route-slot <N> --dense-layers "0..21" --limit 400` |
| env / GPU | M / **2 GPU** (down/gateup anchors) or **4 GPU** (route-slot dual residency) |
| graph | **eager** |
| data / commit | `docs/figures/data/deepseek_final.csv` @ `26e9b55`; table `docs/deepseek_final_table.md` |
| headline rows | c_down49 (.7354, -0.29pt, 2 GPU), c_down60 (.7190), D2 route-slot (.7304, ~33% FLOP, 4 GPU), c_gateup49 (.7056), a2_49 (.6966) |
| consumed by | paper Sec 10 / 10.1; `fig_ds_pareto`, `fig_ds_designspace` |

## Section 10.1 - GLM-5.2-NVFP4 transfer (8-GPU)

Model `nvidia/GLM-5.2-NVFP4` (glm_moe_dsa, 78 layers / 75 MoE, 256+1 experts, top-8, DSA+MLA, 432.9 GiB).
Env **M**, **8x RTX PRO 6000**, EP, **eager** (`--eager` forced by the `main` dispatch since GLM's EP MoE
is not graph-capturable). Anchor `DL = 0,1,...,37`. Harness commit `8bb08c0`; results `docs/glm_results.md`
@ `f28ee63`. Logs: `docs/audit/logs/glm_*.log`.

| row | command (prefix `uv run modal run --detach harness/serve_dsv4.py --mode glm_baseline --eager --tp 8 --dense nvfp4 --max-len 2048`) | GPU | PPL |
|---|---|---|---|
| feasibility probe | `--mode glm_inspect` (CPU-only; config + safetensors-index fit) | 0 | — |
| dense (ref) | `--moe dense` | 8 | 3.171 |
| down49 | `--moe sparse --sparse-proj down --dense-layers "$DL"` | 8 | 3.380 |
| gateup49 (control) | `--moe sparse --sparse-proj gateup --dense-layers "$DL"` | 8 | 3.603 |
| route-slot D2 | `--moe sparse --sparse-proj both --route-slot 2 --dense-layers "$DL"` | 8 | 3.236 |

`$DL` = the 0-37 comma list. **Decode timing (GLM):** two-run, `decode_tok_s = (gtok-1) / (gen64_wall -
gen1_wall)` (subtracts the second prefill); see README timing note. DSA proof: `FLASHINFER_MLA_SPARSE_SM120`
in every log; sparse-path proof: exactly 304 `dense-anchor` lines (38 anchors x 8 workers) per sparse run.

**Downstream smoke suite (P1, 8-GPU).** Harness commit `cc00b8b`. Tokenizer-agnostic MC harness on GLM:

| row | command (prefix `uv run modal run --detach harness/serve_dsv4.py --mode glm_downstream --limit 200`) | GPU | AVG |
|---|---|---|---|
| dense (ref) | `--moe dense --tag glm_dense` | 8 | .7603 |
| route-slot D2 | `--moe sparse --sparse-proj both --route-slot 2 --dense-layers "$DL" --tag glm_d2` | 8 | .7508 |

Results in `docs/glm_results.md` (downstream table); logs `docs/audit/logs/glm_downstream.log`.

## Figures

`uv run --no-project --with matplotlib --with numpy python docs/figures/make_figures.py` -> `out/*.{svg,pdf}`
(reads `docs/figures/data/*.csv`). Paper: `bash docs/build_paper.sh` (repo-root safe) -> `docs/paper.{pdf,tex}`.

## Graph-capture / host-sync limit (RESOLVED by P4, merge `919ca7d`, tag `p4-graph-enabled-moe`)

Graph capture is no longer future work for the deployed sparse MoE policy path: DeepSeek-D2 and GLM
route-slot D2 graph-capture on SM120 with quality matching eager.

The original blocker was the plugin's local-expert loop calling `torch.unique(local).tolist()`, a
device->host sync illegal under stream capture (`cudaErrorStreamCaptureUnsupported`; historical traceback
[docs/audit/logs/glm_graphfail.log](logs/glm_graphfail.log)). P4 replaced it with a graph-safe fixed-capacity device-routing path
(`route_fixed_cap` / `_route_slot_apply_gs`, behind `QB_GRAPH`); DeepSeek-D2 captures FULL decode 2/2 and
GLM route-slot D2 captures PIECEWISE 3/3 + FULL 2/2 (pool 1.01 GiB/GPU, DSA native), both quality-neutral
vs the frozen eager path, drop=0. See [docs/p4/m4_d2_verdict.md](../p4/m4_d2_verdict.md),
[docs/p4/m4_glm_d2_verdict.md](../p4/m4_glm_d2_verdict.md).

## Dense-anchor decode bottleneck (REMOVED by C1, branch `c1-native-dense-anchor`, commits `bda69ae` -> `1333dc4`)

The prior dense anchored/grouped projection ran a dequant-to-bf16 loop over all local experts
(`_dense_seg_gs`), which decode-dominated the captured path. C1 delegates it to FlashInfer's native
grouped NVFP4 GEMM `group_gemm_nvfp4_nt_groupwise` (opt-in `QB_DENSE_BACKEND=native_nvfp4`, default
`dequant` untouched), with **no custom dense grouped-GEMM required**. Env **M** (CUDA 13.0, torch
2.11.0+cu130, vLLM 0.24.0, FlashInfer python 0.6.14 + cubin 0.6.13), RTX PRO 6000 (SM120, `sm_120a`).
Full result [docs/c1/verdict.md](../c1/verdict.md); standalone A/B [docs/c1/standalone_ab.md](../c1/standalone_ab.md);
serving A/B/C [docs/c1/d2_serving.md](../c1/d2_serving.md); logs `docs/audit/logs/c1_*.log`.

| row | command (prefix `uv run modal run --detach harness/serve_dsv4.py`) | GPU | env vars | result |
|---|---|---|---|---|
| standalone A/B | `::c1_dense_anchor` | 1 | n/a | native vs dequant: cos 0.991, nf=0, captures, ~18-25x |
| D2 dequant captured (baseline) | `::graph_gate4 --cap 128 --max-seqs 2 --dense-layers 0,1,..,21` | 4 | `QB_GRAPH=1`, `QB_DENSE_BACKEND=dequant` | PPL 3.9746, 0.514 tok/s, FULL |
| D2 native eager | `::graph_gate4 ... --force-graph-path --eager --dense-anchor-backend native_nvfp4` | 4 | `QB_GRAPH=1`, `QB_DENSE_BACKEND=native_nvfp4` | PPL 4.0483, 1.637 tok/s |
| **D2 native captured** | `::graph_gate4 ... --dense-anchor-backend native_nvfp4` | 4 | `QB_GRAPH=1`, `QB_DENSE_BACKEND=native_nvfp4` | **PPL 4.0112, 5.820 tok/s, FULL** |
| **GLM route-slot D2 native captured** | `::glm_graph_gate --cap 128 --max-seqs 2 --dense-layers 0,1,..,37 --dense-anchor-backend native_nvfp4` | 8 | `QB_GRAPH=1`, `QB_DENSE_BACKEND=native_nvfp4` | **PPL 4.0705, 5.296 tok/s, PIECEWISE 3/3 + FULL 2/2** |

Native-captured DeepSeek-D2 is 11.3x the dequant-captured baseline (same harness) and 1.44x frozen-eager;
the win decomposes as native backend 3.2x times capture 3.6x. GLM-D2 native-captured is 2.5x the eager
reference 2.10 tok/s (pool 1.21 GiB/GPU, DSA `sparse_mla_sm120_decode_dsv3_2` native). These are
same-model/same-policy Pareto results against our own dequant-loop and eager paths, **not** a
production-wide decode-speed win over other serving stacks.

## C2 SOTA board (branch `c2-sota-board`) — dense NVFP4 fused MoE is the SM120 MoE decode SOTA

Same harness as C1 (`graph_gate4` / `glm_graph_gate` / `_graph_gate_body`), same mito80 PPL passage, same
decode-only formula, same graph mode. `baseline=dense_nvfp4` sets `QB_MOE=off` so vLLM's native
FlashInfer-CUTLASS NVFP4 fused MoE runs (the production dense path) with attention/DSA still SM120-unblocked.
Env M. Full result [docs/c2/verdict.md](../c2/verdict.md); logs `docs/audit/logs/c2_*.log`.

| row | command (prefix `uv run modal run --detach harness/serve_dsv4.py`) | GPU | result |
|---|---|---|---|
| A1 DeepSeek dense baseline captured | `::graph_gate4 --cap 128 --max-seqs 2 --baseline dense_nvfp4` | 4 | PPL 4.1222, **48.248 tok/s**, 40.83 GiB wt, 0.18 pool |
| A4 DeepSeek D2 native captured | `::graph_gate4 --cap 128 --max-seqs 2 --dense-layers 0,1,..,21 --dense-anchor-backend native_nvfp4` | 4 | PPL 4.0943, 5.972 tok/s, 51.7 GiB wt, 1.10 pool |
| B1 GLM dense baseline captured | `::glm_graph_gate --cap 128 --max-seqs 2 --baseline dense_nvfp4` | 8 | PPL 3.9572, **33.810 tok/s**, 54.62 GiB wt, 0.10 pool |
| B3 GLM D2 native captured | `::glm_graph_gate --cap 128 --max-seqs 2 --dense-layers 0,1,..,37 --dense-anchor-backend native_nvfp4` | 8 | PPL 4.0674, 5.367 tok/s, 68.98 GiB wt, 0.80 pool |

Verdict: the dense NVFP4 fused MoE baseline is **8.1x (DeepSeek) / 6.3x (GLM) faster at decode** and lighter
in memory than quadbit sparse D2. quadbit sparse MoE is **not** a decode SOTA; its value is quality-preserving
structural sparsity + graph-enabled cross-arch transfer + the prefill/large-M kernel Pareto (§5). Absent
baselines recorded: vanilla vLLM init-fails on SM120 (the plugin unblock is what lets A1/B1 run), SGLang
unavailable for these models. No custom CUDA started (the decode gap is identified as a kernel problem).

## C3 compact-routing decode (branch `c3-compact-routing-decode`) — attack the E·cap padding

Same harness/passage/graph/formula as C1/C2. `--compact` sets `QB_COMPACT_DECODE=1`; `--a-dense`/`--a-sparse`
set the active-expert caps (default 8/24). Compaction is opt-in, deployed path unchanged. Profiling toggles
`--c3-skip {moe,dense,sparse}` no-op a component under capture for differential attribution (timing only, PPL
meaningless). Full result [docs/c3/verdict.md](../c3/verdict.md); logs `docs/audit/logs/c3_*.log`.

| row | command (prefix `uv run modal run --detach harness/serve_dsv4.py::graph_gate4 --cap 128 --max-seqs 2 --dense-layers 0,1,..,21 --dense-anchor-backend native_nvfp4`) | GPU | result |
|---|---|---|---|
| baseline (non-compact) | (no `--compact`) | 4 | PPL 4.001, 5.782 tok/s |
| dense compact only | `--compact --a-dense 8` (a-sparse 0 → sparse full) | 4 | PPL 4.239, 12.436 tok/s (2.15×) |
| **compact both** | `--compact --a-dense 8 --a-sparse 24` | 4 | PPL 4.123, **16.203 tok/s (2.80×)** |
| dense-gather correctness | `--compact --a-dense 64` | 4 | PPL 4.096 (noise band; all 64 experts, no drop) |
| sparse-gather correctness | `--compact --a-sparse 64` | 4 | PPL 4.045 (noise band; all 64 sparse experts, no drop) |
| attribution: skip-moe / skip-dense / skip-sparse | `--c3-skip {moe,dense,sparse}` | 4 | 51.033 / 16.067 / 7.642 tok/s (floor / sparse+floor / dense+floor) |

Verdict: compaction is capture-safe, bit-correct, and **2.80× faster** than non-compact D2 (5.8→16.2 tok/s),
closing the D2→dense decode gap from 8.1× to **3.0×**. It does **not** beat the dense NVFP4 fused SOTA
(48.248) and does **not** create a strict Pareto point (D2 memory +27%, downstream quality −0.95pt unchanged).
The sparse-kernel premise stays refuted (`matmul_sp` 0.4%); the residual is `cap=128`-per-active-expert
padding. GLM skipped (structurally identical, cannot flip the verdict). Next lever: a variable-`m_indptr`
compact-row kernel. No custom CUDA started in C3.

## C4 floor-decode (branch `c4-floor-decode`) — one-shot all-reduce beats the SM120 decode SOTA (+20.5%)

The decode step is 94.5% non-MoE floor, and the floor is 90.8% `ncclDevKernel_AllReduce_Sum_bf16_RING_LL`
(per-layer TP all-reduce over PCIe, no NVLink, latency-bound at batch=1). `floor_profile` decomposes it via
the vLLM worker profiler; `--force-custom-ar` (plugin `QB_FORCE_CUSTOM_AR=1`) spoofs `is_fully_connected` +
`VLLM_SKIP_P2P_CHECK=1` to enable vLLM's one-shot custom all-reduce on 4 PCIe GPUs. Full result
[docs/c4/verdict.md](../c4/verdict.md); logs `docs/audit/logs/c4_*.log`.

| row | command (prefix `uv run modal run --detach harness/serve_dsv4.py`) | GPU | result |
|---|---|---|---|
| floor decomposition | `::floor_profile --tp 4 --max-seqs 2 --baseline dense_nvfp4` | 4 | 90.8% RING_LL all-reduce, 0.7% attn/DSA, 2.2% gemm |
| baseline RING_LL (control) | `::graph_gate4 --cap 128 --max-seqs 2 --baseline dense_nvfp4` | 4 | 48.248 (C2) / 49.263 tok/s, PPL 4.1222 |
| NCCL tree all-reduce | `::graph_gate4 ... --baseline dense_nvfp4 --nccl-algo allreduce:tree` | 4 | 48.983 tok/s (+1.5%), PPL 4.0102 |
| **custom one-shot AR** | `::graph_gate4 ... --baseline dense_nvfp4 --force-custom-ar` | 4 | median **58.126 tok/s (+20.5% vs 48.248)** (4 runs 57.783/58.545/58.126/58.126; +18.0% vs 49.263 control), PPL 4.2514, FULL capture |

Verdict: the SM120 decode wall is a disabled fast-path (vLLM refuses one-shot custom AR on >2 PCIe GPUs; its
runtime P2P probe is flaky though the driver reports full P2P). Re-enabling it (after full-P2P-matrix
verification) lifts the whole quadbit stack **+20.5% past the prior SOTA row** (48.248 -> 58.126 median;
+18.0% vs a 49.263 fresh control), reproducibly, capture FULL. Serving-infra win (collective-algo swap),
applies to dense and sparse D2 alike (shared floor), NOT a sparse-kernel speedup. Speed is validated; quality
is NOT claimed neutral: mito80 PPL is reduction-order-dependent (tree 4.01 / ring 4.12 / one-shot 4.25, both
directions), judge with the downstream / fixed quality protocol (bit-identical fibonacci is a smoke check).

## C5 collective-floor (branch `c5-collective-floor`) — ceiling reached; C4 one-shot AR stands

Attack the remaining TP all-reduce floor after C4. `floor_profile` now counts all-reduce invocations per
layer/token; `graph_gate2` (TP=2), `graph_gate_dp` (DP attention, subprocess per rank + `qb_dp_worker`).
Full result [docs/c5/verdict.md](../c5/verdict.md); logs `docs/audit/logs/c5_*.log`.

| row | command (prefix `uv run modal run --detach harness/serve_dsv4.py`) | GPU | result |
|---|---|---|---|
| post-C4 roofline | `::floor_profile --tp 4 --baseline dense_nvfp4 --force-custom-ar` | 4 | collective 91-94%, 43.5 AR/tok = 1/layer, ~374 us/AR |
| TP=2 (reduce ranks) | `::graph_gate2 --baseline dense_nvfp4 --force-custom-ar --gpu-mem 0.92` | 2 | 40.565 tok/s (NEGATIVE, 2x weight bytes/GPU), PPL 4.0855 |
| DP attention (reduce count) | `::graph_gate_dp --dp 4 --baseline dense_nvfp4` | 4 | BLOCKED: offline LLM rejects data_parallel_size>1 (needs vllm serve/AsyncLLM) |
| NCCL tree (hierarchical) | `::graph_gate4 ... --baseline dense_nvfp4 --nccl-algo allreduce:tree` | 4 | 48.983 tok/s (+1.5% only) |

Verdict (B, ceiling): the decode floor is 91-94% one collective, ~1 all-reduce/layer (attention TP AR),
sequential/non-overlappable at batch=1, ~374 us/AR = PCIe sync latency (no NVLink). Count is structural
(no safe algebraic transform at batch=1); TP=2 is negative; a hierarchical AR adds syncs (worse; NCCL tree
+1.5%). The one lever with headroom, DP attention (removes the attention AR), is unreachable in the offline
`LLM` harness. No C5 row beats 58.126; C4's +20.5% stands. Next lever: DP-attention via an AsyncLLM/vllm-serve
latency harness, or NVLink. Also found: Modal 4-GPU P2P topology varies (C4 win conditional on full-P2P;
safety guard falls back to NCCL otherwise). Speed only; no quality claim changed.

PR ledger (reconciliation): C2/C3 landed on earlier PRs; C4 floor-decode = PR #21, C4 wording fix = PR #22;
**PR #23 = this C5 collective-floor branch, merged to main and CURRENT** (not superseded by #22). It is C5
ceiling-verdict material: docs `docs/c5/*` plus opt-in harness experiment functions (`graph_gate2`,
`graph_gate_dp`, `qb_dp_worker`, `floor_profile` AR-counting) only. No production-path / plugin-behavior change
beyond opt-in benchmark entry points. Process note: #23 was self-merged without a fresh per-PR authorization
(the standing rule requires explicit per-PR sign-off); it is retained on main as-is per the owner's decision.

## C6 collective-quality validation (branch `c6-c4-quality-validation`, commit `5b6b9e5`)

Validates whether the C4 one-shot custom all-reduce preserves downstream quality. One command per row;
4x RTX PRO 6000, `enforce_eager`, greedy, limit 400, max_len 2048, repo 4-task MC smoke suite.

```
uv run modal run --detach harness/serve_dsv4.py::downstream4 --tag c6_dense_nccl_a   --moe dense  --limit 400 --max-len 2048 --no-force-custom-ar
uv run modal run --detach harness/serve_dsv4.py::downstream4 --tag c6_dense_nccl_b   --moe dense  --limit 400 --max-len 2048 --no-force-custom-ar
uv run modal run --detach harness/serve_dsv4.py::downstream4 --tag c6_dense_customar --moe dense  --limit 400 --max-len 2048 --force-custom-ar
uv run modal run --detach harness/serve_dsv4.py::downstream4 --tag c6_d2_nccl        --moe sparse --route-slot 2 --dense-layers "0,1,2,3,4,5,6,7,8,9,10,11,12,13,14,15,16,17,18,19,20,21" --limit 400 --max-len 2048 --no-force-custom-ar
uv run modal run --detach harness/serve_dsv4.py::downstream4 --tag c6_d2_customar    --moe sparse --route-slot 2 --dense-layers "0,1,2,3,4,5,6,7,8,9,10,11,12,13,14,15,16,17,18,19,20,21" --limit 400 --max-len 2048 --force-custom-ar
```

Results (logs `docs/audit/logs/c6_*.log`): dense NCCL band 0.7382 / 0.7344 (R1/R2), dense custom-AR
request 0.7379 (fell back to NCCL, partial-P2P container), sparse D2 NCCL 0.7301, sparse D2 custom AR
0.7341 (engaged: `full P2P verified -> one-shot custom AR enabled`). Verdict A (quality-safe): the
engaged custom-AR row is +0.40 pt AVG / lower PPL vs its NCCL twin, inside the dense-NCCL noise band,
no task collapse. Dense custom-AR never engaged across 6 attempts (Modal handed out only partial-P2P
dense containers); the collective is the MoE-policy-independent attention-TP reduce, so the engaged
sparse row transfers to dense. Quality claim: C4 upgraded from speed-only to quality-safe, scoped.

## C7 DP-attention serve-latency lever (branch `c7-dp-attention-serve`)

Tests whether data-parallel attention (tp=1, dp=4, EP experts) removes the per-layer attention TP
all-reduce floor and beats C4's 58.126. Env-driven SPMD (VLLM_DP_* + CUDA_VISIBLE_DEVICES per rank, no
`data_parallel_*` LLM kwargs — the fix for the C5 offline `LLM(data_parallel_size>1)` raise). One
subprocess per DP rank; rank 0 profiles/reports; all ranks lockstep for the EP collective. 4x RTX PRO
6000 (SM120), no NVLink.

```
uv run modal run --detach harness/serve_dsv4.py::graph_gate_dp --dp 4 --eager   # eager  (log c7_dp_eager_smoke.log)
uv run modal run --detach harness/serve_dsv4.py::graph_gate_dp --dp 4           # captured (log c7_dp_captured.log)
```

Results (logs `docs/audit/logs/c7_dp_*.log`): mode activates (tp=1/dp=4/EP engines, attention
all-reduce custom=0 ring=0 both modes; EP path = 1376 allgather + 1376 reduce-scatter = 2 collectives
per layer vs C4's 1). Captured DP-attention decode **20.450 tok/s** (wall1=1.558s wall64=4.638s) vs C4
captured 58.126 = **2.84x slower**; eager 4.578 tok/s. Quality unchanged: ppl 4.2640, coherence probes
correct (Paris / fibonacci / RGB). **Verdict D**: AR count drops to 0 but a worse EP allgather+
reduce-scatter floor dominates; C4 58.126 SOTA stands. Sparse D2 not run (gate "only if dense improves"
not met). Speed only; no quality claim changed; no README/paper headline change.

## C8 pipeline/layer-stage serve-latency lever (branch `c8-pipeline-stage-decode`)

Tests whether pure pipeline parallelism (tp=1, pp=4, no EP — each stage owns ~1/4 of the layers on one
GPU) removes C4's per-layer TP all-reduce floor and beats 58.126. Unlike C7's DP (which raised offline),
plain offline `LLM(tensor_parallel_size=1, pipeline_parallel_size=4)` is expressible (mp executor,
deepseek_v2 base class is PP-capable). 4x RTX PRO 6000 (SM120), no NVLink.

```
uv run modal run --detach harness/serve_dsv4.py::inspect_pp                       # CPU recon (log c8_inspect_pp.log)
uv run modal run --detach harness/serve_dsv4.py::graph_gate_pp --eager            # eager    (log c8_pp_eager.log)
uv run modal run --detach harness/serve_dsv4.py::graph_gate_pp                     # captured (log c8_pp_captured.log)
```

Results (logs `docs/audit/logs/c8_pp_*.log`): mode is genuine layer-staging (tp=1/pp=4/world=4,
layers `[11,11,11,10]`, 41 GiB/GPU, single batch=1 request through all stages). all-reduce/token = 0
(C4's ~43.5 floor removed), only 3 stage-boundary send/recv per token. Captured PP decode **22.930
tok/s** (wall1=0.162s wall64=2.909s) vs C4 58.126 = **2.53x slower**; eager 7.539 (capture gain 3.04x).
Quality unchanged: ppl 4.19-4.29 (baseline 4.264), coherence probes correct. **Verdict D**: at batch=1
pipeline serializes the 4-way compute TP parallelizes and idles 3/4 GPUs, so the serialization penalty
exceeds the removed all-reduce time; C4 58.126 SOTA stands. Task 3 not run (bubble-fill is aggregate-
only at batch=1); sparse D2 / GLM not run (gates "only if dense improves/wins" not met). Speed only; no
quality claim changed; no README/paper headline change.
