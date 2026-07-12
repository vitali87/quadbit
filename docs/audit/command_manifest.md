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
route-slot D2 graph-capture on SM120 with quality matching eager. The remaining speed limitation is the
dense anchored/grouped projection path, which lacks a fused dense NVFP4 grouped-GEMM.

The original blocker was the plugin's local-expert loop calling `torch.unique(local).tolist()`, a
device->host sync illegal under stream capture (`cudaErrorStreamCaptureUnsupported`; historical traceback
[docs/audit/logs/glm_graphfail.log](logs/glm_graphfail.log)). P4 replaced it with a graph-safe fixed-capacity device-routing path
(`route_fixed_cap` / `_route_slot_apply_gs`, behind `QB_GRAPH`); DeepSeek-D2 captures FULL decode 2/2 and
GLM route-slot D2 captures PIECEWISE 3/3 + FULL 2/2 (pool 1.01 GiB/GPU, DSA native), both quality-neutral
vs the frozen eager path, drop=0. See [docs/p4/m4_d2_verdict.md](../p4/m4_d2_verdict.md),
[docs/p4/m4_glm_d2_verdict.md](../p4/m4_glm_d2_verdict.md). This is
a graph-correctness and graph-enablement result, **not** a production-wide decode-speed win.
