# Campaign B freeze record

Frozen baseline for the P4 graph-capture work. **P4 must not regress any number here.** The entire
frozen tree is captured immutably by the git tag; this file consolidates the checkpoint IDs, Modal
environment, and headline results that P4 compares against.

- **Tag:** `campaign-b-freeze-a91c5d9`
- **Commit:** `a91c5d94e40af79fd782fe11958804346f003d25` (main, merge of PR #15)
- **PRs in the frozen set:** #12 (audit package), #13 (GLM downstream smoke), #14 (paper narrative), #15 (README sync)
- **P4 branch:** `p4-graph-capture` (off this commit; main is not mutated)

## Archived artifacts (all in-repo at the tag)

| artifact | path |
|---|---|
| paper (source + built) | `docs/paper.md`, `docs/paper.tex`, `docs/paper.pdf` |
| README (front door) | `README.md` |
| claims checklist | `docs/claims_checklist.md` |
| command manifest (per-table commands, toolchains, env) | `docs/audit/command_manifest.md` |
| audit package index + timing methodology | `docs/audit/README.md` |
| GLM downstream raw logs (dense + D2) | `docs/audit/logs/glm_downstream.log` |
| GLM serving/PPL raw logs | `docs/audit/logs/glm_runs.log` |
| GLM graph-capture failure traceback | `docs/audit/logs/glm_graphfail.log` |
| DeepSeek downstream raw logs | `docs/audit/logs/deepseek_downstream.log` |
| review-fix summary | `docs/audit/logs/greptile_fixes.md` |
| GLM policy results | `docs/glm_results.md` |
| DeepSeek final table + data | `docs/deepseek_final_table.md`, `docs/figures/data/deepseek_final.csv` |
| Llama serving results | `docs/crossover_result.md`, `docs/graph_serving_result.md`, `docs/frozen_serving_result.md`, `docs/accuracy_pareto.md` |

To recover the exact frozen state: `git checkout campaign-b-freeze-a91c5d9`.

## Checkpoint IDs

| model | id / path |
|---|---|
| Llama dense/sparse serving | `nvidia/Llama-3.1-8B-Instruct-NVFP4` |
| Llama recovered sparse ckpt | `/cache/recovered_Llama-3.1-8B-Instruct_P30000_p25000_2sh_lr3e-05.pt` (Modal volume) |
| DeepSeek MoE | `nvidia/DeepSeek-V4-Flash-NVFP4` (43 MoE layers, 256+1 experts, top-6, MXFP4 experts) |
| GLM MoE | `nvidia/GLM-5.2-NVFP4` (glm_moe_dsa, 78 layers / 75 MoE, 256+1 experts, top-8, DSA+MLA, 432.9 GiB) |

## Modal environment (two toolchains — they cannot coexist)

| id | used by | base image | CUDA | torch | vLLM | FlashInfer | capture |
|----|---------|-----------|------|-------|------|-----------|---------|
| **build** | sparse `.so` compile | CUDA 12.8.1 | 12.8 | (nvcc only) | — | — | `nvcc -arch=sm_120a`; `.so` staged to Modal volume |
| **L** | Llama serving (Sec 4-9) | vLLM image | 12.9/13.0 | 2.14.0.dev+cu130 | 0.21.0 | shipped w/ vLLM | **graph ON** (`enforce_eager=False`), gpu_mem 0.8 |
| **M** | DeepSeek + GLM MoE (Sec 10) | `nvidia/cuda:13.0.0-devel-ubuntu22.04` +py3.12 | 13.0 | 2.11.0+cu130 | 0.24.0 (V1) | python 0.6.14 + cubin 0.6.13 (force-reinstalled) | **eager only** (`enforce_eager=True`); quadbit `general_plugins` entry-point |

**Env M standing vars:** `VLLM_USE_DEEP_GEMM=0`, `FLASHINFER_DISABLE_VERSION_CHECK=1`, `kv_cache_dtype=fp8`,
`enable_expert_parallel=True`. DSA backend `FLASHINFER_MLA_SPARSE_SM120` is auto-selected by vLLM (not an
env var); MoE backend `FLASHINFER_CUTLASS`. Host: Modal-managed RTX PRO 6000 (Blackwell, compute
capability 12.0, `sm_120a`); 8-GPU capacity queues on demand.

## Headline results P4 must preserve

- **Llama serving (graph-vs-graph, Table C):** sparse split-K FP4 MLP wins total request latency in 81/112
  regimes; decode +9.7/+7.2/+2.2% at B=8/32/64; prefill trails 3-5%; serving PPL 10.27 vs dense NVFP4 7.97.
- **DeepSeek structural sparsity (eager):** c_down49 downstream AVG .7354 (-0.29pt) training-free; route-slot
  D2 .7304; per-expert weight repair fails.
- **GLM-5.2 transfer (eager, 8 GPU):** down49 +0.209 / gateup49 +0.432 / route-slot D2 +0.065 held-out PPL;
  **GLM D2 downstream AVG .7508 vs dense .7603 (-0.95pt, 4-task smoke suite, no task collapsing)**.

## P4 known blocker (the thing P4 attacks)

The quadbit vLLM plugin's expert-parallel local-expert loop calls `torch.unique(local).tolist()`
(`harness/qb_vllm_plugin/qb_sm120_plugin.py`), a device->host sync illegal under CUDA-graph stream capture
(`cudaErrorStreamCaptureUnsupported`; traceback in `docs/audit/logs/glm_graphfail.log`). All DeepSeek/GLM
rows are eager because of this. The Llama sparse-MLP path (`quadbit::fused_mlp` torch.library op) is already
graph-captured (Table C). P4 = make the MoE/plugin path graph-safe or document the blocker at the exact
operation level. See [host_sync_audit.md](../p4/host_sync_audit.md).
