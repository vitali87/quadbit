# C8 Task 1: pipeline/layer-stage mode feasibility

**Goal.** Can DeepSeek-V4-Flash-NVFP4 run with `tensor_parallel_size=1` and layer-stage (pipeline)
placement across 4 GPUs, so per-layer TP all-reduces are replaced by stage-boundary transfers? This
gates C8: if the mode cannot be expressed, or silently degrades to replicas / aggregate DP throughput,
C8 stops.

Branch `c8-pipeline-stage-decode`. Model `nvidia/DeepSeek-V4-Flash-NVFP4`, 4x RTX PRO 6000 (SM120), no
NVLink. Same checkpoint / quant path / prompt / decode metric as C4/C7.

## Launch-API feasibility (CPU recon, `inspect_pp`, no GPU)

Log: [c8_inspect_pp.log](../audit/logs/c8_inspect_pp.log). vLLM 0.24.0.

| question | finding | verdict |
|----------|---------|---------|
| Does offline `LLM(pipeline_parallel_size>1)` raise like DP did? | `'single-process usage'` guard hits: **none** in `entrypoints/llm.py` / `arg_utils.py` / `config/parallel.py` | **not blocked** ✓ |
| Does `EngineArgs` accept the PP field in single-process mode? | `EngineArgs.pipeline_parallel_size` present = **True**; `pipeline_parallel_size: int = Field(default=1, ge=1)`; `world_size = pipeline_parallel_size * tensor_parallel_size` | **yes** ✓ |
| Does the DeepSeek model class implement PP? | `deepseek_v2.py` declares `SupportsPP` + `make_layers` (start_layer/end_layer slicing) + `get_pp_missing_layer_names` / `is_pp_missing_parameter`. `deepseek_v4.py`/`v3.py` do not exist as separate files, so V4-Flash maps onto the PP-capable base class. | **yes** ✓ |

This is the key contrast with C7: **DP** hard-raised offline (`LLM(data_parallel_size>1)` →
"not supported for single-process usage"), forcing the env-driven SPMD workaround. **PP has no such
offline guard** — a plain `LLM(tensor_parallel_size=1, pipeline_parallel_size=4)` is expressible, and
vLLM uses the `mp` (multiprocessing) executor backend for `pp*tp=4` on a single 4-GPU node. So C8 needs
neither `vllm serve`/AsyncLLM nor a custom stage harness (options 2/3 of the spec) — option 1 works.

## Planned mode

- `tensor_parallel_size=1` — each stage runs full-width layers, **no per-layer TP all-reduce**.
- `pipeline_parallel_size=4` — 4 stages, each owns ~1/4 of the layers on one physical GPU.
- `enable_expert_parallel=False` — pure PP: each stage holds the full expert set for its own layers
  (experts placed with their layer, not EP-sharded across GPUs). `ep=True` is available as a stacked
  variant if pure PP is memory-bound.
- Cross-GPU traffic = only `(pp-1)` stage-boundary activation send/recv per forward.

Memory note: PP naturally shards the ~142GB model by layer, so each GPU holds ~1/4 (~35GB) — this is
why 4 GPUs are needed to LOAD the model regardless, and PP fits comfortably within 102GB/GPU.

Harness: `graph_gate_pp` (single-process `LLM`, mp backend spawns the stage workers), commit `fcd46c6`.

## Runtime mode proof (GPU run `bhokhb5x5`, eager)

Log: [c8_pp_eager.log](../audit/logs/c8_pp_eager.log). RTX PRO 6000:4, SM120, no NVLink.

| proof point | observed | source line |
|-------------|----------|-------------|
| resolved parallel config | `tp=1 pp=4 dp=1 world_size=4 backend=mp ep=False` | `# C8-PP MODE` |
| per-GPU layer ownership | hidden layers partitioned `[11,11,11,10]` (Worker_PP0..PP3 each own a distinct contiguous range) | `utils.py:144` ×4 |
| weights staged vs replicated | **staged** — model load = **41.0 GiB per GPU** (~1/4 of the model), not the full ~142GB | `gpu_model_runner.py:5255` |
| all-reduce per token | **0** (C4 ~43.5) — the per-layer TP all-reduce floor is gone | `# C8-PP all-reduce=0 per-token=0.0` |
| stage-boundary transfers | 108 `ncclDevKernel_SendRecv` kernels over `decode_fwd~36` = **3 per token = pp-1** boundaries | `# C8-PP TRANSFERS` |
| EP collectives | allgather=0 reduce-scatter=0 all-to-all=0 (pure PP, no EP) | `# C8-PP other` |
| all-reduce backend for `pp:0` group | `['PYNCCL']` (used only for the tiny PP embed/logits sync, fires 0× per decode token) | `cuda_communicator.py:245` |
| graph status | eager (`enforce_eager=True`, `cudagraph_mode=NONE`) | engine config |
| coherent generation | 3/3 prompts coherent (Paris / fibonacci body / primary colors) | `# GEN` ×3 |
| quality | ppl **4.1923** (C4/C7 baseline 4.264, same checkpoint/quant) | `# C8-PP PASS` |
| eager decode | **7.539 tok/s** (wall1=0.420s wall64=8.776s) | `# C8-PP decode tok/s` |

**Guardrail check — PASS.** Genuine layer-staging, not replicas or aggregate DP throughput: one model
sharded across 4 GPUs by distinct layer range (`[11,11,11,10]`, 41 GiB/GPU vs 142GB whole), a single
batch=1 request flowing through all 4 stages, **all-reduce-per-token = 0** (C4's dominant floor
removed), and only pp-1 stage-boundary send/recv per token. C8 does **not** stop at verdict C — the
mode is real and the C4 all-reduce floor is eliminated. Proceed to Task 2 (captured stage baseline).

The open question the eager number cannot answer: eager decode (7.539) is launch-overhead-bound, not
weight-bandwidth-bound. C7's DP-attention gained 4.5x from graph capture (4.578 → 20.450). Whether C8
captured can close the gap to C4's 58.126 depends on (a) how much of C4's per-token time was the
now-removed all-reduce vs the weight stream PP serializes 4-way at batch=1, and (b) whether vLLM can
CUDA-graph-capture cross-process pipeline send/recv cleanly. Both are empirical — Task 2 measures them.
