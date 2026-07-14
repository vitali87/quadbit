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

## Runtime mode proof (GPU run — to be filled from `c8_pp_eager.log`)

_Pending run `bhokhb5x5` (eager). Will record: resolved `tp`/`pp`/`dp`/`ep`/`world_size`/backend,
per-GPU layer ownership, per-GPU memory, weights replicated vs staged, all-reduce count per token
(expect ~0), stage-boundary send/recv per token (expect ~pp-1), graph status, backend selected._

**Guardrail check (must pass before continuing):** the mode must be genuine layer-staging, not 4
independent replicas or aggregate DP throughput. Proof = one model sharded across 4 GPUs by layer
(each GPU has a distinct layer range, not the whole model), a single batch=1 request flowing through
all stages, and all-reduce-per-token dropping toward 0. If instead the run shows 4 full-model replicas
or the collective floor does not drop, C8 stops with verdict C.
