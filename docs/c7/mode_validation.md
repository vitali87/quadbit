# C7 Task 1: DP-attention mode validation

**Goal.** Prove that a data-parallel-attention execution mode (`tensor_parallel_size=1` per rank,
experts expert-parallel) actually *activates* on this SM120 stack and *removes the per-token
attention TP all-reduce* — the 91-94% decode floor C5 measured. This is the gate before any speed
claim: if the all-reduce count per token does not drop, DP-attention is not the lever and C7 stops.

Branch `c7-dp-attention-serve`. Model `nvidia/DeepSeek-V4-Flash-NVFP4`, 4x RTX PRO 6000 (SM120), no
NVLink. Same checkpoint / quant path / prompt as C4/C6.

## The launch fix (why C5 was blocked, how C7 activates it)

C5 could not run DP at all: vLLM 0.24 `LLM(data_parallel_size>1)` hard-raises at
`entrypoints/llm.py:293-303` ("not supported for single-process usage") unless
`distributed_executor_backend=="external_launcher"`. C5's `qb_dp_worker` passed
`data_parallel_size=`/`data_parallel_rank=` as `LLM()` kwargs, which is exactly what trips the raise.

The supported offline path is **env-driven SPMD**: vLLM `config/parallel.py:852-857` reads the DP
size/rank from `VLLM_DP_*` env when no `data_parallel_*` kwarg is present, and the `is_moe_model`
guard (`:859`) clears for this MoE model. So C7's worker:

- launches one fresh interpreter per rank (`graph_gate_dp`, subprocess, not `multiprocessing.spawn`);
- sets `CUDA_VISIBLE_DEVICES=<rank>` (one physical GPU per process) + `VLLM_DP_SIZE/RANK/RANK_LOCAL/
  MASTER_IP/MASTER_PORT` **before** importing torch;
- calls a plain `LLM(tensor_parallel_size=1, enable_expert_parallel=True, ...)` with **no**
  `data_parallel_*` kwargs.

Result: all 4 ranks construct their engine with zero `data_parallel_size` rejection (the C5 wall).

## Lockstep / single-request-latency semantics (the guardrail)

The EP all-to-all is collective over the DP group, so every `generate()` must be matched across all
ranks or rank 0 deadlocks on idle peers. C7 therefore runs **identical** generate calls on all four
ranks; only rank 0 profiles and reports. Each rank is its own **batch=1** request, so rank 0's
per-token latency is a genuine single-request decode latency — **not** aggregate replica QPS. The
speed rows in Task 2/3 are labeled accordingly; no 4-replica aggregate throughput is compared
against a single-request decode number.

## Mode proof (what actually ran)

Confirmed from the eager validation log (all 4 ranks constructed their engine past the C5 wall):

- **tensor_parallel_size: 1** per rank — each worker prints `LLM constructed dp=4 tp=1 EP=on
  device=cuda:<rank>` (`[dp0]`..`[dp3]`).
- **data_parallel_size: 4**, env-driven — no `data_parallel_*` kwarg was passed; vLLM read
  `VLLM_DP_SIZE=4`/`VLLM_DP_RANK` from env and spun up `EngineCore_DP0`..`EngineCore_DP3`.
- **engines/workers: 4 independent engine processes**, one physical GPU each — worker banners
  `Worker_DP0_EP0`..`Worker_DP3_EP3` (pids 272/275/278/281), each pinned via
  `CUDA_VISIBLE_DEVICES=<rank>`.
- **requests replicated vs model repartitioned:** attention weights replicated per rank (tp=1, no
  attention all-reduce); experts EP-sharded across the 4 ranks (`EP=on`), so the EP all-to-all is
  the sole cross-GPU collective. Each rank drives its own batch=1 request in lockstep → rank 0's
  latency is single-request, not aggregate replica QPS.
- **graph status: eager** (this validation run; captured is measured in Task 2/3).

## All-reduce count per token (the gate)

Measured by parsing rank 0's torch trace and counting collective kernels (same method as C5
`floor_profile`): `cross_device_reduce`/`allreduce_` = TP all-reduce; `alltoall` = EP; the
`sparse_mla_decode` kernel is the once-per-layer-per-forward anchor for the per-token normalization.

| config | all-reduce / token | all-to-all / token | verdict |
|--------|-------------------|--------------------|---------|
| C4/C5 baseline (tp=4) | ~43.5 (measured C5) | ~0 | attention AR is the floor |
| C7 DP-attention (tp=1, dp=4) | <!-- FILL --> | <!-- FILL --> | <!-- FILL --> |

**Gate:** AR/token must collapse toward ~0 for DP-attention to be the lever. If it does not, stop and
report DP-attention not activated (verdict C).

## Raw log

`docs/audit/logs/c7_dp_eager_smoke.log`
