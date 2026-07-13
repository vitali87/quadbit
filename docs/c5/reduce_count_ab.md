# C5 Task 2: reduce the TP all-reduce count

Post-C4 roofline: **43.5 all-reduces per decode token = ~1 per layer** (43 layers), 91-94% of the step,
sequential and non-overlappable at batch=1. This is the **attention TP all-reduce** (the MoE uses EP
all-to-all, which is cheap here). Task 2 asks whether any of these can be removed/fused/delayed safely.

## Algebraic transforms (all N/A at batch=1 decode)

| candidate | verdict |
|---|---|
| remove redundant all-reduces (outputs not yet consumed globally) | none: each layer's all-reduced hidden state IS consumed by the next layer's RMSNorm |
| delay all-reduce across residual boundaries | no: RMSNorm is computed over the full hidden dim, which is TP-sharded, so the reduce must complete before the norm |
| fuse adjacent reductions | no adjacency: only 1 all-reduce per layer, separated by attention/MoE |
| reduce only the required slice | the next layer consumes the full hidden state; no slice suffices |
| logits absorb a delayed reduce | only the last layer's 1 all-reduce could fold into the head = 1 of 43.5, negligible |

The count is **structural** for tensor-parallel attention at decode. vLLM's `fuse_allreduce_rms` /
`enable_sp` compilation passes fuse the norm into the collective or convert AR->RS+AG, but they do not reduce
the *count* and require the inductor compilation pipeline (mode != NONE), which conflicts with our CUDA-graph
capture path. Not a count reducer.

## Reduce ranks (TP=2): NEGATIVE

Fewer TP ranks -> a cheaper per-layer all-reduce (2-GPU = 1 peer read). Measured
(`docs/audit/logs/c5_tp2_dense.log`, full 2x2 P2P, custom AR native at world_size==2):

| config | decode tok/s | ms/step | PPL | capture |
|---|---:|---:|---:|---|
| TP=4 baseline | 48.248 | 20.73 | 4.1222 | FULL |
| **TP=2** | **40.565** | 24.65 | 4.0855 | FULL |

TP=2 is **slower** by +3.9 ms/step. The decode is not purely AR-latency-bound: halving the shard count
**doubles the weight bytes each GPU reads per token**, and that memory cost outweighs the faster 2-GPU AR.
**Fewer ranks is a losing lever.**

## Remove the attention all-reduce entirely (DP attention + EP MoE)

The one structural way to drop the ~43 attention all-reduces: run attention **data-parallel** (tp=1,
data_parallel_size=4) so each rank has replicated attention (no reduction) while experts stay EP-sharded (the
MoE all-to-all is unchanged and was already cheap). This does not add MoE collectives — it removes the
attention AR.

**Launch:** vLLM offline DP rejects single-process `LLM(data_parallel_size>1)`; the self-launched path needs
each rank as its own process with `data_parallel_rank=r, data_parallel_size_local=1,
data_parallel_external_lb=True` + `VLLM_DP_MASTER_IP/PORT`. Implemented as `graph_gate_dp` (subprocess per
rank, worker in the installed `qb_dp_worker` module). Result: _see below / final_board.md_.

<!-- DP_RESULT -->

## Task 2 summary

Safe algebraic count reduction: **none** at batch=1 decode (structural). Reduce-ranks (TP=2): **negative**.
The only count lever is **DP attention** (removes the attention AR); its measured decode is the deciding
number for the C5 verdict.
