# C5 verdict: ceiling reached — C4's one-shot all-reduce stands as the SM120 decode SOTA

**Goal:** beat C4's 58.126 tok/s by reducing the remaining TP all-reduce floor. **Outcome: not beaten. C5 is
a ceiling for this harness/config.** No softening: every C5 lever lost or was unreachable.

## Measured remaining floor

The decode step is **91-94% one collective**: **43.5 all-reduces per token = ~1 per layer** (43 layers),
the attention TP all-reduce (MoE uses EP all-to-all, cheap). The all-reduces are **sequential** (layer N+1
needs layer N's reduced output) and **non-overlappable** at batch=1 (compute is <3%). On C4's custom-AR path
each one-shot all-reduce is **~374 us** for a ~14 KB payload = almost entirely **cross-GPU synchronization
latency over PCIe** (no NVLink), not data transfer. This is the wall.

## Why the levers did not move it

- **Reduce all-reduce count (algebraic):** none available at batch=1 decode. The ~1 AR/layer is structural
  for TP attention: RMSNorm over the TP-sharded hidden dim requires the reduce to complete first; no slice,
  delay, or fusion removes it. vLLM's `fuse_allreduce_rms`/`enable_sp` passes do not reduce the count and
  need the inductor pipeline (conflicts with our graph capture).
- **Reduce ranks (TP=2): NEGATIVE.** 40.565 tok/s (vs 48.248), even with the fastest 2-GPU custom AR. Halving
  the shard count doubles the weight bytes each GPU reads per token; that memory cost outweighs the faster
  all-reduce. Decode is not purely AR-latency-bound.
- **Faster collective (hierarchical, Task 3):** the one-shot is already **single-sync** (each rank reads N-1
  peers in parallel, one barrier). A hierarchical/tree adds a stage (more syncs) and is worse for tiny
  latency-bound payloads; NCCL tree confirms this (+1.5% only vs the one-shot's +20.5%). No variant built.

## The one lever with real headroom, and why it is blocked

**DP attention + EP MoE (tp=1, data_parallel_size=4)** would remove the ~43 attention all-reduces entirely
(attention runs replicated per DP rank, no reduction; experts stay EP-sharded, MoE collectives unchanged).
This is the correct structural fix for the 94.5% floor. It is **not reachable in this offline harness**:
vLLM's offline `LLM` class hard-rejects `data_parallel_size>1` in every mode (single-process, external-LB
with `data_parallel_rank`/`data_parallel_external_lb`, and env-driven `VLLM_DP_*` per subprocess). DP requires
the `vllm serve` API server / `AsyncLLM` multi-process launcher.

## Next lever (exact)

1. **DP-attention decode via an `AsyncLLM`/`vllm serve` latency harness** (not the offline `LLM` two-run
   measurement). This is the highest-headroom lever: it targets the 94.5% attention all-reduce directly.
   Open risk it must resolve: the MoE EP all-to-all becomes the sole collective, measure whether it is
   cheaper than the 43 attention all-reduces it replaces (the roofline suggests yes, but it is unmeasured).
2. **NVLink hardware** (or a topology-aware partial-P2P hierarchical AR): the ~374 us/AR is the PCIe sync
   round-trip. NVLink collapses it; a partial-P2P hierarchical AR would instead widen *coverage* (recover the
   one-shot win on the partial-P2P containers that currently fall back to NCCL) rather than raise peak speed.

## Standing claim (unchanged)

C4's one-shot custom all-reduce remains the SM120 decode SOTA: **48.248 -> median 58.126 tok/s = +20.5%**
(fully-connected P2P; partial-P2P safely falls back to NCCL). C5 did not beat it. Speed only; C5 changes no
quality claim. This is a **serving-infra collective** result, not a sparse-MoE or kernel contribution.
