# C7 Task 5: sparse D2 transfer — NOT RUN

**Spec gate.** "Sparse D2 transfer ONLY if dense improves."

Dense DP-attention did **not** improve — captured decode is 20.450 tok/s vs the C4 dense SOTA of
58.126 (2.84x slower, see [serve_baseline.md](serve_baseline.md) / [dp_attention_ab.md](dp_attention_ab.md)).
The gate is not met, so the sparse
D2 transfer run was **not executed**. Running it would only re-confirm the same structural loss (the EP
allgather+reduce-scatter floor is identical for the sparse MoE path; sparsity changes expert FLOPs, not
the per-layer cross-GPU collective count) while burning 4 GPUs.

No sparse D2 number is claimed for C7.
