# C4 verdict: beat the SM120 decode SOTA (+19%) by re-enabling the one-shot all-reduce

**Premise (from C3):** compact routing made sparse D2 decode 2.80x faster but stayed 3x under the dense
NVFP4 fused SOTA (48.248 tok/s), because the sparse-vs-dense MoE decode fight is a 5.5% slice of the step.

**Where the headroom actually is (measured):** the decode step is **94.5% non-MoE floor**, and the floor is
**90.8% one NCCL kernel** (`ncclDevKernel_AllReduce_Sum_bf16_RING_LL`). This is 4-GPU TP with no NVLink;
every layer does a per-layer all-reduce over PCIe, latency-bound at batch=1, on the worst algorithm (ring).
Attention + DSA + GEMM together are under 3%. ([floor_decomposition.md](floor_decomposition.md))

**Fix:** the wall is a **disabled fast-path**, not a hardware limit. vLLM disables its one-shot custom
all-reduce on >2 PCIe GPUs (an NVLink policy), and its runtime P2P probe fails spuriously on Modal though
the driver reports full P2P. Re-enable it: spoof `is_fully_connected -> True` + `VLLM_SKIP_P2P_CHECK=1`
(`QB_FORCE_CUSTOM_AR=1`, opt-in). One hop replaces the ring's six. ([custom_allreduce.md](custom_allreduce.md))

**Result (numeric):** DeepSeek dense baseline, 4 GPU, captured, same harness as C2:
- custom one-shot AR = **57.783 / 58.545 / 58.126 / 58.126 tok/s** (4 runs) = **~58.1**, PPL 4.2514, FULL capture.
- baseline RING_LL = 48.248 (C2) / 49.263 (fresh control), PPL 4.1222.
- **+19% over the prior SM120 decode SOTA, reproducible, correct** (coherent generation, bit-identical
  fibonacci; identical 4.2514 PPL across all 4 runs = the deterministic one-shot reduction).

**Quality:** mito80 PPL swings with reduction order (tree 4.01 / ring 4.12 / one-shot 4.25, both directions),
i.e. bf16-summation-order noise on an 80-token greedy passage, not a regression. The AR is a correct sum;
downstream 4-task eval is the real quality check (future work). No quality-neutral claim on mito80.

**Scope / honesty:**
- This is a **serving-infra** win (a collective-algorithm swap), NOT a sparse-MoE or kernel contribution. It
  applies to the dense fused path and to sparse D2 alike (shared floor).
- quadbit already *owned* the SM120 decode number (48.248) because the plugin is what boots DeepSeek-V4 on
  SM120 at all (vanilla vLLM init-fails); C4 lifts that same stack to ~58.1.
- Not claimed: a general multi-GPU AR improvement (this targets small latency-bound decode all-reduces on
  PCIe-only topologies; vLLM's disable is correct for bandwidth-bound large-tensor cases). Requires driver
  P2P support (verified all-connected here); on a genuinely P2P-blocked host the AR safely disables.

**Next lever:** the floor is still ~85%+ collective after this (the one-shot AR is faster but PCIe is still
the medium). A tree/hierarchical one-shot, or reducing TP all-reduce count (fold the per-layer reduces, or
lower TP with more DP/EP), is the next attack. C4 already clears the SOTA; those extend the margin.
