# C3 verdict: beat the SM120 MoE decode SOTA by attacking the measured bottleneck

**Original sparse-kernel premise:** *refuted, and it stays refuted.* The profile
(`profile_decode.md`) showed the sparse 2:4 `matmul_sp` kernel is **0.4%** of the decode step. The
captured differential attribution (`captured_attribution.md`) confirmed the sparse *group* is 24% of the
step but that its cost is per-row overhead + padded compute on 8192 padded rows, **not** the MMA kernel.
Building `fused_sparse_grouped_decode_nvfp4_2lvl` would have optimized a 0.4% line. Not built.

**Actual bottleneck (measured):** the MoE apply is **89%** of the captured decode step, and it is the
**E·cap = 64×128 = 8192-row fixed-capacity padding** in both expert groups, dense-anchor **64%**
(A−B differential), sparse tail **24%** (A−C). At decode ~a dozen rows are real, so ~680× of the padded
rows are pure waste (gather / per-group quant / scatter + padded compute). The non-MoE floor
(attention/DSA/EP/norms) is a cheap **11%**, removing the whole MoE hits 51 tok/s, above the dense
baseline, so attention/DSA is not the wall.

**Fix attempted:** *active-expert compaction* (`compact_routing_ab.md`), process only the `A_max` most
token-loaded experts per group (A_max·cap rows) with gathered per-expert weights, instead of all E=64.
Capture-safe (fixed A_max shape, device topk/gather, static guard so it only engages on the small decode
graphs) and **bit-correct**: the A_max=E correctness runs reproduce baseline PPL within the mito80 noise
band for both the dense weight gather (PPL 4.096) and the sparse 2:4 4-tuple gather (PPL 4.045). This also
shrinks the dense-anchor per-group quant loop from 64 to A_max iterations (Task 1C is subsumed here, the
E→A_max reduction is the quant cleanup; no separate kernel rewrite was needed).

**DeepSeek result (numeric):** compact-both (A_dense=8, A_sparse=24) = **16.203 tok/s**, captured FULL,
PPL 4.123 (noise band). That is **2.80×** the same-build non-compact D2 (5.782) and **2.71×** the C2 D2
point (5.972), it closes the D2→dense decode gap from **8.1× to 3.0×**. But it does **not** beat the dense
NVFP4 fused SOTA (48.248 tok/s; still **3.0× slower**) and does **not** create a strict Pareto point: D2
weights stay dual-resident at 51.7 GiB/GPU (+27% vs dense 40.83, compaction doesn't change residency), and
downstream quality is D2 AVG .7508 vs dense .7603 = **−0.95 pt** (4-task, P1/PR#13). Dense wins all three
axes (speed, memory, quality). **No SOTA claim, no Pareto claim vs dense.**

**GLM decision:** *skipped.* Per the gate (GLM only if DeepSeek shows a result that could change the
standing), DeepSeek already establishes that compaction is a real intra-D2 speedup but cannot beat or
Pareto-dominate the dense fused baseline. GLM (8 GPUs) is structurally identical, its own dense fused
baseline is likewise far faster, same dual-residency memory, same quality tax, so running it cannot flip
the verdict and would only burn 8-GPU hours. Skipped with reason, matching the C2 precedent.

**Next lever (exact):** a **custom compact-row decode GEMM/layout** that packs *arbitrary* compact rows
(the ~dozen real decode tokens) with **per-row expert routing**, killing the residual `cap=128`-per-active-
expert padding. Active-expert compaction removed the *inactive*-expert padding (E→A_max groups); the
remaining 3.0× lives entirely in the *within-active* 128-row block that carries ~1 real token. This is the
grouped-GEMM variant with a data-dependent-but-bounded row count per group (m_indptr from device counts,
padded to a small multiple of the MMA tile, not to 128), targeting `group_gemm_nvfp4` and the 2:4 seg
kernel. Concretely: replace the fixed `cap=128` in `route_fixed_cap` / `_route_compact` with a per-active
`ceil(count/tile)*tile` layout and teach both kernels a variable `m_indptr`, capture-legal via a fixed
upper-bound buffer with device-set group offsets. Not the refuted sparse-only kernel: this attacks the
padding in **both** groups, which is where the measured cost is.

## One-line campaign summary

C3 refuted the sparse-kernel premise, measured the real wall (E·cap padding, 89% of the step), and shipped
capture-safe active-expert compaction that makes quadbit D2 decode **2.80×** faster (5.8→16.2 tok/s), a
real intra-D2 win, but the dense NVFP4 fused MoE remains the SM120 decode SOTA (3.0× faster, less memory,
better downstream quality). The next lever is a variable-`m_indptr` compact-row kernel to kill the residual
cap=128 padding.
