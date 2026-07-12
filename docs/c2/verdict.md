# C2 verdict: SM120 sparse-FP4 MoE SOTA board

Branch `c2-sota-board` off `main` @ `e28300f`. Board + protocol: `docs/c2/sota_board.md`; DeepSeek detail
`docs/c2/deepseek_sota.md`; GLM detail `docs/c2/glm_sota.md`. Every number is from a raw log under
`docs/audit/logs/c2_*.log`. All rows share one harness, one PPL passage (mito80), one decode-only formula,
one graph mode (captured), one memory-accounting method — an apples-to-apples board, not a mixed bag.

## The board (captured, same harness)

| model | GPUs | dense NVFP4 fused (baseline) | quadbit D2 native captured | dense decode advantage |
|---|---|---|---|---|
| DeepSeek-V4-Flash | 4 | **48.248 tok/s**, 40.83 GiB wt, 0.18 pool, PPL 4.1222 | 5.972 tok/s, 51.7 GiB wt, 2.08 pool, PPL 4.0943 | **8.1× decode, −21% weight mem** |
| GLM-5.2 | 8 | **33.810 tok/s**, 54.62 GiB wt, 0.10 pool, PPL 3.9572 | 5.367 tok/s, 68.98 GiB wt, 0.80 pool, PPL 4.0674 | **6.3× decode, −21% weight mem** |

DSA native on both baselines and both quadbit rows (`sparse_mla_sm120_decode_dsv4` /
`FLASHINFER_MLA_SPARSE_SM120`). All rows coherent, drop=0.

## V1. DeepSeek decode — **NO.**

quadbit D2 native captured 5.972 tok/s vs the dense NVFP4 fused baseline 48.248 tok/s: **the dense baseline
is 8.1× faster at decode.** quadbit does not beat the best dense/NVFP4 graph baseline on decode.

## V2. GLM decode — **NO.**

quadbit D2 native captured 5.367 tok/s vs dense NVFP4 fused baseline 33.810 tok/s: **dense is 6.3× faster.**
The graph-enabled sparse-policy transfer holds (captures, DSA native, downstream smoke intact), but with
no decode advantage.

## V3. Memory — **sparse INCREASES total memory.**

Route-slot D2's dual residency (raw NVFP4 dense slots + 2:4 sparse codes co-resident) costs **+27%
(DeepSeek: 51.7 vs 40.83) / +26% (GLM: 68.98 vs 54.62) weight memory per GPU**, a larger graph pool
(2.08 vs 0.18; 0.80 vs 0.10 GiB), and collapses KV capacity (GLM: 236,672 vs 629,760 tokens, −62%). The
sparse policy does not reduce memory; it raises it.

## V4. Quality — **matched-to-slightly-worse; report deltas, not adjectives.**

- mito80 PPL is noise-dominated (80 tokens) and a wash: DeepSeek D2 4.0943 vs dense 4.1222; GLM D2 4.0674
  vs dense 3.9572. **Not used for ranking** (protocol rule 1).
- Downstream MC smoke (the real evidence): DeepSeek D2 .7304 vs dense .7383 = **−0.79 pt** (`paper.md`
  §10); GLM D2 .7508 vs dense .7603 = **−0.95 pt** (`glm_results.md`). Acceptable under the Pareto
  framing, but a small loss, not a gain.

## V5. SOTA label — **"graph-enabled transfer result, not speed SOTA."**

On SM120, the **dense NVFP4 fused MoE (vLLM native FlashInfer-CUTLASS) is the MoE decode SOTA** — faster
at decode (6–8×) and lighter in memory than the quadbit sparse route-slot D2 policy, at comparable-or-
better quality. None of the three headline success conditions is met: no decode win (V1/V2 NO), no
speed-or-memory Pareto point at decode (V3: sparse costs more memory), and the GLM transfer, while real,
carries no measurable decode/precompute advantage.

**What quadbit's sparse MoE genuinely is (unchanged, and not overclaimed):**
1. The only *deployed* 2:4-sparse FP4 MoE on SM120 (FI/SGLang/vLLM ship none; CUTLASS 80b is an unwrapped
   sparse-kernel example — prior art preserved).
2. Training-free, capability-preserving structural sparsity that **transfers across architectures**
   (DeepSeek → GLM) and **graph-captures** with native SM120 DSA, downstream smoke within ~1 pt.
3. A prefill / large-M *kernel* Pareto point (`paper.md` §5: the 2:4 sparse GEMM beats the best dense FP4
   kernel 1.07–1.38× on prefill shapes). **This board did not measure serving prefill throughput**, so no
   serving-prefill claim is made here.

## Next bottleneck (precise, per the mixed-result instruction — not softened)

At decode (M=1–2 rows) the quadbit sparse serving path — fixed-capacity device routing + per-expert 2:4
`sparse_moe_mm_2lvl` + the dual-residency `group_gemm_nvfp4` dense anchor, driven by Python-level
per-expert loops — is 6–8× slower than vLLM's single autotuned fused NVFP4 grouped GEMM, and dual
residency costs +26–27% weight memory and up to −62% KV. The 2:4 sparsity advantage is a bandwidth effect
that only appears at large M (prefill); at decode there is no bandwidth to save.

**The only missing piece is a kernel:** a fused *sparse* grouped-GEMM that batches all routed experts into
one launch at tiny M (the decode analogue of the fused dense path), plus dropping dual residency (decode
from 2:4 codes only, no raw-NVFP4 co-residency). That is genuine new CUDA. Per the C2 guardrail it is
**identified, not started** — the board has now proven that closing the MoE decode gap is a kernel problem,
not a policy or graph problem.

## Guardrails honored

No "production-wide SOTA" (the board shows the opposite for decode). No "beats dense FP4 generally" (dense
wins decode + memory). Failed/absent baselines reported (`sota_board.md`: vanilla vLLM init-fails, SGLang
unavailable). CUTLASS 80b sparse prior art preserved. No mismatched-PPL quality claim (mito80 excluded
from ranking; downstream deltas used). No custom CUDA started.
