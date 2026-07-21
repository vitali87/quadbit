# C10 verdict: does the sparse-FP4 MoE win serving prefill throughput at large M? = KILL

**Question ([design.md](design.md)):** C2 settled that the dense NVFP4 fused MoE is the SM120 *decode*
SOTA (8.1x faster than sparse D2) and flagged that serving *prefill* throughput was never measured. At
prefill the expert GEMMs are compute-bound, so 2:4 sparsity halving expert FLOPs *should* translate to
a throughput win at large per-expert M. C10 measures exactly that gap on DeepSeek-V4-Flash-NVFP4, 4-GPU
EP, eager (prefill needs no graph capture), all-sparse vs dense fused.

**Result: falsified.** Sparse loses at every reachable M, and its *best* point is below dense's *worst*.

| P (prefill_p) | M/expert (~P*6/256) | sparse tok/s | dense tok/s | sparse/dense |
|---|---|---|---|---|
| 2,048  | ~48   | 4735.2 | 7589.0 | 0.62 |
| 16,384 | ~384  | 6712.5 | 7904.0 | 0.85 |
| 32,768 | ~768  | **6781.2** (sparse peak) | 7793.0 | 0.87 |
| 65,536 | ~1536 | 6228.1 (rolled off) | OOM (unmeasured) | — |

- **No crossover anywhere in the measured range (M ~ 48 to 768, three paired points).** Sparse loses at
  every paired M and the ratio *plateaus* rather than climbing to 1.0: 0.62 (M~48) -> 0.85 (M~384) ->
  0.87 (M~768), then sparse turns **down** to 6228 at M~1536. Sparse's maximum measured throughput
  (6781 at M~768) is below dense's *minimum* measured throughput (7589 at M~48), so the curves never
  overlap in the range we can measure.
- **Dense is flat-high**: 7589 / 7904 / 7793 across a 16x M increase (no downward trend). The
  FlashInfer-CUTLASS *fused grouped* GEMM (`group_gemm_nvfp4`) is launch/bandwidth-saturated at small M,
  not FLOP-bound, so halving expert FLOPs buys the sparse arm nothing to compete against.
- **Sparse rises then falls**: 4735 -> 6712 -> 6781 (M 48->384->768, amortizing per-expert seg-loop
  routing/launch overhead over more rows) -> 6228 (M~1536, -8%). The rise is toward a plateau at ~0.87
  of dense, **not** toward a crossover; the per-expert path peaks near M~384-768 and rolls off at very
  large M (kernel efficiency + activation pressure of a 65k-token single-chunk forward).
- **Beyond M~768 is formally unmeasured** (P=65,536 dense OOMs single-chunk, see below), so a crossover
  at M > 768 is not *disproven* by data. But the trend gives no path to one: sparse is already
  *declining* at M~1536 while dense would have to collapse below 6228 to be overtaken, and dense showed
  no downward trend anywhere it ran.

## Why the 2:4 FLOP win does not survive integration

The kernel leaderboard ([[quadbit-fp4-leaderboard]]) win is at the **isolated-GEMM** level. In the
integrated vLLM serving MoE path three things erase it:

1. **The dense baseline is fused, not a naive per-expert loop.** `QB_MOE=off` routes to FlashInfer's
   `group_gemm_nvfp4` (one fused grouped GEMM across all local experts), already near-peak and flat.
   Our sparse arm is a per-expert seg loop (`sparse_moe_mm_2lvl`) with routing scatter/gather and no
   cross-expert fusion, so it runs at well under half the fused kernel's effective per-FLOP throughput.
2. **Prefill is not MoE-dominated.** The metric times end-to-end TTFT (`P/TTFT`), which includes MLA
   attention, per-layer all-reduce, and norms. Dense's flatness across M shows prefill is dominated by
   non-MoE work; the MoE choice only moves the margin, and the sparse path moves it the wrong way.
3. **Dual residency + rolloff.** The sparse path also carries +26-27% weight memory (codes + anchored
   dense weights) and loses efficiency at very large M, so it cannot buy back the deficit by batching
   harder.

## Consequence

- **The dense fused NVFP4 MoE is the SM120 serving SOTA at prefill too, not only decode.** Combined
  with C2 (dense wins decode 8.1x), sparse-FP4 MoE wins served throughput in **no regime we can
  measure** — decode (C2), and prefill across M~48 to 768 (C10). The MoE sparse contribution stays what
  C2/[[quadbit-moe-sparse-campaign]] said: **quality / transfer only, never a serving speed lever.**
- The paired sweep tops out at M~768: at P=65,536 the dense arm OOMs (6.17 GiB over, activation of a
  65k-token single forward) and the chunked-prefill path has an unfixed `KeyError` on this MoE path, so
  the M~1536 dense point is unmeasured. This does not soften the verdict for the measured range: sparse
  loses at all three paired points and is already declining at M~1536 while dense showed no downward
  trend, so no measured evidence points toward a crossover.

Branch `c10-sparse-prefill`. Metric + method in [design.md](design.md).
