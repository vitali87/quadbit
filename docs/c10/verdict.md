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
| 32,768 | ~768  | **6781.2** (sparse peak) | 7793.0 | 0.87 |
| 65,536 | ~1536 | 6228.1 (rolled off) | OOM (dense flat ~7800) | — |

- **No crossover exists.** Sparse's maximum measured throughput (6781 at M~768) is *below* dense's
  minimum measured throughput (7589 at M~48). The two curves never overlap, so no amount of extra
  batching flips the result.
- **Dense is flat**: 7589 -> 7793 over a 16x M increase (+2.7%). The FlashInfer-CUTLASS *fused grouped*
  GEMM (`group_gemm_nvfp4`) is already launch/bandwidth-saturated at small M; it is not FLOP-bound, so
  halving expert FLOPs buys the sparse arm nothing to compete against.
- **Sparse rises then falls**: 4735 -> 6781 (M 48->768, +43% as the per-expert seg loop amortizes its
  routing/launch overhead over more rows) -> 6228 (M~1536, -8%). The rising trend I saw at 0.62->0.87
  was **not** monotonic toward a crossover; the per-expert path has a throughput peak near M~768 and
  rolls off at very large M (kernel efficiency + activation pressure of a 65k-token single-chunk
  forward), never approaching dense's flat line.

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
  with C2 (dense wins decode 8.1x), sparse-FP4 MoE never wins served throughput in **any** regime,
  decode or prefill. The MoE sparse contribution stays what C2/[[quadbit-moe-sparse-campaign]] said:
  **quality / transfer only, never a serving speed lever.**
- Higher M than 65,536 is unreachable single-chunk on this hardware: P=65,536 dense OOMs (6.17 GiB over,
  activation of a 65k-token single forward), and the chunked-prefill path has an unfixed `KeyError` on
  this MoE path. But this does not matter to the verdict: sparse already peaks and falls **below dense's
  floor** at M~768, so the reachable ceiling is not the limiting factor.

Branch `c10-sparse-prefill`. Metric + method in [design.md](design.md).
