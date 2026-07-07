# Figure plan

Figures for the quadbit SM120 sparse-FP4 paper, keyed to the banked results. Every figure lists what
it shows, the data source it is built from, and its axes. All data is measured on one Modal RTX PRO 6000
(SM120) unless noted. No figure may report a number that is not in the cited source.

---

## Fig 1. The Pareto point: sparse FP4 beats the best available dense FP4 (headline)

- **Shows:** wall-clock per Llama-3-8B prefill GEMM for three kernels side by side, so the reader sees
  quadbit two-level sparse beating the best FlashInfer dense backend on every prefill shape. Bars grouped
  by shape (attn 4096-cubed, ffn-up N=14336, ffn-down K=14336, square 8192), one bar each for FlashInfer
  best dense, CUTLASS 80b sparse, quadbit two-level sparse. Annotate the sparse-over-FlashInfer-dense
  ratio (1.07x / 1.16x / 1.29x / 1.38x).
- **Data source:** `harness/cutlass_sparse.py` (quadbit sparse and CUTLASS 80b, CUDA 12.8) and
  `harness/leaderboard_fp4.py` (FlashInfer best dense, CUDA 13), cross-tabulated as in `docs/paper.md`
  Section 5 and `docs/paper_notes.md` (attn 0.100 vs 0.107 ms, ffn-up 0.301 vs 0.350, ffn-down 0.265 vs
  0.342, sq8192 0.557 vs 0.767).
- **Axes:** x = GEMM shape (categorical); y = wall-clock milliseconds (lower is better). Secondary
  annotation = speedup ratio of sparse over FlashInfer dense.

## Fig 2. Dense FP4 loses the SM120 dense race (honest baseline)

- **Shows:** effective TF/s of quadbit deployed two-level dense versus the best FlashInfer backend per
  shape, making the 1.35 to 2.2x deficit explicit. This is the figure that states dense is table stakes,
  not the headline.
- **Data source:** `harness/leaderboard_fp4.py` leaderboard table in `docs/paper.md` Section 4 (square
  8192 1045 vs 1433 cutlass; prefill attn 838 vs 1283 b12x; ffn-up 936 vs 1374; ffn-down 1017 vs 1408;
  serving M=65536 639 vs 1416 cutlass).
- **Axes:** x = shape (categorical); y = effective TF/s (2*M*N*K/wall). Two bars per shape (quadbit dense,
  FlashInfer best); label the backend that won each shape and the FI/quadbit ratio.

## Fig 3. Sparse vs CUTLASS 80b, the only other sparse FP4 kernel

- **Shows:** quadbit two-level sparse over CUTLASS 80b on every shape (1.01 to 1.12x), establishing quadbit
  as the fastest and only deployed sparse FP4 GEMM on SM120.
- **Data source:** `harness/cutlass_sparse.py` (attn 1.08x, ffn-up 1.01x, ffn-down 1.12x, sq8192 1.09x =
  1973 vs 1807 TF/s), `docs/paper.md` Section 5.
- **Axes:** x = shape (categorical); y = speedup ratio quadbit/80b (baseline line at 1.00). All bars above 1.

## Fig 4. Throughput vs cuBLAS bf16, speed-path ceiling

- **Shows:** the ceiling scaling of dense and sparse FP4 over bf16 across square sizes, with the two-level
  deployed points overlaid so the ceiling-vs-deployed gap is visible (not conflated).
- **Data source:** `harness/bench_vs_bf16.py` and `harness/cutlass_fp4.py`, table in `docs/paper.md`
  Section 5 (4096: bf16 372, CUTLASS 1222, dense 1136, sparse 1512; 8192: 423 / 1497 / 1556 / 2207; 16384:
  405 / n/a / 1645 / 1782). Deployed overlay: dense sq8192 1045, sparse sq8192 1973.
- **Axes:** x = matrix size M=N=K (2048, 4096, 8192, 16384); y = TF/s. Lines for bf16, CUTLASS FP4, dense
  ceiling, sparse ceiling; markers for the two-level deployed dense and sparse points.

## Fig 5. Graph-vs-graph serving: sparse split-K decode beats production NVFP4

- **Shows:** decode and prefill tok/s at B=8/32/64 for production dense NVFP4 versus quadbit sparse MLP
  with split-K down, both CUDA-graph captured. Decode is a win (+9.7/+7.2/+2.2%), prefill trails
  (-5.3/-4.0/-3.4%).
- **Data source:** `docs/graph_serving_result.md` table, `harness/quadbit_serve.py --graph --splits 8`
  (decode 1147/4543/8567 vs 1046/4237/8384; prefill 62914/77605/115069 vs 66469/80825/119083).
- **Axes:** grouped bars, x = batch (8, 32, 64) split into a decode panel and a prefill panel; y = tok/s.
  Annotate the percent delta per group.

## Fig 6. Split-K down projection fills the machine (decode diagnosis)

- **Shows:** microseconds per MLP stage at decode, with the plain down (109 us, 16 CTAs) versus split-K
  down (56.5 us, 128 CTAs) so the 1.94x underfill fix is legible.
- **Data source:** `harness/quadbit_serve.py --mode profile_decode`, `docs/graph_serving_result.md`
  (activation quant 15, gate_up 52 at 112 CTAs, SwiGLU 30, plain down 109 at 16 CTAs, split-K down 56.5 at
  128 CTAs).
- **Axes:** x = MLP stage (categorical); y = microseconds per layer. Overlay CTA count as a label per bar.

## Fig 7. End-to-end request crossover heatmap

- **Shows:** where sparse wins total request latency across the batch x prompt x generation matrix. One
  heatmap panel per batch (B=1/8/32/64), cell color = total-latency ratio NVFP4/sparse (>1 sparse wins).
- **Data source:** `docs/crossover_result.md` heatmaps and `/cache/crossover_nvfp4.csv`,
  `/cache/crossover_sparse.csv` (112 cells each). Tally 81 sparse wins, 2 ties, 29 NVFP4.
- **Axes:** per panel, x = prompt length (128, 512, 2048, 8192); y = generation length (16..1024); color =
  ratio (diverging around 1.00). Mark the crossover boundary (min gen for sparse to win).

## Fig 8. Split-factor sweep for the decode down kernel

- **Shows:** decode tok/s at B=8/32/64 across split factors 4, 8, 16 and the plain (no split-K) kernel,
  justifying splits=8 as the deployed default.
- **Data source:** `docs/graph_serving_result.md` split-factor table (splits 4: 1120/4573/8476; 8:
  1147/4543/8567; 16: 1079/4233/8067; plain: 981/3863/7363).
- **Axes:** x = split factor (plain, 4, 8, 16); y = decode tok/s; one line per batch.

## Fig 9. Accuracy Pareto: reverse densification trades speed for accuracy 1:1

- **Shows:** PPL versus a densification policy sweep, with the "keep down sparse" frontier topping out at
  9.750 PPL and the endpoints (all-sparse 10.256, all-dense 7.974) marked, alongside the decode tok/s cost
  of densifying gate_up, so the 1:1 tradeoff and lack of a free knee are visible.
- **Data source:** `docs/accuracy_pareto.md` (policy/PPL table and the gate_up-dense decode -7 to -9%
  speed cost).
- **Axes:** x = policy along the densification axis (all-sparse -> dense gate_up subsets -> all-dense);
  y-left = PPL; y-right = decode tok/s. Show that accuracy gain and speed loss move together.

## Fig 10. Verification / multi-token decode does not favor sparse (refuted)

- **Shows:** the sparse/NVFP4 decode throughput margin shrinking as effective M grows, refuting the
  larger-M-favors-sparse hypothesis.
- **Data source:** `docs/crossover_result.md` Section 4B, `/cache/versweep_nvfp4.csv`,
  `/cache/versweep_sparse.csv` (M=1 1.134, M=8 1.083, M=16 1.066, M=32 1.051, M=64 1.020, M=128 1.043).
- **Axes:** x = effective M (1, 8, 16, 32, 64, 128); y = sparse/NVFP4 decode tok/s ratio (baseline at 1.00,
  trend declining toward it).

---

## Notes for whoever renders these

- Keep the ceiling numbers (Fig 4) visually distinct from deployed two-level numbers; do not merge them.
- Figures 1, 3, 5, 7 are the load-bearing ones for the sparse-Pareto and serving story. Figures 2, 9, 10
  are the honesty figures (dense loss, accuracy tax, refuted hypothesis) and should not be dropped.
- Every CSV path above is on the Modal volume, not in the repo; regenerate via the harness commands in
  `docs/repro_appendix.md` if the CSV is absent.
