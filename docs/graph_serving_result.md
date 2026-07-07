# Production-graph serving result (the honest serving headline, 2026-07-07)

This supersedes the eager result in `frozen_serving_result.md` as the **production-representative**
serving comparison. vLLM's production path CUDA-graph-captures the whole forward; this measures
quadbit sparse under that same regime (graph-vs-graph), so it is the number the paper should lead with.

## Headline (honest)

> quadbit implements a **correct, graph-capturable sparse-FP4 vLLM path on SM120**, but **production
> dense NVFP4 remains faster end-to-end**: CUDA graphs plus CUTLASS small-M scheduling erase the sparse
> eager advantage. quadbit sparse serves at **~88–97% of production NVFP4 throughput** with correct
> sparse accuracy (PPL 10.27).

## Environment / provenance
- **Branch:** `graph-capture` (HEAD `619ea39`); **main frozen** at `df83c83` (eager fallback, unchanged).
- **Route-B commit** (custom op): `1f96cb7`. **Profiler commit:** `619ea39`.
- **NVFP4 model:** `nvidia/Llama-3.1-8B-Instruct-NVFP4` (vLLM binds `modelopt_fp4` CUTLASS; non-MLP 4-bit).
- **Recovered-Instruct ckpt:** `/cache/recovered_Llama-3.1-8B-Instruct_P30000_p25000_2sh_lr3e-05.pt`
  (through-2:4-sparse-FP4-kernel PPL 10.029 offline).
- **.so:** `cuda/sparse_fp4_lib.cu`, `nvcc -arch=sm_120a -O3 --default-stream per-thread --cudart shared`
  in the CUDA 12.8.1 build image, ctypes-loaded in the CUDA 12.9.0 vLLM process (staged via volume).
- **GPU:** RTX PRO 6000 (SM120). **vLLM:** 0.21.0 V1 engine, `enforce_eager=False` (production graphs,
  PIECEWISE capture, 51 graphs), `gpu_memory_utilization=0.8`.
- **Protocol:** WT-2 15×2048 for PPL; prefill = B×S=2048 (1 out tok); decode = GEN=128 ignore_eos;
  distinct per-request prompts. Same harness for both rows.

## How the sparse MLP gets into the production graph (Route B)
Option A (class-patch + Dynamo graph break) is **impossible**: vLLM V1 compiles with
`aot_compile_fullgraph`, which forbids graph breaks (`torch.compiler.disable` → `Unsupported`). The
sanctioned path is a **`torch.library` custom op** (`quadbit::fused_mlp`): Dynamo emits it as an opaque
op node (no inline, no break) and cudagraph bakes the kernel launches it makes **at capture time**. The
sparse weights are therefore bound **before** `LLM()` (vLLM captures once at init; a post-init patch is
invisible to the frozen graph), and the layer is resolved by a call-order counter. See
`_install_graph_customop` in `harness/quadbit_serve.py`.

### Proof the sparse path actually ran (not a silent fall-back to dense NVFP4)
- `Capturing CUDA graphs (mixed prefill-decode, PIECEWISE): 100%` — capture succeeded with the op inside.
- `SPARSE_CALLS = 7264` (>0 → the custom op executed).
- `PPL_THROUGH_SERVING = 10.2709` — **sparse** accuracy, NOT the 7.97 dense value. (A dense bypass would
  read ~7.97 and match NVFP4's speed exactly; the PPL check is the guard against that false positive.)

## Serving table (graph-vs-graph, the main serving result)

| metric | vLLM NVFP4 (graph, production) | quadbit sparse MLP (graph) | Δ |
|--------|-------------------------------|----------------------------|---|
| WT-2 PPL | 7.97 | 10.2709 | +2.30 |
| prefill B=8/32/64 (tok/s) | 66469 / 80825 / 119083 | 63274 / 77896 / 115334 | **−4.8% / −3.6% / −3.1%** |
| decode B=8/32/64 (tok/s) | 1046 / 4237 / 8384 | 981 / 3863 / 7363 | **−6.2% / −8.8% / −12.2%** |

(B=1 prefill discarded as cold-capture warmup noise; B=1 decode 129 vs 123.)

## Decode diagnosis (why sparse loses decode — the systems lesson)

`--mode profile_decode`, CUDA events, µs/layer. Kernel time is **graph-invariant** (a cudagraph removes
launch gaps, not kernel duration), so this explains the graph-vs-graph decode loss directly.

| stage | decode µs | note |
|-------|-----------|------|
| activation quant | 16 | |
| gate_up sparse GEMM | 53 | **112 CTAs** (M=28672 → grid.y=112) |
| fused SwiGLU (3 kernels) | 31 | 27% of full — not the bottleneck |
| **down sparse GEMM** | **111** | **16 CTAs** (M=4096 → grid.y=16) |
| full fused MLP | 173 | vs NVFP4 dense MLP ≈ 137 |

**The sparse down projection underfills the RTX PRO 6000 at decode.** `matmul_sp` launches
`grid=(N/BN, M/2BM)`, BM=BN=128 (M=output features, N=tokens). At decode tp=128 the token dimension is
one tile, so occupancy is set by the output dimension: gate_up (28672) → 112 CTAs, but down (4096) →
**16 CTAs on ~188 SMs (~8% of the GPU)**. down is 2× gate_up despite **half** the weight nonzeros (29.4M
vs 58.7M) purely because 16 CTAs cannot fill the machine (K=14336 crammed onto few SMs). Sparse-FP4 still
beats dense-**bf16** at decode (full/bf16 = 0.72–0.77), so a dense-bf16 decode fallback loses; dense-NVFP4
would only reach ~parity while defeating the 2:4 purpose (storing dense = no sparsity benefit).

**Split-K / stream-K sparse scheduling is the required next kernel family.** The standalone kernel bench
already shows split-K fixes this shape (decode ffn-down 128/4096/14336: 0.49×→**1.40×** vs bf16, s=8); the
deployed `fused_mlp_2lvl` uses the plain (non-split-K) `matmul_sp`, so serving decode underfills. Wiring
the split-K decode kernel into the fused/graph serving path (plus a graph-friendly reduction and
two-level-scale correctness) is future work, not a final-paper cleanup.

## Exact commands
```bash
uv run modal run --detach harness/quadbit_serve.py --mode store_so
# NVFP4 production-graph baseline
uv run modal run --detach harness/quadbit_serve.py --mode hybrid --util 0.8 --baseline --graph --no-do-ppl
# quadbit sparse MLP in the production graph (Route B custom op)
uv run modal run --detach harness/quadbit_serve.py --mode hybrid --util 0.8 --fused --do-ppl --graph \
  --recovered-ckpt /cache/recovered_Llama-3.1-8B-Instruct_P30000_p25000_2sh_lr3e-05.pt
# decode kernel breakdown
uv run modal run --detach harness/quadbit_serve.py --mode profile_decode
```
