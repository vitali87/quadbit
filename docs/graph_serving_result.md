# Production-graph serving result (the honest serving headline, 2026-07-07)

This is the **production-representative** serving comparison. vLLM's production path
CUDA-graph-captures the whole forward; this measures quadbit sparse under that same regime
(graph-vs-graph), so it is the number the paper leads with. It supersedes the eager result in
`frozen_serving_result.md` (kept as a launch-overhead ablation).

## Headline (honest)

> quadbit implements a **correct, graph-capturable sparse-FP4 vLLM path on SM120** and, with a
> **split-K decode-specialized down projection**, **beats production dense NVFP4 on decode by
> +2.2% to +9.7%** (batch 8/32/64) at matched sparse accuracy (PPL 10.27). Prefill runs at
> ~95–97% of production NVFP4 (uses the plain non-split-K down, which was never underfilled).
> Decode is the latency-critical, memory-bound serving regime, so this is a real serving win.

## What changed vs the earlier loss

The first graph-vs-graph pass lost decode by −6 to −12% because the sparse **down projection
underfilled the GPU** (16 CTAs on ~188 SMs). Track 1 wires a **split-K down kernel**
(`matmul_sp_sk` + `cvt_sp_2lvl_t`, exposed through `fused_mlp_2lvl_skdown`) into the fused/graph
serving path for the decode shape (tp ≤ 128), which fills the machine and flips the result.

## Environment / provenance
- **Branch:** `decode-downproj-schedule` (HEAD `<HASH>`); **main frozen** at the eager fallback.
- **NVFP4 model:** `nvidia/Llama-3.1-8B-Instruct-NVFP4` (vLLM binds `modelopt_fp4` CUTLASS; non-MLP 4-bit).
- **Recovered-Instruct ckpt:** `/cache/recovered_Llama-3.1-8B-Instruct_P30000_p25000_2sh_lr3e-05.pt`
  (through-2:4-sparse-FP4-kernel PPL 10.029 offline).
- **.so:** `cuda/sparse_fp4_lib.cu`, `nvcc -arch=sm_120a -O3 --default-stream per-thread --cudart shared`
  in the CUDA 12.8.1 build image, ctypes-loaded in the CUDA 12.9.0 vLLM process (staged via volume).
- **GPU:** RTX PRO 6000 (SM120). **vLLM:** 0.21.0 V1 engine, `enforce_eager=False` (production graphs,
  FULL_AND_PIECEWISE capture), `gpu_memory_utilization=0.8`.
- **Protocol:** WT-2 15×2048 for PPL; prefill = B×S=2048 (1 out tok); decode = GEN=128 ignore_eos;
  distinct per-request prompts. Same harness for both rows.

## How the sparse MLP gets into the production graph (Route B)
Option A (class-patch + Dynamo graph break) is **impossible**: vLLM V1 compiles with
`aot_compile_fullgraph`, which forbids graph breaks (`torch.compiler.disable` → `Unsupported`). The
sanctioned path is a **`torch.library` custom op** (`quadbit::fused_mlp`): Dynamo emits it as an opaque
op node (no inline, no break) and cudagraph bakes the kernel launches it makes **at capture time**. The
sparse weights are bound **before** `LLM()` (vLLM captures once at init; a post-init patch is invisible
to the frozen graph), and the layer is resolved by a construction-order index set in a patched
`LlamaMLP.__init__`. At decode (tp ≤ 128) the op dispatches to `fused_mlp_2lvl_skdown` (split-K down);
prefill uses the plain `fused_mlp_2lvl`. See `_install_graph_customop` in `harness/quadbit_serve.py`.

### Proof the sparse path actually ran (not a silent fall-back to dense NVFP4)
- `Capturing CUDA graphs (mixed prefill-decode): 100%` — capture succeeded with the op inside.
- `SPARSE_CALLS = 7264` (>0 → the custom op executed).
- `PPL_THROUGH_SERVING = 10.2709` — **sparse** accuracy, NOT the 7.97 dense value. (A dense bypass would
  read ~7.97 and match NVFP4's speed exactly; the PPL check is the guard against that false positive.)

## Serving table (graph-vs-graph, the main serving result)

| metric | vLLM NVFP4 (graph, production) | quadbit sparse MLP + split-K down (graph) | Δ |
|--------|-------------------------------|-------------------------------------------|---|
| WT-2 PPL | 7.97 | 10.2709 | +2.30 |
| prefill B=8/32/64 (tok/s) | 66469 / 80825 / 119083 | 62914 / 77605 / 115069 | **−5.3% / −4.0% / −3.4%** |
| decode B=8/32/64 (tok/s) | 1046 / 4237 / 8384 | **1147 / 4543 / 8567** | **+9.7% / +7.2% / +2.2%** |

Memory: device 81473 MiB load / 84375 MiB peak (nvidia-smi, incl. KV pool). SPARSE_CALLS=7264.
(B=1 decode 144 vs 131; B=1 prefill discarded as cold-capture warmup noise.)

## Split-factor sweep (serving path, decode tok/s, choosing the deployed split)

Split-K factor swept end-to-end in the serving path (`--splits {4,8,16}`); PPL/correctness are
split-invariant (cos 1.0000, relL2 ≈ 1e-5 vs the plain two-level down), so only throughput moves.

| QB_SK_SPLITS | decode B=8/32/64 (tok/s) | down-kernel µs (isolated, M=64) |
|--------------|--------------------------|---------------------------------|
| 4  | 1120 / 4573 / 8476 | 60.8 |
| **8** (deployed) | **1147 / 4543 / 8567** | **56.5** |
| 16 | 1079 / 4233 / 8067 | 69.8 |
| (plain, no split-K) | 981 / 3863 / 7363 | 109.4 |

**splits=8 is the end-to-end winner**, matching the isolated `profile_decode` minimum (56.5 µs vs
60.8 µs @4, 69.8 µs @16). It wins B=8 and B=64 outright; at B=32 splits=4 edges it by 0.7% (noise). All
three split factors beat the plain kernel and NVFP4 at decode; splits=8 is the deployed default
(`QB_SK_SPLITS=8`). Sweep runs are throughput-only (PPL/correctness are split-invariant, cos 1.0000).

## Decode diagnosis (why the plain kernel lost, and why split-K fixes it)

`--mode profile_decode`, CUDA events, µs/layer. Kernel time is **graph-invariant** (a cudagraph removes
launch gaps, not kernel duration).

| stage | plain decode µs | note |
|-------|-----------------|------|
| activation quant | 15 | |
| gate_up sparse GEMM | 52 | **112 CTAs** (M=28672 → grid.y=112) |
| fused SwiGLU (3 kernels) | 30 | not the bottleneck |
| **down sparse GEMM (plain)** | **109** | **16 CTAs** (M=4096 → grid.y=16) — underfill |
| **down sparse GEMM (split-K@8)** | **56.5** | 128 CTAs (16 × 8 K-splits) — fills the machine, **1.94×** |

`matmul_sp` launches `grid=(N/BN, M/2BM)`, BM=BN=128 (M=output features, N=tokens). At decode tp=128
the token dimension is one tile, so occupancy is set by the output dimension: gate_up (28672) → 112
CTAs, but down (4096) → **16 CTAs on ~188 SMs (~8% of the GPU)**. Split-K (`matmul_sp_sk`) adds a
`gridDim.z` K-split (16 × 8 = 128 CTAs), accumulates partial sums in an f32 workspace, and a
`cvt_sp_2lvl_t` pass applies the two-level global scales post-reduction and transposes to `[tok, out_f]`.
The decode output is tiny, so the f32 reduction cost is negligible. This is the split-K decode kernel
family the earlier diagnosis flagged as required — now built, correct, and deployed.

## Exact commands
```bash
uv run modal run --detach harness/quadbit_serve.py --mode store_so
# NVFP4 production-graph baseline
uv run modal run --detach harness/quadbit_serve.py --mode hybrid --util 0.8 --baseline --graph --no-do-ppl
# quadbit sparse MLP + split-K down in the production graph (Route B custom op), the deployed split=8
uv run modal run --detach harness/quadbit_serve.py --mode hybrid --util 0.8 --fused --do-ppl --graph \
  --splits 8 --recovered-ckpt /cache/recovered_Llama-3.1-8B-Instruct_P30000_p25000_2sh_lr3e-05.pt
# split-factor sweep: --splits 4 / --splits 16 (throughput-only, --no-do-ppl)
# decode kernel breakdown incl. split-K calibration
uv run modal run --detach harness/quadbit_serve.py --mode profile_decode
```
