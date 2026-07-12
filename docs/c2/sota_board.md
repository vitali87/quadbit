# C2 SOTA board: SM120 sparse-FP4 MoE decode, controlled same-harness comparison

The C1 result (native FlashInfer grouped-NVFP4 delegation removes the dense-anchor bottleneck) put on a
direct SOTA board against the strongest dense/NVFP4 baseline that actually runs on this hardware, at the
**same** hardware, model, graph mode, PPL protocol, decode-only formula, and memory accounting.

## Toolchain (all rows, env M)

| field | value |
|---|---|
| GPU | RTX PRO 6000 (Blackwell, compute cap 12.0, `sm_120a`), Modal-managed |
| CUDA | 13.0 (`nvidia/cuda:13.0.0-devel-ubuntu22.04`) |
| torch | 2.11.0+cu130 (vLLM-resolved) _(confirm from run log)_ |
| vLLM | 0.24.0 (V1) |
| FlashInfer | python 0.6.14 + cubin 0.6.13 (force-reinstalled `--no-deps`) |
| SGLang | not installed in the serve image (see "Baselines that do not run") |
| plugin | quadbit `qb_sm120` vLLM `general_plugins` entry point, commit _(C2 branch head)_ |

## Protocol (identical across every row, so the board is apples-to-apples)

- **Harness:** `serve_dsv4.py::graph_gate4` (DeepSeek, 4 GPU) / `glm_graph_gate` (GLM, 8 GPU), both via
  `_graph_gate_body`. Every row below is the same code path with only the MoE selector changed.
- **PPL protocol — "mito":** teacher-forced NLL over one fixed passage (the "The mitochondria is the
  powerhouse of the cell…" passage in `_graph_gate_body`), `prompt_logprobs=0`, finite-NLL mean, `exp()`.
  It tokenizes to **80 tokens on DeepSeek** ("mito80"). **All rows on a given model share this exact
  passage**, so their PPL is directly comparable within a model. PPL from any other protocol (the GLM
  114-token policy-sweep passage, the P4 short passages, the DeepSeek downstream calib passages) is marked
  **protocol-mismatched** and excluded from quality ranking. Note: an 80-token PPL is noise-dominated;
  the real quality evidence is the downstream MC smoke suite, not this micro-PPL.
- **Decode-only tok/s:** two-run TTFT-subtracted, `63 / (wall64 - wall1)` from timing a 64-token and a
  1-token greedy generation of a fixed prompt (subtracts the shared prefill). Never prefill+decode.
- **Graph mode:** captured rows run `enforce_eager=False` + `QB_GRAPH=1`; eager rows `enforce_eager=True`.
  A row that cannot capture is reported as such, never silently compared eager-to-graph.
- **Memory accounting:** parsed from vLLM's own per-worker logs (V1 multiproc: driver-side `torch.cuda`
  is not representative) — model-weights load, KV-cache reserve, graph-capture pool, and peak.
- **DSA / route / drop / backend:** read from plugin + vLLM stdout in each raw log.

## The board

Decode tok/s, PPL (mito62), graph mode, GPUs, memory. **PENDING** = run in flight; filled on completion.
Raw logs `docs/audit/logs/c2_*.log`.

### DeepSeek-V4-Flash (4 GPU EP, route-slot D2 policy where sparse)

| # | row | MoE path | graph | PPL (mito80) | decode tok/s | weights GiB/GPU | pool GiB | DSA | log |
|---|---|---|---|---:|---:|---:|---:|---|---|
| A1 | **dense NVFP4 baseline** | vLLM native FlashInfer-CUTLASS fused NVFP4 (QB_MOE=off) | captured (FULL+PIECE) | 4.1222 | **48.248** | 40.83 | 0.18 | native | `c2_ds_dense_baseline_C.log` |
| A4 | **quadbit D2 native captured** | route-slot D2, `group_gemm_nvfp4` anchor | captured (FULL) | 4.0943 | **5.972** | 51.7 | 2.08² | native | `c2_ds_d2_native_C.log` |
| A5 | quadbit D2 native eager (ablation) | route-slot D2, native anchor, eager | eager | 4.0483¹ | 1.637¹ | — | — | native | C1 `c1_d2_native_B.log` |
| A6 | dequant captured (historical bottleneck) | route-slot D2, dequant `_dense_seg_gs` | captured | 3.9746¹ | 0.514¹ | — | — | native | C1 `c1_d2_dequant_C.log` |

¹ C1 numbers, measured on the identical `_graph_gate_body` passage + decode formula, directly comparable.
² D2 pool 2.08 GiB/GPU (C1). D2 weights 51.7 GiB/GPU = +27% over the dense baseline's 40.83 (dual
residency: raw NVFP4 dense slots + 2:4 sparse codes co-resident); D2 KV headroom 25.98 GiB vs dense 42.64.

**Headline (DeepSeek, decode + memory): the dense NVFP4 fused MoE baseline dominates.** It decodes at
**48.248 tok/s vs quadbit D2's 5.972** (same C2 build, same harness) = the dense baseline is **8.1× faster
at decode**, using **less** weight memory (40.83 vs 51.7 GiB/GPU, D2 +27% from dual residency) and a
smaller graph pool (0.18 vs 2.08 GiB). mito80 PPL is a wash (D2 4.0943 vs dense 4.1222, 80-token noise).
C1's 11.3× was measured against the crippled dequant loop (0.514 tok/s), not this production dense path.
The quadbit sparse advantage is **not** a decode-speed or decode-memory win; see the verdict for where it
does live (training-free quality-preserving structural sparsity + graph-enabled cross-arch transfer; and
the prefill/large-M kernel Pareto, `paper.md` §5, not re-measured here).

### GLM-5.2 (8 GPU EP, route-slot D2 policy where sparse)

| # | row | MoE path | graph | PPL (mito80) | decode tok/s | weights GiB/GPU | pool GiB | DSA | log |
|---|---|---|---|---:|---:|---:|---:|---|---|
| B1 | **dense NVFP4 baseline** | vLLM native FlashInfer-CUTLASS fused NVFP4 (QB_MOE=off) | captured | 3.9572 | **33.810** | 54.62 | 0.10 | native | `c2_glm_dense_baseline_C.log` |
| B3 | **quadbit route-slot D2 native captured** | D2, `group_gemm_nvfp4` anchor | captured (FULL) | 4.0674 | **5.367** | 68.98 | 0.80 | native | `c2_glm_d2_native_C.log` |
| B2 | quadbit route-slot D2 eager (ref) | D2, eager | eager | — | 2.10² | — | — | native | C1 / glm_results |
| B5 | dequant captured (ablation) | D2, dequant loop | captured | 4.157²·ᵖ | — | — | — | native | C1 / P4 |

² GLM D2 pool 1.21 GiB/GPU, eager decode ref 2.10 tok/s (C1). ᵖ GLM dequant-captured PPL 4.157 is on a
different short passage vs mito80 → protocol-mismatched, excluded from the quality ranking.

**Headline (GLM, decode + memory): dense NVFP4 fused MoE baseline wins decode 33.810 vs D2 5.367 = 6.30×**
(same C2 build), DSA `FLASHINFER_MLA_SPARSE_SM120` native both. Dense 54.62 GiB weights + 0.10 pool, KV
629,760 tok; D2 68.98 GiB weights (+26%) + 0.80 pool, KV only 236,672 tok. Same shape as DeepSeek: dense
wins decode and memory. mito80 PPL a wash (D2 4.0674 vs dense 3.9572, noise). GLM downstream smoke (real
quality, `glm_results.md`): D2 .7508 vs dense .7603 = −0.95 pt, intact but a small loss.

## Baselines that do not run on SM120 (reported, not hidden)

| baseline | model | status | reason |
|---|---|---|---|
| Vanilla vLLM (no plugin) | DeepSeek-V4-Flash NVFP4 | **fails to init** | ue8m0 W8A8 FP8 attention: DeepGEMM SF-transform asserts, CUTLASS c3x scaled_mm no-dispatch on SM120 (claims_checklist row 120; serve3/serve4 logs). The quadbit plugin's attention/DSA unblock is what makes A1/B1 run at all — so the dense baseline **is** vLLM's native NVFP4 fused MoE, just with SM120 attention unblocked. |
| SGLang NVFP4 MoE | DeepSeek-V4-Flash / GLM-5.2 | **unavailable** | SGLang not in the serve image; no SM120 DSA / FP8-attention path for these models. `harness/sglang_fp4.py` covers only Llama dense-FP4 GEMM microbench, not this MoE model. |
| FlashInfer-native fused NVFP4 MoE, standalone | — | **subsumed** | The FlashInfer-CUTLASS fused NVFP4 MoE **is** exactly what A1/B1 (QB_MOE=off) invoke in-serving. A standalone call would measure the same kernel without the real routing/KV/graph context, so the in-serving A1/B1 row is the stronger, fairer baseline. |

## Verdicts

See `docs/c2/verdict.md` for V1–V5 once the board is filled. DeepSeek detail: `docs/c2/deepseek_sota.md`;
GLM detail: `docs/c2/glm_sota.md`.
