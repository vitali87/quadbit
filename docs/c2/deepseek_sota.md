# C2 DeepSeek-V4-Flash SOTA detail (4 GPU EP)

Same harness (`graph_gate4` / `_graph_gate_body`), same mito80 PPL passage, same decode-only formula
(`63/(wall64-wall1)`), same graph mode (captured), same memory accounting (vLLM per-worker logs). The
only thing that changes between A1 and A4 is the MoE selector. Toolchain: env M (CUDA 13.0, vLLM 0.24.0,
FlashInfer 0.6.14/cubin 0.6.13, torch 2.11.0+cu130). Raw logs `docs/audit/logs/c2_ds_*.log`.

## Full row metrics

| metric | A1 dense NVFP4 baseline | A4 quadbit D2 native captured |
|---|---:|---:|
| MoE path | vLLM native FlashInfer-CUTLASS fused NVFP4 (QB_MOE=off) | route-slot D2, `group_gemm_nvfp4` anchor + 2:4 tail |
| MoE backend logged | `FLASHINFER_CUTLASS` NvFp4 | quadbit sparse + native-nvfp4 anchor (`gu=y dn=y` layers 22+) |
| graph | captured, `cudagraph_mode=FULL_AND_PIECEWISE`, PIECEWISE=3/FULL=2 | captured FULL, PASS |
| PPL (mito80) | 4.1222 | 4.0943 |
| decode-only tok/s | **48.248** (wall1 0.223, wall64 1.528) | **5.972** (wall1 0.619, wall64 11.168) |
| weights GiB/GPU | 40.83 | 51.7 (+27%, dual residency raw NVFP4 + 2:4 codes) |
| KV cache GiB/GPU | 42.64 (120,416 tok, 58.8× concurrency) | 25.98 |
| graph-capture pool GiB | 0.18 | 2.08 (C1) |
| DSA | `sparse_mla_sm120_decode_dsv4` native (autotuned), `DEEPSEEK_SPARSE_SWA`, `fp8_ds_mla` KV | same, native |
| route/drop/overflow | n/a (dense, no fixed-cap routing) | cap=128, max_seqs=2, drop=0 |
| load+capture time | 956 s | ~1100 s (C1) |
| command | `graph_gate4 --cap 128 --max-seqs 2 --baseline dense_nvfp4` | `graph_gate4 --cap 128 --max-seqs 2 --dense-layers 0..21 --dense-anchor-backend native_nvfp4` |
| commit | c2-sota-board head | c2-sota-board head |

## Ablations (C1, identical passage + formula — directly comparable)

| row | PPL | decode tok/s | note |
|---|---:|---:|---|
| A5 D2 native eager | 4.0483 | 1.637 | native anchor, no capture |
| A6 D2 dequant captured | 3.9746 | 0.514 | the C1 "baseline" — the crippled all-E dequant loop |

C1's headline 11.3× (5.82/0.514) was A4-vs-A6: a within-sparse-path speedup over the pathologically slow
dequant loop, **not** a comparison to the production dense path A1.

## Verdicts (DeepSeek)

- **V1 decode: NO.** Dense NVFP4 fused baseline 48.248 tok/s vs quadbit D2 native captured 5.972 = the
  dense baseline is **8.1× faster at decode**. quadbit does not beat the best dense/NVFP4 graph baseline
  on decode. At decode (M=1–2 rows) the sparse path's multi-stage machinery (fixed-cap device routing +
  2:4 sparse experts + dual-residency dense anchor) cannot match vLLM's single autotuned fused NVFP4
  grouped GEMM; the 2:4 sparsity advantage is a prefill/large-M bandwidth effect, absent at decode.
- **V3 memory: dense wins.** Dense 40.83 GiB weights + 0.18 GiB pool; D2 51.7 GiB weights (+27%) + 2.08
  GiB pool, and D2's KV headroom is 25.98 vs 42.64 GiB. The sparse policy **increases** total memory
  (dual residency), it does not reduce it.
- **V4 quality: matched-to-slightly-worse.** mito80 PPL is a wash (4.0943 vs 4.1222, 80-token noise, not
  a quality win for either — do not rank on it). The real downstream evidence (400-item MC, `paper.md`
  §10): DeepSeek D2 .7304 vs dense .7383 = **−0.79 pt** — acceptable under the Pareto framing, but a small
  loss, not a gain.

**DeepSeek bottom line:** on decode and memory the production dense NVFP4 fused MoE is the SM120 SOTA;
quadbit sparse D2 trails on both at slightly worse downstream quality. Not a decode SOTA, not a decode
Pareto point. The next bottleneck is precisely a decode-time kernel/path-overhead problem (§ verdict).
