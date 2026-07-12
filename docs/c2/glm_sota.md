# C2 GLM-5.2 SOTA detail (8 GPU EP)

Same harness (`glm_graph_gate` / `_graph_gate_body`), same mito80 PPL passage, same decode-only formula,
same graph mode, same memory accounting as the DeepSeek board. Only the MoE selector changes between B1
and B3. Toolchain env M. Raw logs `docs/audit/logs/c2_glm_*.log`.

## Full row metrics

| metric | B1 dense NVFP4 baseline | B3 quadbit route-slot D2 native captured |
|---|---:|---:|
| MoE path | vLLM native FlashInfer-CUTLASS fused NVFP4 (QB_MOE=off) | route-slot D2, `group_gemm_nvfp4` anchor + 2:4 tail |
| MoE backend logged | `FLASHINFER_CUTLASS` NvFp4 | quadbit sparse + native-nvfp4 anchor (layers 0–37 `gu=y dn=y`) |
| graph | captured (FULL+PIECEWISE), PASS | captured FULL, PASS |
| PPL (mito80) | 3.9572 | 4.0674 |
| decode-only tok/s | **33.810** (wall1 0.166, wall64 2.029) | **5.367** (wall1 0.549, wall64 12.287) |
| weights GiB/GPU | 54.62 | 68.98 (+26%, dual residency) |
| KV cache GiB/GPU (tokens) | 31.64 (629,760) | 11.89 (236,672) |
| graph-capture pool GiB | 0.10 | 0.80 |
| DSA | `FLASHINFER_MLA_SPARSE_SM120` native, `fp8_ds_mla` KV, `DEEPSEEK_V32_INDEXER` | same, native |
| route/drop/overflow | n/a | cap=128, max_seqs=2, drop=0 |
| coherent gen | ✓ (Paris+distances / fibonacci / RGB / H2O) | ✓ (same) |
| command | `glm_graph_gate --cap 128 --max-seqs 2 --baseline dense_nvfp4` | `glm_graph_gate --cap 128 --max-seqs 2 --dense-layers 0..37 --dense-anchor-backend native_nvfp4` |
| commit | c2-sota-board head | c2-sota-board head |

## Verdicts (GLM)

- **V2 decode: NO.** Dense NVFP4 fused baseline 33.810 tok/s vs quadbit D2 native captured 5.367 = dense
  is **6.30× faster at decode**. Same mechanism as DeepSeek: the sparse serving path cannot match the
  single fused NVFP4 grouped GEMM at decode (M=1–2).
- **V3 memory: dense wins decisively.** Dense 54.62 GiB weights + 0.10 pool with 629,760 KV tokens; D2
  68.98 GiB weights (+26%) + 0.80 pool with only 236,672 KV tokens (−62%). Dual residency both raises
  weight memory and collapses KV capacity.
- **V4 quality: intact but a small loss.** mito80 PPL is a wash (4.0674 vs 3.9572, 80-token noise). The
  real downstream evidence (`glm_results.md`, tokenizer-agnostic MC, `limit=200`): D2 .7508 vs dense
  .7603 = **−0.95 pt**, no task collapse — acceptable under the Pareto framing, but a loss, not a gain.

**GLM bottom line:** the graph-enabled sparse-policy transfer is real (D2 captures, DSA native, downstream
smoke intact), but there is **no** decode or memory advantage over the dense NVFP4 fused baseline — the
"measurable decode advantage" required by success-condition #3 is absent (dense is 6.3× faster). GLM
confirms the DeepSeek finding on a second architecture and GPU count.
