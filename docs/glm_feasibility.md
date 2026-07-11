# GLM-5.2 transfer feasibility (Phase 2)

> **RESOLVED (Phase 3):** the open load-gate risk below cleared. GLM-5.2 loads and generates
> coherently on 8x RTX PRO 6000 (SM120); DSA runs natively (`FLASHINFER_MLA_SPARSE_SM120`); the
> structural sparse policies transfer. Full result: `docs/glm_results.md`. Only caveat found is
> graph capture (eager-only, plugin host-sync in the EP loop), not a DSA/kernel/memory blocker.

Probe: `serve_dsv4.py --mode glm_inspect` (CPU-only, `nvidia/GLM-5.2-NVFP4`). No assumptions carried
over from DeepSeek; every row below is measured from the checkpoint config + safetensors index or the
installed vLLM registry.

## Architecture vs DeepSeek-V4-Flash

| field | GLM-5.2-NVFP4 | DeepSeek-V4-Flash-NVFP4 | transfer |
|---|---|---|---|
| model_type / arch | glm_moe_dsa / GlmMoeDsaForCausalLM | deepseek_v4 | different class |
| hidden layers | 78 (first 3 dense, 75 MoE) | 43 MoE | more layers |
| hidden size | 6144 | 4096 | bigger |
| routed experts | 256 | 256 | same |
| top-k | **8** | 6 | route-slot math changes (8 slots) |
| moe_intermediate | 2048 | 2048 | same |
| shared experts | 1 | 1 | same |
| attention | DSA (Deep Sparse Attn) + MLA (kv_lora 512, q_lora 2048) | MLA + indexer | MLA transfers; DSA indexer TBD |
| quant | NVFP4 modelopt, group 16, weight+act 4-bit, KV fp8 | NVFP4 modelopt, group 16 | **identical scheme** |
| quantized modules | routed experts (Linear); attn/shared/layers0-2/embeds excluded | routed experts | **same target** |

## Weight format (from safetensors index)

- Checkpoint total = **432.9 GiB across 47 shards**.
- Tensor-name buckets: routed_experts 231,168 entries (the NVFP4 uint8 experts = quadbit sparse-op target), attention 742, shared_expert 228, other 247.
- Routed experts are NVFP4 modelopt, identical two-level scheme to DeepSeek -> **quadbit dequant/pack/seg_gemm should attach unchanged** (same per-16 e4m3 blockscale + per-tensor scale, moe_int 2048).

## vLLM / serving support

- **vLLM 0.24.0 (the pinned image) already registers `GlmMoeDsaForCausalLM`.** No upgrade needed to instantiate the model class (avoids the plugin-compat risk a vLLM bump would carry).
- Vendor serving commands (model card): vLLM `--tensor-parallel-size 8 --enable-expert-parallel --kv-cache-dtype fp8_e4m3`; SGLang TP=8. Tested on B200/B300.

## Memory fit on RTX PRO 6000 (95 GiB usable/GPU)

| GPUs | capacity | weights | verdict |
|---|---|---|---|
| 2 | 190 GiB | 433 GiB | **DOES NOT FIT** (−243) |
| 4 | 380 GiB | 433 GiB | **DOES NOT FIT** (−53) |
| 8 | 760 GiB | 433 GiB | **FITS** (327 GiB left for KV + activations) |

## Verdict

- **Architecturally compatible.** Same NVFP4 modelopt routed-expert format, same 256+1 experts / moe_int 2048, MLA attention, and `glm_moe_dsa` is supported by the pinned vLLM. The quadbit sparse-FP4 expert op and the DeepSeek-discovered structural policies (down-anchor, route-slot) should transfer to the expert path with only the top-8 (vs top-6) slot-count change.
- **Hard requirement: 8x RTX PRO 6000.** GLM-5.2-NVFP4 is ~433 GiB and does NOT fit on 2 or 4; it needs 8 GPUs (EP), 4x DeepSeek's footprint. The user's "2 or 4 GPU" Phase-4 target is not reachable for GLM; 8-GPU is the floor.
- **Open risk requiring a load test (not resolvable from config):** whether the DSA (Deep Sparse Attention) indexer and GlmMoeDsa forward path run on SM120 under the quadbit plugin. DeepSeek's MLA + indexer needed SM120-specific patches (block-FP8 attention GEMM, cooperative-cluster topk); GLM's DSA may hit the same gaps or new ones. Gate: an 8-GPU dense-MoE load + coherent-generation smoke.
- **Blocker to confirm before Phase 3/4:** 8x RTX PRO 6000 availability on the Modal account.
