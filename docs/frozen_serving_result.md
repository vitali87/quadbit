# FROZEN serving result (fallback paper result, 2026-07-07)

This is the correct, proven eager-serving result. The CUDA-graph-capture work happens on the
`graph-capture` branch and MUST NOT mutate this path on `main` until a graph result is proven.

## Commit / checkpoint / environment
- **Frozen commit:** `c60f179f1f921aec9c4299b5d6cdcea8ffecbde9` (branch `main`)
- **Recovered-Instruct checkpoint:** `/cache/recovered_Llama-3.1-8B-Instruct_P30000_p25000_2sh_lr3e-05.pt`
  (Modal volume `quadbit-hf-cache`; recipe p1=30000, p2=5000, both_shards, lr 3e-5, WikiText corpus;
  through-2:4-sparse-FP4-kernel PPL 10.029 offline)
- **NVFP4 model:** `nvidia/Llama-3.1-8B-Instruct-NVFP4` (vLLM binds `modelopt_fp4` cutlass; non-MLP 4-bit)
- **.so:** `cuda/sparse_fp4_lib.cu` compiled `nvcc -arch=sm_120a -O3 --default-stream per-thread
  --cudart shared` in the 12.8.1 image, ctypes-loaded in the 12.9 vLLM process (staged via volume)
- **GPU:** RTX PRO 6000 (SM120); vLLM 0.21 V1 engine; `enforce_eager=True`; `gpu_memory_utilization=0.8`
- **Protocol:** WT-2 16x2048 for PPL; prefill = B x S=2048 (1 out tok); decode = GEN=128 ignore_eos;
  distinct per-request prompts

## Result (same eager harness vs full vLLM NVFP4)
| metric | NVFP4 (dense) | quadbit sparse MLP + NVFP4 | Δ |
|--------|---------------|----------------------------|---|
| WT-2 PPL | 7.97 | 10.27 | +2.30 |
| prefill B=8/32/64 (tok/s) | 61118/74916/110600 | 63409/79051/116748 | +3.7% / +5.5% / +5.6% |
| decode B=8/32/64 (tok/s) | 228/897/1750 | 282/1102/2157 | +23.7% / +22.9% / +23.3% |

Enablers (all on `main` at the frozen commit): zero-copy transposed epilogue (`sparse_fp4_mm_2lvl_t`),
two-level fused SwiGLU (`swiglu_amax`/`finalize`/`swiglu_quant_g`), single no-sync `fused_mlp_2lvl`.

## Exact commands
```bash
# rebuild + stage the .so (per-thread stream)
uv run modal run --detach harness/quadbit_serve.py --mode store_so
# NVFP4 eager baseline
uv run modal run --detach harness/quadbit_serve.py --mode hybrid --util 0.8 --baseline --no-do-ppl
# unified row (recovered-Instruct sparse MLP + NVFP4, PPL + tok/s)
uv run modal run --detach harness/quadbit_serve.py --mode hybrid --util 0.8 --fused --do-ppl \
  --recovered-ckpt /cache/recovered_Llama-3.1-8B-Instruct_P30000_p25000_2sh_lr3e-05.pt
# accuracy control (recovered base, bf16 non-MLP, all-MLP sparse)
uv run modal run --detach harness/quadbit_serve.py --mode recovered --util 0.8 --fused
```

## Caveat this result carries
Eager-vs-eager. NVFP4's production path uses CUDA graphs (higher absolute, e.g. decode 8465 @B64 in
Table A). The `graph-capture` branch turns this into a production-comparable result, especially decode.
