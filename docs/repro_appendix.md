# Reproducibility appendix

Exact commands, checkpoints, model ids, and the staged-build recipe behind every measured claim in
`docs/paper.md`. All runs are on a Modal cloud RTX PRO 6000 (SM120, no tcgen05). Kernels are compiled
`nvcc -arch=sm_120a`. Where a claim's status is `needs-rerun` in `docs/claims_checklist.md`, the artifact
is on the Modal volume only and should be regenerated with the command here before it is cited as backed.

---

## Environment

- **GPU:** Modal cloud RTX PRO 6000 (SM120).
- **vLLM:** 0.21.0, V1 engine, `enforce_eager=False` for production graphs (FULL_AND_PIECEWISE capture),
  `gpu_memory_utilization=0.8` unless noted.
- **SGLang:** 0.5.
- **Package management:** `uv` (Astral). Modal jobs launched with `uv run modal run --detach ...` so long
  runs survive as detached apps (verify with `modal app list` / `modal app logs`).
- **CUDA toolchain split (load-bearing):** quadbit's `sm_120a` block-scale mma
  (`kind::mxf4nvf4`/`block_scale`/`scale_vec::4X`) assembles only under CUDA <= 12.8 (ptxas 13 rejects it),
  while FlashInfer's `b12x` path needs CUDA 13. The two cannot coexist in one container, which forces the
  staged build below and the two-container cross-table for the sparse-vs-FlashInfer-dense Pareto.

## Models and checkpoints

- **NVFP4 dense reference (production baseline):** `nvidia/Llama-3.1-8B-Instruct-NVFP4` (vLLM binds the
  `modelopt_fp4` CUTLASS method; non-MLP linears stay 4-bit).
- **bf16 base models:** `meta-llama/Llama-3.1-8B-Instruct`, `meta-llama/Meta-Llama-3-8B`.
- **Recovered sparse checkpoint (serving/crossover):**
  `/cache/recovered_Llama-3.1-8B-Instruct_P30000_p25000_2sh_lr3e-05.pt` on Modal volume
  `quadbit-hf-cache`. Recipe: phase-1 30000 steps, phase-2 5000 steps, both shards, lr 3e-5, WikiText
  corpus; through-2:4-sparse-FP4-kernel PPL 10.029 offline, 10.27 through serving.
- **Accuracy corpus:** WikiText-2, 15x2048 or 16x2048 windows (stated per table).

## Staged .so build recipe (the CUDA 12.8 compile / 12.9 vLLM load split)

The sparse kernels are compiled into a shared object in a CUDA 12.8.1 image and ctypes-loaded into the
CUDA 12.9.0 vLLM process; the `.so` is passed between them on a shared Modal volume.

```bash
# compile + stage the .so (built in the CUDA 12.8.1 image, written to the volume)
uv run modal run --detach harness/quadbit_serve.py --mode store_so
```

The compile command inside the 12.8.1 image:

```
nvcc -arch=sm_120a -O3 --default-stream per-thread --cudart shared cuda/sparse_fp4_lib.cu -o <staged>.so
```

`--default-stream per-thread` binds the kernels to vLLM's stream so no cross-stream synchronize is needed;
`--cudart shared` lets the 12.9 vLLM process load it against its own runtime. The vLLM process then
ctypes-loads the staged `.so`; the sparse weights are bound before `LLM()` (vLLM captures the graph once at
init, so a post-init patch would be invisible), and the sparse MLP is exposed as the `torch.library` custom
op `quadbit::fused_mlp` so Dynamo emits it as an opaque node that CUDA-graph capture bakes at capture time.

## Kernels

- `cuda/matmul_sp_wide_swz2.cu` unit sparse (2731k ceiling).
- `cuda/matmul_sp_full_wide.cu` deployable sparse (2116k).
- `cuda/matmul_fp4_pp_bf16.cu` dense (1503k).
- `cuda/dense_fp4_lib.cu`, `cuda/sparse_fp4_lib.cu` PyTorch-callable libs plus fused quantizer;
  `sparse_fp4_lib.cu` holds `matmul_sp`, the split-K `matmul_sp_sk`, `cvt_sp_2lvl_t`, the two-level
  transposed epilogue `sparse_fp4_mm_2lvl_t`, and the fused entries `fused_mlp_2lvl` /
  `fused_mlp_2lvl_skdown`.
- `cuda/dense_nvfp4_fast_lib.cu` deployed two-level dense with async scale prefetch.

## Commands by result

### Dense leaderboard (Section 4): quadbit dense loses to FlashInfer 1.35 to 2.2x
```bash
uv run modal run --detach harness/leaderboard_fp4.py    # all FlashInfer backends + quadbit, fp32-ref cos>0.97 gate
```

### CUTLASS baselines (Sections 4 and 5)
```bash
uv run modal run --detach harness/cutlass_fp4.py        # dense vs CUTLASS 79b (square), SASS dissect
uv run modal run --detach harness/cutlass_shapes.py     # rectangular Llama-3-8B shapes vs 79b/80b
uv run modal run --detach harness/cutlass_sparse.py     # sparse two-level vs CUTLASS 80b (80b ref-verify passes every size)
```

### Throughput vs bf16 (Section 5)
```bash
uv run modal run --detach harness/bench_vs_bf16.py      # M=N=K ceiling table
uv run modal run --detach harness/bench_llm_shapes.py   # real Llama-3-8B GEMM shapes
```

### Deployment / fusion (Section 6)
```bash
uv run modal run --detach harness/quadbit_linear.py     # nn.Linear drop-in, packer maxrel
uv run modal run --detach harness/real_model.py         # frontier-model tiling + full fused Qwen3-8B block
```

### Dense accuracy W4A4 (Section 7)
```bash
uv run modal run --detach harness/recovery_worth.py     # dense W4A4 PPL, zero-calibration two-level per-16
```

### Sparse recovery and deploy gap (Section 8)
```bash
uv run modal run --detach harness/sparsegpt_pair.py     # one-shot pair-granular SparseGPT mask
uv run modal run --detach harness/finetune_pair.py      # phase-1 distill + phase-2 QAT recovery
uv run modal run --detach harness/verify_sparse_2lvl.py # two-level sparse kernel correctness
uv run modal run --detach harness/ab_sparse_semantics.py # single-level (11.89) vs two-level (8.95) vs fake-quant (8.96)
uv run modal run --detach harness/sensitivity_sparse.py # training-free hybrid placement (negative)
```

### Real serving engine baselines (Section 9 Table A)
```bash
uv run modal run --detach harness/vllm_nvfp4.py serve   # vLLM bf16 and NVFP4
uv run modal run --detach harness/sglang_fp4.py --mode bench  # SGLang NVFP4
uv run modal run --detach harness/dense_e2e.py          # quadbit dense prototype (prefill-only, Table B)
```

### Production-graph serving (Section 9 Table C, decode win)
```bash
# NVFP4 production-graph baseline
uv run modal run --detach harness/quadbit_serve.py --mode hybrid --util 0.8 --baseline --graph --no-do-ppl
# quadbit sparse MLP + split-K down inside the production graph, deployed split=8
uv run modal run --detach harness/quadbit_serve.py --mode hybrid --util 0.8 --fused --do-ppl --graph \
  --splits 8 --recovered-ckpt /cache/recovered_Llama-3.1-8B-Instruct_P30000_p25000_2sh_lr3e-05.pt
# split-factor sweep: --splits 4 / --splits 16 (throughput-only, --no-do-ppl; PPL/correctness split-invariant)
# decode kernel breakdown incl. split-K calibration
uv run modal run --detach harness/quadbit_serve.py --mode profile_decode
```

### End-to-end request crossover (Section 9): 81/112 sparse wins
```bash
# baseline (NVFP4) crossover matrix
uv run modal run --detach harness/quadbit_serve.py --mode hybrid --util 0.8 --baseline --graph --crossover --no-do-ppl
# sparse crossover matrix (prefix caching OFF is built in; per-(B,P) warmup)
uv run modal run --detach harness/quadbit_serve.py --mode hybrid --util 0.8 --fused --graph --crossover \
  --recovered-ckpt /cache/recovered_Llama-3.1-8B-Instruct_P30000_p25000_2sh_lr3e-05.pt
```
Output CSVs: `/cache/crossover_nvfp4.csv`, `/cache/crossover_sparse.csv` (112 cells each).

### Verification / multi-token sweep (Section 9 Track 4B, refuted)
```bash
uv run modal run --detach harness/quadbit_serve.py --versweep   # effective M = B*k sweep
```
Output: `/cache/versweep_nvfp4.csv`, `/cache/versweep_sparse.csv`.

### Reverse densification accuracy Pareto (Section 9, negative)
```bash
uv run modal run --detach harness/quadbit_serve.py --mode densify --policy <p>
# p in {none, all, down, gateup, L<a>-<b>, gu:<a>-<b>, dn:<a>-<b>}
```

### Phase-adaptive same-weight (Section 9 Track 4C, refuted)
```bash
uv run modal run --detach harness/quadbit_serve.py --phase-adaptive       # crossover matrix for dense-prefill/sparse-decode
uv run modal run --detach harness/quadbit_serve.py --mode phase_bench     # us/layer dense-flashinfer vs native NVFP4 vs sparse
```
Output: `/cache/crossover_phaseadaptive.csv`. SM120 dense NVFP4 recipe used by the dense phase:
`nvfp4_quantize(t, (448*6)/amax, sfLayout=layout_128x4, do_shuffle=False)` on both operands, then
`mm_fp4(a, b.T, a_sf, b_sf.T, 1/(gsa*gsw), backend="cutlass")` (`do_shuffle=True` is trtllm-only and
trtllm refuses SM120, so cutlass with `do_shuffle=False` is the viable pairing).

### Layout / probe derivation (Section 3)
```bash
uv run modal run --detach harness/probe_ldmatrix.py     # ldmatrix negative result
uv run modal run --detach harness/verify_sparse_2lvl.py # metadata/scale layout verification (maxrel 0)
```

## Data source index

| Artifact | Produced by | Consumed by |
|----------|-------------|-------------|
| `/cache/crossover_nvfp4.csv`, `/cache/crossover_sparse.csv` | `quadbit_serve.py --crossover` | `docs/crossover_result.md`, Fig 7 |
| `/cache/versweep_nvfp4.csv`, `/cache/versweep_sparse.csv` | `quadbit_serve.py --versweep` | `docs/crossover_result.md` Section 4B, Fig 10 |
| `/cache/crossover_phaseadaptive.csv` | `quadbit_serve.py --phase-adaptive` | `docs/crossover_result.md` Section 4C |
| `/cache/recovered_Llama-3.1-8B-Instruct_P30000_p25000_2sh_lr3e-05.pt` | `finetune_pair.py` | serving, crossover, densify, phase-adaptive |
| staged `sparse_fp4_lib.so` | `quadbit_serve.py --mode store_so` | graph serving, crossover |

## Full chronological build log

Memory note `quadbit-raw-ptx.md` (kept outside the repo) records the complete probe-and-verify build order,
the breakthroughs (wide-TMA-plus-swizzle, two-level rescale, split-K down), and the dead ends
(stream-K, persistent pipeline, cluster TMA multicast, real-scale decode kernel).
