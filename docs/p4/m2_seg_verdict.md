# P4 Milestone 2 (seg path) — verdict: PASS

The segmented sparse-FP4 MoE path is CUDA-graph-capturable with zero host sync in the captured region.
Standalone harness `harness/p4_capture.py` (no vLLM), raw log `docs/audit/logs/p4_m2seg.log`.

Env: torch 2.12.0.dev20260408+cu128, NVIDIA RTX PRO 6000 Blackwell Server Edition, compute cap 12.0,
driver/CUDA 12.8 (the sm_120a build toolchain; the same `.so` ctypes-loads into the CUDA-13 serve image).

## What changed (all additive, frozen path untouched)

1. `cuda/sparse_fp4_lib.cu`: added `quantize_act_nvfp4_2lvl_s(..., void *stream)` — identical kernel to
   `quantize_act_nvfp4_2lvl`, launched on the caller's stream. The stream-less symbol and its ~40 callers
   are unchanged.
2. Capture path passes `torch.cuda.current_stream().cuda_stream` to both the quantizer and
   `sparse_moe_mm_2lvl` (so line 1030's `stream ? 0 : cudaDeviceSynchronize()` never syncs).
3. `route_fixed_cap`: fixed-capacity **device** routing replacing `build_routing`. `eblk` is a
   compile-time constant (`block // (cap/BN)`); slot destinations come from device prefix-sums; overflow
   rows are dropped deterministically. No `.item()`, no host loop, no data-dependent allocation.

## Results (graph replay vs eager, all 12 shapes)

Every shape: **capture OK, cos = 1.000000, relL2 = 0, maxrel = 0, nonfinite = 0** — graph replay is
bit-exact to eager.

- **Decode B=1/8/32/64** (all pad to Rp=1024): graph **0.113–0.116 ms vs eager 0.176–0.179 ms ≈ 1.5×
  faster** — the CPU-launch-overhead win, in the latency-critical decode regime.
- **Prefill 2048/8192/16384/65536**: graph ≈ eager (compute-bound; the win is launch-overhead, which is
  negligible at large M). prefill-65536 uses 71.6 GB (fixed capacity 98304 rows).
- **Pathological** (T=4096): all-to-one (18432 rows dropped), uniform, empty-experts (12288 dropped),
  near-capacity — all cos = 1.0, capture OK; overflow drop is deterministic.

`capture OK 12/12; graph==eager 12/12 → M2-SEG PASS.`

## Acceptance gates (all met)

- graph replay numerically matches eager (bit-exact, rel = 0);
- no hidden syncs (a host sync in the captured region would raise `cudaErrorStreamCaptureUnsupported`);
- no dynamic allocation in the captured region (fixed capacity, persistent workspace);
- no shape-changing tensors during replay;
- stable over repeated replays (3 replays + a 100-iter timing loop per shape);
- frozen Llama fused path untouched.

## Honest scope note

The prefill sweep stops at **M=65536** (71.6 GB). **M=131072 is omitted**: at fixed capacity it needs
~143 GB of workspace, over the RTX PRO 6000's ~96 GB — a memory limit of this standalone fixed-capacity
microbench, not a capture-safety limit (capture succeeded at every size run). In real serving, chunked
prefill bounds the per-step token count well below this, so the omission does not affect the vLLM
integration (M3/M4).

## Next (M3/M4)

M2-seg passing unblocks vLLM integration in the plan's order: (A) plugin-only MoE call in-process, (B)
one-layer MoE, (C) DeepSeek/GLM route-slot D2, (D) full GLM serving — each recording capture status,
exact failure locus if any, and correctness vs eager. The plugin's `patched_moe_apply` adopts the same
three changes (stream-safe quantizer, current-stream seg, fixed-capacity device routing), plus the A6
indexer `.cpu().tolist()` fix for attention capture.
