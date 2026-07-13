# C5 final board: decode collective levers vs the C4 SOTA (58.126 tok/s)

DeepSeek-V4-Flash-NVFP4 dense baseline (`QB_MOE=off`), 4 GPU unless noted, captured, mito80 PPL, two-run
TTFT-subtracted decode. Success condition: **beat 58.126 tok/s reproducibly**.

| row | config | decode tok/s | vs 48.248 | vs 58.126 | AR/tok | collective path (trace) | PPL | graph | log |
|---|---|---:|---:|---:|---:|---|---:|---|---|
| 1 | C2 dense NVFP4 fused baseline | 48.248 | 1.00x | -17.0% | 43.5 ring | `ncclDevKernel_AllReduce_RING_LL` | 4.1222 | FULL | c2 |
| 2 | **C4 one-shot custom AR** | **58.126** (median of 4) | **+20.5%** | 1.00x | 43.5 one-shot | `vllm::cross_device_reduce_1stage` | 4.2514 | FULL | c4 |
| 3a | C5 reduce-count: TP=2 | 40.565 | -15.9% | -30.2% | 43.5 (2-GPU) | `cross_device_reduce_1stage` (w=2) | 4.0855 | FULL | `c5_tp2_dense` |
| 3b | C5 reduce-count: DP attention | BLOCKED | - | - | ~0 attn AR (target) | not measurable (offline LLM rejects DP>1) | - | - | `c5_dp_attention` |
| 4 | C5 hierarchical: NCCL tree | 48.983 | +1.5% | -15.7% | 43.5 tree | `ncclDevKernel_...Tree` | 4.0102 | FULL | `c4_ar_scoped_tree` |
| 5 | **C5 combined best** | **= row 2 (58.126)** | +20.5% | 1.00x | 43.5 one-shot | `cross_device_reduce_1stage` | 4.2514 | FULL | - |

Memory (per GPU): dense TP=4 weights 40.83 GiB, pool 0.18; TP=2 weights ~81.7 GiB (2x, why it is slower and
near the 96 GiB limit). Custom AR adds two small IPC buffers (`meta_size + max_size`, max_size 8 MB) per rank,
negligible vs weights.

## Result

**No C5 row beats 58.126.** The best decode remains C4's one-shot custom AR (row 2 = row 5). Every C5 lever
either lost (TP=2, tree) or was unreachable in this harness (DP attention). Success condition **not met**;
the C4 +20.5% SOTA stands. See `verdict.md`.

## Reproducibility / topology note

C4's 58.126 is 4 runs on **fully-connected** P2P containers; Modal 4-GPU allocations vary and a partial-P2P
container falls back to NCCL (48.248) via the safety guard. C5's TP=2 and roofline runs independently
confirmed both a full 2x2 matrix and a partial 4-GPU matrix, so the variance is real (see
`post_c4_roofline.md`). This is a coverage caveat on C4, not a correctness issue.
