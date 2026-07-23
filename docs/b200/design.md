# B200 head-to-head: is the GLM-5.2 decode gap interconnect or kernels?

Pre-registered before the run returned. Branch `b200-glm-headtohead`, PR #38.

## Question

Aster (YC) advertises GLM-5.2 at **281 tok/s**; Baseten advertises **280+ tok/s** on Blackwell, measured
by Artificial Analysis 2026-06-22. Our frozen GLM-5.2 dense-NVFP4 row is **33.810 tok/s**
([c2/glm_sota.md](../c2/glm_sota.md)), 8x RTX PRO 6000, PCIe, no NVLink. That is roughly **8.3x**.

C4 ([floor_decomposition.md](../c4/floor_decomposition.md)) already attributed the SM120 decode step:
**90.8%** of GPU-kernel time is one NCCL ring all-reduce, **94.5%** is the shared non-MoE floor, and
GEMM+MoE together are **2.2%**. Each all-reduce is ~374 us for a ~14 KB payload, i.e. PCIe sync latency,
not transfer. So the standing hypothesis is that the gap is the **interconnect**, and that essentially
none of it is our kernels.

That hypothesis has never been tested directly, because every quadbit serving measurement to date is on
SM120. This run tests it by changing only the silicon.

## Method

Same checkpoint (`nvidia/GLM-5.2-NVFP4`), same harness (`_graph_gate_body`), same mito80 passage, same
two-run TTFT-subtracted decode formula `63/(wall64-wall1)`, same graph mode (captured), same
`cap=128 max_seqs=2 max_len=2048 gpu_mem=0.92`, same `baseline=dense_nvfp4`.

`native=True` disables every quadbit patch (`QB_DENSE=off`) and stops forcing `VLLM_USE_DEEP_GEMM=0`.
Both exist to route around SM120 gaps that SM100 does not have; leaving them on would measure our
workarounds rather than the hardware. The B200 row is therefore **vanilla vLLM** on the vendor's own
tested config (the GLM-5.2 model card specifies B200/B300).

A **same-commit 8x RTX PRO 6000 control** runs alongside it, so the comparison is not confounded by
toolchain drift since 33.810 was frozen.

## Pre-registered predictions

SM120 reference: 33.810 tok/s = **29.58 ms/step**, of which ~90% (~26.6 ms) is collective. NVLink 5
carries a tiny-payload all-reduce in roughly 20-40 us instead of 374 us, which would leave ~1.4-2.8 ms
of collective plus the ~3.0 ms of everything else, i.e. a **4.4-5.8 ms step = roughly 170-230 tok/s**,
holding the non-collective part constant (conservative: B200 also has 4.5x the memory bandwidth and
selects a different, faster NVFP4 MoE backend).

| outcome | reading |
|---|---|
| **>= 150 tok/s** (>= 4.4x) | **H1 confirmed.** The collective share largely evaporates; the gap was the interconnect, not our kernels. |
| **60-150 tok/s** | **Partial.** Hardware helps materially but another floor takes over, as in C7 where the floor moved rather than vanished. |
| **< 60 tok/s** (< 1.8x) | **H1 refuted.** The gap was not primarily interconnect and the limiter is our stack. |

## What this run does NOT claim

It is **not** a like-for-like answer to 281 tok/s. Baseten's published stack adds **NVFP4 + MTP
speculative decoding + prefill/decode disaggregation (2x on their own measurement) + KV-aware routing**
on a custom engine. Ours is vanilla vLLM, aggregated, no MTP. So a B200 row landing below 281 is
expected and is not a loss; the decomposition is:

- **33.810 -> B200 vanilla** = the hardware/interconnect effect (what this run measures).
- **B200 vanilla -> 281** = the serving-stack optimizations we have not applied.

## Confounds to state when citing

1. **GPU count differs (4 vs 8).** 433 GiB does not fit on 8x95 GiB minus overhead but fits on
   4x191.5 GiB, and 4xB200 is hourly-cost-matched to 8xRTX PRO 6000. Fewer ranks makes an all-reduce
   cheaper independently of the link, so rank count and link are not fully separated here. Note that
   fewer ranks is not automatically better: C5 tested TP=2 on SM120 and it **lost** (40.565 vs 48.248)
   because per-GPU weight bytes doubled.
2. **MoE backend differs.** SM120 selects `FLASHINFER_CUTLASS`; B200 selects `FLASHINFER_TRTLLM`. That
   is a hardware-driven backend upgrade, part of the platform difference rather than a bug.
3. **Metric.** Ours is decode-only at B=1 with prefill subtracted. Artificial Analysis output speed is
   per-request over a long generation, where prefill amortizes to near zero, so these are close but not
   identical.

## Follow-up already wired

`glm_b200 --spec N` enables MTP speculative decoding. C9 killed MTP on SM120 (**-17.8%**, captured
spec=2 47.792 vs no-spec 58.126) because each draft step pays the PCIe all-reduce that dominates the
step there. Published B200 stacks use MTP as a win, which is consistent with that KILL being a property
of the interconnect rather than of MTP. If H1 confirms, the C9 rematch on NVLink is the natural next row.
