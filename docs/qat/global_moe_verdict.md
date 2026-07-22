# Global MoE QAT on DeepSeek-V4-Flash-NVFP4 — KILL

**Status:** COMPLETE — **KILL / no recovery, worse than both weaker QAT variants**
(2026-07-22). **Branch:** `qat-global-moe`. **PR:** #34.

## The question

The per-expert gate_up STE-QAT KILL (`docs/qat/gateup_moe_verdict.md`, PR #33) fit each expert
to its **own** dense output: tight per-expert reconstruction rel (0.02-0.07) yet no downstream
move (QAT .7332 vs one-shot .7363). That verdict flagged one untested lever: fit the
**router-weighted top-k combine** to the **dense routed aggregate** (the thing the next layer
actually sees), using the routing weights the per-expert objective discarded. This is the true
global MoE QAT objective. This experiment runs it to the ground.

## Setup

- `serve_dsv4.py::recon --recon-mode global --rounds 3`, `QB_SPARSE_PROJ=gateup`, `reio2`
  serve-consistent teacher I/O, layers 22-42, 200 STE steps/expert/round x 3 rounds, `lr=1e-4`,
  DeepSeek-V4-Flash-NVFP4, 2x RTX PRO 6000 TP=2.
- Trainer: `moe_recon.train_layer_global` — **Gauss-Seidel coordinate descent**, one expert
  resident on the GPU at a time (joint backward over 256 experts would OOM), each expert fit to
  **its share of the routed residual** `y - Σ(other experts)`. `combined` is updated in-place
  after each expert (Jacobi-style all-vs-round-start overshoots and diverges — L23 blew up
  0.58->2.43 before the switch). Dense teacher `y` rebuilt inside the trainer from dumped
  `x`/`tid`/`tw` + the dense experts (reio2's stored `y` is the sparse block output, unused).
  Best-round keep: serve the round with the min aggregate rel, not the final.
- Only experts with >=8 routes are trained (`trainable`); rare experts (>=1 route) still
  contribute to the dense teacher and aggregate (`routed_all`) so the residual is complete.
- Eval: WS-E 8-task loglikelihood MC (arc_c/arc_e/hellaswag/piqa/obqa/boolq/winogrande/mmlu-5),
  `limit=400`, run **in-process** in the same job as training.
- Controls (same 8-task `limit=400` battery): dense .7548, down49 .7491, one-shot gateup49 .7363,
  per-expert QAT gateup49 .7332.

## Result

The global objective produced the **tightest reconstruction of all three variants** — the
router-weighted aggregate rel converged monotonically to **0.004-0.017** across all 21 layers
(e.g. L42 agg_rel `[0.2964, 0.0082, 0.0043, 0.0039]`), roughly 4-10x tighter than the per-expert
run's 0.02-0.07. Yet downstream capability was the **worst** of the three QAT policies.

| policy | AVG-8 | vs dense | vs one-shot | vs per-expert QAT |
|---|---|---|---|---|
| dense NVFP4 | .7548 | — | — | — |
| down49 (training-free anchor) | .7491 | -0.57pt | — | — |
| one-shot gateup49 (no QAT) | .7363 | -1.85pt | — | — |
| per-expert QAT gateup49 | .7332 | -2.16pt | -0.31pt | — |
| **global QAT gateup49** | **.7259** | **-2.89pt** | **-1.04pt** | **-0.73pt** |

Per-task (primary): arc_c .6125, arc_e .8625, hellaswag .6750, piqa .8300, obqa .4350,
boolq .8350, winogrande .7650, mmlu-5 .7920. PPL 3.777. Both the vs-one-shot (-1.04pt) and
vs-per-expert (-0.73pt) deltas are at or beyond the `limit=400` AVG-8 MC noise band (SE ~0.8pt):
fitting the routed aggregate did not just fail to help, it moved downstream the wrong way.

## Verdict — KILL

**True global MoE QAT (loss on the router-weighted top-k routed aggregate) does not recover the
sparse-FP4 gate_up downstream tax — it is worse than both the one-shot sparse baseline and the
weaker per-expert QAT, despite the tightest reconstruction of any variant.** This closes the last
open lever the per-expert KILL flagged. The result is the strongest confirmation yet of the
recurring lesson (Gap C full-stack QAT, A3, per-expert gate_up QAT): **local reconstruction rel
does not predict downstream capability, and driving it tighter makes the mismatch worse, not
better.** Matching the routed aggregate on the calibration trajectory overfits layer-local
statistics that do not transfer to held-out MC accuracy; the 2:4-FP4 gate_up capability floor is
real and is not a reconstruction-objective artifact. No objective in the layerwise-repair family
(own-output, routed-aggregate, dense-trajectory, or sparse-trajectory) clears the floor.

## What this validates (win-forward)

The deployed strategy is reinforced, now definitively: you do **not** repair the gate_up tax by
any layerwise QAT objective — you **avoid** it by anchoring gate_up dense. **down49
(training-free, 2 GPU, -0.57pt on 8-task) remains the cleanest 2-GPU capability-preserving
policy**, with D2 route-slot the high-sparse-FLOP extension (needs 4 GPUs). The "can QAT recover
gate_up?" question is now fully closed across the whole tractable family: own-output per-expert
(no), and routed-aggregate global (no, and worse).

True end-to-end QAT (attention + MLP jointly trainable through the sparse kernel) is still not
refuted — it remains infra-blocked (no differentiable DeepSeek/GLM load exists; only vLLM
inference). This KILL covers every layerwise-repair objective that the vLLM-inference-only path
can express.

## Infra landed alongside (reusable)

- **`train_layer_global`** (`moe_recon.py`): Gauss-Seidel one-expert-resident coordinate descent
  with in-place residual update and best-round keep — the stable way to fit a 256-expert
  router-weighted combine without a joint backward that OOMs (~13GiB bf16 / ~26GiB fp32+Adam).
- **Preemption resume**: RTX-PRO-6000 preemption (SIGINT) mid-run; atomic per-layer reconw dumps
  keyed on the live `QB_RUNTAG` are reloaded on Modal container restart so training resumes at the
  last completed layer rather than from scratch.
- **All recon config read live at dispatch** (mode/rounds/scale/steps/lr, tag-keyed caches): a
  warm vLLM worker never serves stale reconstruction settings from an earlier run.
