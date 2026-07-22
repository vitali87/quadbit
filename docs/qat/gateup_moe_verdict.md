# gate_up-only STE-QAT layerwise repair on DeepSeek-V4-Flash-NVFP4 — KILL

**Status:** COMPLETE — **KILL / no recovery** (2026-07-22). **Branch:** `qat-gateup-recovery`.

## The question

WS-C proved the sparse-FP4 downstream tax lives in the **gate_up** projection: anchoring
`down` dense and sparsifying only `gate_up` at 49% of MoE layers (`c_gateup49`) missed the
`.718` capability bar (4-task `.7056`, `-3.27pt` from dense), while the mirror `c_down49`
recovered training-free (`.7354`, `-0.29pt`). A3's per-expert layerwise repair failed at
`lr=1e-3` (diverges under the FP4 STE) on the **dense-trajectory** teacher (`reio1`).

**This experiment** attacks two of A3's weaknesses while holding its granularity: fit the surviving
2:4-FP4 gate_up weights of layers 22-42 (`down` held dense-exact) to the **sparse-trajectory,
serve-consistent** teacher I/O (`reio2`, so each layer trains on inputs that reflect upstream
sparsity) at the FP4-STE-stable `lr=1e-4` with best-true-rel checkpoint keep. **Scope note (per
reviewer):** the objective is still **per-expert** — `train_expert` fits each expert to *its own*
dense output (`_dense_expert`, `moe_recon.py:164`); it does NOT backprop through the router-weighted
top-k combine. So this tests whether a *stabilized, serve-consistent per-expert* STE-QAT recovers
the tax; true global MoE QAT (loss on the routed aggregate) is a separate, harder, still-untested
lever.

## Setup

- `serve_dsv4.py::recon`, `QB_SPARSE_PROJ=gateup`, `reio2` teacher I/O (sparse trajectory),
  layers 22-42, 300 STE steps/expert, `lr=1e-4` (the FP4-STE-stable rate; A3's `1e-3` diverges),
  DeepSeek-V4-Flash-NVFP4, 2x RTX PRO 6000 TP=2.
- Eval: WS-E 8-task loglikelihood MC (arc_c/arc_e/hellaswag/piqa/obqa/boolq/winogrande/mmlu-5),
  `limit=400`, run **in-process** in the same job as training.
- Control: **one-shot** `gateup49` (identical config, no QAT) on the **same** 8-task `limit=400`
  battery — this control did not previously exist (only a 4-task `limit=200` `.7056`).

## Result

Per-layer STE fit was tight (gate_up-vs-dense-teacher rel 0.02-0.07 across all 21 layers), yet
downstream did not move. On the identical 8-task `limit=400` battery:

| policy | AVG-8 | vs dense | vs one-shot |
|---|---|---|---|
| dense NVFP4 | .7548 | — | — |
| down49 (training-free anchor) | .7491 | -0.57pt | — |
| **one-shot gateup49 (no QAT)** | **.7363** | -1.85pt | — |
| **QAT gateup49** | **.7332** | -2.16pt | **-0.31pt** |

Per-task (primary), QAT − one-shot: arc_c **-2.0pt**, arc_e +0.3, hellaswag -1.0, piqa 0,
obqa 0, boolq +0.3, winogrande +0.3, mmlu -0.2. **AVG-8 delta = -0.31pt** — neutral-to-slightly-
negative, inside the `limit=400` AVG-8 MC noise band (SE ~0.8pt). No task shows a meaningful QAT
gain; the only move outside noise is arc_c, where QAT is *worse*.

## Verdict — KILL

**Per-expert gate_up STE-QAT layerwise repair does not recover downstream capability.** A tight
per-expert reconstruction rel (0.02-0.07) buys no downstream gain over one-shot gate_up sparse. This
is the same lesson as the Gap C full-stack QAT KILL (`docs/qat/design.md`) and A3: **local
per-expert reconstruction rel does not predict downstream capability; the 2:4-FP4 gate_up capability
floor is real, not a repair artifact.** Fitting each expert to its own fixed dense output — even with
serve-consistent sparse-trajectory inputs and stable optimization — misses both the router-weighted
combine and the cross-layer interactions that set downstream MC accuracy. Whether true global MoE
QAT (loss on the routed top-k aggregate) would clear the floor is untested and remains the harder,
infra-gated lever below.

## What this validates (win-forward)

The result *reinforces* the deployed strategy: you do not repair the gate_up tax, you **avoid** it
by anchoring gate_up dense. **down49 (training-free, 2 GPU, -0.57pt on 8-task) remains the cleanest
2-GPU capability-preserving policy** (DeepSeek D2 edges it on 8-task quality, +0.03pt, but needs
4 GPUs; preferred sparse policy varies by model), and D2 route-slot the high-sparse-FLOP extension. The open
"can QAT recover gate_up?" question (`paper_notes.md:708`, `glm_results.md:104` — gateup49 was
PPL-only) is now closed with a downstream measurement: **no**, at least via layerwise STE-QAT.

True end-to-end QAT (attention + MLP jointly trainable through the sparse kernel) is not refuted
here — it remains infra-blocked (no differentiable DeepSeek/GLM load exists; only vLLM inference).
This KILL is specifically for the tractable layerwise-repair lever.

## Infra fixes landed alongside (reusable regardless of verdict)

- **Atomic reconw dumps** (`qb_sm120_plugin.py::dump_recon_w`, tmp + `os.replace`): the first
  train+eval run hit the 6h Modal timeout during eval warmup; the SIGKILL corrupted the mid-write
  `torch.save`, truncating the zip central directory ("failed finding central directory" on reload)
  and losing 5.8h of training. Atomic writes make a mid-write kill leave the old complete file or
  nothing, never a truncated zip.
- **`recon()` timeout 360->720 MIN**: train (~5.8h) + in-process eval must fit ONE job, so the AVG
  is produced without depending on a corruptible volume reload.
- **`recon_file=<tag>` load-only path**: reload durably-saved repaired weights and eval on any
  battery without retraining (used to re-eval cheaply once a clean checkpoint exists).
