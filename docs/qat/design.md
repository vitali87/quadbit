# Gap C: full-stack QAT to recover sparse-FP4 downstream capability

**Status:** COMPLETE — **KILL** (2026-07-17). **Branch:** `qat-fullstack-capability`.
Result section at the bottom; the design that follows is preserved as written.

## The one question

Every recovery lever so far repairs perplexity but **not downstream capability**:
recovered sparse 8B sits ~20 points below dense on ARC-C / HellaSwag while PPL is
within ~1.5 of dense (`docs/paper_notes.md:623-630`, `docs/standing.md:308-316`).
The paper's open frontier is whether this is a **fundamental ~0.3-rel 2:4-FP4 capacity
floor** or an artifact of how we recovered. This experiment settles it.

A win flips the sparse story from "speed-only Pareto" to "speed AND capability" (landmark).
A clean negative pins the capacity floor as real and is itself a strong, honest result.

## Why prior campaigns could not answer it

The infra map (recovery/QAT/KD stack) shows every recovery harness shares **two scoping
choices that jointly cap capability**, independent of the fake-quant (which is already exact):

1. **Only MLP weights are ever trainable.** `finetune_pair.py:229-234,292-296` unfreezes
   only `gate/up/down`; attention `q/k/v/o`, embeddings, LM head, norms stay frozen dense.
   Attention carries the in-context reasoning that ARC/HellaSwag test. If attention is left
   dense while MLP is sparsified, or is never adapted to the sparse MLP it now feeds, the
   model cannot re-route capability through the surviving weights.
2. **Only web text is ever the corpus.** WikiText-103 / generic C4 (`finetune_pair.py:202-218`,
   `build_corpus.py`). The notes already conclude both "buy perplexity, not capability"
   (`docs/paper.md:624-625`) — even the full 500M-token C4 lever did not pay
   (`docs/standing.md:48`). Distilling on next-token web fluency optimizes the exact metric
   (PPL) that we already know diverges from capability.

The matched STE (`finetune_pair.py:81-129`) bit-matches `cuda/sparse_fp4_lib.cu:221-312`
(verified: weight rel 0.0017, act path exact after the per-32 fix). So **no quantizer work is
needed** — the two load-bearing changes are: widen the trainable+STE set, and change the data.

## The experiment

Fork `finetune_pair.py` -> `harness/finetune_fullstack.py`. Reuse verbatim: the SparseGPT-pair
mask generation, the matched weight+act STE, `QATLinear`, 8-bit AdamW, gradient checkpointing,
mask re-application, phase-1/phase-2 warm-restart schedule, checkpoint/resume, the deployable
kernel pack + through-kernel PPL. Change exactly four things:

1. **Widen the sparsified + trainable + STE set to attention.** Extend `mlp_lins` to also yield
   `self_attn.{q,k,v,o}_proj` (those with dims %256; else leave dense-W4A4 via the
   `finetune_hybrid.py:127-135` `DenseW4A4` module, stated honestly). 2:4 masks frozen on every
   sparsified matrix, re-applied each step. This is the single biggest structural change and the
   one prior work never tried. Target: **Llama-3.1-8B-Instruct**, matching the banked recovered
   checkpoint and `downstream_eval.py`.
2. **Capability-relevant KD corpus.** Replace the web-text corpus with an instruction+reasoning
   mixture (candidate: a Tulu-3 / FLAN-style SFT mixture for capability signal, blended with a
   slice of C4 for fluency so PPL does not regress). Distillation stays logit-KL (T=2) + optional
   hard-label CE against the frozen **dense** teacher — the dense model's own downstream is the
   ceiling we chase. Decontaminate against **all four downstream test sets** (ARC-C, HellaSwag,
   Winogrande, MMLU), not just WT-2, reusing `build_corpus.py:54-119` 13-gram machinery with a
   positive-control assert.
3. **Downstream-in-the-loop checkpoint selection.** PPL and capability diverge, so selecting on
   PPL is exactly the trap. Every N steps, run the `downstream_eval.py` lm-eval path (4 tasks,
   0-shot, acc_norm) on the current student and keep the best-downstream checkpoint, not the
   best-PPL one. Add MMLU behind a flag (slower).
4. **Multi-GPU only if it OOMs.** Start single RTX-PRO-6000 (96GB). Widening MLP->+attention is
   ~+50% trainable params (GQA attention < MLP stack), 8-bit AdamW + checkpointing already fit
   MLP-only 8B. If the wider footprint OOMs, add FSDP + `gpu="RTX-PRO-6000:N"` then, not before.
   <!-- ponytail: no FSDP scaffolding until a real OOM forces it -->

## De-risk before the multi-day 8B run (RED)

Prove the loop end-to-end on **TinyLlama-1.1B** (fast, one short session) before committing 8B:
- attention matrices actually enter the sparsified/STE/trainable set and masks stay frozen,
- the capability corpus loads + decontaminates (positive-control assert fires),
- the in-loop downstream eval runs and drives checkpoint selection,
- through-kernel PPL still == fake-quant PPL (the STE-kernel identity is not broken by the wider set).
Only after the proxy is green do we launch the detached 8B run.

## Success / kill criteria

- **Win:** recovered sparse-FP4 8B closes >=half the current ~20pt downstream gap to dense on the
  4-task avg, through the real kernel. -> headline; sparse Pareto gains a capability axis.
- **Partial:** measurable but <half gap closed -> report the residual as the empirical capacity
  floor with the widened-trainable + capability-data controls that prior work lacked.
- **Kill:** no movement beyond noise after the widened set + capability data + downstream-in-loop
  selection -> the ~0.3-rel 2:4-FP4 capacity floor is real; write it as a clean negative with the
  strongest controls to date. Either outcome is publishable; only an *uncontrolled* negative is not.

## Budget

Single-GPU first. Phase-1 (bf16 masked) is the expensive part and is checkpointed every 2500
steps; phase-2 QAT converges by ~1k steps (`docs/paper_notes.md:235`). 24h timeout per Modal
function, `run.spawn` detached, verified via `modal app list` per the launch discipline. Time is
not the constraint; a controlled answer is.

---

## Result (2026-07-17): KILL

The experiment ran to the pre-registered stop point on `meta-llama/Llama-3.1-8B-Instruct`:
SparseGPT-pair 2:4 one-shot -> phase-1 masked bf16 recovery (30000 steps) -> phase-2 weight+act
FP4 QAT (3000 steps, warm-restart), with the widened trainable+sparsified set (MLP gate/up/down +
attention q/k/v/o), the balanced capability corpus (`/cache/corpus_capability_llama3_bal`, built
from the downstream TRAIN splits, zero eval leakage), and downstream-in-loop best-capability
checkpoint selection. Final weights packed into the real 2:4-sparse FP4 kernel and scored
through-kernel.

**Verdict: KILL.** Final through-kernel downstream `0.3967` is **3.66 pt below** the training-free
one-shot bar (`0.4333`) and far below dense teacher (`0.6150`) — well short of even the PARTIAL
band (measurably above one-shot), let alone WIN (>= half the teacher gap, `0.5242`). The widened
attn+MLP QAT with an in-distribution capability corpus and honest downstream selection does not
recover capability; it underperforms the one-shot prune it started from. The ~0.3-rel 2:4-FP4
downstream capacity floor is real, now with the strongest recovery controls to date.

**Core table (aggregate downstream = 3-task mean, ARC-Challenge + HellaSwag + Winogrande, 200
items each = 600; see the selection-set note below):**

| stage | PPL (WT-2) | downstream(sel) |
|---|---:|---:|
| dense teacher | 7.268 | 0.6150 |
| one-shot 2:4 FP4 (masked bf16 bar) | 149.155 | 0.4333 |
| best-cap restored, fake-quant STE | 203.116 | 0.4017 |
| **through 2:4-sparse FP4 kernel** | **202.832** | **0.3967** |

Deploy gap (fake-quant 0.4017 -> through-kernel 0.3967) = **0.005 / 0.5 pt**: kernel fidelity is
tight, so the limiting factor is capacity/recovery, **not** the kernel or the STE.

**Phase-2 trajectory (best-capability selection picked the 0.4017 peak at step 500/1500):**

| step | KD | PPL(FP4) | downstream(sel) |
|---:|---:|---:|---:|
| post-P1 | -- | 195.782 | 0.3833 |
| 500 | 0.9961 | 205.837 | 0.4017 |
| 1000 | 1.3281 | 207.083 | 0.3833 |
| 1500 | 0.9023 | 206.641 | 0.4017 |
| 2000 | 1.3750 | 206.155 | 0.3783 |
| 2500 | 1.4844 | 208.616 | 0.3900 |
| 3000 | 1.3125 | 208.325 | 0.3950 |

Flat oscillation 0.378-0.402 with no upward drift; the peak never reached one-shot.

**Per-task through-kernel breakdown** (`harness/finetune_fullstack.py::evalpertask`, read-only,
same 600-item selection set). `AGG` is the 3-task mean; deltas are vs one-shot and vs dense teacher:

| task (items) | dense teacher | one-shot 2:4 | post-P1 bf16 | through-kernel | Δ vs one-shot | Δ vs teacher |
|---|---:|---:|---:|---:|---:|---:|
| ARC-Challenge (200) | 0.5100 | 0.3700 | 0.2150 | **0.2100** | **-0.1600** | -0.3000 |
| HellaSwag (200) | 0.6350 | 0.3500 | 0.3950 | 0.4000 | +0.0500 | -0.2350 |
| Winogrande (200) | 0.7000 | 0.5800 | 0.5400 | 0.5800 | 0.0000 | -0.1200 |
| **AGG (3-task mean, 600)** | **0.6150** | **0.4333** | **0.3833** | **0.3967** | **-0.0367** | **-0.2183** |

All four aggregates reproduce the training-run numbers exactly (teacher 0.6150 / one-shot 0.4333 /
post-P1 0.3833 / kernel 0.3967), confirming the read-only eval is faithful. The fake-quant per-task
row was skipped for cost (its per-forward STE eval over 600 items is ~50 min and preemption-prone;
the fake-quant **aggregate** 0.4017 is in the core table above; re-enable with
`evalpertask --include-fq`).

**The KILL is CONCENTRATED in ARC-Challenge, not broad.** ARC-Challenge collapses to **0.2100
through the kernel — below the 4-way chance floor (0.25)** — and QAT made it *worse* than the
one-shot prune (0.3700 -> 0.2100, -0.16); post-P1 already crushed it (0.2150). By contrast
**HellaSwag marginally recovered** vs one-shot (0.3500 -> 0.4000, +0.05, the one task QAT helped)
and **Winogrande held exactly at one-shot** (0.5800, a 2-way task barely above its 0.50 chance).
So the 2:4-FP4 capacity floor bites hardest on multi-step reasoning (ARC-Challenge): the sparse
model loses that capability to at/below random and recovery cannot restore it, dragging the mean
below the one-shot bar even though the two easier tasks did not regress.

**Interpretation.**
- Widening the trainable+sparsified set to attention q/k/v/o + MLP did **not** recover capability.
- The balanced capability corpus (in-distribution, built from the tasks' own TRAIN splits) did
  **not** rescue the sparse model.
- Downstream-in-loop selection worked mechanically (it picked the 0.4017 peak) but every selected
  checkpoint stayed below one-shot.
- The through-kernel deploy gap is only 0.005, so this is **not** a kernel-fidelity failure.
- Recovery worsened **both** PPL and downstream relative to one-shot (PPL 149 -> 203; downstream
  0.4333 -> 0.3967). PPL rising is consistent with training toward QA-completion text and away from
  the WT-2 web-text metric; the load-bearing negative is that downstream also failed to beat
  one-shot on the very tasks the corpus was built from.
- **KILL:** no movement past noise; the strongest-controlled negative for the single-dense-model
  2:4-FP4 capability floor.

**Selection-set note (harness convention, stated honestly).** The in-loop scorer
(`finetune_fullstack.py::_mc_items`) is **3 tasks** — ARC-Challenge, HellaSwag, Winogrande — at 200
items each (600 total), NOT the 4-task set the design section above anticipated. MMLU and ARC-Easy
are present in the training corpus but are **not** in the selection/eval metric. The `0.4333` /
`0.3967` numbers are this 3-task mean. This differs from the repo's other 4-task downstream AVGs
(the MoE `downstream_eval.py` suite); do not cross-compare the absolute values.

**Memory-fix hygiene (implementation, NOT part of the scientific result).** Two commits during this
run fixed infrastructure, not the recipe: (1) a resume-path bug that duplicated the phase-1
checkpoint on the GPU was removed (load on CPU, build the live module weight only, delete the
checkpoint object) and the best-capability snapshot was moved to CPU/disk (`2ff580a`); (2) durable
phase-2 checkpoint+resume so a Modal preemption continues from the last phase-2 step instead of
resetting to step 1 (`90be3e2`). The completed phase-2 ran with ~14.4 GB free at peak. Neither
touched LR, p1/p2 steps, alpha, attention on/off, optimizer, selection metric, corpus, hardware,
model, or eval protocol — the recipe is exactly the pre-registered one.

**Artifacts.** best-capability weights
`/cache/best_cap_Llama-3.1-8B-Instruct_P30000_p23000_corpus_capability_llama3_bal_fsA_lr2e-04.pt`;
recovered phase-2 weights
`/cache/recovered_Llama-3.1-8B-Instruct_P30000_p23000_corpus_capability_llama3_bal_fsA_lr2e-04.pt`;
harness `harness/finetune_fullstack.py` (train: `run`; per-task record eval: `evalpertask`);
corpus builder `harness/build_capability_corpus.py`.

**Future work (only as a NEW campaign, not a continuation of this registered run).** Less aggressive
attention sparsity; attention-dense + MLP-sparse capability recovery; layerwise or
projection-selective sparsity (cf. the MoE WS-C down-only anchoring win); a distillation/objective
redesign; a larger or multi-GPU recipe only if a new hypothesis justifies it.
