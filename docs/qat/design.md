# Gap C: full-stack QAT to recover sparse-FP4 downstream capability

**Status:** design. **Branch (planned):** `qat-fullstack-capability`.

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
