# B200 verdict: interconnect worth **3.27x**, and MTP (a C9 KILL on SM120) adds **+93.5%** on top — **6.32x** total

Branch `b200-glm-headtohead`, PR #38. Pre-registration: [design.md](design.md) (written before results).
Logs `docs/audit/logs/b200_glm_*.log`.

## Result

Same checkpoint (`nvidia/GLM-5.2-NVFP4`), same harness (`_graph_gate_body`), same PPL passage (both rows
score `ntok=96` tokens, confirming identical tokenized input), same two-run TTFT-subtracted decode
formula, same captured graph mode, same `cap=128 max_seqs=2 max_len=2048`, same commit.

This is a **platform comparison, not a silicon-only A/B.** Alongside the hardware, the B200 row also
changes rank count (4 vs 8), plugin usage (`QB_DENSE=off` vs the SM120 patches), the DeepGEMM setting,
and the NVFP4 MoE backend vLLM selects (`FLASHINFER_TRTLLM` vs `FLASHINFER_CUTLASS`). Each of those is
the configuration that platform requires rather than a free variable, but they are **not** separated
from the interconnect by this experiment. What *is* held exactly fixed is quadbit itself: no kernel,
quantization, or policy changed, and on the B200 row quadbit is disabled outright.

| row | GPUs | link | stack | decode tok/s | ms/step | PPL | graph | log |
|---|---|---|---|---:|---:|---:|---|---|
| control | 8x RTX PRO 6000 (sm_120) | PCIe, no NVLink | vLLM + qb plugin | **34.305** | 29.15 | 3.9168 | captured PASS | `b200_glm_rtx_control.log` |
| **B200** | 4x B200 (sm_100) | NVLink, full P2P | **vanilla vLLM** (`QB_DENSE=off`) | **112.079** | 8.92 | 3.7352 | captured PASS | `b200_glm_headtohead.log` |

**3.27x, with zero change to quadbit.** The control reproduces the frozen 33.810
([c2/glm_sota.md](../c2/glm_sota.md)) to within 1.5%, so the comparison is not toolchain drift.

**On quality:** the two PPLs are close (3.7352 vs 3.9168) but they are **not** "matched", and 96 scored
tokens is far too short to resolve a 0.18 difference. The defensible claim is only that **neither row
collapsed** and both generate coherently; the small delta is unattributed and is as likely to come from
the different MoE backend and reduction order as from anything else. Do not cite this as a quality
result in either direction.

## Verdict against the pre-registered bands

Pre-registered: `>=150` confirms, `60-150` partial, `<60` refutes. **112.079 is PARTIAL.**

The prediction of 170-230 tok/s assumed NVLink would carry a tiny-payload all-reduce in 20-40 us against
PCIe's 374 us. Working backward from C4's attribution (90.8% of the SM120 step is the collective, i.e.
26.47 ms of 29.15, leaving 2.68 ms of everything else), and holding the non-collective work constant:

- B200 collective = 8.92 - 2.68 = **6.24 ms** across 78 layers = **~80 us per all-reduce**.
- That is **4.7x** cheaper than PCIe's 374 us, not the 10-20x assumed.
- The collective is therefore **still ~70% of the B200 step**.

Treat the 80 us as **inferred, not measured**: it assumes the non-collective work costs the same on both
machines, which is not established (B200 has 4.5x the memory bandwidth and selects a different NVFP4 MoE
backend, but tp=4 also doubles per-GPU weight bytes versus tp=8). A B200 `floor_profile` would settle it;
that harness is currently DeepSeek-hardcoded and would need a GLM + SM100 path.

## What this does and does not settle

**Settles: the SM120 decode numbers were never a kernel deficit.** No kernel, quantization, or policy
changed between these two rows and the number tripled. This is direct confirmation of C4's profiler
finding that GEMM+MoE are 2.2% of the decode step. Anyone reading quadbit's 33.810 as evidence about our
kernels was reading the interconnect.

**Does not settle: "get better hardware" is not the whole answer either.** The floor **moved but did not
vanish** — the same pattern as C7, where removing the attention all-reduce simply promoted the EP
collective. Decode remains collective-dominated on NVLink. The lever that actually pays is reducing the
collective *count* or overlapping it, which is precisely what the published stacks do.

## Against the advertised numbers

Aster advertises GLM-5.2 at **281 tok/s**; Baseten claims the same 280+ on Blackwell (Artificial
Analysis, 2026-06-22). We are at **112.079**, i.e. **2.51x behind** — on vanilla vLLM against a tuned
production stack. Baseten disclose the recipe, and every item is a named technique rather than hardware
or kernel advantage:

- **NVFP4 weights** — we already use this, same checkpoint format.
- **MTP speculative decoding** — the only one of these we have now measured ourselves (below).
- **Prefill/decode disaggregation** — 2x *on their stack, on their benchmark*.
- **KV-aware routing** — prefill-side cache hit rate, largely orthogonal to single-stream decode.

**These multipliers are theirs, not ours, and must not be composed onto our number.** In particular the
PD-disaggregation 2x is measured on a concurrent serving benchmark, where it works by keeping prefill
from interrupting decode; our measurement is single-stream at batch 1-2 with no competing prefill, so
its expected contribution here is closer to nothing than to 2x. Any arithmetic of the form
"112.079 x 2 = 224, therefore we would beat them" is invalid and is exactly the soft-target reasoning
this repo bans. The only defensible statement is directional: **the residual gap is serving-stack work
we have not done, and those techniques are named and public**, with their magnitudes unverified here.

A further caveat on the target itself: 281 tok/s is a vendor marketing figure whose measurement
conditions (batch size, concurrency, input/output lengths, whether output speed is per-request or
aggregate) we do not control and have not reproduced. It is a **directional** reference point, not a
calibrated baseline. Everything in this repo's own tables is same-harness; this one number is not.

## C9 rematch: the MTP KILL **REVERSES** on NVLink

C9 killed MTP speculative decoding on SM120 at **-17.8%** (captured spec=2 47.792 vs no-spec 58.126)
because each draft step pays the per-layer all-reduce that dominates the step there, and spec=1
**deadlocked** (66 min at 0 tok/s). That per-collective cost just fell 4.7x. Rerun on B200:

| row | GPUs | spec | decode tok/s | vs no-spec | PPL | log |
|---|---|---|---:|---:|---:|---|
| B200 no-spec | 4x B200 | 0 | 112.079 | — | 3.7352 | `b200_glm_headtohead.log` |
| B200 MTP | 4x B200 | 1 | 181.278 | +61.7% | 3.7352 | `b200_glm_mtp_spec1.log` |
| **B200 MTP** | 4x B200 | **2** | **216.828** | **+93.5%** | 3.7352 | `b200_glm_mtp_spec2.log` |

- **At C9's exact speculation count (spec=2) the sign flips: SM120 -17.8% -> B200 +93.5%.** This is the
  clean comparison, holding k fixed; only the machine differs.
- **Lossless: PPL is identical to four decimals across all three B200 rows (3.7352)**, the expected
  signature of speculative decoding under greedy verification. All captured, all PASS.
- **Cumulative: 34.305 -> 216.828 = 6.32x**, from a platform change plus one config flag, with no
  kernel, quantization, or policy change anywhere in quadbit.
- **The C9 KILL was a property of the interconnect, not of MTP.** Drafting only loses when each draft
  step pays a 374 us collective; at ~80 us it pays for itself, twice over.
- **Deeper speculation keeps paying on NVLink** (spec=2 is 1.196x spec=1), where the SM120 run warned
  that k>1 lowers acceptance on the recursive MTP layer. The acceptance penalty is real but on NVLink it
  is outweighed by the amortization; on PCIe it was not.

**Caveat: this does not isolate the interconnect as the cause.** C9 was **DeepSeek-V4-Flash on SM120**;
these rows are **GLM-5.2 on B200**. Model, hardware and platform configuration all differ, so the sign
flip is *consistent with* the interconnect explanation and *predicted by* it, but it does not prove that
the interconnect rather than the model (different draft head, different acceptance rate, different
expert count) caused the reversal. The two models' MTP heads are not equivalent: C9 specifically noted
DeepSeek's draft head is a full MoE+MLA+DSA block.

The clean isolation test is not run here and would be **GLM-5.2 with MTP on SM120** (same model, same
spec count, PCIe). That row is cheap and would settle it. Until then, treat "the C9 KILL was an
interconnect property" as the leading hypothesis with strong supporting evidence, not as established.
The measured claim that stands unconditionally is narrower and still useful: **on B200, MTP is a large
lossless win for GLM-5.2 at batch 1-2, where the published stacks also use it.**

## Standing after the rematch

Ratios in the last column are against a **vendor marketing figure measured under conditions we do not
control** (see the caveat above); treat them as directional, not as a same-harness comparison.

| stage | tok/s | vs the PCIe row | ratio to the advertised 281 |
|---|---:|---:|---:|
| 8x RTX PRO 6000, PCIe (where the campaign lived) | 34.305 | 1.00x | 8.19x |
| 4x B200, NVLink, vanilla vLLM | 112.079 | 3.27x | 2.51x |
| 4x B200 + MTP spec=1 | 181.278 | 5.28x | 1.55x |
| **4x B200 + MTP spec=2** | **216.828** | **6.32x** | **1.30x** |

The remaining 1.30x is prefill/decode disaggregation plus KV-aware routing plus a custom engine. Note
that Baseten's 2x for PD disaggregation should **not** be multiplied onto this number: PD disaggregation
removes prefill interference from decode, which mostly buys aggregate throughput and TTFT under
concurrent load, whereas this measurement is single-stream at batch 1-2 with no competing prefills.

## The broader lesson for the campaign

Several campaign verdicts (C7 DP-attention, C8 pipeline staging, C9 MTP) concluded that **no software
lever moves SM120 decode**. C9 has now flipped on hardware alone, at the same speculation count, from
-17.8% to +93.5%. Every one of those KILLs was measured under a 374 us per-layer collective that
dominated the step, and each should be read as **conditional on that floor** rather than as a general
property of the technique. C7 and C8 are the obvious re-test candidates.
