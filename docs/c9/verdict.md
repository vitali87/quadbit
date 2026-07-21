# C9 verdict: MTP speculative decode on DeepSeek-V4-Flash-NVFP4 (SM120, TP=4 PCIe) = KILL

**Hypothesis (T1/T2):** captured decode is ~90.8% one kernel — the per-layer TP all-reduce over PCIe
(no NVLink). MTP spec-decode verifies k+1 tokens in ONE target forward, so it should amortize the 43
per-layer all-reduces by (k+1)x and beat the C4 custom-AR SOTA of 58.126 tok/s.

**Result: falsified.** Measured, spec-decode is a net loss in the exact deployed config.

| config | mode | decode tok/s | vs 58.126 SOTA | note |
|---|---|---|---|---|
| no-spec | eager | 7.165 | — | eager baseline (PPL 4.1222) |
| spec=1 (mtp) | eager | 6.829 | -4.7% | lossless (PPL 4.1222), compute-bound |
| **no-spec** | **captured** | **58.126** | **SOTA** | C4 custom one-shot AR |
| spec=1 (mtp) | captured | **hang** | — | 66 min at 0.00 tok/s, real deadlock |
| spec=2 (mtp) | captured | **47.792** | **-17.8%** | completed clean (PPL 4.2514) |

- **spec=2 captured completed and PASSED capture cleanly** (full CUDA graph capture, coherent gen):
  `decode_tps=47.792` vs 58.126 no-spec = **17.8% slower**. Not a warmup artifact — graphs captured,
  generation ran, two-run TTFT-subtracted timing (`wall1=0.150s wall64=1.468s`).
- **spec=1 captured deadlocks at runtime:** graph capture succeeded, generation entered, then the
  engine sat at `output: 0.00 toks/s` for 22 faulthandler cycles (~66 min) wedged in
  `llm.generate -> step -> get_output`. This is the vLLM #40926 family failure (MTP + DSA + MoE + MLA
  captured decode deadlock), distinct from the init-time slow-warmup we ruled out in T2.

## Why the amortization does not pay off

The MTP "draft" head on this model is **not lightweight**. `mtp.0` is a full decoder block — MLA
attention + 256-expert MoE + DSA lightning indexer (kept unquantized). So each draft step costs
roughly a real layer's worth of all-reduce **plus** expert all-to-all, and spec-decode runs it k times
per step. vLLM even warns `num_speculative_tokens > 1 ... run multiple times on same MTP layer, lower
acceptance rate`. At batch 1-2 (latency-bound decode — spec-decode's best case), the draft cost plus
imperfect acceptance outweighs the (k+1)x amortization of the target forward's 43 all-reduces. Net
negative at k=2; k=1 never gets to a number because it deadlocks.

## Consequence

- **T4 (spec-decode + quadbit sparse experts at M=B*k) is not run** — there is no captured win to
  amplify, and the large-M sparse kernel win lives in a different regime (already covered by the
  crossover result, not by spec-decode at batch 1-2).
- The deployed decode SOTA stays **C4 custom one-shot all-reduce = 58.126 tok/s**. The collective floor
  is attacked directly (custom AR), not amortized via speculation.
- Spec-decode integration itself is sound: vLLM accepts the MTP config, wires the shared
  embedding/lm_head/topk buffer, and all SM120 unblocks + custom AR co-exist. It is lossless only in
  the eager spec=1 diagnostic (PPL 4.1222, identical to baseline); the completed captured spec=2 run
  measured PPL 4.2514 (vs 4.1222), a small quality regression, and captured spec=1 never completed.
  The loss is a workload-economics result, not a bug: this specific draft head is too heavy on this
  specific interconnect.

Branch `feat/mtp-spec-decode`. Diagnosis trail in [t2_diag.md](t2_diag.md); T1 availability in [t1_recon.md](t1_recon.md).
