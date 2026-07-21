# C9 T2: MTP spec-decode integration — first run stalled, bisect in progress

Branch `feat/mtp-spec-decode`. T1 (`t1_recon.md`) established MTP spec-decode is available and the
NVFP4 checkpoint ships the `mtp.0` head. T2 wires `speculative_config` into `_graph_gate_body`
(`--spec N --spec-method mtp`) and measures decode tok/s vs the C4 dense SOTA (58.126 tok/s captured).

## Run 1 (captured + custom-AR + spec=1): STALLED in the profiling forward

Command: `::graph_gate4 --cap 128 --max-seqs 2 --baseline dense_nvfp4 --force-custom-ar --spec 1`
(the exact C4 SOTA config + `--spec 1`).

**Everything up to the profiling forward worked:**
- vLLM accepted `SpeculativeConfig(method='mtp', num_spec_tokens=1)` for `DeepseekV4ForCausalLM`.
- Spec-decode path active (`min_p/logit_bias won't work with speculative decoding` warnings).
- `Loading drafter model...` → `Detected MTP model. Sharing target embedding / lm_head /
  topk_indices_buffer with the draft model.` — the `mtp.0` head wired as the draft correctly.
- All SM120 unblocks selected WITH spec on: FLASHINFER_CUTLASS MoE, DSA Lightning Indexer,
  fp8_ds_mla KV, custom one-shot AR. `Breakable CUDA graph enabled`.
- Model+draft load 41.79 GiB / 130.7s. TileLang JIT-compiled the MTP hyper-connection kernels
  (`hc_prenorm_gemm`, `mhc_pre_big_fuse_with_norm`, `mhc_post`).

**Then it hung:** after the last kernel compile (10:41:15), the worker went silent and EngineCore
printed the 60s `shm_broadcast` "no available block / processes hanging or doing time-consuming work"
heartbeat **6 times (~5 min)** with NO `Capturing CUDA graphs` line, NO KV-profiling result, NO
error. Without spec-decode this same stage (C2/C4 dense baseline) completes in ~20-30s. So the stall
is in the **spec-decode profiling/warmup forward**, before capture, and MTP is the differentiator
(the non-MTP DeepSeek-V4 path runs fine on this exact stack, vLLM 0.24). Killed at 6 heartbeats.

## Corroborating: vLLM issue #40926 (OPEN)

`[Bug] V1 engine + MTP + GLM-5.1 (DSA + MoE + MLA) — workers hang, EngineCore shm_broadcast stuck.`
Same DSA+MoE+MLA family (their `GlmMoeDsaForCausalLM`, our sibling `DeepseekV4ForCausalLM`), same
`shm_broadcast` stall symptom. Two load-bearing hints:
1. Their working baseline **requires `--disable-custom-all-reduce`** (torch-2.11 custom-AR failure).
   We *added* custom-AR (`--force-custom-ar`, the C4 win) — the prime suspect.
2. The DSA+MoE+MLA+MTP combo is a known-fragile class (a related report hits a
   `context_lens.is_contiguous()` assertion on the GLM MTP+DSA path). Version/model-specific, but the
   family instability is real.

## Bisect (running)

Two eager runs, no capture (removes capture as a variable), no custom-AR (tests the top suspect):
- **SPEC** `--eager --spec 1` (no `--force-custom-ar`): does the MTP forward clear the profiling
  stage at all without custom-AR?
- **CONTROL** `--eager` (no spec, no custom-AR): the clean eager-dense baseline to compare the
  eager-spec decode tok/s against (and confirms eager-nospec runs).

## Bisect result + RECALIBRATION (do not over-conclude)

- **SPEC (eager + spec, no custom-AR): accumulated heartbeats identically to Run 1** (hb 1→6),
  never reached KV/gen. Killed at hb=6. So capture and custom-AR are NOT the differentiator.
- **BUT CONTROL (eager + NO spec, no custom-AR) accumulates the SAME heartbeat pattern** (hb→6+),
  **with `[qb_sm120] o_proj bf16 path ACTIVE` still printing on all workers** = the forward IS
  executing, not frozen. A true collective deadlock would go silent. So the heartbeats are likely
  the **slow eager DSA profiling forward** (bf16 indexer fallback + 2048-token profiling batch +
  first-run TileLang JIT), NOT necessarily a hang.
- **Consequence:** killing Run 1 / SPEC at 6 heartbeats may have been premature (same
  conclude-before-measuring error as the earlier occupancy episode). Captured+nospec (C2/C4) works
  fast because its profiling/capture path differs from the slow eager path.

## Arbiter: QB_FAULT_DUMP run (running)

`::graph_gate4 ... --eager --spec 1 --fault-dump 120` arms `faulthandler.dump_traceback_later(120s,
repeat=True)` in every worker (confirmed `QB_FAULT_DUMP on` in pid=2). The periodic all-thread stack
dump is definitive:
- stacks show a **collective wait** (nccl all-reduce / all-to-all `dequeue`/`wait`) → real deadlock;
  trace which op and why it diverges across ranks under spec-decode.
- stacks show **compute** (a matmul / TileLang kernel / python forward) → merely slow; the fix is
  patience + measuring captured (deployed) config, not eager, and re-timing spec-vs-nospec captured.

Also letting CONTROL run uninterrupted: if eager-nospec breaks through to gen, that alone proves the
heartbeats are slowness, not a hang. Verdict pending the stacks — NOT concluded.

## RESOLVED: it was slow eager warmup, NOT a hang. Runs were killed prematurely.

**CONTROL (eager, no-spec) broke through and PASSED**: `Available KV cache memory 43.69 GiB` at hb=8,
then coherent generation (Paris / fibonacci / cyan-magenta-yellow / H2O EN+ZH), **PPL 4.1222, decode
7.165 tok/s**. So the 6-8 repeating `shm_broadcast` heartbeats are the **slow eager DSA profiling
forward** (~8 min: bf16 indexer fallback + 2048-token profiling batch + first-run TileLang JIT), not a
deadlock. The forward was progressing the whole time (`o_proj bf16 path ACTIVE` printing).

**Correction:** Run 1 (captured+custom-AR+spec) and SPEC (eager+spec) were killed at hb=6 BEFORE they
could finish — the spec path does MORE work (draft model ~doubles the forward), so it needs even
longer than CONTROL's 8 heartbeats. This is the same conclude-before-measuring error as the earlier
occupancy episode; the fix is patience (let spec runs go 15-20+ min). vLLM #40926 is a real but
DIFFERENT failure (sustained-traffic runtime deadlock on v0.20.0), not our init-time slowness.

Eager decode is ~7× slower than captured (7.165 vs the 48.248 captured dense baseline), as expected
for the no-graph DSA path — so eager is only a diagnostic; the deployment number must come from the
CAPTURED config.

## Re-runs (patient this time)

- **FAULTDUMP** `--eager --spec 1 --fault-dump 120`: the eager-spec point + hang tripwire; expected to
  break through slower than CONTROL. Compare decode tok/s to the 7.165 eager-nospec baseline
  (the eager amortization signal).
- **CAPTURED-SPEC** `--force-custom-ar --spec 1 --fault-dump 180`: Run 1's real config, let it run to
  completion. This is the deployment measurement vs the 58.126 captured SOTA. If the faulthandler ever
  prints a collective-wait stack, it's a real hang; otherwise it's just slow capture and will finish.

## Eager result: spec=1 is slightly SLOWER eager, as expected (numbers)

| config (eager, no custom-AR) | decode tok/s | PPL |
|---|---|---|
| no-spec (CONTROL) | 7.165 | 4.1222 |
| spec=1 (mtp) | 6.829 | 4.1222 |

Spec-decode is **lossless** (identical PPL) but **net-negative in eager** (-4.7%). Correct and
expected: in EAGER, compute dominates (no graph), so the extra MTP draft forward — a full MoE+DSA
block — costs more than the collective-floor amortization saves at k=1. **The amortization thesis is
about the CAPTURED collective floor** (C4/C5: 90.8% of captured decode is the per-layer TP all-reduce).
Captured, the target forward verifies k+1 tokens in ONE pass → amortizes its 43 per-layer all-reduces
by (k+1)×, while the 1-layer MTP draft adds ~1. So the win can only show CAPTURED, and DeepSeek trains
MTP for k=2. Hence the two captured runs: `--spec 1` and `--spec 2`. The eager run also disproves the
hang (it completed) — patience was the fix, exactly as CONTROL predicted.

## CAPTURED result (the deployment measurement): spec-decode LOSES — see verdict.md

Both captured runs let go to completion (patient this time; faulthandler armed at 180s as tripwire).

| config | mode | decode tok/s | vs 58.126 SOTA |
|---|---|---|---|
| no-spec | captured | 58.126 | SOTA (C4 custom-AR) |
| spec=1 (mtp) | captured | **hang** (66 min @ 0.00 tok/s) | — |
| spec=2 (mtp) | captured | **47.792** (PPL 4.2514) | **-17.8%** |

- **spec=2 PASSED capture cleanly** and measured 47.792 tok/s = 17.8% *slower* than no-spec. The
  earlier `Timeout (0:03:00)!` dumps in its log were the 180s faulthandler firing on schedule during
  the slow profiling forward (stacks show TileLang compile + `shm_broadcast` wait = compute-slow), NOT
  a deadlock — the run finished normally with full graph capture.
- **spec=1 genuinely deadlocked:** graph capture succeeded, generation entered (`Processed prompts 0/4,
  output: 0.00 toks/s`), then wedged in `llm.generate -> step -> get_output` for 22 faulthandler
  cycles (~66 min) at zero progress. Killed via `modal app stop`. This is the real vLLM #40926 family
  runtime deadlock (MTP+DSA+MoE+MLA captured decode), distinct from the T2 init-time slow-warmup.

**Verdict: C9 = KILL.** The MTP draft head is a full MLA + 256-expert MoE + DSA block, so its per-step
all-reduce + expert all-to-all cost, times k, plus imperfect acceptance, outweighs the (k+1)x
amortization of the target forward's 43 all-reduces at batch 1-2. T4 not run (no captured win to
amplify). Deployed decode SOTA stays C4 custom-AR 58.126 tok/s. Full write-up in `verdict.md`.
