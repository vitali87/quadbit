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

Verdict logic:
- SPEC eager clears the forward → **custom-AR is the culprit**; measure eager spec-vs-control decode
  (the amortization signal), then fix/omit custom-AR for a captured comparison against the ring-AR
  dense baseline (48.248) rather than the custom-AR SOTA (58.126).
- SPEC eager stalls too → the **MTP draft forward itself** hangs on our SM120 DSA/MoE/MLA plugin
  path; trace the specific op/collective (candidate: an EP all-to-all or DSA indexer call on the
  draft forward not covered by the plugin's class-level patches).

(To be finalized when the bisect returns.)
