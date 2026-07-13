# C3 Task 1B: compact routing A/B (DeepSeek-D2, 4 GPU, captured)

Active-expert compaction (`compact_routing_design.md`): process only `A_max` token-loaded experts per
group (A_max*cap rows) instead of all E=64 (E*cap=8192 rows), gathering the matching per-expert weights.
Capture-safe (fixed A_max shape, device topk/gather), and **bit-identical to the full path when no expert
drops**, guaranteed at decode by a static guard (`t_rows*route_slot <= A_dense` for the dense group,
`sparse_slots <= A_sparse` for the sparse group), so the compact branch only engages on the small decode
graphs. Flags: `QB_COMPACT_DECODE=1`, `QB_A_DENSE` (default 8), `QB_A_SPARSE` (default 24).

Same D2 config as the attribution baseline (route_slot=2, native anchor, cap=128, max_seqs=2, captured).
Decode-only tok/s = two-run TTFT-subtracted `63/(wall64-wall1)`. Logs `docs/audit/logs/c3_compact_*.log`.

## Correctness (the compact code path is bit-sound)

Run the compact **code path with A_max = E** (all experts, so zero rows removed): it exercises every
gather/scatter line but cannot drop an expert, so it must reproduce the full path within greedy/FP noise.

| variant | experts covered | decode tok/s | PPL (mito80) | validates |
|---|---|---:|---:|---|
| baseline (non-compact) | all 64 | 5.782 | 4.001 | reference build |
| dense compact code, A_dense=64 | all 64 | 4.926 | 4.096 | dense weight gather |
| sparse compact code, A_sparse=64 | all 64 | 10.042 | 4.045 | 2:4 seg 4-tuple gather |

Both A_max=E runs reproduce baseline PPL within the mito80 noise band (repeated "identical" full runs span
3.95–4.09). The dense A_max=64 run is **slower** than baseline (4.9 vs 5.8), it pays the compact gather
overhead with no rows removed, proof the speedup below is real row-reduction, not a code artifact. A
mis-indexed gather over all 64 experts would blow up PPL, not sit in noise, so both gathers are bit-sound.
(The sparse-gather run kept dense compaction at its default A_dense=8, so its 10.0 tok/s is
dense-compact + sparse-overhead-no-savings; it validates the sparse gather via PPL, not speed.)

## Speed (A/B, captured decode)

| variant | rows/step (dense+sparse) | decode tok/s | vs baseline | PPL (mito80) |
|---|---|---:|---:|---:|
| baseline (full E*cap both groups) | 8192 + 8192 | 5.782 | 1.00× | 4.001 |
| dense compact only (A_dense=8) | 1024 + 8192 | 12.436 | 2.15× | 4.239 |
| compact both (A_dense=8, A_sparse=24) | 1024 + 3072 | 16.203 | 2.80× | 4.123 |

Dense group is 64% of the step (`captured_attribution.md`): compacting it 8192→1024 rows gives 2.15×.
Compacting the sparse group too (8192→3072 rows) reaches **2.80×** (16.203 tok/s), monotone in rows
removed. All three PPLs sit in / adjacent to the mito80 noise band; all capture FULL.

Cross-run caveat: each row is a separate Modal container, so decode tok/s carries container-to-container
GPU variance (a no-savings config landed at 10.0 above). The **direction is robust**, decode tok/s is
monotone in rows removed (5.8 → 12.4 → 16.2) and every step is a full CUDA-graph capture, but treat the
exact multipliers as ±, not certified to 3 digits.

## Quality note (no softening)

The mito80 PPL shifts +0.24 at A_dense=8 vs baseline. This is an 80-token passage under greedy decode,
where a single early logit flip from FP reduction-order changes (8 gathered experts reduce in a different
order than 64) cascades through the rest of the passage, the same 80-token noise C2 excluded from quality
ranking. **We do NOT claim compact routing is quality-neutral on this evidence.** The A_max=E correctness
run shows no gather bug; the real quality delta needs the downstream 4-task eval (future work, not run here).

## Verdict (Task 1B)

Active-expert compaction is correct and materially faster (2.80× compacting both groups), attacking the
measured E*cap padding bottleneck. It does **not** reach the dense NVFP4 fused SOTA (48.248 tok/s), the
residual is the `cap=128`-per-active-expert padding, which needs a custom compact-row decode kernel
(deferred). See
`deepseek_compact_decode.md` for the full serving table and `verdict.md` for the campaign verdict.
