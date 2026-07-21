# WS-E: broaden the downstream quality evidence (8-task battery, both models, all deployed policies)

**Branch:** `ws-e-downstream-breadth`. **Harness:** `serve_dsv4.py::_downstream_impl` (served-model
loglikelihood MC eval) + `downstream` / `downstream4` / `glm_downstream` entrypoints. **Verdict:**
[verdict.md](verdict.md).

## Why this

The sparse-FP4 MoE serving-speed thesis is settled negative (C2 decode, C9 spec-decode, C10 prefill all
KILL). The surviving contribution of the sparse work is **quality / capability preservation at reduced
expert FLOP**, so that is the front to harden. Two concrete gaps in the current quality evidence:

1. **The battery is only 4 tasks.** DeepSeek's `deepseek_final.csv` and GLM's table both average ARC-C,
   HellaSwag, Winogrande, MMLU-5. **PIQA was dropped** because `ybisk/piqa` needs a gated loading
   *script* the serve image can't run. A 4-task AVG is a thin capability claim, and the drop was a
   symptom fix (delete the block) not a root-cause fix.
2. **GLM has no full downstream table.** GLM-5.2 downstream accuracy exists only for **dense + route-slot
   D2** (`limit=200`); **down49 and gateup49 are PPL-only**. The large-model transfer story, the headline
   "the DeepSeek structural rule holds on GLM", rests on PPL for two of its four policies.

## What WS-E changes

**Root-cause PIQA + widen to an 8-task battery**, run at `limit=400` on both models for every deployed
policy:

| task | loader (no gated script) | metric | added by |
|---|---|---|---|
| ARC-Challenge | `allenai/ai2_arc` / ARC-Challenge | acc_norm | (existing) |
| ARC-Easy | `allenai/ai2_arc` / ARC-Easy | acc_norm | **WS-E** |
| HellaSwag | `Rowan/hellaswag` | acc_norm | (existing) |
| PIQA | `ybisk/piqa` @ `refs/convert/parquet` | acc_norm | **WS-E (restored)** |
| OpenBookQA | `allenai/openbookqa` / main | acc_norm | **WS-E** |
| BoolQ | `google/boolq` | acc | **WS-E** |
| Winogrande | `allenai/winogrande` / winogrande_xl | acc | (existing) |
| MMLU-5 | `cais/mmlu` (5 subjects) | acc | (existing) |

- **PIQA root-cause fix:** load the HF server-side parquet-convert branch
  (`revision="refs/convert/parquet"`) directly instead of the gated builder script. Verified on a CPU
  probe: n=1838 validation rows, keys `goal/sol1/sol2/label` (label is a string, `int()`-cast). No
  `trust_remote_code`, so it runs on the serve image.
- **Templates match lm-eval** (comparability): PIQA/ARC `"Question: {q}\nAnswer:"`; OpenBookQA bare
  `question_stem`; BoolQ `"{passage}\nQuestion: {question}?\nAnswer:"` with `["no","yes"]` choices, acc
  primary (yes/no lengths differ, so acc_norm would bias).

## Runs (all `limit=400`, `max_len=2048`, eager)

| model | policy | mode | GPUs | moe / proj / route-slot / anchor |
|---|---|---|---|---|
| DeepSeek-V4 | dense (ref) | `downstream` | 2 | off |
| DeepSeek-V4 | c_down49 | `downstream` | 2 | sparse / down / — / 0..21 |
| DeepSeek-V4 | D2 route-slot | `downstream4` | 4 | sparse / both / 2 / 0..21 |
| GLM-5.2 | dense (ref) | `glm_downstream` | 8 | off |
| GLM-5.2 | down49 | `glm_downstream` | 8 | sparse / down / — / 0..37 |
| GLM-5.2 | D2 route-slot | `glm_downstream` | 8 | sparse / both / 2 / 0..37 |

Anchor comma-lists reproduce the frozen-table policies exactly (DeepSeek first-22 MoE layers dense; GLM
MoE layers 0-37 dense). DeepSeek dense runs first as the shared-harness smoke: it exercises the identical
`_downstream_impl` 8-task path GLM uses, so a clean DeepSeek AVG validates the loaders before the 3x
8-GPU GLM jobs.

## What to report (honesty guards)

- Re-anchor every headline row to the 8-task AVG; keep the 4-task AVG alongside so the tables reconcile
  with the frozen paper numbers (the new AVG is a different battery, not a correction of the old one).
- GLM down49/D2 AVG vs dense: does the DeepSeek structural rule (down-tolerant, ~-0.3pt) hold on GLM at
  full battery, and does the "PPL-only" caveat on down49 close?
- If PIQA or any added task shifts a policy's rank vs dense, say so; do not average away a task collapse.
