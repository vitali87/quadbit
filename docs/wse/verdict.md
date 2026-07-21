# WS-E verdict: 8-task downstream battery, both models, all deployed sparse policies

**Branch:** `ws-e-downstream-breadth`. **Design:** [design.md](design.md). **Harness:**
`serve_dsv4.py::_downstream_impl` (served-model loglikelihood MC) + `downstream` / `downstream4` /
`glm_downstream`. Six runs, `limit=400`, `max_len=2048`, eager. Per-run CSVs:
`/cache/qb_downstream_wse_{ds,glm}_{dense,down49,d2}.csv`.

## Result: capability preservation holds on the wider battery, and GLM's PPL-only caveat closes

Broadening from 4 tasks to 8 (restored PIQA + ARC-Easy + OpenBookQA + BoolQ) does not overturn any
capability claim. Every deployed sparse policy on both models stays within ~0.8 pt AVG of its dense
reference, and **no single task collapses** on any policy.

### 8-task AVG (normalized primary)

| policy | DeepSeek-V4 | Δ dense | GLM-5.2 | Δ dense |
|---|---|---|---|---|
| dense (ref) | **0.7548** | — | **0.7841** | — |
| down49 | 0.7491 | −0.57 pt | 0.7826 | **−0.15 pt** |
| D2 route-slot | 0.7551 | **+0.03 pt** | 0.7762 | −0.79 pt |

### Per-task primary (acc_norm; acc for winogrande/mmlu/boolq)

**DeepSeek-V4** (dense / down49 / D2):

| task | dense | down49 | D2 |
|---|---|---|---|
| arc_c | .655 | .640 | .640 |
| arc_e | .898 | .905 | .895 |
| hellaswag | .703 | .718 | .705 |
| piqa | .848 | .843 | .840 |
| openbookqa | .453 | .460 | .450 |
| boolq | .880 | .860 | .883 |
| winogrande | .788 | .758 | .803 |
| mmlu-5 | .816 | .810 | .826 |
| **AVG-8** | **.7548** | **.7491** | **.7551** |
| AVG-4¹ | .7403 | .7313 | .7434 |

**GLM-5.2** (dense / down49 / D2):

| task | dense | down49 | D2 |
|---|---|---|---|
| arc_c | .700 | .698 | .680 |
| arc_e | .878 | .868 | .875 |
| hellaswag | .778 | .758 | .770 |
| piqa | .868 | .878 | .868 |
| openbookqa | .500 | .513 | .488 |
| boolq | .928 | .925 | .920 |
| winogrande | .773 | .768 | .758 |
| mmlu-5 | .850 | .856 | .852 |
| **AVG-8** | **.7841** | **.7826** | **.7762** |
| AVG-4¹ | .7750 | .7696 | .7649 |

¹ AVG-4 = the frozen paper battery subset (arc_c, hellaswag, winogrande, mmlu-5) recomputed from the
**same** 8-task run, so the two averages are one consistent measurement. This is a fresh `limit=400`
draw and is NOT a correction of the frozen `deepseek_final.csv` / `limit=200` GLM rows; it is a separate,
wider battery. Cross-reference within a battery, not across.

## What the broadening adds

1. **PIQA restored at root cause, not symptom.** The prior 4-task battery dropped PIQA because
   `ybisk/piqa` needs a gated loading *script* the serve image cannot run. WS-E loads the HF
   server-side parquet-convert branch (`revision="refs/convert/parquet"`, n=1838 validation) directly.
   PIQA is now a real column on both models and is **neutral-to-positive under sparsity** (DeepSeek D2
   .840 vs dense .848; GLM down49 .878 *above* dense .868), so it never masked a regression.
2. **GLM gets its first full downstream table.** Before WS-E, GLM had downstream accuracy only for
   dense + route-slot D2 (`limit=200`); down49 was **PPL-only**. WS-E measures all three deployed GLM
   policies at `limit=400`. **down49 lands at −0.15 pt AVG** (near-flat), which directly confirms the
   "down-projection is tolerant" rule on GLM with downstream accuracy, not just PPL. The PPL-only caveat
   on the GLM down policy is closed.

## Honesty guards ([design.md](design.md))

- **No task collapse anywhere.** The largest single-task move is DeepSeek down49 winogrande −3.0 pt and
  GLM D2 arc_c −2.0 pt; both sit inside the per-400 sampling band and neither drags any policy off the
  ~0.8 pt AVG envelope. Restored/added tasks do not flip a policy's rank vs dense.
- **The best sparse policy differs by model.** On DeepSeek, D2 route-slot (+0.03) edges out down49
  (−0.57) on the 8-task battery; on GLM, down49 (−0.15) beats D2 (−0.79). Both models keep both policies
  near dense, but the Pareto winner is not the same policy across models. State this, do not average it
  away.
- **The 8-task battery is generally *kinder* than the 4-task, because the added tasks (arc_e, piqa,
  boolq, obqa) are ones sparse holds up on.** DeepSeek D2 in particular reads +0.03 pt on 8 tasks vs the
  frozen −0.79 pt 4-task row. That gap is run-to-run MC variance plus battery composition, not a fixed
  improvement; the frozen table remains the paper reference and this run corroborates "no collapse on a
  wider battery," it does not upgrade the headline number.
- **Metric of record is downstream AVG, not PPL** (unchanged from the frozen tables). DeepSeek dense PPL
  reproduced at 3.537 (matches frozen); the other rows' PPL tracks their frozen counterparts.
