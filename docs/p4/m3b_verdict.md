# P4 Milestone 3-B (one real MoE layer) — verdict: PASS

A full one-layer sparse-FP4 MoE forward (real gate → topk router → softmax weights → experts →
weighted token scatter) runs through the plugin's graph-safe apply, captures bit-exact, and matches the
frozen `build_routing` eager path on every no-drop routing regime.

Harness `harness/p4_m3b.py`, raw log `docs/audit/logs/p4_m3b.log`. Env: torch 2.12.0.dev+cu128, RTX PRO
6000 Blackwell, cap 12.0, driver 12.8. Single rank (identity `expert_map`); EP off-rank is M3-C.

## What M3-B adds over M3-A

- A **real router**: `gate = x @ Wgate`, `topk`, `softmax` weights per slot (not a fixed route pattern).
- The plugin's new **`_seg_apply_gs`** — a graph-safe `both`-branch MoE apply (additive; the frozen
  eager `patched_moe_apply` is untouched). It calls `route_fixed_cap` + `quant_into` + `seg_into`
  entirely **inside** the captured region (M3-A routed outside capture), which forced three ops in
  `route_fixed_cap` to be made capture-legal (below).
- **router-weight-on-input** semantics tested (`uniform-oninput`).

## `route_fixed_cap` hardened to run INSIDE capture

Running routing inside the captured forward surfaced three host-sync/dynamic-shape ops that were dormant
while M2/M3-A routed outside capture. All fixed, output identical:

| op | why it broke capture | fix |
|---|---|---|
| `torch.bincount(assign, minlength=e)` | reads the max element to host to size output → CPU↔CUDA copy | `torch.zeros(e).scatter_add_(0, assign, ones)` — E constant, static |
| `src[dest[keep]] = order[keep]` (boolean index) | boolean mask calls `.nonzero()` → data-dependent size → invalidates capture | `torch.where(keep, dest, sink)` + unconditional `src.scatter_(0, dest, order)`, slice off the trash slot — identical `src` |
| `dropped = int(keep.sum())` / `bool(keep.all())` | `.item()`/`.all()` host sync | return `dropped = r - keep.sum()` as a **device scalar**; callers `.item()` it outside capture |

A separate harness-only fix: `torch.randn` called to build a case's inputs **after** a prior capture
trips the philox offset bookkeeping ("Offset increment outside graph capture"), so all randomness is
hoisted before the capture loop. This is a test-harness property, not a plugin/kernel issue.

## Results (T=4096; NEWvsOLD = graph-safe vs frozen eager; GvsE = graph replay vs new eager)

| regime | topk | cap | drop | NEWvsOLD cos | GvsE cos | nf | eager ms | graph ms | mem MB |
|---|---|---|---|---|---|---|---|---|---|
| single-e0 (all→e0, overflow) | 1 | 1024 | 3072 | 0.499790¹ | 1.000000 | 0 | 2.029 | 1.734 | 1573.4 |
| uniform | 6 | 6144 | 0 | 1.000000 | 1.000000 | 0 | 11.122 | 10.790 | 4225.6 |
| empty-experts (4 busy) | 6 | 6144 | 0 | 1.000000 | 1.000000 | 0 | 11.121 | 10.769 | 4225.6 |
| near-capacity | 6 | 6784 | 0 | 1.000000 | 1.000000 | 0 | 12.301 | 11.994 | 4557.1 |
| overflow (cap ≪ load) | 6 | 2048 | 8192 | 0.816465¹ | 1.000000 | 0 | 3.712 | 3.367 | 2104.2 |
| uniform-oninput | 6 | 6144 | 0 | 1.000000 | 1.000000 | 0 | 11.520 | 11.174 | 4225.6 |

¹ The two overflow regimes intentionally drop the rows `build_routing` (OLD) keeps, so NEW≠OLD is
correct here; the drop count is deterministic and the **capture is still bit-exact** (GvsE = 1.000000).

## Acceptance gates (all met)

- **capture OK 6/6**; **graph == eager 6/6** (GvsE cos = 1.0, nonfinite = 0 everywhere);
- **NEW == OLD on all 4 no-drop regimes** (cos = 1.000000; uniform, empty-experts, near-capacity, and
  router-weight-on-input);
- single-expert, empty-experts, near-capacity, and **overflow** all capture cleanly with deterministic
  drop — the routing-regime coverage the milestone asked for;
- no host sync / no dynamic shape inside the captured region.

`M3-B PASS. Do not proceed to full model until one-layer MoE is correct — it is.`

## Next (M3-C)

Multi-rank / EP: the on-rank slot count is dynamic (off-rank slots masked out), which must be made
static for capture (a sink expert bucket, not a boolean compaction). Prove multi-rank plugin graph
safety, per-rank routing capacity stability, and no collective/DSA interaction breaks capture, on the
smaller DeepSeek 2-GPU route-slot path before 8-GPU GLM.
