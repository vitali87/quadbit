# P4 Milestone 3-C (expert-parallel graph safety) — verdict: PASS (collective-in-capture deferred to M4)

The graph-safe apply captures **per EP shard** with off-rank slots sink-routed (static shape despite a
data-dependent on-rank count), matches the frozen compacting EP path, and the shards sum back to the full
MoE output. Routing capacity is stable and balanced across shards.

Harness `harness/p4_m3c.py`, raw log `docs/audit/logs/p4_m3c.log`. Env: torch 2.12.0.dev+cu128, RTX PRO
6000 Blackwell, cap 12.0, driver 12.8. EG=8 experts, world=2 (4/shard), T=4096, topk=6.

## Scope split (why this is single-process, no NCCL)

M3-C owns three questions. Two are pure compute/routing and one is the collective:

1. **multi-rank plugin graph safety** — does the graph-safe apply capture on an EP *shard* (partial
   expert set, off-rank slots present)? ✅ proven here.
2. **per-rank routing-capacity stability** — do fixed-capacity counts stay balanced and non-overflowing
   across shards? ✅ proven here.
3. **collective-in-capture** — does an NCCL all-reduce capture inside the graph? → **deferred to M4.**

(1) and (2) need no collective, so they are simulated in one process: each shard runs sequentially with
its own local experts + `expert_map`, and the cross-shard reduce is an **in-process sum** — exactly what
an all-reduce computes. This isolates the routing/compute graph-safety from NCCL.

### The NCCL deferral is a container issue, not a capture issue

A genuine 2-GPU `torch.distributed` version (git history of this file) **hung on the FIRST plain-eager
`all_reduce`, before any graph capture** — the ProcessGroupNCCL watchdog aborted after its timeout.
Reproduced across three configs (default transport; `device_id`-bound comms + no parent CUDA context;
forced TCP/loopback with `NCCL_P2P/SHM/IB_DISABLE`). Since the failing collective was **eager**, this is
a Modal `mp.spawn` NCCL-bootstrap/transport problem, **not** a CUDA-graph or SM120 limitation. vLLM's own
EP NCCL is known-good on this exact hardware (Campaign-B GLM/DeepSeek 8-GPU serves), so the
collective-in-capture question is answered correctly in M4 on vLLM's working communicator, not fought in
a standalone harness. Precise blocker recorded below.

## Results (LOCAL contribution per shard; NEWvsOLD = graph-safe vs frozen eager EP; GvsE = graph vs eager)

| shard | lo | el | cap | on-rank | drop | counts (per local expert) | NEWvsOLD | GvsE | nf | eager ms | graph ms | mem MB |
|---|---|---|---|---|---|---|---|---|---|---|---|---|
| shard0 | 0 | 4 | 6144 | 12328 | 0 | [3090, 3075, 3125, 3038] | 1.000000 | 1.000000 | 0 | 3.281 | 2.927 | 1142.3 |
| shard1 | 4 | 4 | 6144 | 12248 | 0 | [3083, 3078, 3093, 2994] | 1.000000 | 1.000000 | 0 | 3.282 | 2.928 | 1159.1 |
| full-8 | 0 | 8 | 6144 | 24576 | 0 | [3090, 3075, 3125, 3038, 3083, 3078, 3093, 2994] | 1.000000 | 1.000000 | 0 | 6.218 | 5.858 | 1981.6 |

**EP-reduce:** `cos(shard0 + shard1, full-8) = 0.999994` — the two shards partition the 8 experts and
their sum reconstructs the single all-experts output (the in-process stand-in for all-reduce).

## Acceptance gates (all met)

- **capture OK 3/3**; **graph == eager 3/3** (GvsE cos = 1.0, nonfinite = 0);
- **NEW == OLD 3/3** — the sink-routed EP path (valid mask, off-rank → sink bucket) matches the frozen
  compacting EP path exactly (no drop at 2× cap headroom);
- **routing-capacity stability**: both shards balanced (~3050–3125 rows/expert) at cap 6144, 0 overflow;
- **EP correctness**: shard-sum reconstructs the full MoE output (cos 0.999994).

`M3-C PASS.` The `route_fixed_cap` `valid`-mask sink routing (added this milestone) is the EP primitive
M4 will use in the plugin's real `patched_moe_apply` graph-safe branch.

## Precise blocker table (standalone NCCL, deferred to M4)

| operation | where | failure mode | class |
|---|---|---|---|
| first eager `dist.all_reduce(y)` | `mp.spawn` worker, before capture | ProcessGroupNCCL watchdog hang → SIGABRT after timeout, across default / device_id-bound / TCP-forced transports | **collective bootstrap** (Modal `mp.spawn` container transport), NOT plugin / routing / capture / SM120 |

## Next (M4)

Wire the plugin's `patched_moe_apply` sparse `both`/route-slot branch to `_seg_apply_gs` behind an
explicit `QB_GRAPH` flag (+ the A6 indexer `.cpu().tolist()` fix for DSA capture), then run DeepSeek /
GLM route-slot D2 under vLLM with graph capture enabled — where vLLM's working EP NCCL runs the
collective, answering the deferred question in the real serving path.
