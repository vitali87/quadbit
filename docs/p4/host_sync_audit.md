# P4 Milestone 1 — host-sync / graph-safety audit

Scope: `harness/qb_vllm_plugin/qb_sm120_plugin.py` (1301 lines) and `moe_recon.py`, on the frozen
baseline `campaign-b-freeze-a91c5d9`. Goal: classify every graph-breaking operation in the plugin's
**forward** path, separate correctness-critical from debug-only, and produce a replacement plan. This
audit does not change behavior; it scopes M2-M5.

## Method

Grepped for: `.item()`, `.tolist()`, `.cpu()`, `.numpy()`, `torch.unique`, `.synchronize`,
`cudaMemcpy`, `bincount`/`nonzero`, ctypes calls, in-forward `torch.empty`/`zeros`, Python branching on
device values, and STATS/counter reads. Then read each hit to decide whether it runs in the steady-state
serving forward (capture-relevant) or only at load / behind a debug flag (capture-irrelevant).

## The capture-critical forward paths (per policy)

`ModelOptNvFp4FusedMoE.apply` is patched to `patched_moe_apply` (line 1163). It branches by policy, and
**every branch currently contains at least one device->host sync or data-dependent allocation**:

| policy (env) | forward branch | primary graph-breaker |
|---|---|---|
| `QB_MOE=dense` | dense NVFP4 dequant-per-expert loop (1255) | `torch.unique(local).tolist()` (1255) — **the documented blocker** (`glm_graphfail.log`) |
| `QB_MOE=sparse` `proj=both` (c_both) | `build_routing` + `seg_gemm` (1197-1217) | `build_routing`: `int(padc.sum().item())` + per-expert `.item()` loop + data-dependent alloc (750-757) |
| `QB_MOE=sparse` `proj=down`/`gateup` (c_down49/gateup) | sparse seg for one proj + `_dense_route` for the anchored proj (1211/1219) | `_dense_route`: `torch.unique(eblk).tolist()` (99) **and** `build_routing` |
| `QB_MOE=sparse` `route_slot=N` (D2, headline) | `_route_slot_apply` (1179) | dense-slot loop `torch.unique(d_ids).tolist()` (135) + `build_routing` for the tail |
| DSA attention (all policies) | indexer mqa-logits bf16 replacement | `seq_lens.reshape(-1).cpu().tolist()` (995) |

So fixing only line 1255 unblocks the **dense** control, not the sparse deployment policies. The headline
route-slot D2 path needs the routing/`_dense_route`/`_route_slot_apply` syncs fixed too.

## Full sync inventory

### A. Forward-path, correctness-critical (must fix or pad for capture)

| # | line | operation | why it exists | replacement plan | capture effect |
|---|------|-----------|---------------|------------------|----------------|
| A1 | 1255 | `torch.unique(local).tolist()` | dense fallback iterates the local experts present this microbatch | precompute a fixed local-expert list (EP shard is static) or a padded fixed-capacity expert loop; iterate a Python-constant range, mask empty | removes the documented `cudaErrorStreamCaptureUnsupported` for the dense control |
| A2 | 750 | `r_pad = int(padc.sum().item())` sizes `src`/`eblk` | routed rows padded to `_BN` blocks; total pad count sizes the segment buffers | **pad to fixed capacity**: allocate `src`/`eblk` at a capture-time max (`E*ceil(cap/_BN)*_BN`), fill by scatter, no host read | removes host sync **and** the data-dependent allocation (two blockers in one) |
| A3 | 755-757 | per-expert `int(counts[ex].item())`/`int(padc[ex].item())` in a `range(E)` Python loop | builds `src`/`eblk` segment offsets one expert at a time | vectorize: `cumsum` of padded counts on device -> scatter `order` into `src`, `repeat_interleave` expert ids into `eblk`; no Python-visible values | eliminates up to 2*E=512 syncs/layer/forward |
| A4 | 99 | `_dense_route`: `torch.unique(eblk).tolist()` loop | dense (anchored) projection over routed rows, per resident expert | same fix as A1 (fixed/padded local-expert range) | unblocks down-only/gateup-only sparse policies |
| A5 | 135 | route-slot dense-slot `torch.unique(d_ids).tolist()` loop | top-N dense slots iterated per expert | same fix as A1 | unblocks the D2 headline path |
| A6 | 995 | indexer `seq_lens.reshape(-1).cpu().tolist()` | bf16 mqa-logits replacement iterates per-request seq lens | precompute request layout on device (cumulative seq lens), index without host copy; or capture with fixed seq-len layout | unblocks DSA attention capture (plugin-owned, not the native FlashInfer core) |

### B. Forward-path, debug/telemetry only (move behind a debug flag — the M1 "first target")

| # | line | operation | class | replacement plan |
|---|------|-----------|-------|------------------|
| B1 | 1234 | `STATS["sparse_expert_calls"] += int(valid.sum().item())` | counter, runs **every** apply (ungated) | gate behind `if _INSTR:`; capture path must not read it. Counters are proof-of-execution for eager, not needed under graph |
| B2 | 1226-1231 | imbalance `.item()` + `_flush_metrics()` | already `_INSTR`-gated + `%100` | leave gated; ensure `_INSTR=0` under capture (it is off by default) |
| B3 | 826-1141 | many `print(...)` first-call/anchor logs | string ops, mostly first-call-gated | harmless (no device read) but ensure none are in the hot replay path; first-call gates already handle this |
| B4 | 1205,1222-1223 | `_ev_start`/`_ev_end` CUDA-event instrumentation | `_INSTR`-gated | leave gated (events are capture-legal anyway, but off by default) |

### C. Load-time / debug-flag paths (NOT in the steady-state forward — no capture action)

These run at weight load (`process_weights_after_loading`) or only under `QB_DUMP`/`QB_CALIB`/`QB_QMAP`/
`QB_RECON`, which are off during normal serving and complete before graph capture:

- `process_weights` packing / `_dense`/`empty_cache` (1159-1160): load-time, one-shot.
- recon `torch.cuda.synchronize()` (451), nonfinite `.item()` (483-485): `QB_RECON` only.
- calib `.cpu()` (531-533): `QB_CALIB` only.
- qmap `.item()`/cosine/`torch.unique` (577,612,638-647): `QB_QMAP` only.
- dump `.cpu()` (318-321): `QB_DUMP` only.

### D. Not host syncs, but capture-hostile (structural — M2/M3)

| # | lines | issue | plan |
|---|-------|-------|------|
| D1 | 733,741-743 | ctypes `lib.quantize_act_nvfp4_2lvl` / `lib.sparse_moe_mm_2lvl` launch kernels; if they use the **default stream**, they are invisible to torch's capture stream (not captured, or capture error) | verify the `.so` launch stream; **Route B**: wrap `fused_mlp`/`seg_gemm` as a proper `torch.library`/C++ custom op that takes and launches on the current stream (mirrors the already-graph-safe Llama `quadbit::fused_mlp`) |
| D2 | 730-732,740,751-752,1167 | in-forward `torch.empty`/`zeros` sized by data-dependent `r`/`r_pad`/`t` | with A2/A3 padded to fixed capacity and a persistent workspace, allocations become fixed-shape; PyTorch's capture memory pool then handles them |

## Prioritized plan

1. **M1 code (low-risk, this milestone):** gate B1 behind `_INSTR` so the counter sync leaves the steady
   path; confirm B2-B4 are off by default under capture. No correctness change; validate with one eager
   serving run that numbers are byte-identical to the frozen baseline.
2. **M2 (standalone):** build the fixed-capacity, host-sync-free routing (A2/A3 vectorized) + persistent
   workspace + Route-B custom-op wrapper (D1), and capture a single sparse MLP standalone (no vLLM).
3. **M3:** replay the captured sparse-MLP subgraph inside the vLLM Llama forward (Route A/B).
4. **M4:** extend to the MoE apply — fixed local-expert range (A1/A4/A5) + padded routing — on 1 rank, then
   fixed route, padded capacity, DSA, 8-GPU GLM. Fix A6 for indexer capture.
5. **M5:** final graph-vs-graph comparison rows.

**Note on DSA:** A6 is the plugin's own bf16 indexer-logits replacement, not the native FlashInfer
sparse-MLA core (which the frozen result shows runs natively on SM120). P4 does not touch the native DSA
path; it removes the plugin's host sync around it.
