# C3 Task 1D: DeepSeek-V4-Flash compact-decode serving table (4 GPU, captured)

The C3 lever is **active-expert compaction** ([compact_routing_ab.md](compact_routing_ab.md)): route only the token-loaded experts
per group (A_max*cap rows) instead of all E=64 (E*cap=8192), capture-safe and bit-correct. This table puts
the compact D2 path next to the two C2 reference points on the same harness / model / graph-mode /
PPL-protocol.

| row | MoE path | graph | decode tok/s | vs dense SOTA | PPL (mito80) | weights GiB/GPU | pool GiB |
|---|---|---|---:|---:|---:|---:|---:|
| dense NVFP4 baseline (C2 A1) | vLLM native FlashInfer-CUTLASS fused NVFP4 (`QB_MOE=off`) | captured | **48.248** | 1.00× | 4.122 | 40.83 | 0.18 |
| quadbit D2 native captured (C2 A4) | route-slot D2, `group_gemm_nvfp4` anchor | captured | 5.972 | 8.1× slower | 4.094 | 51.7 | 1.10 |
| **quadbit D2 compact-both (C3)** | route-slot D2 + active-expert compaction (A_dense=8, A_sparse=24) | captured (FULL) | **16.203** | **3.0× slower** | 4.123 | 51.7 | 1.10 |

Compact D2's own in-session non-compact baseline was 5.782 tok/s, so compaction is **2.80×** on the same
build; against the C2 D2 point (5.972) it is 2.71×. Either way it closes the D2→dense decode gap from
**8.1× to 3.0×**, roughly a third of the way, while capturing FULL and holding PPL in the mito80 noise
band (3.95–4.09).

## Does it beat the SOTA or create a strict Pareto point? No.

- **Decode speed:** 16.203 < 48.248. Still **3.0× slower** than the dense fused baseline. Not a win.
- **Memory:** unchanged by compaction, the D2 weights are still **dual-resident** (raw NVFP4 dense slots +
  2:4 codes) at 51.7 GiB/GPU (+27% over dense's 40.83), with less KV headroom. Dense still wins this axis.
- **Quality:** mito80 is an 80-token wash; the real downstream signal (P1/PR#13) is D2 AVG .7508 vs dense
  .7603 = **−0.95 pt** (4-task). D2 is slightly *worse*, not better. Dense wins this axis too.

So compact D2 is faster than the *old* D2 but is dominated by the dense NVFP4 fused baseline on all three
axes (speed, memory, quality). **No SOTA claim, no strict-Pareto claim vs dense.**

## Where the remaining 3.0× lives

Attribution ([captured_attribution.md](captured_attribution.md)) after compaction: the wall is now the **`cap=128`-per-active-expert
padding**, each active expert still processes a full 128-row block for ~1 real decode token. Active-expert
compaction removed the *inactive*-expert padding (E→A_max groups); it cannot remove the *within-active*
padding without a decode kernel/layout that packs arbitrary compact rows with per-row expert routing (the
deferred custom compact-row path). That kernel is the exact next lever.
