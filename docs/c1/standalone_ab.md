# C1 validation-ladder A: standalone dense-anchor A/B

**Harness:** `serve_dsv4.py::c1_dense_anchor` (1 GPU). **Log:** `docs/audit/logs/c1_dense_anchor.log`.
**Command:** `uv run modal run --detach harness/serve_dsv4.py::c1_dense_anchor`

**Question:** can `flashinfer.gemm.group_gemm_nvfp4_nt_groupwise` express the dense-anchor projection that
`_dense_seg_gs` runs as a decode-slow `range(E)` dequant-to-bf16 loop — correctly, graph-capturably, and
faster? Same fixed-capacity seg layout (`cap=128` rows/expert), DeepSeek MoE shapes (H=4096, I=2048).

- **A (old):** per-expert dequant NVFP4→bf16 then bf16 matmul (mirror of plugin `_dense_seg_gs`).
- **B (native):** one `group_gemm_nvfp4_nt_groupwise` call over the seg buffer (acts also NVFP4-quantized),
  inputs built with the verbatim flashinfer recipe (per-group `nvfp4_quantize` layout_128x4 do_shuffle=
  False, 128-aligned padded `a_scale`, `alpha=1/(a_gsf·b_gsf)`).

`cos` and `relL2` are vs the **true-bf16** grouped matmul (ceiling — both operands full precision), so the
~0.99 cos is the inherent NVFP4 tax of quantising *both* operands, not an error. `eager A` = dequant-loop
median ms; `eager B` / `graph B` = native eager / graph-replay median ms.

| proj | E | cos vs bf16 | relL2 | max abs | nonfin | capture | replay==eager | eager A (ms) | eager B (ms) | graph B (ms) | speedup (A/graphB) |
|---|---|---|---|---|---|---|---|---|---|---|---|
| gate_up | 32 | 0.9910 | 0.134 | 0.070 | 0 | OK | yes | 5.652 | 0.285 | 0.245 | **23.0×** |
| gate_up | 64 | 0.9910 | 0.134 | 0.070 | 0 | OK | yes | 11.016 | 0.520 | 0.479 | **23.0×** |
| down | 32 | 0.9910 | 0.134 | 0.071 | 0 | OK | yes | 3.276 | 0.176 | 0.140 | **23.3×** |
| down | 64 | 0.9910 | 0.134 | 0.072 | 0 | OK | yes | 6.552 | 0.305 | 0.266 | **24.6×** |

**Verdict (gate A): PASS.** The native grouped NVFP4 GEMM expresses the dense-anchor branch for both
projections at both EP-local expert counts, is CUDA-graph-capturable with bit-identical replay, is finite,
and is **~18–25× faster** than the `_dense_seg_gs` dequant loop. This refutes the P4 verdict's stated
limitation ("the dense anchored/grouped projection path lacks a fused dense NVFP4 grouped-GEMM"): FlashInfer
0.6.14 ships `group_gemm_nvfp4_nt_groupwise` and it works on SM120.

**Caveat carried to serving (gate B):** the native path NVFP4-quantises the *activations* on the anchored
projection (the old dequant path kept them bf16). Op-level cos vs true-bf16 is 0.991; whether that extra
activation-quant erodes the anchor's quality benefit is exactly what the DeepSeek-D2 serving A/B/C measures.
