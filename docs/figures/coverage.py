"""M4 Phase-2: quadbit sparse coverage of DeepSeek-V4-Flash (parameter accounting).

What quadbit sparsifies: the routed + shared EXPERT MLPs (gate/up/down). What stays dense NVFP4:
MLA attention (q/kv LoRA + o_proj + sparse index), the router/gate, embeddings, lm_head, MTP head.
Parameter coverage is exact from config dims; active-FLOP coverage is per-token (top-k experts).
Emits data/coverage.csv + prints the table. Run: uv run --with numpy python docs/figures/coverage.py
"""

import csv
from pathlib import Path

# DeepSeek-V4-Flash config (verified from HF config.json)
H = 4096                # hidden_size
I = 2048                # moe_intermediate_size (per expert)
N_ROUTED = 256
N_SHARED = 1
TOPK = 6
LAYERS = 43
FIRST_K_DENSE = 3       # DeepSeek convention: first-k layers use a dense MLP (rest are MoE)
DENSE_INTER = 18432     # dense-layer MLP intermediate (V3-family default; labeled estimate)
VOCAB = 129280
# MLA attention linear params/layer (estimate from LoRA ranks; q_lora=o_lora=1024, kv compressed)
ATTN_PER_LAYER = H * 1536 + 1536 * (64 * 192) // 64 + H * 576 + H * H  # rough MLA q_a/q_b/kv_a/o
MOE_LAYERS = LAYERS - FIRST_K_DENSE

expert = 2 * I * H + I * H  # w1(I,H)+w3(I,H)+w2(H,I) MACs-worth of params = 3*I*H
routed = N_ROUTED * expert
shared = N_SHARED * expert
router = N_ROUTED * H
moe_mlp_params = MOE_LAYERS * (routed + shared)            # SPARSIFIED by quadbit
router_params = MOE_LAYERS * router                        # dense
dense_mlp_params = FIRST_K_DENSE * (3 * DENSE_INTER * H)    # dense MLP layers (not sparsified here)
attn_params = LAYERS * ATTN_PER_LAYER                       # dense
embed_params = 2 * VOCAB * H                                # embed + lm_head (untied)

PUBLISHED_TOTAL = 284e9   # DeepSeek-V4-Flash reported total params (anchor; our attn/embed are estimates)
total = PUBLISHED_TOTAL
sparsified = moe_mlp_params

# active linear FLOPs per token (2*MACs): MLP = (topk routed + shared) experts; attention = MLA per layer
mlp_active_macs = MOE_LAYERS * (TOPK + N_SHARED) * expert + FIRST_K_DENSE * (3 * DENSE_INTER * H)
attn_active_macs = LAYERS * ATTN_PER_LAYER
mlp_sparse_active = MOE_LAYERS * (TOPK + N_SHARED) * expert   # the part quadbit runs sparse

rows = [
    ("routed experts (gate/up/down)", moe_mlp_params - MOE_LAYERS * shared, "SPARSE FP4 (2:4)"),
    ("shared expert (gate/up/down)", MOE_LAYERS * shared, "SPARSE FP4 (2:4)"),
    ("router / gate", router_params, "dense"),
    ("dense-MLP layers (first-k)", dense_mlp_params, "dense (est.)"),
    ("MLA attention (q/kv/o, est.)", attn_params, "dense"),
    ("embeddings + lm_head", embed_params, "dense"),
]
Path(__file__).parent.joinpath("data").mkdir(exist_ok=True)
with open(Path(__file__).parent / "data" / "coverage.csv", "w", newline="") as f:
    w = csv.writer(f); w.writerow(["component", "params_B", "precision"])
    for n, p, prec in rows:
        w.writerow([n, round(p / 1e9, 3), prec])

print("# M4 Phase-2: quadbit sparse coverage of DeepSeek-V4-Flash")
print(f"{'component':34s} {'params (B)':>12s}  precision")
for n, p, prec in rows:
    print(f"{n:34s} {p / 1e9:12.2f}  {prec}")
print(f"{'-' * 60}")
print(f"{'TOTAL (published)':34s} {total / 1e9:12.2f}")
print(f"\nquadbit sparsifies {sparsified / 1e9:.1f}B / {total / 1e9:.0f}B params "
      f"= {100 * sparsified / total:.0f}% of parameters (all expert MLPs)")
print(f"active linear FLOPs in sparse modules (per token) = "
      f"{100 * mlp_sparse_active / (mlp_active_macs + attn_active_macs):.1f}% "
      f"(top-{TOPK}/{N_ROUTED} experts + shared; attention + router stay dense)")
print("\nComparison: in dense Llama-3-8B, the MLP is ~66% of params; here the expert MLPs are the "
      "dominant parameter mass, so sparse coverage is HIGHER for the large MoE, not lower.")
