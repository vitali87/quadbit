"""Publication figures for the quadbit SM120 sparse-FP4 paper (banked single-GPU results).

Generates SVG + PDF into out/ from the source CSVs in data/. Colorblind-safe (Okabe-Ito) palette,
two-column width, vector output. Figures 1/2 (system + dataflow schematics) are drawn separately;
Figure 6 (distributed scaling) is emitted by the M4 sweep. Run:
    uv run --with matplotlib,numpy python docs/figures/make_figures.py
Every number here traces to docs/graph_serving_result.md, docs/accuracy_pareto.md, or data/*.csv.
"""

import csv
from pathlib import Path

import matplotlib
import numpy as np

matplotlib.use("Agg")
import matplotlib.pyplot as plt

HERE = Path(__file__).parent
DATA, OUT = HERE / "data", HERE / "out"
OUT.mkdir(exist_ok=True)

# Okabe-Ito colorblind-safe palette
OK = {"blue": "#0072B2", "orange": "#E69F00", "green": "#009E73", "red": "#D55E00",
      "purple": "#CC79A7", "sky": "#56B4E9", "yellow": "#F0E442", "grey": "#999999"}
plt.rcParams.update({
    "font.size": 9, "font.family": "sans-serif", "axes.spines.top": False,
    "axes.spines.right": False, "axes.grid": True, "grid.alpha": 0.25,
    "figure.dpi": 150, "savefig.bbox": "tight", "axes.axisbelow": True,
})


def save(fig, name):
    for ext in ("svg", "pdf"):
        fig.savefig(OUT / f"{name}.{ext}")
    plt.close(fig)
    print(f"  wrote out/{name}.svg + .pdf", flush=True)


def read_csv(path):
    with open(path) as f:
        return list(csv.DictReader(f))


def fig3_decode_win():
    # docs/graph_serving_result.md, harness/quadbit_serve.py --graph --splits 8
    B = ["8", "32", "64"]
    decode = {"NVFP4 graph": [1046, 4237, 8384], "quadbit sparse split-K": [1147, 4543, 8567]}
    prefill = {"NVFP4 graph": [66469, 80825, 119083], "quadbit sparse split-K": [62914, 77605, 115069]}
    with open(DATA / "fig3_graph_serving.csv", "w", newline="") as f:
        w = csv.writer(f); w.writerow(["batch", "phase", "backend", "tok_s"])
        for ph, d in (("decode", decode), ("prefill", prefill)):
            for bk, vals in d.items():
                for b, v in zip(B, vals):
                    w.writerow([b, ph, bk, v])
    fig, axes = plt.subplots(1, 2, figsize=(6.8, 2.9))
    x = np.arange(len(B)); wd = 0.38
    for ax, (title, d, deltas) in zip(axes, [
            ("Decode (latency-critical)", decode, ["+9.7%", "+7.2%", "+2.2%"]),
            ("Prefill", prefill, ["-5.3%", "-4.0%", "-3.4%"])]):
        ax.bar(x - wd / 2, d["NVFP4 graph"], wd, label="NVFP4 graph", color=OK["grey"])
        ax.bar(x + wd / 2, d["quadbit sparse split-K"], wd, label="quadbit sparse split-K", color=OK["blue"])
        for i, dl in enumerate(deltas):
            ax.annotate(dl, (i, max(d["NVFP4 graph"][i], d["quadbit sparse split-K"][i])),
                        ha="center", va="bottom", fontsize=8,
                        color=OK["green"] if dl.startswith("+") else OK["red"])
        ax.set_title(title, fontsize=9); ax.set_xticks(x); ax.set_xticklabels([f"B={b}" for b in B])
        ax.set_ylabel("tok/s"); ax.margins(y=0.15)
    axes[0].legend(fontsize=7.5, loc="upper left", frameon=False)
    fig.suptitle("Fig 3. Graph-vs-graph serving: sparse split-K decode beats production NVFP4", fontsize=9.5)
    save(fig, "fig3_decode_win")


def fig4_crossover_heatmap():
    sp = {(r["B"], r["prompt"], r["gen"]): float(r["total_tps"]) for r in read_csv(DATA / "crossover_sparse.csv")}
    nv = {(r["B"], r["prompt"], r["gen"]): float(r["total_tps"]) for r in read_csv(DATA / "crossover_nvfp4.csv")}
    Bs = ["1", "8", "32", "64"]; prompts = ["128", "512", "2048", "8192"]; gens = ["16", "32", "64", "128", "256", "512", "1024"]
    fig, axes = plt.subplots(1, 4, figsize=(9.0, 2.7), sharey=True)
    wins = losses = ties = 0
    cmap = plt.cm.RdYlGn
    for ax, B in zip(axes, Bs):
        M = np.full((len(prompts), len(gens)), np.nan)
        for i, p in enumerate(prompts):
            for j, g in enumerate(gens):
                k = (B, p, g)
                if k in sp and k in nv and nv[k] > 0:
                    M[i, j] = (sp[k] / nv[k] - 1.0) * 100
        im = ax.imshow(M, aspect="auto", cmap=cmap, vmin=-8, vmax=8, origin="upper")
        ax.set_title(f"B={B}", fontsize=9); ax.set_xticks(range(len(gens)))
        ax.set_xticklabels(gens, rotation=90, fontsize=6.5); ax.set_xlabel("gen len")
        if B == "1":
            ax.set_yticks(range(len(prompts))); ax.set_yticklabels(prompts, fontsize=7); ax.set_ylabel("prompt len")
        for i in range(len(prompts)):
            for j in range(len(gens)):
                if not np.isnan(M[i, j]):
                    v = M[i, j]  # end-to-end tok/s delta %, no tie band (banked crossover rule)
                    if v > 0:
                        wins += 1
                    elif v < 0:
                        losses += 1
                    else:
                        ties += 1
    fig.colorbar(im, ax=axes, fraction=0.02, pad=0.01, label="sparse total tok/s vs NVFP4 (%)")
    fig.suptitle(f"Fig 4. Request-regime crossover (112 regimes): {wins} sparse wins / {losses} losses / {ties} ties", fontsize=9.5)
    save(fig, "fig4_crossover_heatmap")
    print(f"    crossover tally: {wins} wins / {losses} losses / {ties} ties", flush=True)


def fig5_pareto():
    # docs/accuracy_pareto.md (PPL) ; serving axis = keeps split-K decode win (categorical position)
    pts = [
        ("dense NVFP4 (baseline)", 7.974, 0, OK["grey"], "o"),
        ("all-sparse (banked)", 10.256, 1, OK["blue"], "o"),
        ("distilled sparse", 9.10, 1, OK["green"], "s"),
        ("gate_up-dense frontier", 9.750, 1, OK["orange"], "^"),
    ]
    with open(DATA / "fig5_pareto.csv", "w", newline="") as f:
        w = csv.writer(f); w.writerow(["policy", "ppl", "keeps_decode_win"])
        for n, ppl, win, _, _ in pts:
            w.writerow([n, ppl, win])
    fig, ax = plt.subplots(figsize=(4.2, 3.2))
    for n, ppl, win, c, mk in pts:
        ax.scatter(win, ppl, s=90, color=c, marker=mk, zorder=3, edgecolor="white", linewidth=0.8, label=n)
    ax.set_xticks([0, 1]); ax.set_xticklabels(["forfeits\ndecode win", "keeps split-K\ndecode win"])
    ax.set_ylabel("WikiText-2 PPL (lower better)"); ax.invert_yaxis()
    ax.axhline(7.974, color=OK["grey"], ls=":", lw=0.8)
    ax.legend(fontsize=7, loc="center right", frameon=False)
    ax.set_title("Fig 5. Accuracy vs serving: distillation cuts the\nsparse tax 10.27->9.10 but does not close it", fontsize=9)
    ax.margins(x=0.4)
    save(fig, "fig5_pareto")


def fig_ds_pareto():
    # DeepSeek-V4-Flash downstream Pareto: x = active sparse expert-FLOP %, y = downstream AVG.
    # Source: data/deepseek_final.csv (frozen final table). Gates at .718 / .728.
    rows = read_csv(DATA / "deepseek_final.csv")
    style = {
        "dense": (OK["grey"], "o", "dense NVFP4"),
        "c_down49": (OK["green"], "*", "c_down49 (down-only, 2 GPU)"),
        "c_down60": (OK["sky"], "^", "c_down60 (down-only, 2 GPU)"),
        "d2_slot2": (OK["orange"], "D", "D2 route-slot (2 dense, 4 GPU)"),
        "d1_slot1": (OK["red"], "v", "D1 route-slot (1 dense, 4 GPU)"),
        "c_gateup49": (OK["purple"], "s", "c_gateup49 (gate_up-only)"),
        "a2_49": (OK["blue"], "X", "a2_49 (both-proj)"),
    }
    fig, ax = plt.subplots(figsize=(5.2, 3.6))
    ax.axhline(0.718, color=OK["grey"], ls="--", lw=0.9)
    ax.axhline(0.728, color=OK["grey"], ls=":", lw=0.9)
    ax.text(50, 0.7185, ".718 gate", fontsize=6.5, color=OK["grey"], va="bottom", ha="right")
    ax.text(50, 0.7285, ".728 stretch", fontsize=6.5, color=OK["grey"], va="bottom", ha="right")
    for r in rows:
        key = r["policy"]
        if key not in style:
            continue
        c, mk, lab = style[key]
        x, y = float(r["active_sparse_flop_pct"]), float(r["avg_primary"])
        ax.scatter(x, y, s=130, color=c, marker=mk, zorder=3, edgecolor="white",
                   linewidth=0.8, label=lab)
    ax.set_xlabel("active sparse expert-FLOP % (real sparsity, not layer %)")
    ax.set_ylabel("downstream AVG (higher better)")
    ax.set_title("DeepSeek-V4-Flash: training-free sparse-FP4 Pareto\n"
                 "c_down49 cleanest 2-GPU; D2 max sparse-FLOP (4-GPU)", fontsize=8.5)
    ax.legend(fontsize=6.3, loc="lower left", frameon=False)
    ax.margins(x=0.08, y=0.12)
    save(fig, "fig_ds_pareto")


def fig_ds_designspace():
    # Capability design-space: what recovers training-free downstream AVG. Source:
    # data/wsa_downstream.csv + deepseek_final.csv. Gate at .718; dense ref .7383.
    rows = [
        ("magnitude 2:4 (100% sparse)", 0.5096, False),
        ("routed Wanda alone (100%)", 0.5092, False),
        ("A3 local repair (49%)", 0.6964, False),
        ("both-proj all-sparse a2_49 (49%)", 0.6966, False),
        ("gate_up-only anchor (49%)", 0.7056, False),
        ("route-slot D2 (33% FLOP)", 0.7304, True),
        ("proj-anchor c_down49 (49% lyr)", 0.7354, True),
    ]
    fig, ax = plt.subplots(figsize=(6.6, 3.2))
    y = np.arange(len(rows))
    for yi, (name, avg, ok) in zip(y, rows):
        c = OK["green"] if ok else OK["red"]
        ax.barh(yi, avg, color=c, alpha=0.85, edgecolor="white")
        ax.text(avg + 0.006, yi, f"{avg:.3f}", va="center", ha="left", fontsize=7.5)
        ax.text(0.46, yi, f"  {name}", va="center", ha="left", fontsize=7.5, color="white")
    ax.axvline(0.718, color=OK["grey"], ls="--", lw=1.0)
    ax.axvline(0.7383, color=OK["grey"], ls=":", lw=1.0)
    ax.text(0.718, len(rows) - 0.3, ".718 gate", fontsize=6.5, color=OK["grey"], ha="center")
    ax.text(0.7383, len(rows) - 0.3, "dense", fontsize=6.5, color=OK["grey"], ha="center")
    ax.set_xlim(0.45, 0.76); ax.set_yticks([])
    ax.set_xlabel("downstream AVG (training-free)")
    ax.set_title("DeepSeek design-space: only structural placement recovers capability\n"
                 "(magnitude/Wanda/A3/all-sparse fail; projection anchoring + route-slot pass)",
                 fontsize=8.5)
    save(fig, "fig_ds_designspace")


def fig_glm_transfer():
    # Structural sparse-FP4 mechanism transfers DeepSeek -> GLM-5.2. Same mechanism on the y-axis;
    # left = DeepSeek Δ downstream AVG (pt, from deepseek_final.csv), right = GLM Δ PPL (glm_results.md).
    # Both orderings agree: route-tail and down-proj are safe; gate/up carries the tax.
    # (mechanism, deepseek_delta_pt, glm_delta_ppl, safe)
    rows = [
        ("route tail (D2: top-2 dense)", -0.79, 0.065, True),
        ("down projection (49% lyr)", -0.29, 0.209, True),
        ("gate/up projection (49% lyr)", -3.27, 0.432, False),
    ]
    fig, (axl, axr) = plt.subplots(1, 2, figsize=(7.4, 2.6), sharey=True)
    y = np.arange(len(rows))
    for yi, (_, ds, glm, ok) in zip(y, rows):
        c = OK["green"] if ok else OK["red"]
        axl.barh(yi, ds, color=c, alpha=0.85, edgecolor="white")
        axl.text(ds - 0.05, yi, f"{ds:+.2f}", va="center", ha="right", fontsize=7.5)
        axr.barh(yi, glm, color=c, alpha=0.85, edgecolor="white")
        axr.text(glm + 0.008, yi, f"{glm:+.3f}", va="center", ha="left", fontsize=7.5)
    axl.set_yticks(y)
    axl.set_yticklabels([r[0] for r in rows], fontsize=7.5)
    axl.set_xlabel("DeepSeek Δ downstream AVG (pt)")
    axl.set_title("DeepSeek-V4-Flash\n(downstream AVG)", fontsize=8)
    axl.invert_xaxis()
    axr.set_xlabel("GLM-5.2 Δ PPL (lower better)")
    axr.set_title("GLM-5.2 transfer\n(held-out PPL)", fontsize=8)
    axr.margins(x=0.18)
    fig.suptitle("Structural sparse-FP4 mechanism transfers to GLM-5.2: down + route-tail safe, "
                 "gate/up expensive", fontsize=8.5, y=1.02)
    save(fig, "fig_glm_transfer")


def fig7_designspace():
    rows = [
        ("eager sparse win", "diagnostic only", OK["yellow"]),
        ("graph split-K decode", "REAL win (+2.2..9.7%)", OK["green"]),
        ("phase-adaptive dense-prefill", "refuted (no free Pareto)", OK["red"]),
        ("token-parallel decode", "refuted (~190x slower)", OK["red"]),
        ("distillation repair", "PPL 10.27->9.10, capability limited", OK["orange"]),
    ]
    fig, ax = plt.subplots(figsize=(6.4, 2.6))
    y = np.arange(len(rows))[::-1]
    for yi, (name, verdict, c) in zip(y, rows):
        ax.barh(yi, 1, color=c, alpha=0.85, edgecolor="white")
        ax.text(0.02, yi, f"  {name}: {verdict}", va="center", ha="left", fontsize=8.5)
    ax.set_xlim(0, 1); ax.set_yticks([]); ax.set_xticks([]); ax.spines["left"].set_visible(False); ax.spines["bottom"].set_visible(False)
    ax.set_title("Fig 7. Design-space: what was tried, what survived", fontsize=9.5)
    save(fig, "fig7_designspace")


def _box(ax, xy, w, h, text, fc, tc="black", fs=7.5):
    from matplotlib.patches import FancyBboxPatch
    ax.add_patch(FancyBboxPatch(xy, w, h, boxstyle="round,pad=0.008,rounding_size=0.02",
                                fc=fc, ec="#333", lw=0.8))
    ax.text(xy[0] + w / 2, xy[1] + h / 2, text, ha="center", va="center", fontsize=fs, color=tc, wrap=True)


def _arrow(ax, x0y0, x1y1):
    ax.annotate("", xy=x1y1, xytext=x0y0, arrowprops=dict(arrowstyle="-|>", color="#333", lw=1.0))


def fig1_overview():
    fig, ax = plt.subplots(figsize=(6.8, 3.1)); ax.set_xlim(0, 1); ax.set_ylim(0, 1); ax.axis("off")
    ax.text(0.5, 0.96, "Fig 1. System overview: dense NVFP4 baseline vs quadbit sparse-FP4 path",
            ha="center", fontsize=9.5)
    # baseline row
    ax.text(0.02, 0.78, "dense\nNVFP4\n(baseline)", fontsize=7.5, va="center", color=OK["grey"])
    _box(ax, (0.16, 0.70), 0.20, 0.16, "NVFP4 weights\n(per-16 two-level)", "#eee")
    _box(ax, (0.42, 0.70), 0.20, 0.16, "dense FP4\nGEMM / FusedMoE", "#eee")
    _box(ax, (0.68, 0.70), 0.28, 0.16, "vLLM graph capture\n(FlashInfer CUTLASS)", "#eee")
    _arrow(ax, (0.36, 0.78), (0.42, 0.78)); _arrow(ax, (0.62, 0.78), (0.68, 0.78))
    # quadbit row
    ax.text(0.02, 0.34, "quadbit\nsparse FP4", fontsize=7.5, va="center", color=OK["blue"])
    _box(ax, (0.16, 0.24), 0.20, 0.18, "2:4-sparse FP4\n+ two-level scale", OK["sky"])
    _box(ax, (0.42, 0.24), 0.20, 0.18, "fused sparse MLP /\nsegmented expert GEMM", OK["sky"])
    _box(ax, (0.68, 0.32), 0.28, 0.11, "split-K decode down", OK["yellow"])
    _box(ax, (0.68, 0.19), 0.28, 0.11, "vLLM graph custom-op\n(staged .so)", OK["sky"])
    _arrow(ax, (0.36, 0.33), (0.42, 0.33)); _arrow(ax, (0.62, 0.33), (0.68, 0.37)); _arrow(ax, (0.62, 0.31), (0.68, 0.25))
    ax.text(0.5, 0.06, "attention / router / embeddings stay dense NVFP4 (the ecosystem baseline); "
            "quadbit sparsifies the MLP / experts (~91% of params)", ha="center", fontsize=7, color="#555")
    save(fig, "fig1_overview")


def fig2_dataflow():
    fig, ax = plt.subplots(figsize=(6.8, 2.4)); ax.set_xlim(0, 1); ax.set_ylim(0, 1); ax.axis("off")
    ax.text(0.5, 0.9, "Fig 2. Sparse MLP / expert dataflow (one graph-capturable launch chain)", ha="center", fontsize=9.5)
    stages = [("X", "#eee"), ("act quant\nNVFP4", OK["sky"]), ("gate/up\nsparse GEMM", OK["blue"]),
              ("fused SwiGLU\n+ quant", OK["green"]), ("down\nsparse GEMM", OK["blue"]), ("combine\n(+shared)", OK["orange"])]
    n = len(stages); w = 0.13; gap = (1 - 0.04 - n * w) / (n - 1); x = 0.02
    for i, (t, c) in enumerate(stages):
        _box(ax, (x, 0.38), w, 0.26, t, c, fs=7)
        if i < n - 1:
            _arrow(ax, (x + w, 0.51), (x + w + gap, 0.51))
        x += w + gap
    ax.text(0.5, 0.12, "MoE: routed rows sorted by expert; each column-block carries its expert id, "
            "indexing stacked 2:4-sparse weights (routed-row scheduling, not per-expert launches)",
            ha="center", fontsize=7, color="#555")
    save(fig, "fig2_dataflow")


def fig6_dist_scaling():
    rows = read_csv(DATA / "dist_scaling.csv")
    world = [int(r["world"]) for r in rows]
    kernel = [float(r["expert_kernel_ms"]) for r in rows]
    comm = [float(r["all_reduce_comm_ms"]) for r in rows]
    speed = [float(r["kernel_speedup_vs_1gpu"]) for r in rows]
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(6.8, 2.9))
    x = np.arange(len(world))
    ax1.bar(x - 0.2, kernel, 0.4, label="expert kernel", color=OK["blue"])
    ax1.bar(x + 0.2, comm, 0.4, label="all-reduce (PCIe)", color=OK["orange"])
    ax1.set_yscale("log"); ax1.set_xticks(x); ax1.set_xticklabels([f"{w} GPU" for w in world])
    ax1.set_ylabel("ms (log)"); ax1.set_title("Compute vs communication", fontsize=9)
    ax1.legend(fontsize=7.5, frameon=False)
    ax2.plot(world, speed, "-o", color=OK["green"], label="measured")
    ax2.plot(world, world, "--", color=OK["grey"], lw=0.9, label="ideal linear")
    ax2.set_xticks(world); ax2.set_xlabel("GPUs"); ax2.set_ylabel("expert-kernel speedup")
    ax2.set_title("Expert-parallel scaling", fontsize=9); ax2.legend(fontsize=7.5, frameon=False)
    for xi, s in zip(world, speed):
        ax2.annotate(f"{s:.2f}x", (xi, s), textcoords="offset points", xytext=(4, -10), fontsize=8)
    fig.suptitle("Fig 6. Distributed sparse-FP4 MoE (DeepSeek-V4-Flash, expert-parallel, no NVLink)", fontsize=9.5)
    save(fig, "fig6_dist_scaling")


if __name__ == "__main__":
    print("# building banked paper figures", flush=True)
    fig1_overview()
    fig2_dataflow()
    fig3_decode_win()
    fig4_crossover_heatmap()
    fig5_pareto()
    fig6_dist_scaling()
    fig7_designspace()
    fig_ds_pareto()
    fig_ds_designspace()
    fig_glm_transfer()
    print("# done", flush=True)
