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
    fig3_decode_win()
    fig4_crossover_heatmap()
    fig5_pareto()
    fig6_dist_scaling()
    fig7_designspace()
    print("# done", flush=True)
