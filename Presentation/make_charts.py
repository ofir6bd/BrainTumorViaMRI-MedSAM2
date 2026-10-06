"""Charts for the class presentation, drawn from the saved run results (no GPU).

Run from the repo root:
    .venv\\Scripts\\python.exe Presentation\\make_charts.py
Writes PNGs to Presentation/images/.
"""
import json
import os

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
OUT = os.path.join(ROOT, "Presentation", "images")
MS = os.path.join(ROOT, "MedSAM2_Finetune", "runs")
YO = os.path.join(ROOT, "YOLO_finetune", "runs")
BEST_MS, BEST_YOLO = "20260930-234255", "20260929-074550"

BG, PANEL, TEXT, MUTED, GRID = "#0f1220", "#171b2e", "#e8eaf2", "#9aa0b8", "#2a2f47"
YOLO_C, MS_C, GOOD, BAD = "#3987e5", "#e06a2e", "#2fb35c", "#e0534c"
plt.rcParams.update({"figure.facecolor": BG, "axes.facecolor": BG, "axes.edgecolor": GRID, "axes.labelcolor": TEXT,
                     "xtick.color": MUTED, "ytick.color": MUTED, "text.color": TEXT, "font.size": 13,
                     "axes.spines.top": False, "axes.spines.right": False, "font.family": "DejaVu Sans"})


def dice(i, p, g):
    return 1.0 if p + g == 0 else 2 * i / (p + g)


def summary(run, split):
    return json.load(open(os.path.join(MS, run, "eval", "summary.json")))[split]


def final_comparison():
    """Test 3D Dice: YOLO's own best setup, the plain YOLO hint, and MedSAM2 — same 324 patients."""
    y = json.load(open(os.path.join(YO, BEST_YOLO, "eval", "test.json")))
    k = y["thresholds"].index(0.05)
    yolo_best = {p["id"]: dice(*p["sweep"][k]) for p in y["patients"]}
    ms = {p["id"]: p for p in summary(BEST_MS, "test")["patient_scores"]}
    ids = sorted(set(yolo_best) & set(ms))
    vals = [np.mean([ms[i]["yolo_dice3d"] for i in ids]), np.mean([yolo_best[i] for i in ids]),
            np.mean([ms[i]["dice3d"] for i in ids])]
    labels = ["YOLO alone\n(conf 0.25)", "YOLO best\n(conf 0.05 + mirror\n+ speck filter)", "YOLO → MedSAM2\n(ours)"]
    fig, ax = plt.subplots(figsize=(7.5, 4.6))
    bars = ax.bar(labels, vals, color=[YOLO_C, YOLO_C, MS_C], width=0.6)
    bars[0].set_alpha(0.55)
    for b, v in zip(bars, vals):
        ax.text(b.get_x() + b.get_width() / 2, v + 0.001, f"{v:.4f}", ha="center", va="bottom", fontsize=15, fontweight="bold")
    ax.set_ylim(0.86, 0.905)
    ax.set_ylabel("Mean 3D Dice (test, 324 patients)")
    ax.grid(axis="y", color=GRID)
    ax.set_axisbelow(True)
    fig.tight_layout()
    fig.savefig(os.path.join(OUT, "chart_final_comparison.png"), dpi=180)
    plt.close(fig)


def experiments():
    """Val 3D Dice of the settings experiments on the current pools (same YOLO hints)."""
    runs = [("20260930-183638", "Base: decoder+prompt+memory"), ("20260930-122551", "Train decoder+prompt only"),
            ("20260929-142123", "Also train image encoder"), ("20261001-081623", "Quieter hint (logit_scale 4)"),
            ("20260930-234255", "Firmer 'nothing here' (−20)")]
    vals = [summary(r, "val")["dice3d_mean"] for r, _ in runs]
    base = vals[0]
    fig, ax = plt.subplots(figsize=(8.5, 4.4))
    y = np.arange(len(runs))[::-1]
    # green = the only change a paired per-patient test called real (Wilcoxon p = 0.001); the rest are noise-level
    cols = [GOOD if r == "20260930-234255" else MUTED for r, _ in runs]
    ax.barh(y, vals, color=cols, height=0.6)
    for yy, v in zip(y, vals):
        d = v - base
        ax.text(v + 0.0002, yy, f"{v:.4f}" + ("" if abs(d) < 1e-9 else f"  ({d:+.4f})"), va="center", fontsize=12)
    ax.set_yticks(y, [l for _, l in runs])
    ax.set_xlim(0.888, 0.8955)
    ax.set_xlabel("Mean 3D Dice on val (245 patients)")
    ax.grid(axis="x", color=GRID)
    ax.set_axisbelow(True)
    fig.tight_layout()
    fig.savefig(os.path.join(OUT, "chart_experiments.png"), dpi=180)
    plt.close(fig)


def error_breakdown():
    """Where MedSAM2's remaining wrong voxels are (val, best run)."""
    pts = json.load(open(os.path.join(MS, BEST_MS, "eval", "val.json")))["patients"]
    t = dict(fp_edge=0, fn_edge=0, fp_empty=0, fn_missed=0)
    for r in pts:
        g, p, i = map(np.array, (r["gt"], r["pred"], r["inter"]))
        fp, fn = p - i, g - i
        t["fp_edge"] += fp[g > 0].sum(); t["fn_edge"] += fn[p > 0].sum()
        t["fp_empty"] += fp[g == 0].sum(); t["fn_missed"] += fn[p == 0].sum()
    tot = sum(t.values())
    labels = ["Too much at the tumour edge", "Too little at the tumour edge", "False alarm on a tumour-free slice",
              "Whole tumour slice missed"]
    vals = [t[k] / tot for k in ("fp_edge", "fn_edge", "fp_empty", "fn_missed")]
    fig, ax = plt.subplots(figsize=(8.5, 3.6))
    y = np.arange(4)[::-1]
    ax.barh(y, vals, color=[MS_C, MS_C, MUTED, MUTED], height=0.6)
    for yy, v in zip(y, vals):
        ax.text(v + 0.01, yy, f"{v:.0%}", va="center", fontsize=14, fontweight="bold")
    ax.set_yticks(y, labels)
    ax.set_xlim(0, 0.7)
    ax.xaxis.set_major_formatter(matplotlib.ticker.PercentFormatter(1.0))
    ax.set_xlabel("Share of all wrong voxels (val, 245 patients)")
    ax.grid(axis="x", color=GRID)
    ax.set_axisbelow(True)
    fig.tight_layout()
    fig.savefig(os.path.join(OUT, "chart_errors.png"), dpi=180)
    plt.close(fig)


def side_by_size():
    """Test MedSAM2 3D Dice by side of the brain (rows) and tumour size (columns), with patient counts."""
    stats = json.load(open(os.path.join(ROOT, "outputs", "dashboard", "dataset_stats.json")))["records"]
    side = lambda f: None if f is None else ("Left" if f > 2 / 3 else "Right" if f < 1 / 3 else "Both")
    sizes = [(0, 10), (10, 30), (30, 60), (60, 100), (100, np.inf)]
    rows = ["Left", "Right", "Both"]
    pts = summary(BEST_MS, "test")["patient_scores"]
    V = np.full((3, 5), np.nan); N = np.zeros((3, 5), int)
    for i, r in enumerate(rows):
        for j, (a, b) in enumerate(sizes):
            c = [p["dice3d"] for p in pts if side(stats.get(p["id"], {}).get("left_frac")) == r and a <= p["gt_total"] / 1000 < b]
            N[i, j] = len(c)
            if c:
                V[i, j] = np.mean(c)
    fig, ax = plt.subplots(figsize=(8.2, 3.9))
    cmap = matplotlib.colors.LinearSegmentedColormap.from_list("ms", ["#3a2418", MS_C])
    ax.imshow(np.ma.masked_invalid(V), cmap=cmap, vmin=0.4, vmax=0.95, aspect="auto")
    for i in range(3):
        for j in range(5):
            if N[i, j]:
                ax.text(j, i - 0.12, f"{V[i, j]:.3f}", ha="center", va="center", fontsize=15, fontweight="bold", color="white")
                ax.text(j, i + 0.22, f"n = {N[i, j]}", ha="center", va="center", fontsize=11, color="#f3d9cc")
            else:
                ax.text(j, i, "no patients", ha="center", va="center", fontsize=11, color=MUTED)
    ax.set_xticks(range(5), ["0–10 mL", "10–30 mL", "30–60 mL", "60–100 mL", "≥ 100 mL"])
    ax.set_yticks(range(3), [f"{r} (n={N[i].sum()})" for i, r in enumerate(rows)])
    ax.tick_params(length=0)
    for sp in ax.spines.values():
        sp.set_visible(False)
    ax.set_xticks(np.arange(-.5, 5, 1), minor=True); ax.set_yticks(np.arange(-.5, 3, 1), minor=True)
    ax.grid(which="minor", color=BG, linewidth=3)
    ax.set_xlabel("Tumour size (expert whole tumour)")
    fig.tight_layout()
    fig.savefig(os.path.join(OUT, "chart_side_size.png"), dpi=180)
    plt.close(fig)


if __name__ == "__main__":
    os.makedirs(OUT, exist_ok=True)
    final_comparison()
    experiments()
    error_breakdown()
    side_by_size()
    print("charts written to", OUT)
