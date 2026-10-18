"""What MedSAM2 is handed under the two hint styles, 'max' vs 'max_weighted' (the hint-variants slide).

Run from the repo root:
    .venv\\Scripts\\python.exe Presentation\\make_hint_variants.py
Uses the saved hint cache and the project's own prompt_logits() with the best run's settings, switching only
`prompt.variant`, so every value is exactly what MedSAM2 would be given. Test pool.
Writes Presentation/images/hv_*.png.
"""
import copy
import os
import sys

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import yaml

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, ROOT)
from MedSAM2_Finetune.common import list_patients  # noqa: E402
from MedSAM2_Finetune.yolo_prompts import prompt_logits  # noqa: E402

OUT = os.path.join(ROOT, "Presentation", "images")
PATIENT, Z = "BraTS-GLI-00046-101", 120
MS_RUN, PROMPT_KEY = "20260930-234255", "980735f3de"
BG, TEXT, MUTED, GRID = "#0f1220", "#e8eaf2", "#9aa0b8", "#2a2f47"
MAX_C, W_C = "#3987e5", "#e06a2e"
plt.rcParams.update({"figure.facecolor": BG, "axes.facecolor": BG, "axes.edgecolor": GRID, "axes.labelcolor": TEXT,
                     "xtick.color": MUTED, "ytick.color": MUTED, "text.color": TEXT, "font.size": 12,
                     "axes.spines.top": False, "axes.spines.right": False, "font.family": "DejaVu Sans"})
DIV = matplotlib.colors.LinearSegmentedColormap.from_list("hint", ["#1d3f8f", "#5b8ad6", "#f2f2f2", "#e06a2e", "#a3260f"])


def load(pid):
    return dict(np.load(os.path.join(ROOT, "MedSAM2_Finetune", "cache", PROMPT_KEY, f"{pid}.npz")))


def main():
    cfg = yaml.safe_load(open(os.path.join(ROOT, "MedSAM2_Finetune", "runs", MS_RUN, "run_config.yaml")))
    cfgs = {}
    for v in ("max", "max_weighted"):
        cfgs[v] = copy.deepcopy(cfg); cfgs[v]["prompt"]["variant"] = v

    # 1 ---- one slice, both hints
    e = load(PATIENT)
    k = list(e["z"]).index(Z)
    fig, axes = plt.subplots(1, 2, figsize=(9.2, 4.6))
    for ax, (v, name) in zip(axes, [("max", "max: YOLO's probability"), ("max_weighted", "max_weighted: probability × confidence")]):
        im = ax.imshow(np.rot90(prompt_logits(e, cfgs[v], index=[k])[0]), cmap=DIV, vmin=-44, vmax=44)
        ax.set_title(name, fontsize=13); ax.set_xticks([]); ax.set_yticks([])
    cb = fig.colorbar(im, ax=axes, fraction=0.03, pad=0.02, ticks=[-44, -20, 0, 20, 44])
    cb.set_label("hint value (logit × 8)   > 0 = 'tumour'")
    fig.savefig(os.path.join(OUT, "hv_example.png"), dpi=170, bbox_inches="tight", pad_inches=0.05)
    plt.close(fig)

    # 2 ---- every blob pixel on the test pool
    edges = np.linspace(-45, 45, 91)
    hist = {v: np.zeros(len(edges) - 1) for v in cfgs}
    bins = [0.05, 0.25, 0.5, 0.75, 1.0001]
    pos = {v: np.zeros(len(bins) - 1) for v in cfgs}
    tot = np.zeros(len(bins) - 1)
    pats = list_patients(cfg["data"]["test_pool"])
    for p in pats:
        e = load(p["id"])
        blob = e["prob"] > 0                                             # every pixel some YOLO blob touches
        conf = np.where(blob, e["prob_w"] / np.maximum(e["prob"], 1), 0)  # that blob's confidence
        cb_ = np.digitize(conf[blob], bins) - 1
        tot += np.bincount(cb_.clip(0, len(tot) - 1), minlength=len(tot))
        for v in cfgs:
            h = prompt_logits(e, cfgs[v])[blob]
            hist[v] += np.histogram(h.clip(-44.9, 44.9), edges)[0]
            pos[v] += np.bincount(cb_.clip(0, len(tot) - 1), weights=(h > 0), minlength=len(tot))
    n = hist["max"].sum()
    mid = (edges[:-1] + edges[1:]) / 2
    plt.rcParams["font.size"] = 15                                   # the two charts sit small on the slide
    fig, ax = plt.subplots(figsize=(7.6, 2.75))
    for v, col in (("max", MAX_C), ("max_weighted", W_C)):
        ax.step(mid, hist[v] / n * 100, where="mid", color=col, lw=2, label=v)
        ax.fill_between(mid, hist[v] / n * 100, step="mid", color=col, alpha=0.18)
    ax.axvline(0, color=MUTED, lw=1, ls="--")
    ax.text(1.5, ax.get_ylim()[1] * 0.38, "'tumour' →", color=MUTED, fontsize=14)
    ax.text(-1.5, ax.get_ylim()[1] * 0.38, "← 'not tumour'", color=MUTED, fontsize=14, ha="right")
    ax.set_xlabel("hint value handed to MedSAM2 (logit × 8)"); ax.set_ylabel("% of blob pixels")
    ax.set_xlim(-46, 46); ax.legend(frameon=False, loc="upper left", bbox_to_anchor=(0.06, 1.0))
    fig.savefig(os.path.join(OUT, "hv_hist.png"), dpi=170, bbox_inches="tight", pad_inches=0.05)
    plt.close(fig)

    labels = ["0.05–0.25", "0.25–0.5", "0.5–0.75", "0.75–1"]
    x = np.arange(len(labels))
    fig, ax = plt.subplots(figsize=(7.6, 2.65))
    for j, (v, col) in enumerate((("max", MAX_C), ("max_weighted", W_C))):
        val = pos[v] / tot * 100
        bars = ax.bar(x + (j - 0.5) * 0.38, val, 0.38, color=col, label=v)
        for b_, t in zip(bars, val):
            ax.text(b_.get_x() + b_.get_width() / 2, t + 1.5, f"{t:.0f}%", ha="center", fontsize=14, color=TEXT)
    ax.set_xticks(x, labels); ax.set_ylim(0, 118); ax.set_yticks([0, 25, 50, 75, 100])
    ax.set_xlabel("confidence of the YOLO blob"); ax.set_ylabel("% 'tumour'")
    ax.legend(frameon=False, ncol=2, loc="upper left")
    fig.savefig(os.path.join(OUT, "hv_bins.png"), dpi=170, bbox_inches="tight", pad_inches=0.05)
    plt.close(fig)

    for v in cfgs:
        h = hist[v] / n
        print(f"{v}: blob pixels > 0 = {h[mid > 0].sum()*100:.1f}%  | <= -40 = {h[mid <= -40].sum()*100:.1f}%  | >= 40 = {h[mid >= 40].sum()*100:.1f}%")
    print("blob pixels by conf bin:", dict(zip(labels, (tot / tot.sum() * 100).round(1))))
    for v in cfgs:
        print(v, "share 'tumour' by bin:", dict(zip(labels, (pos[v] / tot * 100).round(1))))
    print(f"{len(pats)} test patients, {int(n):,} blob pixels")


if __name__ == "__main__":
    main()
