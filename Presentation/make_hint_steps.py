"""How YOLO's output becomes MedSAM2's hint, for one patient (the 'hint, step by step' slide).

Run from the repo root:
    .venv\\Scripts\\python.exe Presentation\\make_hint_steps.py
Uses the saved hint cache and the project's own prompt_logits() with the best run's settings,
so the pictures are exactly what MedSAM2 was given. Writes Presentation/images/hint_*.png.
"""
import os
import sys

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import yaml

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, ROOT)
from MedSAM2_Finetune.yolo_prompts import prompt_logits  # noqa: E402

OUT = os.path.join(ROOT, "Presentation", "images")
PATIENT, Z = "BraTS-GLI-00046-101", 115
MS_RUN, PROMPT_KEY = "20260930-234255", "980735f3de"
BG, TEXT, MUTED, GRID, PINK = "#0f1220", "#e8eaf2", "#9aa0b8", "#2a2f47", "#d6619b"
plt.rcParams.update({"figure.facecolor": BG, "axes.facecolor": BG, "axes.edgecolor": GRID, "axes.labelcolor": TEXT,
                     "xtick.color": MUTED, "ytick.color": MUTED, "text.color": TEXT, "font.size": 12,
                     "axes.spines.top": False, "axes.spines.right": False, "font.family": "DejaVu Sans"})


def main():
    e = dict(np.load(os.path.join(ROOT, "MedSAM2_Finetune", "cache", PROMPT_KEY, f"{PATIENT}.npz")))
    cfg = yaml.safe_load(open(os.path.join(ROOT, "MedSAM2_Finetune", "runs", MS_RUN, "run_config.yaml")))
    k = list(e["z"]).index(Z)
    u8 = e["prob"][k]
    head = e["rgb"][k].max(-1) > 0
    scale, empty = float(cfg["prompt"]["logit_scale"]), float(cfg["prompt"]["empty_logit"])
    p = np.clip(u8.astype(np.float32) / 255, 1 / 255, 254 / 255)
    logit8 = np.log(p / (1 - p)) * scale                         # steps B + C, before the overrides
    final = prompt_logits(e, cfg, index=[k])[0]                  # step D: the project's own function
    rot = np.rot90
    flair = rot(e["rgb"][k][:, :, 2])

    fig, axes = plt.subplots(1, 3, figsize=(12, 5.0))
    div = matplotlib.colors.LinearSegmentedColormap.from_list("hint", ["#1d3f8f", "#5b8ad6", "#f2f2f2", "#e06a2e", "#a3260f"])
    # A: stored probability over the FLAIR picture
    ax = axes[0]
    ax.imshow(flair, cmap="gray")
    a = np.ma.masked_where(rot(u8) == 0, rot(u8))
    im = ax.imshow(a, cmap="magma", vmin=0, vmax=255)
    cb = fig.colorbar(im, ax=ax, fraction=0.046, pad=0.02); cb.set_label("stored value (0–255)")
    ax.set_title("A · stored probability", fontsize=14)
    # B + C: logit x scale (same picture, two scales)
    ax = axes[1]
    im = ax.imshow(rot(logit8), cmap=div, vmin=-44, vmax=44)
    cb = fig.colorbar(im, ax=ax, fraction=0.046, pad=0.02, ticks=[-44, -22, 0, 22, 44])
    cb.ax.set_yticklabels([f"{t:+d}  ({t / scale:+.1f})" if t else "0  (0)" for t in (-44, -22, 0, 22, 44)])
    cb.set_label(f"logit × {scale:g}   (logit)")
    ax.set_title(f"B + C · logit, × {scale:g}", fontsize=14)
    # D: after the 'nothing here' overrides
    ax = axes[2]
    im = ax.imshow(rot(final), cmap=div, vmin=-44, vmax=44)
    ax.contour(rot(head.astype(float)), levels=[0.5], colors="white", linewidths=1.0)
    cb = fig.colorbar(im, ax=ax, fraction=0.046, pad=0.02, ticks=[-44, empty, 0, 44]); cb.set_label("value given to MedSAM2")
    ax.set_title(f"D · outside the head = {empty:g}", fontsize=14)
    for ax in axes:
        ax.axis("off")
    fig.tight_layout()
    fig.savefig(os.path.join(OUT, "hint_steps.png"), dpi=170, bbox_inches="tight", pad_inches=0.08)
    plt.close(fig)

    # whole-patient histogram of every non-zero probability YOLO produced
    v = e["prob"][e["prob"] > 0] / 255
    heads = e["rgb"].max(-1) > 0
    zero_in_head = int(((e["prob"] == 0) & heads).sum())
    h, edges = np.histogram(v, bins=50, range=(0, 1))
    fig, ax = plt.subplots(figsize=(7.0, 2.3))
    ax.bar(edges[:-1], h, width=0.018, align="edge", color=PINK)
    ax.text(edges[0] + 0.03, h[0], f"{h[0]:,} unsure (< 0.02)", ha="left", va="center", fontsize=11, color=TEXT)
    ax.text(edges[49] - 0.012, h[49], f"{h[49]:,} sure (> 0.98)", ha="right", va="top", fontsize=11, color=TEXT)
    ax.set_xlim(0, 1); ax.set_xlabel("YOLO's probability per pixel")
    ax.set_ylabel("Pixels")
    ax.yaxis.set_major_formatter(matplotlib.ticker.FuncFormatter(lambda x, _: f"{x / 1000:.0f}k" if x else "0"))
    ax.grid(axis="y", color=GRID); ax.set_axisbelow(True)
    fig.tight_layout()
    fig.savefig(os.path.join(OUT, "hint_hist.png"), dpi=170, bbox_inches="tight", pad_inches=0.05)
    plt.close(fig)

    empty_slices = int((e["prob"].max(axis=(1, 2)) == 0).sum())
    print(f"slices {len(e['z'])}, no blob {empty_slices}; non-zero px {len(v):,}; zero px inside head {zero_in_head:,}; "
          f"sure (>0.98) {h[-1]:,}; z{Z}: {int((u8 > 0).sum())} px > 0")


if __name__ == "__main__":
    main()
