"""Pictures for the 'normalization' slide: raw MRI values -> 0-255 per slice -> one colour picture.

Run from the repo root:
    .venv\\Scripts\\python.exe Presentation\\make_normalization.py
Uses the project's own rgb_slice() (YOLO_finetune/common.py), so the result is exactly what
YOLO and MedSAM2 receive. Writes Presentation/images/norm_*.png.
"""
import glob
import os
import sys

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import nibabel as nib
import numpy as np

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, ROOT)
from YOLO_finetune.common import rgb_slice  # noqa: E402

OUT = os.path.join(ROOT, "Presentation", "images")
PATIENT, Z = "BraTS-GLI-00046-101", 115
BG, TEXT, MUTED, GRID = "#0f1220", "#e8eaf2", "#9aa0b8", "#2a2f47"
SEQ = [("t1c", "T1C", "#e3413c"), ("t2w", "T2", "#3cc43c"), ("t2f", "FLAIR", "#4a6cff")]
plt.rcParams.update({"figure.facecolor": BG, "axes.facecolor": BG, "axes.edgecolor": GRID, "axes.labelcolor": TEXT,
                     "xtick.color": MUTED, "ytick.color": MUTED, "text.color": TEXT, "font.size": 11,
                     "axes.spines.top": False, "axes.spines.right": False, "font.family": "DejaVu Sans"})


def main():
    d = glob.glob(os.path.join(ROOT, "datasets", "BraTS2024", "*", PATIENT))[0]
    load = lambda k: np.asarray(nib.load(glob.glob(os.path.join(d, f"*{k}.nii*"))[0]).dataobj, dtype=np.float32)
    vols = {"T1C": load("t1c"), "T2": load("t2w"), "FLAIR": load("t2f")}
    rgb = rgb_slice(vols, Z, 100)                               # the project's own function
    rot = np.rot90

    # raw vs normalised, one row per sequence, + the final colour picture
    fig = plt.figure(figsize=(11, 6.2))
    gs = fig.add_gridspec(3, 3, width_ratios=[1, 1, 1.9], wspace=0.03, hspace=0.12)
    vmax = max(float(vols[n][:, :, Z].max()) for _, n, _ in SEQ)   # one shared raw scale shows the mismatch
    for i, (key, name, col) in enumerate(SEQ):
        raw = vols[name][:, :, Z]
        nz = raw[raw != 0]
        ax = fig.add_subplot(gs[i, 0]); ax.imshow(rot(raw), cmap="gray", vmin=0, vmax=vmax); ax.axis("off")
        ax.text(0.02, 0.97, f"{name} raw\n{nz.min():,.0f} – {nz.max():,.0f}", transform=ax.transAxes, va="top",
                fontsize=10, color=TEXT, bbox=dict(fc=BG, ec="none", alpha=0.7, pad=2))
        ax = fig.add_subplot(gs[i, 1]); ax.imshow(rot(rgb[:, :, i]), cmap="gray", vmin=0, vmax=255); ax.axis("off")
        ax.text(0.02, 0.97, f"{name} → 0–255", transform=ax.transAxes, va="top", fontsize=10, color=col,
                fontweight="bold", bbox=dict(fc=BG, ec="none", alpha=0.7, pad=2))
    ax = fig.add_subplot(gs[:, 2]); ax.imshow(rot(rgb)); ax.axis("off")
    ax.set_title("One colour picture: red = T1C · green = T2 · blue = FLAIR", fontsize=11, color=TEXT)
    fig.savefig(os.path.join(OUT, "norm_steps.png"), dpi=170, bbox_inches="tight", pad_inches=0.05)
    plt.close(fig)

    # FLAIR histogram with the two percentile cut lines
    raw = vols["FLAIR"][:, :, Z]; nz = raw[raw != 0]
    lo, hi = np.percentile(nz, [0.5, 99.5])
    fig, ax = plt.subplots(figsize=(6.2, 2.3))
    ax.hist(nz, bins=80, color="#4a6cff")
    for x, lab in ((lo, f"0.5th pct = {lo:,.0f} → 0"), (hi, f"99.5th pct = {hi:,.0f} → 255")):
        ax.axvline(x, color="#f0a030", lw=1.8, ls="--")
        ax.text(x, ax.get_ylim()[1] * 0.92, " " + lab, color="#f0a030", fontsize=10, va="top",
                ha="left" if x == lo else "right")
    ax.set_xlabel("FLAIR raw value (brain voxels of this slice)"); ax.set_ylabel("Voxels")
    ax.grid(axis="y", color=GRID); ax.set_axisbelow(True)
    fig.savefig(os.path.join(OUT, "norm_hist.png"), dpi=170, bbox_inches="tight", pad_inches=0.05)
    plt.close(fig)
    print("written norm_steps.png, norm_hist.png; FLAIR cuts", round(lo), round(hi))


if __name__ == "__main__":
    main()
