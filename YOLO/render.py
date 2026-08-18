"""Matplotlib rendering of the YOLO segmentation overlay (README.md Stage F).

Renders the selected slice's RGB frame with predicted vs. expert whole-tumour masks
overlaid, plus the per-slice Dice score. A single view (no multi-step stack), served by
`routes.py` at `/yolo/segment.png`.
"""
import io

import matplotlib
import numpy as np

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402

BG = "#141428"
FG = "white"


def _to_png(fig):
    buf = io.BytesIO()
    fig.savefig(buf, format="png", bbox_inches="tight", facecolor=fig.get_facecolor())
    plt.close(fig)
    buf.seek(0)
    return buf


def _style(ax, title):
    ax.set_title(title, color=FG, fontsize=12, fontweight="bold")
    ax.axis("off")


def render(pipeline, z, slice_no=None):
    pl = pipeline
    slice_label = f"slice #{slice_no}" if slice_no is not None else f"z={z}"

    if not pl.is_processed(z):
        fig, ax = plt.subplots(1, 1, figsize=(6, 6), facecolor=BG)
        ax.imshow(pl.flair[:, :, z], cmap="gray")
        ax.text(
            0.5, 0.06,
            f"{slice_label} skipped: brain {pl.brain_count(z)} px "
            f"< min_fg_voxels ({pl.p['min_fg_voxels']})",
            transform=ax.transAxes, ha="center", color="#ffcc00",
            fontsize=11, fontweight="bold",
            bbox=dict(facecolor="black", alpha=0.6, pad=4),
        )
        _style(ax, f"Skipped {slice_label}")
        return _to_png(fig)

    s = pl.process_slice(z)

    fig, ax = plt.subplots(1, 1, figsize=(6.2, 6.2), facecolor=BG)
    ax.imshow(s["rgb"])

    overlay = np.zeros((*s["gt_wt"].shape, 4))
    overlay[s["gt_wt"]] = (0, 1, 0, 0.35)          # expert GT: green
    overlay[s["pred_wt"]] = (1, 0, 0, 0.45)        # predicted: red
    overlay[s["gt_wt"] & s["pred_wt"]] = (1, 1, 0, 0.6)  # overlap: yellow
    ax.imshow(overlay)

    title = f"Whole tumour  -  Dice={s['dice']:.3f}"
    if s["model_error"]:
        title += "  (no model — GT only)"
    _style(ax, title)
    fig.suptitle(f"YOLO segmentation  -  {slice_label}", color=FG, fontsize=14)
    return _to_png(fig)
