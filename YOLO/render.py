"""Matplotlib rendering of the YOLO segmentation overlay (README.md Stage F).

Renders the selected slice as a 4-panel comparison (input / expert GT / YOLO prediction /
overlap), plus a stats + legend footer (Dice, instance counts, confidence). Served by
`routes.py` at `/yolo/segment.png`.
"""
import io

import matplotlib
import numpy as np

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
from matplotlib.patches import Patch  # noqa: E402

BG = "#141428"
PANEL_BG = "#1b1b36"
FG = "white"
MUTED = "#9aa0b4"
GREEN = (0.20, 0.85, 0.35)
RED = (1.0, 0.25, 0.25)
YELLOW = (1.0, 0.85, 0.15)

_LEGEND_ENTRIES = [
    ("Expert GT only", (*GREEN, 0.9)),
    ("YOLO prediction only", (*RED, 0.9)),
    ("Overlap (GT \u2229 prediction)", (*YELLOW, 0.9)),
]


def _to_png(fig):
    buf = io.BytesIO()
    fig.savefig(buf, format="png", bbox_inches="tight", facecolor=fig.get_facecolor())
    plt.close(fig)
    buf.seek(0)
    return buf


def _style(ax, title, subtitle=None):
    ax.set_facecolor(PANEL_BG)
    ax.text(0.5, 1.16, title, transform=ax.transAxes, ha="center", va="bottom",
            color=FG, fontsize=12, fontweight="bold")
    if subtitle:
        ax.text(0.5, 1.045, subtitle, transform=ax.transAxes, ha="center", va="bottom",
                color=MUTED, fontsize=9.5)
    ax.axis("off")


def _dice_color(dice):
    if dice >= 0.7:
        return "#3ddc84"
    if dice >= 0.4:
        return "#ffcc00"
    return "#ff5a5a"


def render(pipeline, z, slice_no=None):
    pl = pipeline
    slice_label = f"slice #{slice_no}" if slice_no is not None else f"z={z}"
    header = f"YOLO Tumour Segmentation  \u2014  {pl.patient_id}  \u2014  {slice_label} (z={z})"

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
        fig.suptitle(header, color=FG, fontsize=14, fontweight="bold", y=1.02)
        return _to_png(fig)

    s = pl.process_slice(z)
    rgb = s["rgb"]
    gt_wt, pred_wt = s["gt_wt"], s["pred_wt"]

    fig, axes = plt.subplots(1, 4, figsize=(19, 5.6), facecolor=BG)
    ax_in, ax_gt, ax_pred, ax_cmp = axes

    # Panel 1: raw model input --------------------------------------------------
    ax_in.imshow(rgb)
    _style(ax_in, "Input (RGB)", "R=T1C\u2212T1  G=T2  B=FLAIR")

    # Panel 2: expert ground truth ----------------------------------------------
    ax_gt.imshow(rgb)
    ov_gt = np.zeros((*gt_wt.shape, 4))
    ov_gt[gt_wt] = (*GREEN, 0.45)
    ax_gt.imshow(ov_gt)
    ax_gt.contour(gt_wt, levels=[0.5], colors=[GREEN], linewidths=1.4)
    _style(ax_gt, "Expert Ground Truth", f"{s['num_gt_components']} region(s)")

    # Panel 3: YOLO prediction ----------------------------------------------------
    ax_pred.imshow(rgb)
    ov_pred = np.zeros((*pred_wt.shape, 4))
    ov_pred[pred_wt] = (*RED, 0.5)
    ax_pred.imshow(ov_pred)
    if pred_wt.any():
        ax_pred.contour(pred_wt, levels=[0.5], colors=[RED], linewidths=1.4)
    conf_txt = f"conf\u2248{s['mean_conf']:.2f}" if s["mean_conf"] is not None else "no detections"
    _style(ax_pred, "YOLO Prediction", f"{s['num_pred_instances']} instance(s), {conf_txt}")

    # Panel 4: comparison / overlap -----------------------------------------------
    ax_cmp.imshow(rgb)
    ov_cmp = np.zeros((*gt_wt.shape, 4))
    ov_cmp[gt_wt] = (*GREEN, 0.35)
    ov_cmp[pred_wt] = (*RED, 0.4)
    ov_cmp[gt_wt & pred_wt] = (*YELLOW, 0.6)
    ax_cmp.imshow(ov_cmp)
    dice = s["dice"]
    _style(ax_cmp, "Comparison", f"Dice = {dice:.3f}")
    ax_cmp.text(
        0.5, -0.10, f"{dice:.3f}", transform=ax_cmp.transAxes, ha="center", va="top",
        color=_dice_color(dice), fontsize=20, fontweight="bold",
    )

    if s["model_error"]:
        fig.text(0.5, 0.965, f"\u26a0 no model loaded ({s['model_error']}) \u2014 showing GT only",
                  ha="center", color="#ffcc00", fontsize=10, fontweight="bold")

    fig.suptitle(header, color=FG, fontsize=15, fontweight="bold", y=1.06)

    handles = [Patch(facecolor=color, edgecolor="none", label=label)
               for label, color in _LEGEND_ENTRIES]
    leg = fig.legend(
        handles=handles, loc="lower center", bbox_to_anchor=(0.5, -0.04),
        ncol=3, frameon=True, facecolor=PANEL_BG, edgecolor="#555",
        fontsize=10.5, labelcolor=FG,
    )
    leg.get_frame().set_alpha(0.95)

    fig.tight_layout(rect=[0, 0.02, 1, 0.98])
    return _to_png(fig)
