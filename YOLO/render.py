"""Matplotlib rendering of the YOLO segmentation overlay (README.md Stage F).

Renders the selected slice's RGB frame three times side by side — expert ground-truth
mask, YOLO prediction, and the combined overlay (GT/prediction/overlap) — each labeled,
plus the per-slice Dice score. Served by `routes.py` at `/yolo/segment.png`.

The RGB frame's own green/blue channels (T2/FLAIR) can visually clash with a green mask
fill, so overlays here are drawn as a bright outline + light fill on top of a dimmed copy
of the base image, with a colour legend on every panel so it's unambiguous which colour
means what.
"""
import io

import matplotlib
import matplotlib.patches as mpatches
import numpy as np
from scipy import ndimage

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402

BG = "#141428"
FG = "white"

GT_RGB = (0.15, 1.0, 0.15)      # expert GT: bright green
PRED_RGB = (1.0, 0.15, 0.15)    # predicted: bright red
OVERLAP_RGB = (1.0, 0.95, 0.1)  # overlap: bright yellow

FILL_ALPHA = 0.30
DIM_FACTOR = 0.55  # base image darkened so overlay colours stand out


def _to_png(fig):
    buf = io.BytesIO()
    fig.savefig(buf, format="png", bbox_inches="tight", facecolor=fig.get_facecolor())
    plt.close(fig)
    buf.seek(0)
    return buf


def _style(ax, title):
    ax.set_title(title, color=FG, fontsize=12, fontweight="bold")
    ax.axis("off")


def _dimmed(rgb):
    """Darken the base RGB frame so overlaid mask colours remain visible on top of it."""
    return (rgb.astype(np.float32) / 255.0) * DIM_FACTOR


def _mask_outline(mask):
    """1px-thick boundary of a boolean mask (mask minus its binary erosion)."""
    if not mask.any():
        return np.zeros_like(mask)
    eroded = ndimage.binary_erosion(mask, border_value=0)
    return mask & ~eroded


def _paint(base, mask, rgb_color, fill_alpha=FILL_ALPHA):
    """Blend `rgb_color` fill (alpha) + a solid outline of `mask` onto `base` (HxWx3, 0-1)."""
    out = base.copy()
    if mask.any():
        color = np.array(rgb_color, dtype=np.float32)
        out[mask] = out[mask] * (1 - fill_alpha) + color * fill_alpha
        out[_mask_outline(mask)] = color
    return out


def _legend(ax, entries):
    """`entries`: list of (label, rgb_color). Small colour-patch legend under the panel."""
    handles = [mpatches.Patch(color=c, label=lbl) for lbl, c in entries]
    leg = ax.legend(
        handles=handles, loc="upper center", bbox_to_anchor=(0.5, -0.02),
        ncol=len(entries), frameon=False, fontsize=9.5, handlelength=1.2,
        handleheight=1.2, columnspacing=1.2,
    )
    for text in leg.get_texts():
        text.set_color(FG)


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
    gt, pred = s["gt_wt"], s["pred_wt"]
    dim = _dimmed(s["rgb"])

    fig, axes = plt.subplots(1, 3, figsize=(16, 6.6), facecolor=BG)

    # -- Ground truth ---------------------------------------------------
    ax_gt = axes[0]
    ax_gt.imshow(_paint(dim, gt, GT_RGB))
    _style(ax_gt, "Expert ground truth")
    _legend(ax_gt, [("Ground truth", GT_RGB)])

    # -- Prediction -------------------------------------------------------
    ax_pred = axes[1]
    ax_pred.imshow(_paint(dim, pred, PRED_RGB))
    pred_title = "YOLO prediction"
    if s["model_error"]:
        pred_title += "  (no model)"
    _style(ax_pred, pred_title)
    _legend(ax_pred, [("Prediction", PRED_RGB)])

    # -- Combined ---------------------------------------------------------
    ax_comb = axes[2]
    combined = _paint(dim, gt, GT_RGB)
    combined = _paint(combined, pred, PRED_RGB)
    combined = _paint(combined, gt & pred, OVERLAP_RGB, fill_alpha=0.5)
    ax_comb.imshow(combined)
    _style(ax_comb, f"Combined  -  Dice={s['dice']:.3f}")
    _legend(ax_comb, [
        ("Ground truth", GT_RGB), ("Prediction", PRED_RGB), ("Overlap", OVERLAP_RGB),
    ])

    fig.suptitle(f"YOLO segmentation  -  {slice_label}", color=FG, fontsize=14)
    fig.tight_layout(rect=(0, 0, 1, 0.94))
    return _to_png(fig)
