"""Matplotlib rendering for the GT-vs-YOLO tab.

Three panels side by side over the same background: the expert ground truth, MedSAM2
prompted with the GT mask, and MedSAM2 prompted with the YOLO mask — each with its
per-slice Dice. The background is the RGB frame MedSAM2 actually saw (R:T1c / G:T2w / B:T2f),
not the YOLO RGB, so what you look at is what the model looked at.

Anchor slices are called out in the title, including whether YOLO had a mask there —
an anchor with no YOLO mask is the single most informative frame in a run. On an anchor,
the prompt mask actually handed to MedSAM2 is outlined in yellow, so you can see the
input and the resulting propagation in the same panel.
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

GT_RGB = (0.15, 1.0, 0.15)       # expert ground truth: bright green
ARM_GT_RGB = (0.25, 0.65, 1.0)   # MedSAM2 from GT mask: blue
ARM_YOLO_RGB = (1.0, 0.35, 0.15) # MedSAM2 from YOLO mask: orange
PROMPT_RGB = (1.0, 0.95, 0.1)    # prompt mask handed to MedSAM2: yellow

FILL_ALPHA = 0.30
DIM_FACTOR = 0.55


def _to_png(fig):
    buf = io.BytesIO()
    fig.savefig(buf, format="png", bbox_inches="tight", facecolor=fig.get_facecolor())
    plt.close(fig)
    buf.seek(0)
    return buf


def _dimmed(rgb):
    return (rgb.astype(np.float32) / 255.0) * DIM_FACTOR


def _mask_outline(mask):
    if not mask.any():
        return np.zeros_like(mask)
    return mask & ~ndimage.binary_erosion(mask, border_value=0)


def _paint(base, mask, color, fill_alpha=FILL_ALPHA):
    out = base.copy()
    if mask.any():
        c = np.array(color, dtype=np.float32)
        out[mask] = out[mask] * (1 - fill_alpha) + c * fill_alpha
        out[_mask_outline(mask)] = c
    return out


def _panel(ax, image, title, color, dice=None, legend=None):
    ax.imshow(np.clip(image, 0, 1), interpolation="nearest")
    label = title if dice is None else f"{title}\nDice {dice:.4f}"
    ax.set_title(label, color=color, fontsize=12, fontweight="bold")
    ax.axis("off")
    if legend:
        handles = [mpatches.Patch(color=c, label=lbl) for lbl, c in legend]
        leg = ax.legend(handles=handles, loc="upper center", bbox_to_anchor=(0.5, -0.02),
                        ncol=len(legend), frameon=False, fontsize=9,
                        handlelength=1.2, handleheight=1.2, columnspacing=1.0)
        for text in leg.get_texts():
            text.set_color(FG)


def _draw_prompt(ax, mask):
    """Outline the prompt mask that was handed to MedSAM2 on this anchor slice."""
    if mask is None or not mask.any():
        return
    overlay = np.zeros((*mask.shape, 4), dtype=float)
    overlay[_mask_outline(np.asarray(mask, dtype=bool))] = (*PROMPT_RGB, 1.0)
    ax.imshow(overlay, interpolation="nearest")


def render(comp, z, prompts_by_arm=None):
    """Render the 3-panel figure for slice `z` of a `Comparison`.

    `prompts_by_arm`: optional `{"gt": mask2d, "yolo": mask2d}` — the prompt masks fed to
    MedSAM2, outlined when `z` is an anchor slice.
    """
    result = comp.result()
    masks = comp.masks()
    rgb = comp.rgb_volume[:, :, z]
    gt = comp.gt_wt[:, :, z]
    a = masks["gt"][:, :, z]
    b = masks["yolo"][:, :, z]

    from YOLO.pipeline import _dice
    dice_a, dice_b = _dice(a, gt), _dice(b, gt)

    schedule = result["schedule"]
    is_anchor = z in schedule
    anchor_note = ""
    if is_anchor:
        rank = schedule.index(z) + 1
        anchor_note = f"  ·  ANCHOR #{rank}/{len(schedule)}"
        if z in result["yolo_empty_anchors"]:
            anchor_note += "  ·  YOLO: no mask here"

    base = _dimmed(rgb)
    fig, axes = plt.subplots(1, 3, figsize=(16.5, 6), facecolor=BG)
    fig.suptitle(
        f"{result['patient_id']}  ·  z = {z}{anchor_note}",
        color=FG, fontsize=13, fontweight="bold", y=1.02)

    _panel(axes[0], _paint(base, gt, GT_RGB), "Ground truth (WT)", FG,
           legend=[("expert seg > 0", GT_RGB)])
    legend_extra = [("prompt mask", PROMPT_RGB)] if is_anchor else []
    _panel(axes[1], _paint(base, a, ARM_GT_RGB), "MedSAM2 from GT mask",
           "#7fc4ff", dice=dice_a,
           legend=[("prediction", ARM_GT_RGB), ("ground truth", GT_RGB)] + legend_extra)
    _panel(axes[2], _paint(base, b, ARM_YOLO_RGB), "MedSAM2 from YOLO mask",
           "#ffa07a", dice=dice_b,
           legend=[("prediction", ARM_YOLO_RGB), ("ground truth", GT_RGB)] + legend_extra)

    # GT outline on both prediction panels so over/under-segmentation is readable.
    outline = _mask_outline(gt)
    for ax, mask in ((axes[1], a), (axes[2], b)):
        overlay = np.zeros((*gt.shape, 4), dtype=float)
        overlay[outline] = (*GT_RGB, 1.0)
        ax.imshow(overlay, interpolation="nearest")

    if is_anchor and prompts_by_arm:
        _draw_prompt(axes[1], prompts_by_arm.get("gt"))
        _draw_prompt(axes[2], prompts_by_arm.get("yolo"))

    fig.patch.set_facecolor(BG)
    return _to_png(fig)


def render_message(text, title="GT vs YOLO"):
    """Standalone message frame (used before a run has produced anything to draw)."""
    fig, ax = plt.subplots(1, 1, figsize=(9, 5), facecolor=BG)
    ax.axis("off")
    ax.text(0.5, 0.5, text, transform=ax.transAxes, ha="center", va="center",
            color="#ffcc00", fontsize=13, fontweight="bold", wrap=True)
    ax.set_title(title, color=FG, fontsize=12, fontweight="bold")
    return _to_png(fig)
