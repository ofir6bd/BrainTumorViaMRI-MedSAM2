"""Slice figures for the inference-test tab.

Three panels over the same background: the expert ground truth, the reference variant
(O), and whichever variant is selected. On an anchor slice a fourth panel shows the
prompt that variant was actually handed — as a heatmap, because for the graded variants
the *values* are the whole point and an outline would hide them.

The background is the RGB frame MedSAM2 saw (R:T1c / G:T2w / B:T2f), not the YOLO
composite, so what you look at is what produced the segmentation being drawn.
"""
import io

import matplotlib
import matplotlib.patches as mpatches
import numpy as np
from scipy import ndimage

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402

from .variants import COLORS, NEUTRAL, VARIANTS  # noqa: E402

BG = "#141428"
FG = "white"
GT_RGB = (0.15, 1.0, 0.15)
REF_RGB = (1.0, 0.35, 0.15)
FILL_ALPHA = 0.30
DIM_FACTOR = 0.55


def _to_png(fig):
    buf = io.BytesIO()
    fig.savefig(buf, format="png", bbox_inches="tight", facecolor=fig.get_facecolor())
    plt.close(fig)
    buf.seek(0)
    return buf


def _hex_rgb(h):
    h = h.lstrip("#")
    return tuple(int(h[i:i + 2], 16) / 255.0 for i in (0, 2, 4))


def _outline(mask):
    if not mask.any():
        return np.zeros_like(mask)
    return mask & ~ndimage.binary_erosion(mask, border_value=0)


def _paint(base, mask, color, fill_alpha=FILL_ALPHA):
    out = base.copy()
    if mask.any():
        c = np.array(color, dtype=np.float32)
        out[mask] = out[mask] * (1 - fill_alpha) + c * fill_alpha
        out[_outline(mask)] = c
    return out


def _panel(ax, image, title, color, dice=None, legend=None):
    ax.imshow(np.clip(image, 0, 1), interpolation="nearest")
    ax.set_title(title if dice is None else f"{title}\nDice {dice:.4f}",
                 color=color, fontsize=11.5, fontweight="bold")
    ax.axis("off")
    if legend:
        handles = [mpatches.Patch(color=c, label=lbl) for lbl, c in legend]
        leg = ax.legend(handles=handles, loc="upper center", bbox_to_anchor=(0.5, -0.02),
                        ncol=len(legend), frameon=False, fontsize=8.5,
                        handlelength=1.2, handleheight=1.2, columnspacing=1.0)
        for text in leg.get_texts():
            text.set_color(FG)


def _gt_edge(ax, gt):
    """GT contour over a prediction panel, so over- and under-segmentation read at a glance."""
    overlay = np.zeros((*gt.shape, 4), dtype=float)
    overlay[_outline(gt)] = (*GT_RGB, 1.0)
    ax.imshow(overlay, interpolation="nearest")


def _prompt_panel(ax, base, prompt, variant):
    """Heatmap of the prompt values, with the model's 0.5 decision boundary marked.

    MedSAM2 turns a prompt value p into the logit 20p-10, so 0.5 is where "tumour" starts.
    The colour bar is pinned to 0..1 across variants so a weak prompt looks weak.
    """
    ax.imshow(np.clip(base, 0, 1), interpolation="nearest")
    if prompt is None:
        ax.set_title("no prompt on this slice", color="#ffcc00", fontsize=11.5,
                     fontweight="bold")
        ax.axis("off")
        return
    masked = np.ma.masked_where(prompt <= 0, prompt)
    im = ax.imshow(masked, cmap="turbo", vmin=0.0, vmax=1.0, interpolation="nearest",
                   alpha=0.85)
    inside = prompt >= NEUTRAL
    if inside.any():
        overlay = np.zeros((*prompt.shape, 4), dtype=float)
        overlay[_outline(inside)] = (1.0, 1.0, 1.0, 1.0)
        ax.imshow(overlay, interpolation="nearest")
    vals = prompt[prompt > 0]
    rng = (f"{vals.min():.2f}–{vals.max():.2f}" if vals.size else "empty")
    kind = "graded" if VARIANTS[variant].is_soft else "binary"
    ax.set_title(f"prompt handed to MedSAM2\n{kind}, values {rng}",
                 color="#ffd23f", fontsize=11.5, fontweight="bold")
    ax.axis("off")
    cb = plt.colorbar(im, ax=ax, fraction=0.046, pad=0.03)
    cb.set_label("prompt value (0.5 = decision boundary)", color=FG, fontsize=8)
    cb.ax.tick_params(colors=FG, labelsize=7)
    cb.outline.set_edgecolor("#2a2a45")


def render(run, z, variant_id, masks, prompt=None, dice=None):
    """`masks` is `{"O": mask3d, variant_id: mask3d}`; `prompt` the 2-D prompt or None."""
    from YOLO.pipeline import _dice

    gt = run.gt_wt[:, :, z]
    base = (run.rgb_medsam2[:, :, z].astype(np.float32) / 255.0) * DIM_FACTOR
    ref = masks.get("O")
    sel = masks.get(variant_id)
    v = VARIANTS[variant_id]
    sel_rgb = _hex_rgb(COLORS.get(variant_id, "#ffffff"))

    from .pipeline import is_anchor_slice
    is_anchor = is_anchor_slice(run, v, z)
    n_panels = 4 if is_anchor else 3
    fig, axes = plt.subplots(1, n_panels, figsize=(5.5 * n_panels, 6), facecolor=BG)
    anchor_note = ""
    if is_anchor:
        # H prompts every brain slice, so there is no ranked schedule to number against.
        anchor_note = ("  ·  PROMPTED" if v.mode == "all_slices"
                       else f"  ·  ANCHOR #{run.schedule.index(z) + 1}")
    fig.suptitle(f"{run.patient_id}  ·  z = {z}{anchor_note}  ·  {v.label}",
                 color=FG, fontsize=13, fontweight="bold", y=1.02)

    _panel(axes[0], _paint(base, gt, GT_RGB), "Ground truth (WT)", FG,
           legend=[("expert seg > 0", GT_RGB)])

    if ref is not None:
        _panel(axes[1], _paint(base, ref[:, :, z], REF_RGB), "O · reference",
               "#ffa07a", dice=_dice(ref[:, :, z], gt),
               legend=[("prediction", REF_RGB), ("ground truth", GT_RGB)])
        _gt_edge(axes[1], gt)
    else:
        _panel(axes[1], base, "reference not run yet", "#ffcc00")

    if sel is not None:
        d = dice if dice is not None else _dice(sel[:, :, z], gt)
        _panel(axes[2], _paint(base, sel[:, :, z], sel_rgb),
               f"{variant_id} · {v.label}", COLORS.get(variant_id, "#fff"), dice=d,
               legend=[("prediction", sel_rgb), ("ground truth", GT_RGB)])
        _gt_edge(axes[2], gt)
    else:
        _panel(axes[2], base, "variant not run yet", "#ffcc00")

    if is_anchor:
        _prompt_panel(axes[3], base, prompt, variant_id)

    fig.patch.set_facecolor(BG)
    return _to_png(fig)


def render_message(text, title="Inference tests"):
    fig, ax = plt.subplots(1, 1, figsize=(9, 5), facecolor=BG)
    ax.axis("off")
    ax.text(0.5, 0.5, text, transform=ax.transAxes, ha="center", va="center",
            color="#ffcc00", fontsize=13, fontweight="bold", wrap=True)
    ax.set_title(title, color=FG, fontsize=12, fontweight="bold")
    return _to_png(fig)
