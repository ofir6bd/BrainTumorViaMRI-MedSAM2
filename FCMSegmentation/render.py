"""Matplotlib rendering of each FCM segmentation pipeline stage to a PNG buffer."""
import io

import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

BG = "#141428"
FG = "white"

# (key, label, slice_based) in display order; id = index + 1
STEP_DEFS = [
    ("whole",      "Whole-tumor segmentation (FLAIR FCM)",       True),
    ("predgt",     "Prediction vs. ground truth",                True),
    ("summary",    "Volume summary",                   False),
]

# One-sentence explanation shown under each per-slice image in the stacked view.
EXPLANATIONS = {
    "whole": ("Whole-tumor mask is taken from the brightest FCM cluster on FLAIR, "
              "then largest connected component is selected (no opening step)."),
    "predgt": ("Predicted whole-tumor mask is overlaid against BraTS whole-tumor GT (seg>0), "
               "with per-slice Dice score."),
    "summary": ("Per-slice areas are accumulated and multiplied by slice step to estimate "
                "volumes. Detection and Dice metrics are shown at patient level."),
}

STEPS = [(i + 1, lbl, sb) for i, (_, lbl, sb) in enumerate(STEP_DEFS)]
_KEY_BY_ID = {i + 1: key for i, (key, _, _) in enumerate(STEP_DEFS)}
KEY_LABEL_EXPL = [
    (i + 1, key, lbl, sb, EXPLANATIONS.get(key, ""))
    for i, (key, lbl, sb) in enumerate(STEP_DEFS)
]
_slice_ids = [i + 1 for i, (_, _, sb) in enumerate(STEP_DEFS) if sb]
FIRST_SLICE_STEP = min(_slice_ids)
LAST_SLICE_STEP = max(_slice_ids)
SUMMARY_STEP = next(i + 1 for i, (k, _, _) in enumerate(STEP_DEFS) if k == "summary")

STEP_LABEL = {i: lbl for i, lbl, _ in STEPS}


def _fig(nrows=1, ncols=1, w=6, h=6):
    fig, axes = plt.subplots(nrows, ncols, figsize=(w * ncols, h * nrows),
                             facecolor=BG, squeeze=False)
    return fig, axes


def _to_png(fig):
    buf = io.BytesIO()
    fig.savefig(buf, format="png", bbox_inches="tight", facecolor=fig.get_facecolor())
    plt.close(fig)
    buf.seek(0)
    return buf


def _pnorm(sl):
    nz = sl[sl != 0]
    if nz.size == 0:
        return sl
    lo, hi = np.percentile(nz, 1), np.percentile(nz, 99)
    return np.clip((sl - lo) / max(hi - lo, 1e-8), 0, 1)


def _style(ax, title):
    ax.set_title(title, color=FG, fontsize=12, fontweight="bold")
    ax.axis("off")


def render(pipeline, z, step, slice_no=None):
    pl = pipeline
    key = _KEY_BY_ID.get(step)
    slice_label = f"slice #{slice_no}" if slice_no is not None else f"z={z}"

    if key == "summary":
        return _summary(pl)

    # ---- per-slice (Stages B-D) -------------------------------------------
    if not pl.is_processed(z):
        return _skipped(pl, z, slice_no=slice_no)

    s = pl.process_slice(z)
    flair_med = s["flair_med"]

    if key == "whole":
        fig, ax = _fig(1, 3, 5, 5.5)
        ax[0][0].imshow(flair_med, cmap="gray")
        _style(ax[0][0], "FLAIR median")
        ax[0][1].imshow(s["whole_fcm"], cmap="gray")
        _style(ax[0][1], "Whole tumor (FCM-only)")
        ax[0][2].imshow(flair_med, cmap="gray")
        ov = np.zeros((*flair_med.shape, 4))
        ov[s["whole_mask"]] = (1, 0, 0, 0.5)
        ax[0][2].imshow(ov)
        _style(ax[0][2], f"Whole mask (FCM+CC, {s['whole_pixels']} px)")
        fig.suptitle(f"Stage 1: Whole-tumor segmentation  -  {slice_label}", color=FG, fontsize=14)
        return _to_png(fig)

    if key == "predgt":
        fig, axes = _fig(1, 1, 6.2, 6.0)
        a0 = axes[0][0]
        a0.imshow(flair_med, cmap="gray")
        ov0 = np.zeros((*flair_med.shape, 4))
        gt_whole = s["gt_whole"]
        pd_whole = s["whole_mask"]
        ov0[gt_whole] = (0, 1, 0, 0.35)
        ov0[pd_whole] = (1, 0, 0, 0.45)
        ov0[gt_whole & pd_whole] = (1, 1, 0, 0.6)
        a0.imshow(ov0)
        _style(a0, f"Whole: Dice={s['slice_dice_whole']:.3f}")

        fig.suptitle(f"Stage 2: Prediction vs GT  -  {slice_label}", color=FG, fontsize=14)
        return _to_png(fig)

    return _skipped(pl, z)


def _skipped(pl, z, slice_no=None):
    fig, axes = _fig(1, 1, 7, 7)
    a = axes[0][0]
    a.imshow(pl.slice_flair_med(z), cmap="gray")
    label = f"slice #{slice_no}" if slice_no is not None else f"slice z={z}"
    a.text(0.5, 0.06, f"{label} skipped: brain {pl.brain_count(z)} px "
                      f"< N_min_voxel ({pl.p['n_min_voxel']})",
           transform=a.transAxes, ha="center", color="#ffcc00",
           fontsize=11, fontweight="bold",
           bbox=dict(facecolor="black", alpha=0.6, pad=4))
    _style(a, f"Skipped {label}")
    return _to_png(fig)


def _summary(pl):
    vd = pl.volume_dice()
    seg = pl.seg
    sums = [int((seg[:, :, z] > 0).sum()) for z in range(pl.depth)]
    zb = int(np.argmax(sums)) if max(sums) > 0 else pl.depth // 2
    fig, ax = _fig(1, 2, 6.5, 6)
    a = ax[0][0]
    s = pl.process_slice(zb) if pl.is_processed(zb) else None
    flair = s["flair_med"] if s is not None else pl.slice_flair_med(zb)
    a.imshow(flair, cmap="gray")
    gt = (pl.slice_seg(zb) > 0)
    cand = pl.whole_mask(zb)
    H, W = gt.shape
    overlay = np.zeros((H, W, 4))
    overlay[gt] = (0, 1, 0, 0.35)
    overlay[cand] = (1, 0, 0, 0.45)
    overlay[gt & cand] = (1, 1, 0, 0.6)
    a.imshow(overlay)
    _style(a, f"Best GT slice z={zb}")
    b = ax[0][1]
    b.axis("off")
    b.set_facecolor(BG)
    txt = (f"VOLUME SUMMARY\n{pl.patient_id}\n\n"
            f"Whole-tumour Dice : {vd:.3f}\n"
            f"\n"
            f"Slices processed  : {len(pl.slice_indices)} / {pl.depth}")
    b.text(0.02, 0.98, txt, transform=b.transAxes, ha="left", va="top",
           color=FG, fontsize=13, family="monospace")
    fig.suptitle("Volume summary", color=FG, fontsize=15, fontweight="bold")
    return _to_png(fig)
