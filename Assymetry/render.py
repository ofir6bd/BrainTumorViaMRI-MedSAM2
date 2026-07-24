"""Matplotlib rendering of each asymmetry pipeline step to a PNG buffer.

The web UI advances through STEPS one at a time via a "Next" button. Steps are defined
once in STEP_DEFS (key, label, slice_based); ids are 1-based positions, so inserting a
step here automatically renumbers everything and the frontend adapts via /asym/api/steps.

BraTS data is already skull-stripped, so there is no Stage A: the walk starts at the
per-slice asymmetry steps (Stages B-D) and ends with the volume summary.
"""
import io

import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

BG = "#141428"
FG = "white"

# (key, label, slice_based) in display order; id = index + 1
STEP_DEFS = [
    ("midline",    "Midline",                          True),
    ("clustering", "Per-hemisphere GMM clustering",    True),
    ("flip",       "Flip & overlay hemispheres",       True),
    ("diff",       "Difference map + candidate mask",  True),
    ("features",   "Symmetry features (AD / MD / BC)", True),
    ("predgt",     "Prediction vs. ground truth",      True),
    ("summary",    "Volume summary",                   False),
]

# One-sentence explanation shown under each per-slice image in the stacked view.
EXPLANATIONS = {
    "midline": ("The mid-sagittal plane is found by maximising left-right symmetry: a tilt "
                "angle is estimated once for the whole volume, and the slice is rotated so "
                "that plane stands vertical (brain shown upright). A vertical midline is then "
                "placed at the symmetry-optimal column, splitting it into left and right "
                "hemispheres."),
    "clustering": ("Each hemisphere's FLAIR intensities are clustered independently with a "
                   "Gaussian Mixture Model (number of components chosen automatically by "
                   "BIC). Every pixel is repainted with its cluster's mean value, so tissue "
                   "types collapse into flat colour regions."),
    "flip": ("One hemisphere is mirrored across the midline and laid over the other. A "
             "healthy brain is nearly symmetric, so subtracting the two sides leaves a "
             "bright signal only where a tumour breaks the symmetry."),
    "diff": ("The absolute difference between the hemispheres is thresholded with Otsu's "
             "method; the surviving asymmetric region becomes the candidate tumour mask "
             "(red)."),
    "features": ("Three symmetry features summarise this slice: AD = area difference of the "
                 "brightest cluster, MD = difference of mean gray levels, BC = Bhattacharyya "
                 "overlap of the two hemispheres' histograms."),
    "predgt": ("The candidate mask (red) is overlaid on the ground-truth whole tumour "
               "(green); their overlap is yellow. The Dice score reports how well they "
               "match for this slice."),
    "summary": ("Whole-volume result: the per-slice candidate masks are stacked and scored "
                "against the ground-truth whole tumour."),
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


def _full_quant(pl, s):
    """Reassemble the two quantised half-slices into a full-slice image."""
    H, W = pl.slice_flair_corr(s["z"]).shape
    c, hw = s["midline"], s["half_w"]
    full = np.zeros((H, W))
    full[:, c - hw:c] = s["left_quant"]
    full[:, c:c + hw] = s["right_quant"]
    return full


def render(pipeline, z, step):
    pl = pipeline
    key = _KEY_BY_ID.get(step)

    if key == "summary":
        return _summary(pl)

    # ---- per-slice (Stages B-D) -------------------------------------------
    if not pl.is_processed(z):
        return _skipped(pl, z)

    s = pl.process_slice(z)
    fln = pl.slice_flair_corr(z)

    if key == "midline":
        fig, axes = _fig(1, 1, 7, 7)
        a = axes[0][0]
        a.imshow(fln, cmap="gray")
        a.axvline(s["midline"], color="#ffe600", lw=2)
        _style(a, f"Midline col {s['midline']}  ·  tilt {pl.mid_angle:+.1f} deg  (z={z})")
        return _to_png(fig)

    if key == "clustering":
        fig, ax = _fig(1, 3, 5, 5.5)
        ax[0][0].imshow(fln, cmap="gray")
        ax[0][0].axvline(s["midline"], color="#ffe600", lw=1.2)
        _style(ax[0][0], "FLAIR (input)")
        fq = _full_quant(pl, s)
        ax[0][1].imshow(fq, cmap="viridis")
        ax[0][1].axvline(s["midline"], color="w", lw=1.2)
        _style(ax[0][1], "GMM mean-colour quantised")
        ax[0][2].imshow(s["left_quant"], cmap="viridis")
        _style(ax[0][2], "Left hemisphere (quantised)")
        fig.suptitle(f"Per-hemisphere GMM clustering  -  z={z}", color=FG, fontsize=14)
        return _to_png(fig)

    if key == "flip":
        fig, ax = _fig(1, 3, 5, 5.5)
        ax[0][0].imshow(s["left_quant"], cmap="viridis")
        _style(ax[0][0], "Left hemisphere")
        ax[0][1].imshow(s["right_flip_quant"], cmap="viridis")
        _style(ax[0][1], "Right hemisphere (flipped)")
        diff_disp = np.abs(s["diff"])
        ax[0][2].imshow(diff_disp, cmap="magma")
        _style(ax[0][2], "|left - flipped right|")
        fig.suptitle(f"Flip & overlay  -  z={z}", color=FG, fontsize=14)
        return _to_png(fig)

    if key == "diff":
        fig, ax = _fig(1, 2, 6.5, 6.5)
        H, W = fln.shape
        c, hw = s["midline"], s["half_w"]
        full_diff = np.zeros((H, W))
        full_diff[:, c - hw:c] = np.abs(s["diff"])
        ax[0][0].imshow(full_diff, cmap="magma")
        ax[0][0].axvline(c, color="w", lw=1)
        _style(ax[0][0], f"Difference map (Otsu thr = {s['thr']:.3f})")
        ax[0][1].imshow(fln, cmap="gray")
        overlay = np.zeros((H, W, 4))
        overlay[s["candidate"]] = (1, 0, 0, 0.55)
        ax[0][1].imshow(overlay)
        _style(ax[0][1], f"Candidate tumour mask ({int(s['candidate'].sum())} px)")
        fig.suptitle(f"Difference & candidate  -  z={z}", color=FG, fontsize=14)
        return _to_png(fig)

    if key == "features":
        f = s["features"]
        fig, axes = _fig(1, 2, 6, 5.5)
        a = axes[0][0]
        a.imshow(fln, cmap="gray")
        overlay = np.zeros((*fln.shape, 4))
        overlay[s["candidate"]] = (1, 0, 0, 0.5)
        a.imshow(overlay)
        _style(a, f"Candidate  -  z={z}")
        b = axes[0][1]
        b.set_facecolor(BG)
        names = ["AD", "MD", "BC"]
        vals = [f["AD"], f["MD"], f["BC"]]
        bars = b.bar(names, vals, color=["#4fc3f7", "#81c784", "#ffb74d"])
        for bar, v in zip(bars, vals):
            b.text(bar.get_x() + bar.get_width() / 2, v, f"{v:.3f}",
                   ha="center", va="bottom", color=FG, fontsize=11, fontweight="bold")
        b.set_ylim(0, max(1.0, max(vals) * 1.2))
        b.set_title("Symmetry features", color=FG, fontsize=12, fontweight="bold")
        b.tick_params(colors=FG)
        for spine in b.spines.values():
            spine.set_color("#555")
        return _to_png(fig)

    if key == "predgt":
        gt = pl.slice_seg_corr(z) > 0
        cand = s["candidate"]
        fig, axes = _fig(1, 1, 7.5, 7.5)
        a = axes[0][0]
        a.imshow(fln, cmap="gray")
        H, W = fln.shape
        overlay = np.zeros((H, W, 4))
        overlay[gt] = (0, 1, 0, 0.35)                 # ground truth = green
        overlay[cand] = (1, 0, 0, 0.45)               # prediction  = red
        overlay[gt & cand] = (1, 1, 0, 0.6)           # overlap     = yellow
        a.imshow(overlay)
        _style(a, f"Pred (red) vs GT (green), overlap yellow  -  Dice = {s['slice_dice']:.3f}")
        return _to_png(fig)

    return _skipped(pl, z)


def _skipped(pl, z):
    fig, axes = _fig(1, 1, 7, 7)
    a = axes[0][0]
    a.imshow(pl.slice_flair_corr(z), cmap="gray")
    a.text(0.5, 0.06, f"slice z={z} skipped: brain {pl.brain_count(z)} px "
                      f"< N_min_voxel ({pl.p['n_min_voxel']})",
           transform=a.transAxes, ha="center", color="#ffcc00",
           fontsize=11, fontweight="bold",
           bbox=dict(facecolor="black", alpha=0.6, pad=4))
    _style(a, f"Skipped slice (z={z})")
    return _to_png(fig)


def _summary(pl):
    vd = pl.volume_dice()
    vf = pl.volume_features()
    seg = pl.seg
    sums = [int((seg[:, :, z] > 0).sum()) for z in range(pl.depth)]
    zb = int(np.argmax(sums)) if max(sums) > 0 else pl.depth // 2
    fig, ax = _fig(1, 2, 6.5, 6)
    a = ax[0][0]
    a.imshow(pl.slice_flair_corr(zb), cmap="gray")
    gt = pl.slice_seg_corr(zb) > 0
    cand = pl.candidate_corr(zb)
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
           f"Whole-tumour Dice : {vd:.3f}\n\n"
           f"Slices processed  : {len(pl.slice_indices)} / {pl.depth}\n"
           f"Mid-sagittal tilt : {pl.mid_angle:+.1f} deg\n\n"
           f"Mean AD : {vf['AD']:.3f}\n"
           f"Mean MD : {vf['MD']:.3f}\n"
           f"Mean BC : {vf['BC']:.3f}")
    b.text(0.02, 0.98, txt, transform=b.transAxes, ha="left", va="top",
           color=FG, fontsize=13, family="monospace")
    fig.suptitle("Volume summary", color=FG, fontsize=15, fontweight="bold")
    return _to_png(fig)
