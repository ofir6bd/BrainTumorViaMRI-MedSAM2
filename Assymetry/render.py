"""Matplotlib rendering of each asymmetry pipeline stage to a PNG buffer."""
import io

import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

BG = "#141428"
FG = "white"

# (key, label, slice_based) in display order; id = index + 1
STEP_DEFS = [
    ("median",     "Median filtering (FLAIR/T1C)",               True),
    ("symmetry",   "Symmetry split + FCM binary map",            True),
    ("features",   "AD / MD / BC scoring",                       True),
    ("whole",      "Whole-tumor segmentation (FLAIR FCM)",       True),
    ("active",     "Active-core segmentation (T1C FCM + ROI)",   True),
    ("necrotic",   "Necrotic-core estimation",                   True),
    ("areas",      "Slice area estimation",                      True),
    ("predgt",     "Prediction vs. ground truth",                True),
    ("summary",    "Volume summary",                   False),
]

# One-sentence explanation shown under each per-slice image in the stacked view.
EXPLANATIONS = {
    "median": ("Noise reduction is applied first using 3x3 median filtering on FLAIR and T1C. "
               "BraTS is already skull-stripped, so no skull-removal stage is run."),
    "symmetry": ("The FLAIR slice is split into left/right hemispheres across the midline. "
                 "An FCM binary bright-tissue map is computed and used to compare hemisphere areas."),
    "features": ("Symmetry features are computed as AD (area difference), MD (mean gray-level "
                 "difference), and BC-asym = 1-BC. Slice-level tumor detection uses score "
                 "L1+L2+L3 with fixed thresholds."),
    "whole": ("Whole-tumor (edema-inclusive) segmentation is obtained from FLAIR by FCM, "
              "connected components, and morphological opening."),
    "active": ("Active core is segmented inside the whole-tumor ROI using adaptive high-T1C "
               "thresholding + opening, with small-component cleanup."),
    "necrotic": ("Necrotic/cystic core is estimated as closing(active) minus active, bounded "
                 "inside the whole-tumor region."),
    "areas": ("Pixel counts are converted to physical area via A = pixel_area * white_pixels, "
              "preparing per-slice inputs for volume estimation."),
    "predgt": ("Predictions are overlaid against BraTS GT: whole tumor (seg>0) and active-core "
               "GT (labels 1,3,4), with per-slice Dice scores."),
    "summary": ("Per-slice areas are accumulated and multiplied by slice spacing to estimate "
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


def render(pipeline, z, step):
    pl = pipeline
    key = _KEY_BY_ID.get(step)

    if key == "summary":
        return _summary(pl)

    # ---- per-slice (Stages B-D) -------------------------------------------
    if not pl.is_processed(z):
        return _skipped(pl, z)

    s = pl.process_slice(z)
    flair_raw = s["flair_raw"]
    flair_med = s["flair_med"]
    t1c_raw = s["t1c_raw"]
    t1c_med = s["t1c_med"]
    mid = s["midline"]

    if key == "median":
        fig, ax = _fig(2, 2, 4.8, 4.4)
        ax[0][0].imshow(flair_raw, cmap="gray")
        _style(ax[0][0], "FLAIR raw")
        ax[0][1].imshow(flair_med, cmap="gray")
        _style(ax[0][1], "FLAIR median")
        ax[1][0].imshow(t1c_raw, cmap="gray")
        _style(ax[1][0], "T1C raw")
        ax[1][1].imshow(t1c_med, cmap="gray")
        _style(ax[1][1], "T1C median")
        fig.suptitle(f"Stage 1: Median filtering  -  z={z}", color=FG, fontsize=14)
        return _to_png(fig)

    if key == "symmetry":
        fig, ax = _fig(1, 3, 5, 5.5)
        ax[0][0].imshow(flair_med, cmap="gray")
        ax[0][0].axvline(mid, color="#ffe600", lw=1.3)
        _style(ax[0][0], "FLAIR median + midline")
        ax[0][1].imshow(s["symmetry_bright"], cmap="gray")
        ax[0][1].axvline(mid, color="#ffe600", lw=1.0)
        _style(ax[0][1], "FCM bright-tissue binary")
        hemi = np.zeros_like(flair_med)
        hw = s["half_w"]
        hemi[:, mid - hw:mid] = s["left_bin"]
        hemi[:, mid:mid + hw] = s["right_bin"]
        ax[0][2].imshow(hemi, cmap="magma")
        _style(ax[0][2], "Left/Right binary halves")
        fig.suptitle(f"Stage 2: Symmetry preparation  -  z={z}", color=FG, fontsize=14)
        return _to_png(fig)

    if key == "features":
        f = s["features"]
        fig, axes = _fig(1, 2, 6, 5.5)
        a = axes[0][0]
        a.imshow(flair_med, cmap="gray")
        overlay = np.zeros((*flair_med.shape, 4))
        overlay[s["whole_mask"]] = (1, 0, 0, 0.5)
        a.imshow(overlay)
        _style(a, f"Whole-tumor candidate  -  z={z}")
        b = axes[0][1]
        b.set_facecolor(BG)
        names = ["AD", "MD", "1-BC", "Score"]
        vals = [f["AD"], f["MD"], f["BC_asym"], s["asymmetry_score"]]
        bars = b.bar(names, vals, color=["#4fc3f7", "#81c784", "#ffb74d", "#ce93d8"])
        for bar, v in zip(bars, vals):
            b.text(bar.get_x() + bar.get_width() / 2, v, f"{v:.3f}",
                   ha="center", va="bottom", color=FG, fontsize=11, fontweight="bold")
        ymin = min(-3, float(np.min(vals)) - 0.5)
        ymax = max(1.0, float(np.max(vals)) * 1.2 + 0.1)
        b.set_ylim(ymin, ymax)
        b.set_title(
            f"Detected={int(s['tumor_detected'])}  (L1/L2/L3={s['labels']['L1']}/{s['labels']['L2']}/{s['labels']['L3']})",
            color=FG,
            fontsize=11,
            fontweight="bold",
        )
        b.tick_params(colors=FG)
        for spine in b.spines.values():
            spine.set_color("#555")
        return _to_png(fig)

    if key == "whole":
        fig, ax = _fig(1, 3, 5, 5.5)
        ax[0][0].imshow(flair_med, cmap="gray")
        _style(ax[0][0], "FLAIR median")
        ax[0][1].imshow(s["whole_raw"], cmap="gray")
        _style(ax[0][1], "Whole tumor raw (FCM+CC)")
        ax[0][2].imshow(flair_med, cmap="gray")
        ov = np.zeros((*flair_med.shape, 4))
        ov[s["whole_mask"]] = (1, 0, 0, 0.5)
        ax[0][2].imshow(ov)
        _style(ax[0][2], f"Opened whole mask ({s['area_px']['whole']} px)")
        fig.suptitle(f"Stage 3: Whole-tumor segmentation  -  z={z}", color=FG, fontsize=14)
        return _to_png(fig)

    if key == "active":
        fig, ax = _fig(1, 3, 5, 5.5)
        ax[0][0].imshow(t1c_med, cmap="gray")
        _style(ax[0][0], "T1C median")
        ax[0][1].imshow(s["active_bin"], cmap="gray")
        _style(ax[0][1], "T1C active binary (FCM)")
        ax[0][2].imshow(t1c_med, cmap="gray")
        ov = np.zeros((*t1c_med.shape, 4))
        ov[s["whole_mask"]] = (0, 1, 0, 0.25)
        ov[s["active_mask"]] = (1, 0, 0, 0.55)
        ax[0][2].imshow(ov)
        _style(ax[0][2], "Active core (red) within whole ROI (green)")
        fig.suptitle(f"Stage 4: Active-core segmentation  -  z={z}", color=FG, fontsize=14)
        return _to_png(fig)

    if key == "necrotic":
        fig, ax = _fig(1, 2, 6, 6)
        ax[0][0].imshow(t1c_med, cmap="gray")
        ov0 = np.zeros((*t1c_med.shape, 4))
        ov0[s["active_mask"]] = (1, 0, 0, 0.55)
        ax[0][0].imshow(ov0)
        _style(ax[0][0], "Active core")
        ax[0][1].imshow(t1c_med, cmap="gray")
        ov1 = np.zeros((*t1c_med.shape, 4))
        ov1[s["active_mask"]] = (1, 0, 0, 0.45)
        ov1[s["necrotic_mask"]] = (0, 0.8, 1, 0.55)
        ax[0][1].imshow(ov1)
        _style(ax[0][1], "Necrotic estimate (cyan)")
        fig.suptitle(f"Stage 5: Necrotic-core estimation  -  z={z}", color=FG, fontsize=14)
        return _to_png(fig)

    if key == "areas":
        fig, ax = _fig(1, 2, 6.2, 5.8)
        ax[0][0].imshow(flair_med, cmap="gray")
        ov = np.zeros((*flair_med.shape, 4))
        ov[s["whole_mask"]] = (1, 0, 0, 0.35)
        ov[s["active_mask"]] = (1, 1, 0, 0.55)
        ov[s["necrotic_mask"]] = (0, 0.8, 1, 0.55)
        ax[0][0].imshow(ov)
        _style(ax[0][0], "Slice masks")
        b = ax[0][1]
        vals = [
            s["area_mm2"]["whole"],
            s["area_mm2"]["active"],
            s["area_mm2"]["necrotic"],
        ]
        names = ["Whole", "Active", "Necrotic"]
        colors = ["#ef5350", "#ffee58", "#4fc3f7"]
        bars = b.bar(names, vals, color=colors)
        b.set_facecolor(BG)
        b.tick_params(colors=FG)
        for spine in b.spines.values():
            spine.set_color("#555")
        b.set_title(f"Area (mm²)  |  pixel area={pl.pixel_area_mm2:.3f}", color=FG, fontsize=11)
        for bar, v in zip(bars, vals):
            b.text(bar.get_x() + bar.get_width() / 2, v, f"{v:.1f}", ha="center", va="bottom", color=FG)
        fig.suptitle(f"Stage 6: Area estimation  -  z={z}", color=FG, fontsize=14)
        return _to_png(fig)

    if key == "predgt":
        fig, axes = _fig(1, 2, 6.2, 6.0)
        a0, a1 = axes[0][0], axes[0][1]
        a0.imshow(flair_med, cmap="gray")
        ov0 = np.zeros((*flair_med.shape, 4))
        gt_whole = s["gt_whole"]
        pd_whole = s["whole_mask"]
        ov0[gt_whole] = (0, 1, 0, 0.35)
        ov0[pd_whole] = (1, 0, 0, 0.45)
        ov0[gt_whole & pd_whole] = (1, 1, 0, 0.6)
        a0.imshow(ov0)
        _style(a0, f"Whole: Dice={s['slice_dice_whole']:.3f}")

        a1.imshow(t1c_med, cmap="gray")
        ov1 = np.zeros((*t1c_med.shape, 4))
        gt_active = s["gt_active"]
        pd_active = s["active_mask"]
        ov1[gt_active] = (0, 1, 0, 0.35)
        ov1[pd_active] = (1, 0, 0, 0.45)
        ov1[gt_active & pd_active] = (1, 1, 0, 0.6)
        a1.imshow(ov1)
        _style(a1, f"Active core: Dice={s['slice_dice_active']:.3f}")

        fig.suptitle(f"Stage 7: Prediction vs GT  -  z={z}", color=FG, fontsize=14)
        return _to_png(fig)

    return _skipped(pl, z)


def _skipped(pl, z):
    fig, axes = _fig(1, 1, 7, 7)
    a = axes[0][0]
    a.imshow(pl.slice_flair_med(z), cmap="gray")
    a.text(0.5, 0.06, f"slice z={z} skipped: brain {pl.brain_count(z)} px "
                      f"< N_min_voxel ({pl.p['n_min_voxel']})",
           transform=a.transAxes, ha="center", color="#ffcc00",
           fontsize=11, fontweight="bold",
           bbox=dict(facecolor="black", alpha=0.6, pad=4))
    _style(a, f"Skipped slice (z={z})")
    return _to_png(fig)


def _summary(pl):
    vd = pl.volume_dice()
    adv = pl.active_volume_dice()
    vf = pl.volume_features()
    det = pl.detection_metrics()
    vol = pl.volume_estimates()
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
            f"Active-core Dice  : {adv:.3f}\n\n"
            f"Slices processed  : {len(pl.slice_indices)} / {pl.depth}\n"
            f"Detection acc/sens/spec : {det['accuracy']:.3f} / {det['sensitivity']:.3f} / {det['specificity']:.3f}\n\n"
           f"Mean AD : {vf['AD']:.3f}\n"
           f"Mean MD : {vf['MD']:.3f}\n"
            f"Mean BC : {vf['BC']:.3f}\n"
            f"Mean 1-BC : {vf['BC_asym']:.3f}\n\n"
            f"Whole volume (mm3) : {vol['whole_volume_mm3']:.1f}\n"
            f"Active volume (mm3): {vol['active_volume_mm3']:.1f}\n"
            f"GT active (mm3)    : {vol['gt_active_volume_mm3']:.1f}")
    b.text(0.02, 0.98, txt, transform=b.transAxes, ha="left", va="top",
           color=FG, fontsize=13, family="monospace")
    fig.suptitle("Volume summary", color=FG, fontsize=15, fontweight="bold")
    return _to_png(fig)
