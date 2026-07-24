"""Asymmetry pipeline core (Stages B-D of PIPELINE.md).

BraTS volumes arrive **already skull-stripped**, so there is no skull-stripping stage:
the brain region is simply the nonzero FLAIR voxels, and FLAIR is used directly.

Stage B  Per-slice asymmetry: midline, per-hemisphere GMM clustering (mean-colour
         quantisation), flip, subtract -> candidate whole-tumour mask.
Stage C  Symmetry features AD / MD / BC.
Stage D  Dice of the stacked prediction vs. ground-truth whole tumour (seg > 0).

Design decisions (defaults, see PIPELINE.md open questions):
  Q3 midline                 : mid-sagittal axis found by symmetry maximisation -- a tilt
                               angle estimated once per volume (max left-right intensity
                               symmetry) + a per-slice symmetry-optimal offset column. Each
                               slice is rotated into a tilt-corrected frame where the axis
                               is a vertical line; the candidate mask is rotated back for
                               scoring.
  Q4 skipped slices          : reported, rendered as a "skipped" frame by the UI
  Q5 difference threshold    : Otsu on the |difference| map
  Q6 features                : per slice AND aggregated per volume
  Q7 persist outputs         : CSV written under Assymetry/outputs/<patient>/
  Q8 volume Dice denominator : all slices (skipped slices predict empty)

GMM component count is auto-selected per hemisphere by BIC over K = 2..K_max.
"""
import os
import glob

import numpy as np
import nibabel as nib
from scipy import ndimage
from sklearn.mixture import GaussianMixture
from skimage.filters import threshold_otsu

# ---------------------------------------------------------------------------
# Parameters
# ---------------------------------------------------------------------------
PARAMS = {
    "n_min_voxel": 500,   # min brain voxels for a slice to be processed
    "k_max": 4,           # max GMM components per hemisphere (auto-select 2..k_max by BIC)
    "feat_bins": 64,      # bins for the Bhattacharyya coefficient
    "angle_max": 12.0,    # mid-sagittal axis search: +/- degrees
    "angle_step": 1.0,    # angle search step (degrees)
    "offset_win": 15,     # midline column search half-window (pixels)
}

MODALITY_SUFFIX = {"FLAIR": "-t2f"}


def _find_modalities(patient_dir):
    files = glob.glob(os.path.join(patient_dir, "*.nii*"))
    out = {}
    for role, suf in MODALITY_SUFFIX.items():
        for f in files:
            if suf in os.path.basename(f).lower():
                out[role] = f
                break
    seg = next((f for f in files if "seg" in os.path.basename(f).lower()), None)
    out["SEG"] = seg
    return out


def _fit_gmm(vals, k_max):
    """Fit a GMM on 1-D intensities, choosing K in 2..k_max by lowest BIC.

    Returns (labels, means) aligned with ``vals``. Falls back to a single cluster
    when there is too little data.
    """
    x = np.asarray(vals, dtype=np.float64).reshape(-1, 1)
    n_unique = len(np.unique(x))
    if x.shape[0] < 10 or n_unique < 2:
        return np.zeros(x.shape[0], dtype=int), np.array([float(x.mean()) if x.size else 0.0])

    best = None
    for k in range(2, min(k_max, n_unique) + 1):
        try:
            g = GaussianMixture(n_components=k, covariance_type="full",
                                random_state=0, max_iter=100, reg_covar=1e-6).fit(x)
        except Exception:
            continue
        bic = g.bic(x)
        if best is None or bic < best[0]:
            best = (bic, g)
    if best is None:
        return np.zeros(x.shape[0], dtype=int), np.array([float(x.mean())])
    g = best[1]
    return g.predict(x), g.means_.ravel()


def _bhattacharyya(a, b, bins, rng):
    ha, _ = np.histogram(a, bins=bins, range=rng, density=False)
    hb, _ = np.histogram(b, bins=bins, range=rng, density=False)
    ha = ha / max(ha.sum(), 1)
    hb = hb / max(hb.sum(), 1)
    return float(np.sum(np.sqrt(ha * hb)))


class AsymmetryPipeline:
    """Lazy, cached pipeline for one patient directory."""

    def __init__(self, patient_dir, params=None):
        self.patient_dir = patient_dir
        self.patient_id = os.path.basename(os.path.normpath(patient_dir))
        self.p = dict(PARAMS)
        if params:
            self.p.update(params)
        self.paths = _find_modalities(patient_dir)
        self._cache = {}
        self._slice_cache = {}

    # -- volume loaders ----------------------------------------------------
    def _vol(self, role):
        key = f"vol:{role}"
        if key not in self._cache:
            path = self.paths.get(role)
            if not path:
                raise FileNotFoundError(f"{role} modality missing for {self.patient_id}")
            self._cache[key] = np.asarray(nib.load(path).get_fdata(), dtype=np.float32)
        return self._cache[key]

    @property
    def flair(self):
        return self._vol("FLAIR")

    @property
    def seg(self):
        return self._vol("SEG")

    @property
    def depth(self):
        return int(self.seg.shape[2])

    # -- brain region (data is already skull-stripped) ---------------------
    @property
    def brain_mask(self):
        """Brain = nonzero FLAIR voxels (BraTS volumes are pre-skull-stripped)."""
        if "brain_mask" not in self._cache:
            self._cache["brain_mask"] = self.flair > 0
        return self._cache["brain_mask"]

    @property
    def flair_stripped(self):
        """FLAIR normalised to [0, 1] over the brain."""
        if "flair_stripped" not in self._cache:
            fl = self.flair
            hi = float(fl.max()) or 1.0
            self._cache["flair_stripped"] = (fl / hi).astype(np.float32)
        return self._cache["flair_stripped"]

    # -- orientation: put the Left-Right (mid-sagittal) axis on the columns ---
    @property
    def lr_is_rows(self):
        """True if the anatomical Left-Right axis is image rows (axis 0).

        Determined from the NIfTI affine (aff2axcodes). When True, slices are
        transposed into a canonical frame so the mid-sagittal split is a vertical
        line (columns = Left-Right), matching the radiological convention.
        """
        if "lr_is_rows" not in self._cache:
            axis = 1
            try:
                codes = nib.aff2axcodes(nib.load(self.paths["FLAIR"]).affine)
                axis = 0 if codes[0] in ("L", "R") else 1
            except Exception:
                axis = 1
            self._cache["lr_is_rows"] = (axis == 0)
        return self._cache["lr_is_rows"]

    def _canon(self, sl2d):
        """Transpose a 2-D slice into the canonical frame (Left-Right on columns)."""
        return sl2d.T if self.lr_is_rows else sl2d

    def slice_flair(self, z):
        return self._canon(self.flair_stripped[:, :, z])

    def slice_brain(self, z):
        return self._canon(self.brain_mask[:, :, z])

    def slice_seg(self, z):
        return self._canon(self.seg[:, :, z])

    # -- mid-sagittal axis: tilt angle (per volume) + corrected frame ---------
    @staticmethod
    def _best_mirror_ssd(img, brain, win):
        """Min mean-squared reflection error over a small offset window (vertical axis)."""
        W = img.shape[1]
        if brain.sum() == 0:
            return np.inf
        c0 = int(round(brain.sum(0) @ np.arange(W) / max(brain.sum(), 1)))
        best = np.inf
        for c in range(max(c0 - win, 5), min(c0 + win + 1, W - 5)):
            hw = min(c, W - c)
            diff = img[:, c - hw:c] - img[:, c:c + hw][:, ::-1]
            ov = brain[:, c - hw:c] & brain[:, c:c + hw][:, ::-1]
            if ov.any():
                best = min(best, float((diff[ov] ** 2).mean()))
        return best

    @property
    def mid_angle(self):
        """Tilt angle (degrees) that makes the mid-sagittal plane vertical.

        Estimated ONCE per volume by rotating the densest-brain slices over a range of
        angles and picking the one with the most left-right intensity symmetry.
        """
        if "mid_angle" not in self._cache:
            self._cache["mid_angle"] = self._estimate_mid_angle()
        return self._cache["mid_angle"]

    def _estimate_mid_angle(self):
        idx = self.slice_indices
        if not idx:
            return 0.0
        reps = [z for _, z in sorted(((self.brain_count(z), z) for z in idx),
                                     reverse=True)[:12]]
        slices = [self.slice_flair(z) for z in reps]
        win = self.p["offset_win"]
        amax, astep = self.p["angle_max"], self.p["angle_step"]
        best = (0.0, np.inf)
        for th in np.arange(-amax, amax + 1e-6, astep):
            tot, n = 0.0, 0
            for sl in slices:
                r = sl if abs(th) < 1e-6 else ndimage.rotate(sl, th, reshape=False, order=1)
                rb = r > 0.01
                s = self._best_mirror_ssd(r, rb, win)
                if np.isfinite(s):
                    tot += s
                    n += 1
            if n and tot / n < best[1]:
                best = (float(th), tot / n)
        return best[0]

    def _rot(self, sl, angle, order):
        if abs(angle) < 1e-6:
            return sl
        return ndimage.rotate(sl, angle, reshape=False, order=order,
                              mode="constant", cval=0)

    def slice_flair_corr(self, z):
        """Canonical FLAIR slice rotated so the mid-sagittal axis is vertical."""
        return self._rot(self.slice_flair(z), self.mid_angle, 1)

    def slice_brain_corr(self, z):
        return self._rot(self.slice_brain(z).astype(np.float32), self.mid_angle, 0) > 0.5

    def slice_seg_corr(self, z):
        return self._rot(self.slice_seg(z), self.mid_angle, 0)

    # -- slice bookkeeping -------------------------------------------------
    def brain_count(self, z):
        return int(self.brain_mask[:, :, z].sum())

    def is_processed(self, z):
        return self.brain_count(z) >= self.p["n_min_voxel"]

    @property
    def slice_indices(self):
        """Axial slices with enough brain to process (Stage B qualifiers)."""
        if "slice_indices" not in self._cache:
            self._cache["slice_indices"] = [
                z for z in range(self.depth) if self.is_processed(z)
            ]
        return self._cache["slice_indices"]

    def best_slice_index(self):
        """Index (into slice_indices) of the processed slice with the most tumour.

        Falls back to the middle processed slice when the segmentation is empty.
        """
        idx = self.slice_indices
        if not idx:
            return 0
        seg = self.seg
        sums = [int((seg[:, :, z] > 0).sum()) for z in idx]
        return int(np.argmax(sums)) if max(sums) > 0 else len(idx) // 2

    # -- Stage B + C (per slice) ------------------------------------------
    def process_slice(self, z):
        if z in self._slice_cache:
            return self._slice_cache[z]

        fln = self.slice_flair_corr(z)   # tilt-corrected: axis is a vertical line
        bm = self.slice_brain_corr(z)
        H, W = fln.shape

        # Q3 midline: symmetry-optimal column near the brain centroid (tilt already fixed)
        if bm.sum() == 0:
            c = W // 2
        else:
            c0 = int(round(bm.sum(axis=0) @ np.arange(W) / max(bm.sum(), 1)))
            win = self.p["offset_win"]
            best = (c0, -1.0)
            for cc in range(max(c0 - win, 1), min(c0 + win + 1, W - 1)):
                hw = min(cc, W - cc)
                lft, rgt = bm[:, cc - hw:cc], bm[:, cc:cc + hw][:, ::-1]
                inter = np.logical_and(lft, rgt).sum()
                union = np.logical_or(lft, rgt).sum()
                iou = inter / union if union else 0.0
                if iou > best[1]:
                    best = (cc, iou)
            c = best[0]
        c = min(max(c, 1), W - 1)
        half_w = min(c, W - c)

        left_i = fln[:, c - half_w:c]
        right_i = fln[:, c:c + half_w]
        left_b = bm[:, c - half_w:c]
        right_b = bm[:, c:c + half_w]

        # per-hemisphere GMM clustering -> mean-colour quantised image
        lq, l_labels, l_means = self._quantise(left_i, left_b)
        rq, r_labels, r_means = self._quantise(right_i, right_b)

        right_flip_q = rq[:, ::-1]
        right_flip_b = right_b[:, ::-1]
        overlap = left_b & right_flip_b

        d = (lq - right_flip_q) * overlap
        absd = np.abs(d)
        pos = absd[overlap & (absd > 0)]
        if pos.size >= 2 and pos.max() > pos.min():
            thr = float(threshold_otsu(pos))
        else:
            thr = float(pos.max()) + 1.0 if pos.size else 1.0

        cand_left = overlap & (d > thr)      # left hemisphere brighter -> tumour left
        cand_right = overlap & (-d > thr)    # mirrored right brighter  -> tumour right

        candidate = np.zeros((H, W), dtype=bool)
        candidate[:, c - half_w:c] |= cand_left
        candidate[:, c:c + half_w] |= cand_right[:, ::-1]

        # -- Stage C features --------------------------------------------
        lv = left_i[left_b]
        rv = right_i[right_b]
        md = float(abs(lv.mean() - rv.mean())) if lv.size and rv.size else 0.0
        bc = _bhattacharyya(lv, rv, self.p["feat_bins"], (0.0, 1.0)) if lv.size and rv.size else 1.0
        # AD: area of the brightest GMM cluster on each side (fraction of hemisphere)
        la = (l_labels == int(np.argmax(l_means))).mean() if l_labels.size else 0.0
        ra = (r_labels == int(np.argmax(r_means))).mean() if r_labels.size else 0.0
        ad = float(abs(la - ra))

        gt = self.slice_seg_corr(z) > 0
        dice = _dice(candidate, gt)

        out = {
            "z": z, "midline": c, "half_w": half_w,
            "left_quant": lq, "right_quant": rq, "right_flip_quant": right_flip_q,
            "left_brain": left_b, "right_brain": right_b, "overlap": overlap,
            "diff": d, "thr": thr, "candidate": candidate,
            "features": {"AD": ad, "MD": md, "BC": bc},
            "slice_dice": dice, "brain_count": int(bm.sum()),
        }
        self._slice_cache[z] = out
        return out

    def _quantise(self, intensities, brain):
        """Replace each brain pixel by its GMM cluster mean; background -> 0."""
        q = np.zeros_like(intensities)
        idx = np.where(brain)
        vals = intensities[idx]
        if vals.size == 0:
            return q, np.array([], dtype=int), np.array([0.0])
        labels, means = _fit_gmm(vals, self.p["k_max"])
        q[idx] = means[labels]
        return q, labels, means

    # -- Stage D -----------------------------------------------------------
    def candidate_corr(self, z):
        """Candidate tumour mask in the tilt-corrected frame (empty for skipped slices)."""
        if self.is_processed(z):
            return self.process_slice(z)["candidate"]
        return np.zeros(self.slice_seg_corr(z).shape, dtype=bool)

    def volume_dice(self):
        """Dice over all slices, computed in the tilt-corrected frame (Dice is invariant
        to the shared rotation/transpose, so this equals the original-space value)."""
        if "vdice" not in self._cache:
            procset = set(self.slice_indices)
            inter = pa = ga = 0
            for z in range(self.depth):
                gt = self.slice_seg_corr(z) > 0
                cand = self.process_slice(z)["candidate"] if z in procset \
                    else np.zeros_like(gt)
                inter += int(np.logical_and(cand, gt).sum())
                pa += int(cand.sum())
                ga += int(gt.sum())
            self._cache["vdice"] = 1.0 if (pa + ga) == 0 else float(2.0 * inter / (pa + ga))
        return self._cache["vdice"]

    def volume_features(self):
        rows = [self.process_slice(z)["features"] for z in self.slice_indices]
        if not rows:
            return {"AD": 0.0, "MD": 0.0, "BC": 0.0}
        return {k: float(np.mean([r[k] for r in rows])) for k in ("AD", "MD", "BC")}

    def summary(self):
        return {
            "patient_id": self.patient_id,
            "n_slices_processed": len(self.slice_indices),
            "mid_angle": round(self.mid_angle, 1),
            "volume_dice": self.volume_dice(),
            "volume_features": self.volume_features(),
        }

    def feature_table(self):
        """Per-slice AD/MD/BC/Dice for every processed slice."""
        rows = []
        for z in self.slice_indices:
            s = self.process_slice(z)
            rows.append({
                "z": z, "brain_voxels": s["brain_count"],
                "AD": s["features"]["AD"], "MD": s["features"]["MD"],
                "BC": s["features"]["BC"], "dice": s["slice_dice"],
            })
        return rows

    def save_csv(self, out_root):
        out_dir = os.path.join(out_root, self.patient_id)
        os.makedirs(out_dir, exist_ok=True)
        path = os.path.join(out_dir, "features.csv")
        rows = self.feature_table()
        with open(path, "w", encoding="utf-8") as f:
            f.write("z,brain_voxels,AD,MD,BC,dice\n")
            for r in rows:
                f.write(f"{r['z']},{r['brain_voxels']},{r['AD']:.6f},"
                        f"{r['MD']:.6f},{r['BC']:.6f},{r['dice']:.6f}\n")
            vd = self.volume_dice()
            f.write(f"volume,,,,,{vd:.6f}\n")
        return path


def _dice(a, b):
    a = a.astype(bool)
    b = b.astype(bool)
    denom = a.sum() + b.sum()
    if denom == 0:
        return 1.0
    return float(2.0 * np.logical_and(a, b).sum() / denom)
