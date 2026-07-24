"""Asymmetry pipeline core (Stages B-D of PIPELINE.md).

BraTS volumes arrive **already skull-stripped**, so there is no skull-stripping stage:
the brain region is simply the nonzero FLAIR voxels, and FLAIR is used directly.

Stage B  Per-slice asymmetry: midline, per-hemisphere GMM clustering (mean-colour
         quantisation), flip, subtract -> candidate whole-tumour mask.
Stage C  Symmetry features AD / MD / BC.
Stage D  Dice of the stacked prediction vs. ground-truth whole tumour (seg > 0).

Design decisions (defaults, see PIPELINE.md open questions):
  Q3 midline                 : per-slice centroid column of the brain mask
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
from sklearn.mixture import GaussianMixture
from skimage.filters import threshold_otsu

# ---------------------------------------------------------------------------
# Parameters
# ---------------------------------------------------------------------------
PARAMS = {
    "n_min_voxel": 500,   # min brain voxels for a slice to be processed
    "k_max": 4,           # max GMM components per hemisphere (auto-select 2..k_max by BIC)
    "feat_bins": 64,      # bins for the Bhattacharyya coefficient
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

    # -- Stage B + C (per slice) ------------------------------------------
    def process_slice(self, z):
        if z in self._slice_cache:
            return self._slice_cache[z]

        fln = self.flair_stripped[:, :, z]
        bm = self.brain_mask[:, :, z]
        H, W = fln.shape

        # Q3 midline: centroid column of the brain mask (fallback = geometric centre)
        cols = np.where(bm.any(axis=0))[0]
        c = int(round(fln.shape[1] * 0.5)) if cols.size == 0 else int(round(bm.sum(axis=0) @ np.arange(W) / max(bm.sum(), 1)))
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

        gt = self.seg[:, :, z] > 0
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
    @property
    def predicted_volume(self):
        if "pred" not in self._cache:
            pred = np.zeros(self.seg.shape, dtype=bool)
            for z in self.slice_indices:
                pred[:, :, z] = self.process_slice(z)["candidate"]
            self._cache["pred"] = pred
        return self._cache["pred"]

    @property
    def gt_whole(self):
        return self.seg > 0

    def volume_dice(self):
        return _dice(self.predicted_volume, self.gt_whole)

    def volume_features(self):
        rows = [self.process_slice(z)["features"] for z in self.slice_indices]
        if not rows:
            return {"AD": 0.0, "MD": 0.0, "BC": 0.0}
        return {k: float(np.mean([r[k] for r in rows])) for k in ("AD", "MD", "BC")}

    def summary(self):
        return {
            "patient_id": self.patient_id,
            "n_slices_processed": len(self.slice_indices),
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
