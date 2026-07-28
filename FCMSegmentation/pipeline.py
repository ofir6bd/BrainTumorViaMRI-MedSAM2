"""Automatic brain tumor detection and volume estimation pipeline.

This module follows the methodology of:
"Automatic Brain Tumor Detection and Volume Estimation in Multimodal MRI Scans
via a Symmetry Analysis" (Symmetry, 2023), adapted to BraTS where skull stripping
is skipped because inputs are already skull-stripped.

Stages implemented per slice:
1) Whole tumor segmentation on FLAIR (FCM + connected components)
"""
import os
import glob

import numpy as np
import nibabel as nib
from scipy import ndimage

# ---------------------------------------------------------------------------
# Parameters
# ---------------------------------------------------------------------------
PARAMS = {
    "n_min_voxel": 500,
    "fcm_clusters": 3,
    "fcm_m": 2.0,
    "fcm_max_iter": 100,
    "fcm_tol": 1e-5,
    "min_component": 40,
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


def _fcm_1d(values, n_clusters=3, m=2.0, max_iter=100, tol=1e-5):
    values = np.asarray(values, dtype=np.float64).ravel()
    n = values.size
    if n == 0:
        return np.array([], dtype=int), np.array([], dtype=np.float64), np.empty((0, 0), dtype=np.float64)

    n_clusters = int(max(2, min(n_clusters, max(2, n))))
    quant = np.linspace(5, 95, n_clusters)
    centers = np.percentile(values, quant)
    eps = 1e-12
    for _ in range(max_iter):
        dist = np.abs(values[:, None] - centers[None, :]) + eps
        power = 2.0 / (m - 1.0)
        inv = dist ** (-power)
        u = inv / np.maximum(inv.sum(axis=1, keepdims=True), eps)

        um = u ** m
        denom = np.maximum(um.sum(axis=0), eps)
        new_centers = (um * values[:, None]).sum(axis=0) / denom

        if np.max(np.abs(new_centers - centers)) < tol:
            centers = new_centers
            break
        centers = new_centers

    labels = np.argmax(u, axis=1).astype(int)
    return labels, centers.astype(np.float64), u


def _largest_group_pixels(mask2d):
    lab, nlab = ndimage.label(mask2d)
    if nlab == 0:
        return 0
    sizes = ndimage.sum(mask2d, labels=lab, index=np.arange(1, nlab + 1))
    return int(np.max(sizes)) if np.size(sizes) else 0


def _largest_group_mask(mask2d):
    lab, nlab = ndimage.label(mask2d)
    if nlab == 0:
        return np.zeros_like(mask2d, dtype=bool)
    sizes = ndimage.sum(mask2d, labels=lab, index=np.arange(1, nlab + 1))
    if not np.size(sizes):
        return np.zeros_like(mask2d, dtype=bool)
    best = int(np.argmax(sizes)) + 1
    return (lab == best)


class FcmSegmentationPipeline:
    """Lazy, cached paper-style pipeline for one patient directory."""

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
    def flair_norm(self):
        if "flair_norm" not in self._cache:
            fl = self.flair
            hi = float(fl.max()) or 1.0
            self._cache["flair_norm"] = (fl / hi).astype(np.float32)
        return self._cache["flair_norm"]

    @property
    def flair_med(self):
        if "flair_med" not in self._cache:
            self._cache["flair_med"] = self.flair_norm.astype(np.float32)
        return self._cache["flair_med"]

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
        return self._canon(self.flair_norm[:, :, z])

    def slice_flair_med(self, z):
        return self._canon(self.flair_med[:, :, z])

    def slice_brain(self, z):
        return self._canon(self.brain_mask[:, :, z])

    def slice_seg(self, z):
        return self._canon(self.seg[:, :, z])

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

    def _whole_tumor_from_flair(self, flair_med, brain):

        idx = np.where(brain)
        vals = flair_med[idx]
        if vals.size < 3:
            empty = np.zeros_like(brain, dtype=bool)
            return empty, empty, np.array([])

        labels, centers, _ = _fcm_1d(
            vals,
            n_clusters=self.p["fcm_clusters"],
            m=self.p["fcm_m"],
            max_iter=self.p["fcm_max_iter"],
            tol=self.p["fcm_tol"],
        )

        cluster_map = np.full(flair_med.shape, -1, dtype=int)
        cluster_map[idx] = labels

        min_component = int(self.p["min_component"])
        brightest = int(np.argmax(centers)) if centers.size else -1
        fcm_mask = (cluster_map == brightest) if brightest >= 0 else np.zeros_like(brain, dtype=bool)

        labeled, nlab = ndimage.label(fcm_mask)
        best_cc = None
        best_area = -1
        for k in range(1, nlab + 1):
            cc = (labeled == k)
            area = int(cc.sum())
            if area >= min_component and area > best_area:
                best_area = area
                best_cc = cc

        cc_mask = best_cc if best_cc is not None else fcm_mask
        return fcm_mask.astype(bool), cc_mask.astype(bool), centers

    # -- Stages 1-4 (per slice) -------------------------------------------
    def process_slice(self, z):
        if z in self._slice_cache:
            return self._slice_cache[z]

        flair_med = self.slice_flair_med(z)
        bm = self.slice_brain(z)

        whole_fcm, whole_cc, whole_centers = self._whole_tumor_from_flair(flair_med, bm)

        seg2d = self.slice_seg(z)
        gt_whole = seg2d > 0

        whole_dice_fcm_cc = _dice(whole_cc, gt_whole)
        whole_dice = whole_dice_fcm_cc
        largest_group_pixels = _largest_group_pixels(whole_cc)

        whole_pixels = int(whole_cc.sum())

        out = {
            "z": z,
            "brain_count": int(bm.sum()),
            "flair_med": flair_med,
            "brain": bm,
            "fcm_centers": {
                "whole": whole_centers.tolist() if isinstance(whole_centers, np.ndarray) else [],
            },
            "whole_fcm": whole_fcm,
            "whole_raw": whole_cc,
            "whole_mask": whole_cc,
            "gt_whole": gt_whole,
            "slice_dice_whole_fcm_cc": whole_dice_fcm_cc,
            "slice_dice_whole": whole_dice,
            "largest_group_pixels": largest_group_pixels,
            "whole_pixels": whole_pixels,
        }
        self._slice_cache[z] = out
        return out

    @property
    def pixel_area_mm2(self):
        if "pixel_area_mm2" not in self._cache:
            img = nib.load(self.paths["FLAIR"])
            zooms = img.header.get_zooms()
            dx = float(zooms[0]) if len(zooms) > 0 else 1.0
            dy = float(zooms[1]) if len(zooms) > 1 else 1.0
            self._cache["pixel_area_mm2"] = dx * dy
        return self._cache["pixel_area_mm2"]

    @property
    def slice_step_mm(self):
        if "slice_step_mm" not in self._cache:
            img = nib.load(self.paths["FLAIR"])
            zooms = img.header.get_zooms()
            dz = float(zooms[2]) if len(zooms) > 2 else 1.0
            self._cache["slice_step_mm"] = dz
        return self._cache["slice_step_mm"]

    # -- Stage 6 -----------------------------------------------------------
    def whole_mask(self, z):
        if self.is_processed(z):
            return self.process_slice(z)["whole_mask"]
        return np.zeros(self.slice_seg(z).shape, dtype=bool)

    def volume_dice(self):
        if "vdice" not in self._cache:
            procset = set(self.slice_indices)
            inter = pa = ga = 0
            for z in range(self.depth):
                gt = self.slice_seg(z) > 0
                cand = self.process_slice(z)["whole_mask"] if z in procset \
                    else np.zeros_like(gt)
                inter += int(np.logical_and(cand, gt).sum())
                pa += int(cand.sum())
                ga += int(gt.sum())
            self._cache["vdice"] = 1.0 if (pa + ga) == 0 else float(2.0 * inter / (pa + ga))
        return self._cache["vdice"]

    def volume_estimates(self):
        whole_px_sum = 0.0
        for z in self.slice_indices:
            s = self.process_slice(z)
            whole_px_sum += float(s["whole_pixels"])

        step = self.slice_step_mm
        return {
            "whole_volume_mm3": float(whole_px_sum * self.pixel_area_mm2 * step),
            "slice_step_mm": float(step),
            "pixel_area_mm2": float(self.pixel_area_mm2),
        }

    def summary(self):
        return {
            "patient_id": self.patient_id,
            "n_slices_processed": len(self.slice_indices),
            "volume_dice": self.volume_dice(),
        }

    def feature_table(self):
        """Per-slice whole-tumor outputs for every processed slice."""
        rows = []
        for z in self.slice_indices:
            s = self.process_slice(z)
            rows.append({
                "z": z,
                "brain_voxels": s["brain_count"],
                "whole_dice": s["slice_dice_whole"],
                "largest_group_pixels": s["largest_group_pixels"],
            })
        return rows

    def save_csv(self, out_root):
        out_dir = os.path.join(out_root, self.patient_id)
        os.makedirs(out_dir, exist_ok=True)
        path = os.path.join(out_dir, "features.csv")
        rows = self.feature_table()
        with open(path, "w", encoding="utf-8") as f:
            f.write("z,brain_voxels,whole_dice,largest_group_pixels\n")
            for r in rows:
                f.write(
                    f"{r['z']},{r['brain_voxels']},{r['whole_dice']:.6f},{r['largest_group_pixels']}\n"
                )
            summ = self.summary()
            f.write(
                f"volume,,{summ['volume_dice']:.6f},\n"
            )
        return path


def _dice(a, b):
    a = a.astype(bool)
    b = b.astype(bool)
    denom = a.sum() + b.sum()
    if denom == 0:
        return 1.0
    return float(2.0 * np.logical_and(a, b).sum() / denom)
