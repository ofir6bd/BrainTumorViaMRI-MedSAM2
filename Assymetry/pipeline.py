"""Automatic brain tumor detection and volume estimation pipeline.

This module follows the methodology of:
"Automatic Brain Tumor Detection and Volume Estimation in Multimodal MRI Scans
via a Symmetry Analysis" (Symmetry, 2023), adapted to BraTS where skull stripping
is skipped because inputs are already skull-stripped.

Stages implemented per slice:
1) Median filtering (FLAIR + T1C)
2) Symmetry analysis on FLAIR (AD, MD, BC -> asymmetry score)
3) Whole tumor segmentation on FLAIR (FCM + connected components + opening)
4) Active core segmentation on T1C (FCM + opening + whole-mask ROI)
5) Necrotic core estimation (closing(active) - active)
6) Area and volume estimation
"""
import os
import glob

import numpy as np
import nibabel as nib
from scipy import ndimage
from skimage.morphology import opening, closing, disk

# ---------------------------------------------------------------------------
# Parameters
# ---------------------------------------------------------------------------
PARAMS = {
    "n_min_voxel": 500,
    "fcm_clusters": 3,
    "fcm_m": 2.0,
    "fcm_max_iter": 100,
    "fcm_tol": 1e-5,
    "feat_bins": 64,      # bins for the Bhattacharyya coefficient
    "md_threshold": 0.02,
    "bc_threshold": 0.02,
    "ad_threshold": 600.0,
    "whole_open_radius": 2,
    "active_open_radius": 1,
    "active_roi_quantile": 0.70,
    "active_min_component": 20,
    "necrotic_close_radius": 2,
    "min_component": 40,
}

MODALITY_SUFFIX = {"FLAIR": "-t2f", "T1C": "-t1c"}


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


def _bhattacharyya(a, b, bins, rng):
    ha, _ = np.histogram(a, bins=bins, range=rng, density=False)
    hb, _ = np.histogram(b, bins=bins, range=rng, density=False)
    ha = ha / max(ha.sum(), 1)
    hb = hb / max(hb.sum(), 1)
    return float(np.sum(np.sqrt(ha * hb)))


class AsymmetryPipeline:
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
    def t1c(self):
        return self._vol("T1C")

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
    def t1c_norm(self):
        if "t1c_norm" not in self._cache:
            im = self.t1c
            hi = float(im.max()) or 1.0
            self._cache["t1c_norm"] = (im / hi).astype(np.float32)
        return self._cache["t1c_norm"]

    @property
    def flair_med(self):
        if "flair_med" not in self._cache:
            self._cache["flair_med"] = ndimage.median_filter(self.flair_norm, size=(3, 3, 1)).astype(np.float32)
        return self._cache["flair_med"]

    @property
    def t1c_med(self):
        if "t1c_med" not in self._cache:
            self._cache["t1c_med"] = ndimage.median_filter(self.t1c_norm, size=(3, 3, 1)).astype(np.float32)
        return self._cache["t1c_med"]

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

    def slice_t1c(self, z):
        return self._canon(self.t1c_norm[:, :, z])

    def slice_flair_med(self, z):
        return self._canon(self.flair_med[:, :, z])

    def slice_t1c_med(self, z):
        return self._canon(self.t1c_med[:, :, z])

    def slice_brain(self, z):
        return self._canon(self.brain_mask[:, :, z])

    def slice_seg(self, z):
        return self._canon(self.seg[:, :, z])

    def _midline(self, brain):
        if brain.sum() == 0:
            return brain.shape[1] // 2
        cols = np.arange(brain.shape[1])
        c = int(round((brain.sum(axis=0) @ cols) / max(brain.sum(), 1)))
        return int(np.clip(c, 1, brain.shape[1] - 1))

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

    def _binary_fcm_mask(self, image, brain):
        mask = np.zeros(image.shape, dtype=bool)
        idx = np.where(brain)
        vals = image[idx]
        if vals.size < 3:
            return mask, np.array([], dtype=int), np.array([], dtype=np.float64)

        labels, centers, _ = _fcm_1d(
            vals,
            n_clusters=self.p["fcm_clusters"],
            m=self.p["fcm_m"],
            max_iter=self.p["fcm_max_iter"],
            tol=self.p["fcm_tol"],
        )
        if centers.size == 0:
            return mask, labels, centers

        bright = int(np.argmax(centers))
        chosen = labels == bright
        mask[idx] = chosen
        return mask, labels, centers

    def _whole_tumor_from_flair(self, flair_med, brain):

        idx = np.where(brain)
        vals = flair_med[idx]
        if vals.size < 3:
            return np.zeros_like(brain, dtype=bool), np.zeros_like(brain, dtype=bool), np.array([])

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
        bright_mask = (cluster_map == brightest) if brightest >= 0 else np.zeros_like(brain, dtype=bool)

        labeled, nlab = ndimage.label(bright_mask)
        best_cc = None
        best_area = -1
        for k in range(1, nlab + 1):
            cc = (labeled == k)
            area = int(cc.sum())
            if area >= min_component and area > best_area:
                best_area = area
                best_cc = cc

        raw = best_cc if best_cc is not None else bright_mask
        opened = opening(raw, footprint=disk(self.p["whole_open_radius"]))
        if opened.sum() == 0 and raw.sum() > 0:
            opened = raw
        return raw.astype(bool), opened.astype(bool), centers

    def _active_and_necrotic(self, t1c_med, brain, whole_mask):
        active_bin, labels, centers = self._binary_fcm_mask(t1c_med, brain)
        active = np.zeros_like(whole_mask, dtype=bool)
        if whole_mask.sum() > 0:
            roi_vals = t1c_med[whole_mask]
            q = float(self.p["active_roi_quantile"])
            q = float(np.clip(q, 0.05, 0.95))
            thr = float(np.quantile(roi_vals, q))

            active_raw = whole_mask & (t1c_med >= thr)
            active_open = opening(active_raw, footprint=disk(self.p["active_open_radius"]))

            min_cc = int(self.p["active_min_component"])
            labeled, nlab = ndimage.label(active_open)
            kept = np.zeros_like(active_open, dtype=bool)
            for k in range(1, nlab + 1):
                cc = (labeled == k)
                if int(cc.sum()) >= min_cc:
                    kept |= cc
            if kept.sum() == 0 and active_open.sum() > 0:
                kept = active_open
            active = kept.astype(bool)

        closed = closing(active, footprint=disk(self.p["necrotic_close_radius"]))
        necrotic = (closed & (~active) & whole_mask).astype(bool)
        return active_bin.astype(bool), active.astype(bool), necrotic.astype(bool), centers

    # -- Stages 1-5 (per slice) -------------------------------------------
    def process_slice(self, z):
        if z in self._slice_cache:
            return self._slice_cache[z]

        flair_raw = self.slice_flair(z)
        flair_med = self.slice_flair_med(z)
        t1c_raw = self.slice_t1c(z)
        t1c_med = self.slice_t1c_med(z)
        bm = self.slice_brain(z)

        H, W = flair_med.shape
        c = self._midline(bm)
        half_w = min(c, W - c)

        left_i = flair_med[:, c - half_w:c]
        right_i = flair_med[:, c:c + half_w]
        left_b = bm[:, c - half_w:c]
        right_b = bm[:, c:c + half_w]

        # symmetry-binary map via FCM on full FLAIR slice
        bright_full, _, flair_centers = self._binary_fcm_mask(flair_med, bm)
        left_bin = bright_full[:, c - half_w:c]
        right_bin = bright_full[:, c:c + half_w]

        ad = float(abs(int(left_bin.sum()) - int(right_bin.sum())))

        lv = left_i[left_b]
        rv = right_i[right_b]
        md = float(abs(lv.mean() - rv.mean())) if lv.size and rv.size else 0.0
        bc = _bhattacharyya(lv, rv, self.p["feat_bins"], (0.0, 1.0)) if lv.size and rv.size else 1.0
        bc_asym = float(1.0 - bc)

        l1 = 1 if md > self.p["md_threshold"] else -1
        l2 = 1 if bc_asym > self.p["bc_threshold"] else -1
        l3 = 1 if ad > self.p["ad_threshold"] else -1
        score = int(l1 + l2 + l3)
        detected = bool(score > 0)

        whole_raw, whole_mask, whole_centers = self._whole_tumor_from_flair(flair_med, bm)
        active_bin, active_mask, necrotic_mask, t1c_centers = self._active_and_necrotic(t1c_med, bm, whole_mask)

        seg2d = self.slice_seg(z)
        gt_whole = seg2d > 0
        gt_active = np.isin(seg2d, [1, 3, 4])

        whole_dice = _dice(whole_mask, gt_whole)
        active_dice = _dice(active_mask, gt_active)

        area_px_whole = int(whole_mask.sum())
        area_px_active = int(active_mask.sum())
        area_px_necrotic = int(necrotic_mask.sum())

        area_mm2_whole = area_px_whole * self.pixel_area_mm2
        area_mm2_active = area_px_active * self.pixel_area_mm2
        area_mm2_necrotic = area_px_necrotic * self.pixel_area_mm2

        out = {
            "z": z,
            "midline": c,
            "half_w": half_w,
            "brain_count": int(bm.sum()),
            "flair_raw": flair_raw,
            "flair_med": flair_med,
            "t1c_raw": t1c_raw,
            "t1c_med": t1c_med,
            "brain": bm,
            "left_bin": left_bin,
            "right_bin": right_bin,
            "symmetry_bright": bright_full,
            "features": {
                "AD": ad,
                "MD": md,
                "BC": float(bc),
                "BC_asym": bc_asym,
            },
            "labels": {"L1": l1, "L2": l2, "L3": l3},
            "thresholds": {
                "MD": float(self.p["md_threshold"]),
                "BC_asym": float(self.p["bc_threshold"]),
                "AD": float(self.p["ad_threshold"]),
            },
            "asymmetry_score": score,
            "tumor_detected": detected,
            "fcm_centers": {
                "flair": flair_centers.tolist() if isinstance(flair_centers, np.ndarray) else [],
                "whole": whole_centers.tolist() if isinstance(whole_centers, np.ndarray) else [],
                "t1c": t1c_centers.tolist() if isinstance(t1c_centers, np.ndarray) else [],
            },
            "whole_raw": whole_raw,
            "whole_mask": whole_mask,
            "active_bin": active_bin,
            "active_mask": active_mask,
            "necrotic_mask": necrotic_mask,
            "gt_whole": gt_whole,
            "gt_active": gt_active,
            "slice_dice_whole": whole_dice,
            "slice_dice_active": active_dice,
            "area_px": {
                "whole": area_px_whole,
                "active": area_px_active,
                "necrotic": area_px_necrotic,
            },
            "area_mm2": {
                "whole": area_mm2_whole,
                "active": area_mm2_active,
                "necrotic": area_mm2_necrotic,
            },
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

    def active_mask(self, z):
        if self.is_processed(z):
            return self.process_slice(z)["active_mask"]
        return np.zeros(self.slice_seg(z).shape, dtype=bool)

    def necrotic_mask(self, z):
        if self.is_processed(z):
            return self.process_slice(z)["necrotic_mask"]
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

    def active_volume_dice(self):
        if "active_vdice" not in self._cache:
            procset = set(self.slice_indices)
            inter = pa = ga = 0
            for z in range(self.depth):
                gt = np.isin(self.slice_seg(z), [1, 3, 4])
                pred = self.process_slice(z)["active_mask"] if z in procset else np.zeros_like(gt)
                inter += int(np.logical_and(pred, gt).sum())
                pa += int(pred.sum())
                ga += int(gt.sum())
            self._cache["active_vdice"] = 1.0 if (pa + ga) == 0 else float(2.0 * inter / (pa + ga))
        return self._cache["active_vdice"]

    def volume_features(self):
        rows = [self.process_slice(z)["features"] for z in self.slice_indices]
        if not rows:
            return {"AD": 0.0, "MD": 0.0, "BC": 0.0, "BC_asym": 0.0}
        return {k: float(np.mean([r[k] for r in rows])) for k in ("AD", "MD", "BC", "BC_asym")}

    def detection_metrics(self):
        tp = tn = fp = fn = 0
        for z in self.slice_indices:
            s = self.process_slice(z)
            pred = bool(s["tumor_detected"])
            gt = bool((s["gt_whole"]).any())
            if pred and gt:
                tp += 1
            elif (not pred) and (not gt):
                tn += 1
            elif pred and (not gt):
                fp += 1
            else:
                fn += 1
        total = tp + tn + fp + fn
        acc = (tp + tn) / total if total else 1.0
        sens = tp / (tp + fn) if (tp + fn) else 1.0
        spec = tn / (tn + fp) if (tn + fp) else 1.0
        return {
            "tp": tp,
            "tn": tn,
            "fp": fp,
            "fn": fn,
            "accuracy": float(acc),
            "sensitivity": float(sens),
            "specificity": float(spec),
        }

    def volume_estimates(self):
        active_area_sum = 0.0
        gt_active_area_sum = 0.0
        whole_area_sum = 0.0
        for z in self.slice_indices:
            s = self.process_slice(z)
            active_area_sum += s["area_mm2"]["active"]
            whole_area_sum += s["area_mm2"]["whole"]
            gt_active_area_sum += float(np.isin(self.slice_seg(z), [1, 3, 4]).sum()) * self.pixel_area_mm2

        step = self.slice_step_mm
        return {
            "whole_volume_mm3": float(whole_area_sum * step),
            "active_volume_mm3": float(active_area_sum * step),
            "gt_active_volume_mm3": float(gt_active_area_sum * step),
            "slice_step_mm": float(step),
            "pixel_area_mm2": float(self.pixel_area_mm2),
        }

    def summary(self):
        det = self.detection_metrics()
        vol = self.volume_estimates()
        return {
            "patient_id": self.patient_id,
            "n_slices_processed": len(self.slice_indices),
            "volume_dice": self.volume_dice(),
            "active_volume_dice": self.active_volume_dice(),
            "volume_features": self.volume_features(),
            "detection": det,
            "volume_estimation": vol,
            "thresholds": {
                "MD": float(self.p["md_threshold"]),
                "BC_asym": float(self.p["bc_threshold"]),
                "AD": float(self.p["ad_threshold"]),
            },
        }

    def feature_table(self):
        """Per-slice paper-stage outputs for every processed slice."""
        rows = []
        for z in self.slice_indices:
            s = self.process_slice(z)
            rows.append({
                "z": z,
                "brain_voxels": s["brain_count"],
                "AD": s["features"]["AD"],
                "MD": s["features"]["MD"],
                "BC": s["features"]["BC"],
                "BC_asym": s["features"]["BC_asym"],
                "score": s["asymmetry_score"],
                "detected": int(s["tumor_detected"]),
                "whole_dice": s["slice_dice_whole"],
                "active_dice": s["slice_dice_active"],
                "area_mm2_whole": s["area_mm2"]["whole"],
                "area_mm2_active": s["area_mm2"]["active"],
                "area_mm2_necrotic": s["area_mm2"]["necrotic"],
            })
        return rows

    def save_csv(self, out_root):
        out_dir = os.path.join(out_root, self.patient_id)
        os.makedirs(out_dir, exist_ok=True)
        path = os.path.join(out_dir, "features.csv")
        rows = self.feature_table()
        with open(path, "w", encoding="utf-8") as f:
            f.write("z,brain_voxels,AD,MD,BC,BC_asym,score,detected,whole_dice,active_dice,area_mm2_whole,area_mm2_active,area_mm2_necrotic\n")
            for r in rows:
                f.write(
                    f"{r['z']},{r['brain_voxels']},{r['AD']:.6f},{r['MD']:.6f},{r['BC']:.6f},"
                    f"{r['BC_asym']:.6f},{r['score']},{r['detected']},{r['whole_dice']:.6f},"
                    f"{r['active_dice']:.6f},{r['area_mm2_whole']:.6f},{r['area_mm2_active']:.6f},"
                    f"{r['area_mm2_necrotic']:.6f}\n"
                )
            summ = self.summary()
            f.write(
                f"volume,,,,,,,,{summ['volume_dice']:.6f},{summ['active_volume_dice']:.6f},,,\n"
            )
        return path


def _dice(a, b):
    a = a.astype(bool)
    b = b.astype(bool)
    denom = a.sum() + b.sum()
    if denom == 0:
        return 1.0
    return float(2.0 * np.logical_and(a, b).sum() / denom)
