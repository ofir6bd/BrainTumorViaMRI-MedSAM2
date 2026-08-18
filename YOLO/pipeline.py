"""YOLO tumour-segmentation dataset builder + per-patient inference pipeline.

Implements README.md:
- Stage A-C: RGB frame + YOLO-seg label generation, dataset assembly & split
  (`build_dataset`).
- Stage E: per-patient inference pipeline (`YoloPipeline`), used by `render.py` and
  `routes.py` (Stage F, web UI).

Data source (config.yaml -> paths.extract_to, default `data/dataset`):
- Train pool : `training_data1_v2` + `training_data_additional` (split into train/val
  by patient, deterministic via `seed`/`val_fraction`).
- Test       : `validation_data` (fully isolated, never trained on).
"""
import glob
import os
import random

import nibabel as nib
import numpy as np
import yaml
from scipy import ndimage
from skimage import measure

# ---------------------------------------------------------------------------
# Parameters (README.md §4)
# ---------------------------------------------------------------------------
PARAMS = {
    "min_mask_area": 50,
    "min_fg_voxels": 100,
    "val_fraction": 0.2,
    "seed": 42,
    "imgsz": 512,
    "epochs": 100,
    "batch": 16,
    "conf_threshold": 0.25,
    "mask_threshold": 0.5,
}

MODALITY_SUFFIX = {"T1C": "-t1c", "T1": "-t1n", "T2": "-t2w", "FLAIR": "-t2f"}

_HERE = os.path.dirname(os.path.abspath(__file__))
_DATASET_DIR = os.path.join(_HERE, "dataset")
_WEIGHTS_PATH = os.path.join(_HERE, "weights", "best.pt")


# ---------------------------------------------------------------------------
# Data source discovery
# ---------------------------------------------------------------------------
def _config():
    cfg_path = "config.yaml"
    if os.path.exists(cfg_path):
        with open(cfg_path, "r") as f:
            return yaml.safe_load(f) or {}
    return {}


def _data_root():
    """`data/dataset/` — parent of the three split source folders."""
    cfg = _config()
    return cfg.get("paths", {}).get("extract_to", "data/dataset")


def train_pool_dirs():
    """Folders that make up the combined train+val patient pool."""
    root = _data_root()
    return [
        os.path.join(root, "training_data1_v2"),
        os.path.join(root, "training_data_additional"),
    ]


def test_dir():
    """Folder used as the held-out test split (never trained on)."""
    return os.path.join(_data_root(), "validation_data")


def _find_modalities(patient_dir):
    files = glob.glob(os.path.join(patient_dir, "*.nii*"))
    out = {}
    for role, suf in MODALITY_SUFFIX.items():
        for f in files:
            if suf in os.path.basename(f).lower():
                out[role] = f
                break
    seg = next((f for f in files if "-seg" in os.path.basename(f).lower()), None)
    out["SEG"] = seg
    return out


def _list_patients(root, require_seg=True):
    """List patient dirs under `root` that have all four modalities.

    `validation_data` is the official BraTS *challenge* validation split — it ships
    without expert `-seg` masks (held out for leaderboard submission), so callers must
    pass `require_seg=False` for it.
    """
    patients = []
    if not os.path.isdir(root):
        return patients
    required = ("T1C", "T1", "T2", "FLAIR") + (("SEG",) if require_seg else ())
    for name in sorted(os.listdir(root)):
        d = os.path.join(root, name)
        if not os.path.isdir(d):
            continue
        mods = _find_modalities(d)
        if all(mods.get(k) for k in required):
            patients.append({"patient_id": name, "dir": d})
    return patients


def split_patients(params=None, max_patients=None, fraction=None):
    """Deterministic patient-level split: train / val (from the train pool) / test.

    - `train`/`val`: from the combined train pool, both **labeled** (have `-seg`).
    - `test`: `validation_data` patients — **unlabeled** (no `-seg`; BraTS challenge
      validation split). Usable only for qualitative inference, never for Dice/mAP.

    `fraction` (0 < fraction <= 1, default `None` = all): keeps this fraction of each
    split's patients (train, val, test independently), preserving the train/val ratio
    -- e.g. `fraction=0.5` trains on half the data while keeping the same ~80/20
    train/val proportion. Applied on the already-shuffled deterministic order, so it's
    still a reproducible subset.

    `max_patients` caps each split's **absolute** patient count on top of `fraction`.
    It's a smoke-test-only knob (small n) — leave it `None` for a real run; unlike
    `fraction` it does NOT preserve the train/val ratio (a small absolute cap applied
    to both splits equally will disproportionately shrink the larger split).

    Returns {"train": [...], "val": [...], "test": [...]}, each a list of
    {"patient_id", "dir"} dicts.
    """
    p = dict(PARAMS)
    if params:
        p.update(params)

    pool = []
    for root in train_pool_dirs():
        pool.extend(_list_patients(root, require_seg=True))
    pool.sort(key=lambda r: r["patient_id"])

    rng = random.Random(p["seed"])
    shuffled = list(pool)
    rng.shuffle(shuffled)

    n_val = int(round(len(shuffled) * p["val_fraction"]))
    val = shuffled[:n_val]
    train = shuffled[n_val:]
    test = _list_patients(test_dir(), require_seg=False)

    if fraction is not None:
        if not (0 < fraction <= 1):
            raise ValueError("fraction must be in (0, 1]")
        train = train[:max(1, round(len(train) * fraction))]
        val = val[:max(1, round(len(val) * fraction))]
        test = test[:max(1, round(len(test) * fraction))]

    if max_patients is not None:
        train = train[:max_patients]
        val = val[:max_patients]
        test = test[:max_patients]
    return {"train": train, "val": val, "test": test}


# ---------------------------------------------------------------------------
# §2 RGB channel construction
# ---------------------------------------------------------------------------
def _norm_slice_uint8(slice_2d, min_fg_voxels):
    """Per-slice p0.5-p99.5 normalisation to uint8; black if too little foreground."""
    s = slice_2d.astype(np.float32)
    nonzero = s[s != 0]
    if nonzero.size < min_fg_voxels:
        return np.zeros(s.shape, dtype=np.uint8)
    lo = float(np.percentile(nonzero, 0.5))
    hi = float(np.percentile(nonzero, 99.5))
    rng = hi - lo
    if rng < 1e-8:
        return np.zeros(s.shape, dtype=np.uint8)
    return np.clip((s - lo) / rng * 255, 0, 255).astype(np.uint8)


def build_rgb_slice(t1c_slice, t1_slice, t2w_slice, t2f_slice, min_fg_voxels):
    """§2: R = clip(T1C-T1, 0, None), G = T2, B = FLAIR -> HxWx3 uint8."""
    diff = np.clip(t1c_slice.astype(np.float32) - t1_slice.astype(np.float32), 0, None)
    r = _norm_slice_uint8(diff, min_fg_voxels)
    g = _norm_slice_uint8(t2w_slice, min_fg_voxels)
    b = _norm_slice_uint8(t2f_slice, min_fg_voxels)
    return np.stack([r, g, b], axis=-1)


def _has_foreground(t2f_slice, min_fg_voxels):
    return int((t2f_slice != 0).sum()) >= min_fg_voxels


# ---------------------------------------------------------------------------
# Stage B — polygon labels from the expert mask
# ---------------------------------------------------------------------------
def _polygons_from_mask(mask2d, min_mask_area):
    """Connected components of `mask2d` -> list of (N, 2) [row, col] polygons."""
    labelled, n = ndimage.label(mask2d)
    polys = []
    for comp_id in range(1, n + 1):
        comp = (labelled == comp_id)
        if int(comp.sum()) < min_mask_area:
            continue
        contours = measure.find_contours(comp.astype(np.float32), level=0.5)
        if not contours:
            continue
        polys.append(max(contours, key=len))
    return polys


def _yolo_seg_lines(polys, h, w):
    lines = []
    for contour in polys:
        if len(contour) < 3:
            continue
        coords = []
        for row, col in contour:
            x = min(max(col / w, 0.0), 1.0)
            y = min(max(row / h, 0.0), 1.0)
            coords.append(f"{x:.6f} {y:.6f}")
        lines.append("0 " + " ".join(coords))
    return lines


# ---------------------------------------------------------------------------
# Stage A-C — dataset build
# ---------------------------------------------------------------------------
def build_dataset(output_dir=None, params=None, log=print, max_patients=None, fraction=None):
    """Build `YOLO/dataset/images/{train,val,test}` + `labels/{train,val}` + `data.yaml`.

    `test` (validation_data) has no expert `-seg` mask, so no `labels/test/` is written
    for it — see `split_patients`. Writing empty/negative labels there would fabricate
    ground truth that doesn't exist, which this project never does.

    `fraction` (0 < fraction <= 1, default `None` = all): train on this fraction of
    each split's patients while preserving the train/val ratio — see `split_patients`.

    `max_patients` (smoke-test only, default `None`): cap each split to this many
    patients — see `split_patients`.
    """
    from PIL import Image

    p = dict(PARAMS)
    if params:
        p.update(params)
    out_root = output_dir or _DATASET_DIR

    splits = split_patients(p, max_patients=max_patients, fraction=fraction)
    counts = {}
    for split_name, patients in splits.items():
        img_dir = os.path.join(out_root, "images", split_name)
        os.makedirs(img_dir, exist_ok=True)

        n_slices = 0
        for entry in patients:
            pid, pdir = entry["patient_id"], entry["dir"]
            mods = _find_modalities(pdir)
            has_gt = mods.get("SEG") is not None

            lbl_dir = None
            if has_gt:
                lbl_dir = os.path.join(out_root, "labels", split_name)
                os.makedirs(lbl_dir, exist_ok=True)

            t1c = nib.load(mods["T1C"]).get_fdata()
            t1 = nib.load(mods["T1"]).get_fdata()
            t2w = nib.load(mods["T2"]).get_fdata()
            t2f = nib.load(mods["FLAIR"]).get_fdata()
            seg = nib.load(mods["SEG"]).get_fdata() if has_gt else None
            depth = t2f.shape[2]

            for z in range(depth):
                t2f_slice = t2f[:, :, z]
                if not _has_foreground(t2f_slice, p["min_fg_voxels"]):
                    continue

                rgb = build_rgb_slice(
                    t1c[:, :, z], t1[:, :, z], t2w[:, :, z], t2f_slice,
                    p["min_fg_voxels"],
                )
                stem = f"{pid}_z{z + 1:03d}"
                Image.fromarray(rgb, mode="RGB").save(os.path.join(img_dir, f"{stem}.png"))

                if lbl_dir is not None:
                    wt = seg[:, :, z] > 0
                    polys = _polygons_from_mask(wt, p["min_mask_area"])
                    h, w = wt.shape
                    lines = _yolo_seg_lines(polys, h, w)
                    with open(os.path.join(lbl_dir, f"{stem}.txt"), "w", encoding="utf-8") as f:
                        if lines:
                            f.write("\n".join(lines) + "\n")
                n_slices += 1

            log(f"[{split_name}] {pid}: {depth} slices scanned")
        counts[split_name] = n_slices
        log(f"[{split_name}] {len(patients)} patients, {n_slices} slices kept"
            + ("" if split_name != "test" else " (unlabeled, inference-only)"))

    data_yaml = os.path.join(out_root, "data.yaml")
    with open(data_yaml, "w", encoding="utf-8") as f:
        f.write(
            f"path: {out_root}\n"
            "train: images/train\n"
            "val: images/val\n"
            "test: images/test\n"
            "names:\n"
            "  0: tumour\n"
        )
    log(f"Wrote {data_yaml}")
    return {"data_yaml": data_yaml, "counts": counts, "splits": splits}


# ---------------------------------------------------------------------------
# Stage E — per-patient inference pipeline (used by render.py / routes.py)
# ---------------------------------------------------------------------------
def _dice(a, b):
    a = a.astype(bool)
    b = b.astype(bool)
    denom = a.sum() + b.sum()
    if denom == 0:
        return 1.0
    return float(2.0 * np.logical_and(a, b).sum() / denom)


class YoloPipeline:
    """Lazy, cached per-patient inference pipeline for the web UI (Stage E/F)."""

    def __init__(self, patient_dir, params=None, weights_path=None):
        self.patient_dir = patient_dir
        self.patient_id = os.path.basename(os.path.normpath(patient_dir))
        self.p = dict(PARAMS)
        if params:
            self.p.update(params)
        self.paths = _find_modalities(patient_dir)
        self.weights_path = weights_path or _WEIGHTS_PATH
        self._cache = {}
        self._slice_cache = {}
        self._model = None
        self._model_error = None

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
    def t1c(self):
        return self._vol("T1C")

    @property
    def t1(self):
        return self._vol("T1")

    @property
    def t2w(self):
        return self._vol("T2")

    @property
    def flair(self):
        return self._vol("FLAIR")

    @property
    def seg(self):
        return self._vol("SEG")

    @property
    def depth(self):
        return int(self.seg.shape[2])

    # -- slice bookkeeping ---------------------------------------------------
    def brain_count(self, z):
        return int((self.flair[:, :, z] != 0).sum())

    def is_processed(self, z):
        return self.brain_count(z) >= self.p["min_fg_voxels"]

    @property
    def slice_indices(self):
        if "slice_indices" not in self._cache:
            self._cache["slice_indices"] = [
                z for z in range(self.depth) if self.is_processed(z)
            ]
        return self._cache["slice_indices"]

    def best_slice_index(self):
        idx = self.slice_indices
        if not idx:
            return 0
        seg = self.seg
        sums = [int((seg[:, :, z] > 0).sum()) for z in idx]
        return int(np.argmax(sums)) if max(sums) > 0 else len(idx) // 2

    def rgb_slice(self, z):
        return build_rgb_slice(
            self.t1c[:, :, z], self.t1[:, :, z], self.t2w[:, :, z], self.flair[:, :, z],
            self.p["min_fg_voxels"],
        )

    # -- model (lazy, optional) ----------------------------------------------
    def _model_instance(self):
        if self._model is not None or self._model_error is not None:
            return self._model
        if not os.path.exists(self.weights_path):
            self._model_error = (
                f"weights not found: {self.weights_path} (run YOLO/train.py first)"
            )
            return None
        try:
            from ultralytics import YOLO
            self._model = YOLO(self.weights_path)
        except Exception as e:  # pragma: no cover - best effort
            self._model_error = str(e)
        return self._model

    # -- Stage E: per-slice inference + Dice ---------------------------------
    def process_slice(self, z):
        if z in self._slice_cache:
            return self._slice_cache[z]

        rgb = self.rgb_slice(z)
        gt_wt = self.seg[:, :, z] > 0
        pred_wt = np.zeros_like(gt_wt, dtype=bool)

        model = self._model_instance()
        if model is not None:
            results = model.predict(rgb, conf=self.p["conf_threshold"], verbose=False)
            r0 = results[0]
            if r0.masks is not None:
                for m in r0.masks.data.cpu().numpy():
                    mask_bin = m >= self.p["mask_threshold"]
                    if mask_bin.shape != gt_wt.shape:
                        zy = gt_wt.shape[0] / mask_bin.shape[0]
                        zx = gt_wt.shape[1] / mask_bin.shape[1]
                        mask_bin = ndimage.zoom(mask_bin.astype(np.float32), (zy, zx),
                                                order=0) >= 0.5
                    pred_wt |= mask_bin

        out = {
            "z": z,
            "rgb": rgb,
            "gt_wt": gt_wt,
            "pred_wt": pred_wt,
            "dice": _dice(pred_wt, gt_wt),
            "model_error": self._model_error,
        }
        self._slice_cache[z] = out
        return out

    def dice_table(self):
        """Per-slice Dice for all valid (processed) slices."""
        return [{"z": z, "dice": self.process_slice(z)["dice"]} for z in self.slice_indices]
