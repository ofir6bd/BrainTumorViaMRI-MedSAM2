"""YOLO tumour-segmentation dataset builder + per-patient inference pipeline.

Implements README.md:
- Stage A-C: RGB frame + YOLO-seg label generation, dataset assembly & split
  (`build_dataset`).
- Stage E: per-patient inference pipeline (`YoloPipeline`), used by `render.py` and
  `routes.py` (Stage F, web UI).

Data source (config.yaml -> paths.extract_to, default `data/dataset`):
- Patient pool: `training_data1_v2` + `training_data_additional`, split by patient
  (deterministic via `seed`) into train (60%) / val (20%) / test (20%) — all three
  **labeled** (have expert `-seg` masks). `test` is held out from training entirely;
  it's used only for the post-training Dice quality summary (`evaluate_test_split`,
  `YOLO/evaluate.py`) and as the web UI's patient list (`list_test_patients`), never
  for gradient updates or hyperparameter selection.
"""
import glob
import json
import os
import random
import re
from datetime import datetime

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
    "test_fraction": 0.2,
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
_WEIGHTS_DIR = os.path.join(_HERE, "weights")
_WEIGHTS_PATH = os.path.join(_WEIGHTS_DIR, "best.pt")  # legacy fallback, no metadata

_WEIGHTS_NAME_RE = re.compile(
    r"^best_(?P<stamp>\d{8}-\d{6})"
    r"(?:_valloss(?P<valloss>\d+p\d+))?"
    r"(?:_testdice(?P<testdice>\d+p\d+))?"
    r"\.pt$"
)


# ---------------------------------------------------------------------------
# Weight-file discovery (README.md Stage D/F — one `.pt` per training run)
# ---------------------------------------------------------------------------
def weights_metadata_path(pt_path):
    """Sidecar `.json` path for a weights `.pt` path (same stem, see `save_weights_metadata`)."""
    return os.path.splitext(pt_path)[0] + ".json"


def load_weights_metadata(pt_path):
    """Load the sidecar training-metadata JSON for a weights file, if present.

    Written by `YOLO/train.py` right after saving a run's checkpoint — holds the run's
    params (epochs/imgsz/batch/data_fraction/max_patients), final val_loss/test_dice,
    per-patient test Dice, and the full per-epoch `results.csv` history, so the UI's
    "Show training data" button works even after `YOLO/runs/` is deleted. Returns `None`
    if missing/unreadable (e.g. a manually renamed or pre-existing `.pt` with no sidecar).
    """
    meta_path = weights_metadata_path(pt_path)
    if not os.path.exists(meta_path):
        return None
    try:
        with open(meta_path, "r", encoding="utf-8") as f:
            return json.load(f)
    except (OSError, ValueError):
        return None


def save_weights_metadata(pt_path, metadata):
    """Write the sidecar `.json` next to a weights `.pt` (see `load_weights_metadata`)."""
    meta_path = weights_metadata_path(pt_path)
    with open(meta_path, "w", encoding="utf-8") as f:
        json.dump(metadata, f, indent=2)
    return meta_path


def list_weight_files():
    """List available YOLO weight checkpoints in `YOLO/weights/`.

    Each entry: {"filename", "path", "timestamp" (datetime|None), "val_loss" (float|None),
    "test_dice" (float|None), "has_training_info" (bool)}. Metadata is read first from
    the sidecar `.json` (see `load_weights_metadata`) and falls back to parsing the
    `best_<timestamp>[_valloss<v>][_testdice<v>].pt` filename for older/manually-placed
    checkpoints with no sidecar.

    Sorted best-first: highest `test_dice`, then lowest `val_loss`, then most recent —
    entries missing a given metric sort after those that have it.
    """
    if not os.path.isdir(_WEIGHTS_DIR):
        return []
    entries = []
    for name in sorted(os.listdir(_WEIGHTS_DIR)):
        if not name.lower().endswith(".pt"):
            continue
        path = os.path.join(_WEIGHTS_DIR, name)
        if not os.path.isfile(path):
            continue

        meta = load_weights_metadata(path)
        m = _WEIGHTS_NAME_RE.match(name)

        timestamp = None
        val_loss = None
        test_dice = None

        if meta:
            ts = meta.get("timestamp")
            if ts:
                try:
                    timestamp = datetime.fromisoformat(ts)
                except ValueError:
                    timestamp = None
            val_loss = meta.get("val_loss")
            test_dice = meta.get("test_dice")

        if m:
            if timestamp is None:
                try:
                    timestamp = datetime.strptime(m.group("stamp"), "%Y%m%d-%H%M%S")
                except ValueError:
                    timestamp = None
            if val_loss is None and m.group("valloss"):
                try:
                    val_loss = float(m.group("valloss").replace("p", "."))
                except ValueError:
                    pass
            if test_dice is None and m.group("testdice"):
                try:
                    test_dice = float(m.group("testdice").replace("p", "."))
                except ValueError:
                    pass

        entries.append({
            "filename": name,
            "path": path,
            "timestamp": timestamp,
            "val_loss": val_loss,
            "test_dice": test_dice,
            "has_training_info": meta is not None,
        })

    def sort_key(e):
        return (
            0 if e["test_dice"] is not None else 1,
            -(e["test_dice"] if e["test_dice"] is not None else 0.0),
            0 if e["val_loss"] is not None else 1,
            e["val_loss"] if e["val_loss"] is not None else 0.0,
            -(e["timestamp"].timestamp() if e["timestamp"] else 0),
        )

    entries.sort(key=sort_key)
    return entries


def default_weights_path():
    """Best-first choice among `list_weight_files()`; falls back to the fixed
    `weights/best.pt` path (pre-existing behaviour when nothing new has been trained
    yet — the model loader then reports a clear "weights not found" error).
    """
    entries = list_weight_files()
    return entries[0]["path"] if entries else _WEIGHTS_PATH


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
    """Folders that make up the combined train/val/test patient pool (all labeled)."""
    root = _data_root()
    return [
        os.path.join(root, "training_data1_v2"),
        os.path.join(root, "training_data_additional"),
    ]


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
    """List patient dirs under `root` that have all four modalities (and `-seg` if
    `require_seg`)."""
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
    """Deterministic patient-level split: train / val / test — all three **labeled**
    (have `-seg`), from the combined pool `training_data1_v2` + `training_data_additional`.

    Ratios: train 60% / val 20% / test 20% (`PARAMS["val_fraction"]`/`["test_fraction"]`).
    `test` is fully held out from training — used only for the post-training Dice
    quality summary (`evaluate_test_split`) and as the web UI's patient list, never for
    gradient updates or hyperparameter selection.

    `fraction` (0 < fraction <= 1, default `None` = all): keeps this fraction of each
    split's patients (train, val, test independently), preserving the 60/20/20 ratio
    -- e.g. `fraction=0.5` trains on half the data while keeping the same proportions.
    Applied on the already-shuffled deterministic order, so it's still a reproducible
    subset.

    `max_patients` caps each split's **absolute** patient count on top of `fraction`.
    It's a smoke-test-only knob (small n) — leave it `None` for a real run; unlike
    `fraction` it does NOT preserve the split ratio (a small absolute cap applied to
    all splits equally will disproportionately shrink the larger ones).

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

    n = len(shuffled)
    n_test = int(round(n * p["test_fraction"]))
    n_val = int(round(n * p["val_fraction"]))
    test = shuffled[:n_test]
    val = shuffled[n_test:n_test + n_val]
    train = shuffled[n_test + n_val:]

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


def list_test_patients(params=None):
    """Patients in the labeled, held-out test split — used as the YOLO web UI's Patient
    dropdown (only test-split patients are offered, never train/val, so every Dice score
    shown in the UI is on data the model never saw during training).
    """
    p = dict(PARAMS)
    if params:
        p.update(params)
    return split_patients(p)["test"]


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


def build_rgb_slice_raw(t1c_slice, t2w_slice, t2f_slice, min_fg_voxels):
    """Alternate RGB (Explore tab comparison only, not used for training/inference):
    R = raw T1C (no T1 subtraction), G = T2, B = FLAIR -> HxWx3 uint8."""
    r = _norm_slice_uint8(t1c_slice, min_fg_voxels)
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
    """Build `YOLO/dataset/images/{train,val,test}` + `labels/{train,val,test}` +
    `data.yaml`. All three splits are labeled (see `split_patients`), so `labels/test/`
    is written just like `train`/`val` — `test` just isn't trained on.

    `fraction` (0 < fraction <= 1, default `None` = all): train on this fraction of
    each split's patients while preserving the 60/20/20 ratio — see `split_patients`.

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
        log(f"[{split_name}] {len(patients)} patients, {n_slices} slices kept")

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
        self.weights_path = weights_path or default_weights_path()
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
        confs = []

        model = self._model_instance()
        if model is not None:
            # `rgb` is RGB (R=T1C-T1, G=T2, B=FLAIR), matching the RGB PNGs training
            # was built from. Ultralytics assumes a *numpy* array is already BGR and
            # would silently swap R<->B, feeding the model FLAIR where it learned
            # enhancement (great val metrics on the PNGs, poor real inference here).
            # Passing a PIL image makes Ultralytics honour the true RGB order, so
            # inference sees exactly what training did.
            from PIL import Image
            results = model.predict(Image.fromarray(rgb), conf=self.p["conf_threshold"],
                                     verbose=False)
            r0 = results[0]
            if r0.masks is not None:
                if r0.boxes is not None and r0.boxes.conf is not None:
                    confs = [float(c) for c in r0.boxes.conf.cpu().numpy()]
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
            "num_pred_instances": len(confs),
            "mean_conf": float(np.mean(confs)) if confs else None,
            "num_gt_components": int(ndimage.label(gt_wt)[1]) if gt_wt.any() else 0,
        }
        self._slice_cache[z] = out
        return out

    def dice_table(self):
        """Per-slice Dice for all valid (processed) slices."""
        return [{"z": z, "dice": self.process_slice(z)["dice"]} for z in self.slice_indices]


# ---------------------------------------------------------------------------
# Stage D+ — post-training Dice quality summary on the held-out test split
# ---------------------------------------------------------------------------
def evaluate_test_split(weights_path=None, params=None, max_patients=None, fraction=None,
                         log=print):
    """Run inference with `weights_path` (default: best available checkpoint) over every
    patient in the labeled, held-out test split (`list_test_patients`) and report
    per-patient + overall mean Dice — the "quality of the training" summary. Used by
    both `YOLO/train.py` (right after a training run, to tag the saved checkpoint) and
    the standalone `YOLO/evaluate.py` CLI.

    `max_patients`/`fraction`: forwarded to `split_patients` so a training run that
    subsampled its dataset evaluates on the *same* test patients it built, rather than
    the full test split.

    Returns {"weights_path", "n_patients", "per_patient": [{"patient_id", "mean_dice",
    "n_slices"}], "overall_mean_dice"}. `overall_mean_dice` is the mean Dice over every
    processed slice across every test patient (not a mean-of-means), so patients with
    more slices contribute proportionally more. `None` values mean no processed slices
    were found at all (e.g. an empty test split).
    """
    wp = weights_path or default_weights_path()
    p = dict(PARAMS)
    if params:
        p.update(params)
    test_patients = split_patients(p, max_patients=max_patients, fraction=fraction)["test"]

    per_patient = []
    all_dices = []
    for entry in test_patients:
        pl = YoloPipeline(entry["dir"], params=p, weights_path=wp)
        rows = pl.dice_table()
        dices = [r["dice"] for r in rows]
        mean_dice = float(np.mean(dices)) if dices else None
        per_patient.append({
            "patient_id": entry["patient_id"],
            "mean_dice": mean_dice,
            "n_slices": len(dices),
        })
        all_dices.extend(dices)
        log(f"[evaluate] {entry['patient_id']}: mean Dice = "
            + (f"{mean_dice:.4f}" if mean_dice is not None else "n/a")
            + f"  ({len(dices)} slices)")

    overall = float(np.mean(all_dices)) if all_dices else None
    log(f"[evaluate] Overall mean Dice over {len(all_dices)} slices, "
        f"{len(test_patients)} patients: "
        + (f"{overall:.4f}" if overall is not None else "n/a"))
    return {
        "weights_path": wp,
        "n_patients": len(test_patients),
        "per_patient": per_patient,
        "overall_mean_dice": overall,
    }
