"""Shared paths, configuration and image/label helpers for YOLO_finetune.

Everything this folder writes stays inside it: `dataset/` (YOLO-seg images + labels),
`runs/` (one folder per fine-tune run) and `pretrained/` (downloaded COCO weights).
The patient pools are only *read*, from the project's `config.yaml -> paths`.
"""
import hashlib
import json
import os

import nibabel as nib
import numpy as np
import yaml
from scipy import ndimage
from skimage import measure

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.dirname(HERE)
CONFIG_PATH = os.path.join(HERE, "config.yaml")
RUNS_DIR = os.path.join(HERE, "runs")
DATASETS_DIR = os.path.join(HERE, "dataset")
PRETRAINED_DIR = os.path.join(HERE, "pretrained")

FILE_SUFFIX = {"T1C": "-t1c", "T1": "-t1n", "T2": "-t2w", "FLAIR": "-t2f", "SEG": "-seg"}


def load_config(path=CONFIG_PATH):
    with open(path, "r", encoding="utf-8") as f:
        return yaml.safe_load(f)


def project_paths():
    with open(os.path.join(ROOT, "config.yaml"), "r", encoding="utf-8") as f:
        return yaml.safe_load(f)["paths"]


def pool_dir(pool_key):
    return os.path.join(ROOT, project_paths()[pool_key])


def list_patients(pool_key):
    """Patient folders of a pool that have all four modalities and the `-seg` mask."""
    root = pool_dir(pool_key)
    out = []
    for name in sorted(os.listdir(root)):
        d = os.path.join(root, name)
        if not os.path.isdir(d):
            continue
        files = {}
        for f in os.listdir(d):
            for role, suf in FILE_SUFFIX.items():
                if f.endswith(".nii.gz") and suf + "." in f.lower():
                    files[role] = os.path.join(d, f)
        if len(files) == len(FILE_SUFFIX):
            out.append({"id": name, "dir": d, "files": files})
    return out


def load_volumes(patient):
    return {role: np.asarray(nib.load(path).dataobj, dtype=np.float32)
            for role, path in patient["files"].items()}


def brain_slices(flair, min_fg_voxels):
    """Axial slices with enough brain in them to be worth a frame."""
    counts = (flair != 0).sum(axis=(0, 1))
    return [int(z) for z in np.nonzero(counts >= min_fg_voxels)[0]]


def _norm_uint8(sl, min_fg_voxels):
    """Per-slice p0.5-p99.5 of the non-zero voxels -> uint8; black if too little brain."""
    s = sl.astype(np.float32)
    nz = s[s != 0]
    if nz.size < min_fg_voxels:
        return np.zeros(s.shape, dtype=np.uint8)
    lo = float(np.percentile(nz, 0.5))
    hi = float(np.percentile(nz, 99.5))
    if hi - lo < 1e-8:
        return np.zeros(s.shape, dtype=np.uint8)
    return np.clip((s - lo) / (hi - lo) * 255, 0, 255).astype(np.uint8)


def rgb_slice(vols, z, min_fg_voxels):
    """The YOLO input frame: R = clip(T1C - T1, 0) (enhancement), G = T2, B = FLAIR."""
    diff = np.clip(vols["T1C"][:, :, z] - vols["T1"][:, :, z], 0, None)
    return np.stack([_norm_uint8(diff, min_fg_voxels),
                     _norm_uint8(vols["T2"][:, :, z], min_fg_voxels),
                     _norm_uint8(vols["FLAIR"][:, :, z], min_fg_voxels)], axis=-1)


def tumour_polygons(mask, min_mask_area):
    """One outline per connected tumour piece of at least `min_mask_area` pixels,
    as (N, 2) [row, col] arrays."""
    labelled, n = ndimage.label(mask)
    polys = []
    for comp_id in range(1, n + 1):
        comp = labelled == comp_id
        if int(comp.sum()) < min_mask_area:
            continue
        contours = measure.find_contours(np.pad(comp, 1).astype(np.float32), level=0.5)
        if contours:
            polys.append(max(contours, key=len) - 1)  # undo the padding offset
    return polys


def yolo_seg_lines(polys, h, w):
    """YOLO-seg label lines: class 0, then x y pairs normalised to 0..1."""
    lines = []
    for contour in polys:
        if len(contour) < 3:
            continue
        xy = [f"{min(max(c / w, 0.0), 1.0):.6f} {min(max(r / h, 0.0), 1.0):.6f}" for r, c in contour]
        lines.append("0 " + " ".join(xy))
    return lines


def short_hash(obj):
    return hashlib.sha1(json.dumps(obj, sort_keys=True).encode("utf-8")).hexdigest()[:10]


def write_json(path, data):
    """Atomic write, so the UI never reads a half-written file."""
    tmp = path + ".tmp"
    with open(tmp, "w", encoding="utf-8") as f:
        json.dump(data, f)
    os.replace(tmp, path)


def read_json(path, default=None):
    try:
        with open(path, "r", encoding="utf-8") as f:
            return json.load(f)
    except (OSError, ValueError):
        return default
