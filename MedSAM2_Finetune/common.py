"""Shared paths, settings and helpers for MedSAM2_Finetune.

Everything this folder writes stays inside it: `cache/` (the YOLO prompt maps), `runs/`
(one folder per fine-tune run). The patient pools are only *read*, and the vendored
`MedSAM2/` is only *read* (its checkpoints and its `sam2` package) — never modified.

The frame that MedSAM2 sees must be the same frame the prompting YOLO was trained on
(R = T1C, G = T2, B = FLAIR, each scaled to its own p0.5-p99.5). So the frame builder is
*imported* from YOLO_finetune rather than copied: a second copy could drift and silently
feed the YOLO model channels it never saw, which costs Dice without raising anything.
"""
import os
import sys

import numpy as np
import torch
import torch.nn.functional as F
import yaml

# The frame builder, patient walker and atomic writer are YOLO_finetune's, on purpose (see above).
from YOLO_finetune.common import (  # noqa: F401  (re-exported for this package)
    CHANNELS, brain_slices, list_patients, load_volumes, read_json, rgb_slice, short_hash,
    write_json)

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.dirname(HERE)
CONFIG_PATH = os.path.join(HERE, "config.yaml")
RUNS_DIR = os.path.join(HERE, "runs")
CACHE_DIR = os.path.join(HERE, "cache")          # YOLO prompt maps, one .npz per patient

MEDSAM2_DIR = os.path.join(ROOT, "MedSAM2")      # read-only: the vendored repo
CHECKPOINTS_DIR = os.path.join(MEDSAM2_DIR, "checkpoints")
YOLO_RUNS_DIR = os.path.join(ROOT, "YOLO_finetune", "runs")

# How a per-blob YOLO output becomes one number per voxel. The UI offers exactly these.
PROMPT_VARIANTS = {
    "max": "the highest probability any YOLO blob gives that voxel",
    "max_weighted": "same, but each blob's probability is multiplied by its own confidence score",
    "conf_filtered": "blobs below the score threshold are dropped, then the highest probability",
    "binary": "the plain YOLO mask (probability cut at 0.5), 0 or 1 — no soft values",
    "box": "a filled rectangle around each kept blob instead of its outline",
    "none": "no prompt at all — MedSAM2 is asked to find the tumour unaided",
}
UNFREEZE = {
    "decoder": "mask decoder only (fastest, fewest weights changed)",
    "decoder+prompt": "mask decoder and prompt encoder (learns how to read the YOLO prompt)",
    "all": "everything, including the image encoder (slowest, needs the most memory)",
}


def load_config(path=CONFIG_PATH):
    with open(path, "r", encoding="utf-8") as f:
        return yaml.safe_load(f)


def checkpoints():
    """The MedSAM2 / SAM2 checkpoints available to start from."""
    if not os.path.isdir(CHECKPOINTS_DIR):
        return []
    return sorted(f for f in os.listdir(CHECKPOINTS_DIR) if f.endswith(".pt"))


def yolo_runs():
    """YOLO_finetune runs that finished with a usable checkpoint, newest first.

    Each entry carries the numbers the picker shows, so a run can be chosen on its test
    score rather than on its timestamp.
    """
    out = []
    if not os.path.isdir(YOLO_RUNS_DIR):
        return out
    for run_id in sorted(os.listdir(YOLO_RUNS_DIR), reverse=True):
        d = os.path.join(YOLO_RUNS_DIR, run_id)
        for name in ("best.pt", "last.pt"):
            weights = os.path.join(d, "train", "weights", name)
            if os.path.exists(weights):
                cfg = read_json(os.path.join(d, "eval", "summary.json")) or {}
                run_cfg = os.path.join(d, "run_config.yaml")
                imgsz, min_fg = 512, 100
                if os.path.exists(run_cfg):
                    with open(run_cfg, "r", encoding="utf-8") as f:
                        rc = yaml.safe_load(f) or {}
                    imgsz = rc.get("train", {}).get("imgsz", imgsz)
                    min_fg = rc.get("data", {}).get("min_fg_voxels", min_fg)
                out.append({
                    "id": run_id, "weights": weights, "which": name, "imgsz": imgsz,
                    "min_fg_voxels": min_fg,
                    "test_dice3d": (cfg.get("test") or {}).get("dice3d_mean"),
                    "val_dice3d": (cfg.get("val") or {}).get("dice3d_mean"),
                })
                break
    return out


def yolo_run(run_id=""):
    """The chosen YOLO run, or the best-scoring one when nothing is chosen."""
    runs = yolo_runs()
    if not runs:
        raise FileNotFoundError("No YOLO_finetune run has a checkpoint yet — fine-tune YOLO first.")
    if run_id:
        hit = next((r for r in runs if r["id"] == run_id), None)
        if hit is None:
            raise FileNotFoundError(f"YOLO run {run_id} has no checkpoint")
        return hit
    scored = [r for r in runs if r["test_dice3d"] is not None]
    return max(scored, key=lambda r: r["test_dice3d"]) if scored else runs[0]


# ------------------------------------------------------------------ the vendored sam2 package
def import_sam2():
    """Import the vendored sam2 package without installing or changing it.

    `sam2/__init__.py` registers its own Hydra config group on import, which is what makes
    `build_sam2("configs/...")` work.
    """
    if MEDSAM2_DIR not in sys.path:
        sys.path.insert(0, MEDSAM2_DIR)
    from sam2.build_sam import build_sam2  # noqa: E402
    return build_sam2


def build_model(cfg, device="cuda"):
    """A SAM2Base built from the vendored config and one of the MedSAM2 checkpoints."""
    build_sam2 = import_sam2()
    ckpt = os.path.join(CHECKPOINTS_DIR, cfg["model"]["checkpoint"])
    if not os.path.exists(ckpt):
        raise FileNotFoundError(f"no checkpoint at {ckpt}")
    model = build_sam2(cfg["model"]["config"], ckpt, device=device, mode="train")
    model.directly_add_no_mem_embed = True  # single slices: there is no memory to attend to
    return model


# ------------------------------------------------------------------ small numeric helpers
SIGMOID_CLIP = 5.541263545158426  # logit(254/255): the most a uint8-stored probability can say


def prob_to_logit(prob):
    """Undo the uint8 probability quantisation of the cache, without infinities."""
    p = np.clip(prob.astype(np.float32) / 255.0, 1.0 / 255.0, 254.0 / 255.0)
    return np.log(p / (1.0 - p)).astype(np.float32)


def dice(inter, pred, gt):
    """Dice of two masks from their pixel counts; two empty masks agree perfectly."""
    return 1.0 if pred + gt == 0 else 2.0 * inter / float(pred + gt)


def resize_chw(t, size, mode="bilinear"):
    """Resize a (C, H, W) float tensor to size x size."""
    kw = {"antialias": True} if mode == "bilinear" else {}
    return F.interpolate(t[None], size=(size, size), mode=mode,
                         align_corners=False if mode == "bilinear" else None, **kw)[0]


def to_model_image(rgb, size, device=None):
    """An RGB uint8 slice -> the normalised (3, size, size) tensor SAM2 expects."""
    t = torch.from_numpy(np.ascontiguousarray(rgb.transpose(2, 0, 1))).float() / 255.0
    t = resize_chw(t, size)
    mean = torch.tensor([0.485, 0.456, 0.406])[:, None, None]
    std = torch.tensor([0.229, 0.224, 0.225])[:, None, None]
    t = (t - mean) / std
    return t.to(device) if device else t
