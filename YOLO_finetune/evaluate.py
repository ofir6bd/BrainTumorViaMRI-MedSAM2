"""Evaluate a fine-tuned checkpoint on the val and test pools, straight from the NIfTI files.

Each brain slice is predicted once at the lowest sweep threshold; the tumour mask at any
higher threshold is the union of the instances at or above it, so the whole confidence
sweep costs one pass. Masks come back at the slice's own size (`retina_masks`), so they
line up with the expert mask pixel for pixel.

Scores, per patient and per threshold:
- 3D Dice   = 2|P∩G| / (|P|+|G|) over all the patient's brain slices (the headline number)
- per slice: gt / pred / overlap pixels at the configured `conf`, for slice-level metrics:
  tumour-slice Dice (slices with tumour only), slice detection (TP/FN/FP/TN), and the
  legacy mean slice Dice (empty-vs-empty slice = 1.0) that the old YOLO/ 0.8662 used.

val is where a threshold may be chosen; test is only ever reported.
"""
import os
from datetime import datetime

import numpy as np
from PIL import Image

from .common import brain_slices, list_patients, load_volumes, rgb_slice, write_json


def thresholds_of(cfg):
    """The sweep, always including the configured `conf`."""
    e = cfg["evaluate"]
    return sorted(set(e["sweep"]) | {e["conf"]})


def _predict_patient(model, patient, cfg):
    d, e = cfg["data"], cfg["evaluate"]
    thresholds = thresholds_of(cfg)
    vols = load_volumes(patient)
    zs = brain_slices(vols["FLAIR"], d["min_fg_voxels"])
    seg = vols["SEG"] > 0
    sweep = np.zeros((len(thresholds), 3), dtype=np.int64)  # inter, pred, gt per threshold
    per_slice = {"z": zs, "gt": [], "pred": [], "inter": [], "conf": []}
    k_conf = thresholds.index(e["conf"])
    for i in range(0, len(zs), e["batch"]):
        batch_z = zs[i:i + e["batch"]]
        # PIL images, not numpy: Ultralytics treats numpy input as BGR and would swap
        # the T1C-T1 and FLAIR channels relative to the PNGs the model trained on.
        imgs = [Image.fromarray(rgb_slice(vols, z, d["min_fg_voxels"])) for z in batch_z]
        results = model.predict(imgs, conf=min(thresholds), imgsz=cfg["train"]["imgsz"],
                                retina_masks=True, verbose=False, device=cfg["train"]["device"])
        for z, r in zip(batch_z, results):
            gt = seg[:, :, z]
            masks = r.masks.data.cpu().numpy().astype(bool) if r.masks is not None else np.zeros((0, *gt.shape), bool)
            confs = r.boxes.conf.cpu().numpy() if r.boxes is not None else np.zeros(0)
            for k, t in enumerate(thresholds):
                keep = confs >= t
                pred = masks[keep].any(axis=0) if keep.any() else np.zeros_like(gt)
                inter = int(np.logical_and(pred, gt).sum())
                sweep[k] += (inter, int(pred.sum()), int(gt.sum()))
                if k == k_conf:
                    per_slice["gt"].append(int(gt.sum()))
                    per_slice["pred"].append(int(pred.sum()))
                    per_slice["inter"].append(inter)
            per_slice["conf"].append(round(float(confs.max()), 3) if confs.size else 0.0)
    return {"id": patient["id"], "sweep": sweep.tolist(), "slices": per_slice}


def _dice(inter, pred, gt):
    return 1.0 if pred + gt == 0 else 2.0 * inter / (pred + gt)


def summarise(patients, cfg):
    """Split-level numbers from the per-patient records (also used by the UI)."""
    e = cfg["evaluate"]
    thresholds = thresholds_of(cfg)
    k_conf = thresholds.index(e["conf"])
    sweep = []
    for k, t in enumerate(thresholds):
        d3 = [_dice(*p["sweep"][k]) for p in patients]
        sweep.append({"conf": t, "dice3d_mean": float(np.mean(d3)) if d3 else None})
    slice_d, tumour_d = [], []
    det = {"tp": 0, "fn": 0, "fp": 0, "tn": 0}
    for p in patients:
        s = p["slices"]
        for g, pr, it in zip(s["gt"], s["pred"], s["inter"]):
            dsc = _dice(it, pr, g)
            slice_d.append(dsc)
            if g:
                tumour_d.append(dsc)
            det["tp" if g and pr else "fn" if g else "fp" if pr else "tn"] += 1
    best = max(sweep, key=lambda s: s["dice3d_mean"] or 0)
    d3 = [_dice(*p["sweep"][k_conf]) for p in patients]
    return {
        "patients": len(patients),
        "conf": e["conf"],
        "dice3d_mean": float(np.mean(d3)) if d3 else None,
        "dice3d_median": float(np.median(d3)) if d3 else None,
        "tumour_slice_dice": float(np.mean(tumour_d)) if tumour_d else None,
        "legacy_slice_dice": float(np.mean(slice_d)) if slice_d else None,
        "detection": det,
        "sensitivity": det["tp"] / max(1, det["tp"] + det["fn"]),
        "specificity": det["tn"] / max(1, det["tn"] + det["fp"]),
        "sweep": sweep,
        "best_conf": best["conf"],
    }


def evaluate(weights, cfg, out_dir, max_patients=None, progress=None, log=print):
    from ultralytics import YOLO

    model = YOLO(weights)
    os.makedirs(out_dir, exist_ok=True)
    pools = {"val": cfg["data"]["val_pool"], "test": cfg["data"]["test_pool"]}
    todo = {split: list_patients(pool)[:max_patients] if max_patients else list_patients(pool)
            for split, pool in pools.items()}
    total = sum(len(v) for v in todo.values())
    done = 0
    summary = {"weights": weights, "created": datetime.now().isoformat(timespec="seconds")}
    for split, patients in todo.items():
        records = []
        for p in patients:
            records.append(_predict_patient(model, p, cfg))
            done += 1
            if progress:
                progress(done, total, split)
        write_json(os.path.join(out_dir, f"{split}.json"), {"thresholds": thresholds_of(cfg), "patients": records})
        summary[split] = summarise(records, cfg)
        s = summary[split]
        log(f"[eval] {split}: 3D Dice {s['dice3d_mean']:.4f} (median {s['dice3d_median']:.4f}), "
            f"tumour-slice Dice {s['tumour_slice_dice']:.4f}, legacy slice Dice {s['legacy_slice_dice']:.4f} "
            f"@ conf {s['conf']} | best sweep conf on {split}: {s['best_conf']}")
    write_json(os.path.join(out_dir, "summary.json"), summary)
    return summary
