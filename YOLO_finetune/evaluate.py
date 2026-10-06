"""Evaluate a fine-tuned checkpoint on the val and test pools, straight from the NIfTI files.

Each brain slice is predicted once at the lowest sweep threshold; the tumour mask at any
higher threshold is built from the blobs at or above it, so the whole confidence sweep
costs one pass. Masks come back at the slice's own size, so they line up with the expert
mask pixel for pixel.

A blob's mask is kept as a probability per pixel (`common.keep_soft_masks`); "tumour" is
probability > 0.5, which with the two extras off is exactly Ultralytics' own mask. Extras:
- `evaluate.tta_flip`: the left-right mirrored slice is predicted too, flipped back, and the
  two probability maps are averaged (in the same predict call).
- `evaluate.min_component`: 3D pieces smaller than this many voxels are removed.

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
from scipy import ndimage

from .common import brain_slices, keep_soft_masks, list_patients, load_volumes, rgb_slice, write_json


def thresholds_of(cfg):
    """The sweep, always including the configured `conf`."""
    e = cfg["evaluate"]
    return sorted(set(e["sweep"]) | {e["conf"]})


def predict_volume(model, vols, zs, cfg, conf):
    """Every slice through the model. Returns, per view (plain, and mirrored when
    `evaluate.tta_flip`), a list over slices of (blob scores, blob probability maps)."""
    d, e = cfg["data"], cfg["evaluate"]
    flips = [False, True] if e.get("tta_flip") else [False]
    out = [[] for _ in flips]
    for i in range(0, len(zs), e["batch"]):
        frames = [rgb_slice(vols, z, d["min_fg_voxels"]) for z in zs[i:i + e["batch"]]]
        # PIL images, not numpy: Ultralytics treats numpy input as BGR and would swap the T1C
        # and FLAIR channels relative to the PNGs the model trained on. Array axis 0 is the
        # head's left-right axis, so `f[::-1]` is the mirrored head.
        imgs = [Image.fromarray(np.ascontiguousarray(f[::-1]) if fl else f) for fl in flips for f in frames]
        with keep_soft_masks() as soft:
            results = model.predict(imgs, conf=conf, imgsz=cfg["train"]["imgsz"], retina_masks=True,
                                    verbose=False, device=cfg["train"]["device"])
        taken = 0
        for j, r in enumerate(results):
            v = j // len(frames)
            shape = frames[j % len(frames)].shape[:2]
            confs = r.boxes.conf.cpu().numpy() if r.boxes is not None else np.zeros(0, np.float32)
            if r.masks is None:           # no blob at all: it never reached the mask step
                probs = np.zeros((0, *shape), np.float16)
            else:
                probs = soft[taken].cpu().numpy().astype(np.float16)
                taken += 1
                if flips[v]:
                    probs = probs[:, ::-1]
            out[v].append((confs, probs))
    return out


def mask_at(views, t, shape, cfg):
    """The tumour mask of a whole patient [Z, H, W] at confidence `t`, extras applied."""
    prob = np.zeros((len(views[0]), *shape), np.float32)
    for view in views:
        for k, (confs, probs) in enumerate(view):
            keep = confs >= t
            if keep.any():
                prob[k] += probs[keep].max(axis=0)
    mask = prob / len(views) > 0.5
    small = int(cfg["evaluate"].get("min_component") or 0)
    if small > 0 and mask.any():
        lab, n = ndimage.label(mask)
        sizes = np.bincount(lab.ravel(), minlength=n + 1)
        keep = sizes >= small
        keep[0] = False
        mask = keep[lab]
    return mask


def _scored(model, patient, cfg, conf):
    vols = load_volumes(patient)
    zs = brain_slices(vols["FLAIR"], cfg["data"]["min_fg_voxels"])
    gt = np.stack([vols["SEG"][:, :, z] > 0 for z in zs])
    return zs, gt, predict_volume(model, vols, zs, cfg, conf)


def _per_slice(pred, gt):
    n = len(gt)
    return (np.logical_and(pred, gt).reshape(n, -1).sum(1), pred.reshape(n, -1).sum(1),
            gt.reshape(n, -1).sum(1))


def _best_score(views):
    return [round(float(max((c.max() for c, _ in (v[k] for v in views) if c.size), default=0.0)), 3)
            for k in range(len(views[0]))]


def patient_profile(model, patient, cfg, conf):
    """One patient, slice by slice at one threshold: Dice plus expert / predicted pixels."""
    zs, gt, views = _scored(model, patient, cfg, conf)
    inter, pred, g = _per_slice(mask_at(views, conf, gt.shape[1:], cfg), gt)
    out = {"id": patient["id"], "conf": conf, "z": zs, "gt": g.tolist(), "pred": pred.tolist(),
           "inter": inter.tolist(), "score": _best_score(views),
           "dice": [round(_dice(int(i), int(p), int(q)), 4) for i, p, q in zip(inter, pred, g)]}
    out["dice3d"] = round(_dice(int(inter.sum()), int(pred.sum()), int(g.sum())), 4)
    out["gt_total"] = int(g.sum())
    out["pred_total"] = int(pred.sum())
    return out


def patient_dice3d(model, patient, cfg):
    """3D Dice at `evaluate.conf` with every extra on — the after-every-round check."""
    conf = cfg["evaluate"]["conf"]
    _zs, gt, views = _scored(model, patient, cfg, conf)
    pred = mask_at(views, conf, gt.shape[1:], cfg)
    return _dice(int((pred & gt).sum()), int(pred.sum()), int(gt.sum()))


def _predict_patient(model, patient, cfg):
    e = cfg["evaluate"]
    thresholds = thresholds_of(cfg)
    zs, gt, views = _scored(model, patient, cfg, min(thresholds))
    sweep = np.zeros((len(thresholds), 3), dtype=np.int64)  # inter, pred, gt per threshold
    per_slice = {"z": zs, "gt": [], "pred": [], "inter": [], "conf": _best_score(views)}
    for k, t in enumerate(thresholds):
        inter, pred, g = _per_slice(mask_at(views, t, gt.shape[1:], cfg), gt)
        sweep[k] = (inter.sum(), pred.sum(), g.sum())
        if t == e["conf"]:
            per_slice.update(gt=g.tolist(), pred=pred.tolist(), inter=inter.tolist())
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
        "tta_flip": bool(e.get("tta_flip")),
        "min_component": int(e.get("min_component") or 0),
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


def evaluate(weights, cfg, out_dir, max_patients=None, progress=None, log=print, splits=("val", "test")):
    from ultralytics import YOLO

    model = YOLO(weights)
    os.makedirs(out_dir, exist_ok=True)
    pools = {"val": cfg["data"]["val_pool"], "test": cfg["data"]["test_pool"]}
    todo = {split: list_patients(pools[split])[:max_patients] if max_patients else list_patients(pools[split])
            for split in splits}
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
