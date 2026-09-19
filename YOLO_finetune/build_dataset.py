"""Build the YOLO-seg dataset (train + val images and labels) from the NIfTI pools.

A dataset is identified by a hash of its data settings and patient lists, and lives in
`dataset/<hash>/`. A run reuses an existing, complete one (it has `meta.json`), so
re-running with only training settings changed costs no rebuild.
"""
import os
import shutil
import threading
from concurrent.futures import ThreadPoolExecutor, as_completed
from datetime import datetime

import numpy as np
from PIL import Image

from .common import (DATASETS_DIR, brain_slices, list_patients, load_volumes, rgb_slice,
                    short_hash, tumour_polygons, write_json, read_json, yolo_seg_lines)

AREA_BINS = [0, 50, 100, 250, 500, 1000, 2000, 4000, 8000, 16000, 40000]  # mask px per slice
WORKERS = 6


def dataset_splits(cfg, max_patients=None):
    d = cfg["data"]
    splits = {"train": list_patients(d["train_pool"]), "val": list_patients(d["val_pool"])}
    if max_patients:
        splits = {k: v[:max_patients] for k, v in splits.items()}
    return splits


def dataset_key(cfg, splits):
    d = cfg["data"]
    return short_hash({"min_fg_voxels": d["min_fg_voxels"], "min_mask_area": d["min_mask_area"],
                       "splits": {k: [p["id"] for p in v] for k, v in splits.items()}})


def _write_patient(patient, split, out_dir, d):
    vols = load_volumes(patient)
    seg = vols["SEG"] > 0
    stats = {"slices": 0, "positive": 0, "instances": 0, "areas": []}
    for z in brain_slices(vols["FLAIR"], d["min_fg_voxels"]):
        stem = f"{patient['id']}_z{z:03d}"
        Image.fromarray(rgb_slice(vols, z, d["min_fg_voxels"]), mode="RGB").save(
            os.path.join(out_dir, "images", split, stem + ".png"))
        mask = seg[:, :, z]
        lines = yolo_seg_lines(tumour_polygons(mask, d["min_mask_area"]), *mask.shape)
        with open(os.path.join(out_dir, "labels", split, stem + ".txt"), "w", encoding="utf-8") as f:
            f.write("\n".join(lines) + ("\n" if lines else ""))
        stats["slices"] += 1
        if lines:
            stats["positive"] += 1
            stats["instances"] += len(lines)
            stats["areas"].append(int(mask.sum()))
    return stats


def build(cfg, max_patients=None, progress=None, log=print):
    """Return (dataset_dir, meta). Builds it unless an identical complete one exists."""
    splits = dataset_splits(cfg, max_patients)
    key = dataset_key(cfg, splits)
    out_dir = os.path.join(DATASETS_DIR, key)
    meta = read_json(os.path.join(out_dir, "meta.json"))
    if meta:
        log(f"[dataset] reusing {out_dir}")
        return out_dir, meta

    if os.path.isdir(out_dir):  # an earlier build was interrupted: start clean
        shutil.rmtree(out_dir)
    for split in splits:
        os.makedirs(os.path.join(out_dir, "images", split))
        os.makedirs(os.path.join(out_dir, "labels", split))

    d = cfg["data"]
    total = sum(len(v) for v in splits.values())
    done = 0
    lock = threading.Lock()
    per_patient = {}
    log(f"[dataset] building {out_dir} from {total} patients")
    with ThreadPoolExecutor(max_workers=WORKERS) as pool:
        futures = {pool.submit(_write_patient, p, split, out_dir, d): (p["id"], split)
                   for split, pats in splits.items() for p in pats}
        for fut in as_completed(futures):
            pid, split = futures[fut]
            per_patient[pid] = {"split": split, **fut.result()}
            with lock:
                done += 1
                if progress:
                    progress(done, total)

    summary = {}
    for split in splits:
        rows = [v for v in per_patient.values() if v["split"] == split]
        areas = [a for v in rows for a in v["areas"]]
        summary[split] = {
            "patients": len(rows),
            "slices": sum(v["slices"] for v in rows),
            "positive": sum(v["positive"] for v in rows),
            "negative": sum(v["slices"] - v["positive"] for v in rows),
            "instances": sum(v["instances"] for v in rows),
            "area_hist": np.histogram(areas, bins=AREA_BINS + [np.inf])[0].tolist(),  # last = ">= 40000"
            "area_median": float(np.median(areas)) if areas else None,
        }
    meta = {
        "key": key,
        "created": datetime.now().isoformat(timespec="seconds"),
        "data": d,
        "area_bins": AREA_BINS,
        "splits": summary,
        "patients": {pid: {k: v[k] for k in ("split", "slices", "positive", "instances")}
                     for pid, v in per_patient.items()},
    }
    with open(os.path.join(out_dir, "data.yaml"), "w", encoding="utf-8") as f:
        f.write(f"path: {out_dir}\ntrain: images/train\nval: images/val\nnames:\n  0: tumour\n")
    write_json(os.path.join(out_dir, "meta.json"), meta)  # written last = build complete
    log(f"[dataset] done: " + ", ".join(f"{k} {v['slices']} slices ({v['positive']} with tumour)"
                                         for k, v in summary.items()))
    return out_dir, meta
