import io
import json
import os
import sys
from functools import lru_cache

import yaml
import numpy as np
import nibabel as nib
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from flask import Flask, jsonify, send_file, abort, render_template, request

from dataset_stats import StatsStore

# Repo root on sys.path so sibling packages (YOLO_finetune) import as packages.
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from YOLO_finetune.routes import finetune_bp  # noqa: E402

with open("config.yaml", "r") as _f:
    _CFG = yaml.safe_load(_f)
_USE_SAMPLE = bool(_CFG.get("inference", {}).get("use_sample", False))
DATASET_DIR = _CFG["paths"]["sample"] if _USE_SAMPLE else _CFG["paths"]["dataset"]
MANIFEST_PATH = os.path.join(_CFG["paths"]["dataset"], "split_manifest.json")

# Physical dataset pools, in display order — paths come from config.yaml.
POOLS = [
    {"key": "yolo_train", "family": "yolo", "role": "train", "use": "YOLO trains on these"},
    {"key": "yolo_val", "family": "yolo", "role": "val", "use": "YOLO picks its best epoch"},
    {"key": "medsam2_train", "family": "medsam2", "role": "train", "use": "MedSAM2 trains on these"},
    {"key": "medsam2_val", "family": "medsam2", "role": "val", "use": "MedSAM2 picks its best epoch"},
    {"key": "test", "family": "test", "role": "test", "use": "Final test for both, never trained on"},
]
for _pool in POOLS:
    _pool["path"] = _CFG["paths"][_pool["key"]]

LABEL_NAMES = {1: "NETC", 2: "SNFH", 3: "ET", 4: "RC"}
# One colour per label, shared by the slice images and the Analytics charts.
LABEL_COLORS = {1: "#e66767", 2: "#008300", 3: "#9085e9", 4: "#c98500"}
# Images the patient drawer can show: a modality, or T1C - T1 (enhancement).
IMAGE_KINDS = {"t1c": "T1C", "t1": "T1", "t2": "T2", "flair": "FLAIR", "sub": "T1C − T1"}


def image_needs(kind):
    """Modalities an image kind is built from."""
    return ("T1C", "T1") if kind == "sub" else (IMAGE_KINDS[kind],)

app = Flask(__name__)
app.config["TEMPLATES_AUTO_RELOAD"] = True
app.config["SEND_FILE_MAX_AGE_DEFAULT"] = 0
app.register_blueprint(finetune_bp)  # the YOLO_finetune page, at /finetune/


def _pool_of(root):
    for pool in POOLS:
        if os.path.normcase(os.path.abspath(os.path.dirname(root))) == \
                os.path.normcase(os.path.abspath(pool["path"])):
            return pool["key"]
    return None


def find_patients():
    patients = []
    if not os.path.exists(DATASET_DIR):
        return patients
    for root, _, files in os.walk(DATASET_DIR):
        nii = [f for f in files if f.endswith((".nii", ".nii.gz"))]
        seg = [f for f in nii if "seg" in f.lower()]
        if not nii or not seg:
            continue
        modalities = {}
        for f in nii:
            if "seg" in f.lower():
                continue
            name = f.lower()
            if "-t1c" in name:
                modalities["T1C"] = os.path.join(root, f)
            if "-t1n" in name:
                modalities["T1"] = os.path.join(root, f)
            if "-t2w" in name:
                modalities["T2"] = os.path.join(root, f)
            if "-t2f" in name:
                modalities["FLAIR"] = os.path.join(root, f)
        patients.append({
            "patient_id": os.path.basename(root),
            "pool": _pool_of(root),
            "seg_file": os.path.join(root, seg[0]),
            "modalities": modalities,
        })
    patients.sort(key=lambda p: p["patient_id"])
    return patients


PATIENTS = find_patients()
stats_store = StatsStore(os.path.join(_CFG["paths"]["outputs"], "dashboard", "dataset_stats.json"))


@lru_cache(maxsize=12)
def load_volume(path):
    return nib.load(path).get_fdata()


def get_patient(idx):
    if idx < 0 or idx >= len(PATIENTS):
        abort(404)
    return PATIENTS[idx]


def best_slice(seg):
    sums = [int(np.sum(seg[:, :, z] > 0)) for z in range(seg.shape[2])]
    return int(np.argmax(sums)) if max(sums) > 0 else seg.shape[2] // 2


def norm_slice(sl):
    """p1-p99 of the non-zero (brain) voxels -> 0..1 for display."""
    sl = sl.astype(np.float32)
    nz = sl[sl != 0]
    lo = float(np.percentile(nz, 1)) if nz.size else 0.0
    hi = float(np.percentile(nz, 99)) if nz.size else 1.0
    return np.clip((sl - lo) / max(hi - lo, 1e-8), 0, 1)


def label_overlay(seg_2d, alpha=0.55):
    overlay = np.zeros((*seg_2d.shape, 4), dtype=float)
    for label_id, hexcol in LABEL_COLORS.items():
        overlay[seg_2d == label_id] = (*matplotlib.colors.to_rgb(hexcol), alpha)
    return overlay


def figure_to_png(fig):
    buf = io.BytesIO()
    fig.savefig(buf, format="png", bbox_inches="tight",
                facecolor=fig.get_facecolor())
    plt.close(fig)
    buf.seek(0)
    return buf


@app.route("/")
def index():
    return render_template("index.html")


@app.route("/thumb.png")
def thumb_png():
    """One axial slice for the patient drawer: a modality or T1C - T1, optional labels."""
    p = get_patient(int(request.args.get("id", -1)))
    kind = request.args.get("mod", "flair")
    if kind not in IMAGE_KINDS:
        abort(400)
    if not all(m in p["modalities"] for m in image_needs(kind)):
        abort(404)
    seg = load_volume(p["seg_file"])
    z = max(0, min(int(request.args.get("z", best_slice(seg))), seg.shape[2] - 1))
    if kind == "sub":
        # Enhancement: what T1C gains over T1 (contrast uptake); negative change -> 0.
        sl = np.clip(load_volume(p["modalities"]["T1C"])[:, :, z]
                     - load_volume(p["modalities"]["T1"])[:, :, z], 0, None)
    else:
        sl = load_volume(p["modalities"][IMAGE_KINDS[kind]])[:, :, z]
    fig, ax = plt.subplots(figsize=(3.2, 3.8), facecolor="#12122a")
    # Axial slice shown in radiological orientation: patient's right on the left.
    ax.imshow(np.rot90(norm_slice(sl)), cmap="gray", vmin=0, vmax=1)
    if request.args.get("ov", "1") == "1":
        ax.imshow(np.rot90(label_overlay(seg[:, :, z])), interpolation="none")
    ax.axis("off")
    fig.subplots_adjust(0, 0, 1, 1)
    return send_file(figure_to_png(fig), mimetype="image/png")


def _manifest_report():
    """Compare split_manifest.json with where each patient folder actually is."""
    if not os.path.exists(MANIFEST_PATH):
        return {"found": False}
    with open(MANIFEST_PATH, "r", encoding="utf-8") as f:
        manifest = json.load(f)
    assignment = manifest.get("assignment", {})
    on_disk = {p["patient_id"]: p["pool"] for p in PATIENTS}
    moved = [{"id": pid, "manifest": assignment[pid], "disk": pool}
             for pid, pool in sorted(on_disk.items())
             if pid in assignment and assignment[pid] != pool]
    return {
        "found": True,
        "note": manifest.get("note"),
        "seed": manifest.get("seed"),
        "counts": manifest.get("counts"),
        "moved": moved,
        "not_in_manifest": sorted(set(on_disk) - set(assignment)),
        "missing_on_disk": sorted(set(assignment) - set(on_disk)),
    }


@app.route("/api/dashboard/status")
def api_dashboard_status():
    # Opening the dashboard fills in whatever is not cached yet.
    stats_store.start(PATIENTS, log=print)
    return jsonify(stats_store.status(PATIENTS))


@app.route("/api/dashboard/recompute", methods=["POST"])
def api_dashboard_recompute():
    started = stats_store.start(PATIENTS, force=True, log=print)
    return jsonify({"started": started, **stats_store.status(PATIENTS)})


@app.route("/api/dashboard/data")
def api_dashboard_data():
    records = []
    for i, p in enumerate(PATIENTS):
        rec = stats_store.records.get(p["patient_id"])
        if rec is None:
            continue
        rec = {k: v for k, v in rec.items() if k != "_key"}
        rec["idx"] = i
        rec["pool"] = p["pool"]
        rec["images"] = [k for k in IMAGE_KINDS
                         if all(m in p["modalities"] for m in image_needs(k))]
        records.append(rec)
    return jsonify({
        "records": records,
        "pools": POOLS,
        "labels": [{"id": i, "name": LABEL_NAMES[i], "color": LABEL_COLORS[i]}
                   for i in LABEL_NAMES],
        "image_kinds": [{"key": k, "label": v} for k, v in IMAGE_KINDS.items()],
        "dataset_dir": DATASET_DIR,
        "n_patients": len(PATIENTS),
        "manifest": _manifest_report(),
        **stats_store.status(PATIENTS),
    })
