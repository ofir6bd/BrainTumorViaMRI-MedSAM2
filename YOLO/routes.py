"""Flask blueprint exposing the YOLO segmentation pipeline to the web UI.

Mirrors the FCMSegmentation/Assymetry blueprints: reads patients from `data/sample`
(config.yaml -> paths.sample), pipelines cached per patient. Wiring this blueprint into
`Frontend/app.py` (plus the sidebar button) is the Stage F integration step in README.md
and is deliberately left for a follow-up once `YOLO/weights/best.pt` exists.
"""
import os

import yaml
from flask import Blueprint, abort, jsonify, request, send_file

from . import render as R
from .pipeline import YoloPipeline

yolo_bp = Blueprint("yolo", __name__, url_prefix="/yolo")


def _sample_dir():
    cfg_path = "config.yaml"
    sample = "data/sample"
    if os.path.exists(cfg_path):
        with open(cfg_path, "r") as f:
            cfg = yaml.safe_load(f) or {}
        sample = cfg.get("paths", {}).get("sample", sample)
    return sample


def _discover():
    root = _sample_dir()
    patients = []
    if not os.path.isdir(root):
        return patients
    for name in sorted(os.listdir(root)):
        d = os.path.join(root, name)
        if not os.path.isdir(d):
            continue
        files = [f.lower() for f in os.listdir(d)]
        has_seg = any("-seg" in f for f in files)
        has_all_mod = all(any(suf in f for f in files)
                          for suf in ("-t1c", "-t1n", "-t2w", "-t2f"))
        if has_seg and has_all_mod:
            patients.append({"patient_id": name, "dir": d})
    return patients


PATIENTS = _discover()
_PIPELINES = {}


def _pipeline(idx):
    if idx < 0 or idx >= len(PATIENTS):
        abort(404)
    if idx not in _PIPELINES:
        _PIPELINES[idx] = YoloPipeline(PATIENTS[idx]["dir"])
    return _PIPELINES[idx]


@yolo_bp.route("/api/patients")
def api_patients():
    return jsonify([{"id": i, "label": p["patient_id"]} for i, p in enumerate(PATIENTS)])


@yolo_bp.route("/api/patient/<int:idx>")
def api_patient(idx):
    pl = _pipeline(idx)
    return jsonify({
        "patient_id": pl.patient_id,
        "depth": pl.depth,
        "slice_indices": pl.slice_indices,
        "best_slice_index": pl.best_slice_index(),
        "min_fg_voxels": pl.p["min_fg_voxels"],
    })


@yolo_bp.route("/segment.png")
def segment_png():
    idx = int(request.args.get("id", -1))
    pl = _pipeline(idx)
    z = max(0, min(int(request.args.get("z", 0)), pl.depth - 1))
    try:
        slice_no = pl.slice_indices.index(z) + 1
    except ValueError:
        slice_no = None
    buf = R.render(pl, z, slice_no=slice_no)
    return send_file(buf, mimetype="image/png")


@yolo_bp.route("/api/dice")
def api_dice():
    idx = int(request.args.get("id", -1))
    pl = _pipeline(idx)
    return jsonify({"patient_id": pl.patient_id, "rows": pl.dice_table()})
