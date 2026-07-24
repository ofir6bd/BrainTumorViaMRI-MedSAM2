"""Flask blueprint exposing the asymmetry pipeline to the web UI.

Registered by Frontend/app.py under the /asym prefix. Reads patients from the
`data/sample` folder (config.yaml -> paths.sample). Pipelines are cached per patient.
"""
import os

import yaml
from flask import Blueprint, jsonify, send_file, abort, request

from .pipeline import AsymmetryPipeline
from . import render as R

asym_bp = Blueprint("asym", __name__, url_prefix="/asym")

_HERE = os.path.dirname(os.path.abspath(__file__))
_OUT_DIR = os.path.join(_HERE, "outputs")


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
        files = os.listdir(d)
        has_seg = any("seg" in f.lower() for f in files)
        has_flair = any("-t2f" in f.lower() for f in files)
        if has_seg and has_flair:
            patients.append({"patient_id": name, "dir": d})
    return patients


PATIENTS = _discover()
_PIPELINES = {}


def _pipeline(idx):
    if idx < 0 or idx >= len(PATIENTS):
        abort(404)
    if idx not in _PIPELINES:
        _PIPELINES[idx] = AsymmetryPipeline(PATIENTS[idx]["dir"])
    return _PIPELINES[idx]


@asym_bp.route("/api/patients")
def api_patients():
    return jsonify([{"id": i, "label": p["patient_id"]} for i, p in enumerate(PATIENTS)])


@asym_bp.route("/api/steps")
def api_steps():
    return jsonify({
        "steps": [
            {"id": i, "key": key, "label": lbl, "slice_based": sb, "explanation": expl}
            for i, key, lbl, sb, expl in R.KEY_LABEL_EXPL
        ],
        "first_slice_step": R.FIRST_SLICE_STEP,
        "last_slice_step": R.LAST_SLICE_STEP,
        "summary_step": R.SUMMARY_STEP,
    })


@asym_bp.route("/api/patient/<int:idx>")
def api_patient(idx):
    pl = _pipeline(idx)
    return jsonify({
        "patient_id": pl.patient_id,
        "depth": pl.depth,
        "slice_indices": pl.slice_indices,
        "best_slice_index": pl.best_slice_index(),
        "n_min_voxel": pl.p["n_min_voxel"],
    })


@asym_bp.route("/api/summary/<int:idx>")
def api_summary(idx):
    pl = _pipeline(idx)
    summ = pl.summary()
    try:
        summ["csv"] = pl.save_csv(_OUT_DIR)
    except Exception as e:  # pragma: no cover - best effort
        summ["csv_error"] = str(e)
    return jsonify(summ)


@asym_bp.route("/step.png")
def step_png():
    idx = int(request.args.get("id", -1))
    step = int(request.args.get("step", 1))
    pl = _pipeline(idx)
    z = max(0, min(int(request.args.get("z", 0)), pl.depth - 1))
    if step == R.SUMMARY_STEP:
        try:
            pl.save_csv(_OUT_DIR)
        except Exception:
            pass
    buf = R.render(pl, z, step)
    return send_file(buf, mimetype="image/png")
