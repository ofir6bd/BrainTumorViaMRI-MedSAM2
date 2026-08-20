"""Flask blueprint exposing the YOLO segmentation pipeline to the web UI.

Patients come from the labeled, held-out **test** split (`pipeline.list_test_patients`)
— never train/val — so every Dice score shown here is on data the model never saw
during training. Wiring this blueprint into `Frontend/app.py` (plus the sidebar button)
is the Stage F integration step in README.md.
"""
from flask import Blueprint, abort, jsonify, request, send_file

from . import render as R
from .pipeline import YoloPipeline, list_test_patients, list_weight_files, load_weights_metadata

yolo_bp = Blueprint("yolo", __name__, url_prefix="/yolo")

PATIENTS = list_test_patients()
_PIPELINES = {}


def _resolve_weights_path(weights_file):
    """Map a `weights` query-string filename to a real path in `YOLO/weights/`.

    Only filenames returned by `list_weight_files()` are accepted (prevents path
    traversal via the query string). Returns `None` for empty/unknown values, meaning
    "use the default" (best-by-val_loss, resolved inside `YoloPipeline`).
    """
    if not weights_file:
        return None
    for entry in list_weight_files():
        if entry["filename"] == weights_file:
            return entry["path"]
    abort(404)


def _pipeline(idx, weights_file=None):
    if idx < 0 or idx >= len(PATIENTS):
        abort(404)
    weights_path = _resolve_weights_path(weights_file)
    key = (idx, weights_path)
    if key not in _PIPELINES:
        _PIPELINES[key] = YoloPipeline(PATIENTS[idx]["dir"], weights_path=weights_path)
    return _PIPELINES[key]


@yolo_bp.route("/api/weights")
def api_weights():
    """List available trained checkpoints for the "Model" dropdown, best-first."""
    out = []
    for e in list_weight_files():
        label = e["timestamp"].strftime("%Y-%m-%d %H:%M") if e["timestamp"] else e["filename"]
        parts = []
        if e["val_loss"] is not None:
            parts.append(f"val_loss {e['val_loss']:.4f}")
        if e["test_dice"] is not None:
            parts.append(f"test_dice {e['test_dice']:.3f}")
        if parts:
            label += " · " + " · ".join(parts)
        out.append({
            "filename": e["filename"],
            "label": label,
            "val_loss": e["val_loss"],
            "test_dice": e["test_dice"],
            "has_training_info": e["has_training_info"],
        })
    return jsonify(out)


@yolo_bp.route("/api/weights/<path:filename>/info")
def api_weights_info(filename):
    """Full training metadata for the "Show training data" button — per-epoch metrics,
    run params, and per-patient test Dice, read from the sidecar `.json` next to
    `filename` (written by `YOLO/train.py`). 404s for unknown filenames; returns
    `{"available": false}` (200) for known weights with no sidecar (legacy files).
    """
    match = next((e for e in list_weight_files() if e["filename"] == filename), None)
    if match is None:
        abort(404)
    meta = load_weights_metadata(match["path"])
    if meta is None:
        return jsonify({"available": False, "filename": filename})
    meta = dict(meta)
    meta["available"] = True
    meta["filename"] = filename
    return jsonify(meta)


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
    pl = _pipeline(idx, request.args.get("weights"))
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
    pl = _pipeline(idx, request.args.get("weights"))
    return jsonify({"patient_id": pl.patient_id, "rows": pl.dice_table()})
