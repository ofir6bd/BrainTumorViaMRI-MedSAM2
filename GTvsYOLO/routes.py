"""Flask blueprint for the GT-vs-YOLO comparison tab (served under `/gtvsyolo`).

Patients come from the labeled, held-out **test** split only (`pipeline.test_patients`),
the same source the YOLO tab uses — so neither arm is being scored on data YOLO trained
on.

`/api/patient/<idx>` runs the comparison synchronously on first request for a given
(patient, weights) pair. That is minutes of GPU work; every later request for the same
pair is served from `outputs/gtvsyolo/`. The JS shows a running state while it waits.
"""
import os

from flask import Blueprint, abort, jsonify, request, send_file

from YOLO.pipeline import list_weight_files
from . import render as R
from .pipeline import ARMS, Comparison, is_cached, test_patients

gtvsyolo_bp = Blueprint("gtvsyolo", __name__, url_prefix="/gtvsyolo")

PATIENTS = test_patients()
_COMPARISONS = {}


def _resolve_weights_path(weights_file):
    """Map a `weights` query-string filename to a real path in `YOLO/weights/`.

    Only filenames returned by `list_weight_files()` are accepted, so the query string
    cannot be used for path traversal. `None` means "use the pipeline default".
    """
    if not weights_file:
        return None
    for entry in list_weight_files():
        if entry["filename"] == weights_file:
            return entry["path"]
    abort(404)


def _comparison(idx, weights_file=None):
    if idx < 0 or idx >= len(PATIENTS):
        abort(404)
    weights_path = _resolve_weights_path(weights_file)
    key = (idx, weights_path)
    if key not in _COMPARISONS:
        _COMPARISONS[key] = Comparison(PATIENTS[idx]["dir"], weights_path=weights_path)
    return _COMPARISONS[key]


@gtvsyolo_bp.route("/api/patients")
def api_patients():
    """Test-split patients, flagged with whether a cached run already exists."""
    weights_path = _resolve_weights_path(request.args.get("weights"))
    return jsonify([
        {"id": i, "label": p["patient_id"],
         "cached": is_cached(p["patient_id"], weights_path)}
        for i, p in enumerate(PATIENTS)
    ])


@gtvsyolo_bp.route("/api/weights")
def api_weights():
    """Checkpoints for the Model dropdown (the YOLO arm's prompt source), best first."""
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
        out.append({"filename": e["filename"], "label": label})
    return jsonify(out)


@gtvsyolo_bp.route("/api/patient/<int:idx>")
def api_patient(idx):
    """Run (or load) the comparison and return the full summary.

    Blocks for the duration of the run on a cache miss — both MedSAM2 arms, up to 7
    anchors each.
    """
    comp = _comparison(idx, request.args.get("weights"))
    force = request.args.get("force", "").lower() in ("1", "true")
    try:
        result = comp.result(force=force)
    except FileNotFoundError as e:
        return jsonify({"error": str(e)}), 500
    except Exception as e:  # noqa: BLE001 - surface model/CUDA failures in the UI
        return jsonify({"error": f"{type(e).__name__}: {e}"}), 500

    result = dict(result)
    result["best_slice_index"] = comp.best_slice_index()
    result["cached"] = comp.is_cached()
    return jsonify(result)


@gtvsyolo_bp.route("/api/cached/<int:idx>")
def api_cached(idx):
    """Whether this (patient, weights) pair already has a result on disk — lets the UI
    warn before kicking off a multi-minute run."""
    comp = _comparison(idx, request.args.get("weights"))
    return jsonify({"cached": comp.is_cached(), "patient_id": comp.patient_id})


@gtvsyolo_bp.route("/segment.png")
def segment_png():
    """3-panel figure for one slice: GT | MedSAM2-from-GT | MedSAM2-from-YOLO.

    On an anchor slice the prompt mask each arm was given is outlined on its panel.
    """
    idx = int(request.args.get("id", -1))
    comp = _comparison(idx, request.args.get("weights"))
    if not comp.is_cached() and comp._result is None:
        return send_file(
            R.render_message("No run yet for this patient — select it to start."),
            mimetype="image/png")

    result = comp.result()
    z = max(0, min(int(request.args.get("z", 0)), result["depth"] - 1))

    prompts = None
    if z in result["schedule"]:
        # Show the masks that were actually fed to MedSAM2 on this anchor: GT always,
        # YOLO only if that arm reached this anchor and had something there.
        prompts = {"gt": comp.gt_wt[:, :, z]}
        used_by_yolo = any(z in rnd["prompted_slices"]
                           for rnd in result["arms"]["yolo"]["rounds"])
        if used_by_yolo:
            from YOLO.pipeline import YoloPipeline
            yp = YoloPipeline(comp.dir, weights_path=comp.weights_path)
            prompts["yolo"] = yp.process_slice(z)["pred_wt"]

    return send_file(R.render(comp, z, prompts_by_arm=prompts), mimetype="image/png")


@gtvsyolo_bp.route("/api/summary/<int:idx>")
def api_summary(idx):
    """Per-slice Dice rows for both arms — the summary table and two-line chart."""
    comp = _comparison(idx, request.args.get("weights"))
    result = comp.result()
    return jsonify({
        "patient_id": result["patient_id"],
        "rows": result["slices"],
        "arms": {a: {k: result["arms"][a][k] for k in
                     ("label", "anchors_used", "anchors_available", "anchor_z",
                      "stop_reason", "final_dice", "seconds", "rounds")}
                 for a in ARMS},
        "schedule": result["schedule"],
        "yolo_empty_anchors": result["yolo_empty_anchors"],
        "comparable_rounds": result["comparable_rounds"],
        "total_seconds": result["total_seconds"],
    })
