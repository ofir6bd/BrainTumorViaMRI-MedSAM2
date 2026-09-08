"""Flask blueprint for the inference-test tab (served under `/inftest`).

One variant per request. A whole patient is eleven MedSAM2 runs and can take minutes, so
the UI drives the sweep itself — it asks for O, draws the row, asks for GT, draws that
row, and so on. That keeps every request short enough to survive a browser timeout and
lets the table fill in while the GPU works.

Only one `PatientRun` is kept alive at a time. It holds the loaded volumes, the JPEG
frames and the MedSAM2 inference state, all of which are shared by every variant of that
patient; switching patients releases the previous one.
"""
import os

from flask import Blueprint, abort, jsonify, request, send_file

from YOLO.pipeline import list_weight_files
from . import render as R
from .pipeline import (PatientRun, aggregate, cached_variants, is_anchor_slice,
                       patient_cache_dir, test_patients)
from .variants import COLORS, ORDER, REFERENCE, VARIANTS

inftest_bp = Blueprint("inftest", __name__, url_prefix="/inftest")

PATIENTS = test_patients()
_ACTIVE = {"key": None, "run": None}


def _resolve_weights_path(weights_file):
    """Map a `weights` filename to a real path. Only names `list_weight_files()` returns
    are accepted, so the query string cannot be used for path traversal."""
    if not weights_file:
        return None
    for entry in list_weight_files():
        if entry["filename"] == weights_file:
            return entry["path"]
    abort(404)


def _run(idx, weights_file=None):
    if idx < 0 or idx >= len(PATIENTS):
        abort(404)
    weights_path = _resolve_weights_path(weights_file)
    key = (idx, weights_path)
    if _ACTIVE["key"] != key:
        if _ACTIVE["run"] is not None:
            _ACTIVE["run"].release()
        _ACTIVE["key"] = key
        _ACTIVE["run"] = PatientRun(PATIENTS[idx]["dir"], weights_path=weights_path)
    return _ACTIVE["run"]


@inftest_bp.route("/api/variants")
def api_variants():
    """The registry, in run order — the UI builds its table from this."""
    return jsonify([{
        "id": v.id, "label": v.label, "changes": v.changes, "why": v.why,
        "color": COLORS.get(v.id),
        "source": v.source, "conf": v.conf, "mode": v.mode, "tau": v.tau,
        "max_anchors": v.max_anchors, "early_stop": v.early_stop,
        "soft": v.is_soft, "reference": v.id == REFERENCE,
    } for v in (VARIANTS[i] for i in ORDER)])


@inftest_bp.route("/api/patients")
def api_patients():
    weights_path = _resolve_weights_path(request.args.get("weights"))
    out = []
    for i, p in enumerate(PATIENTS):
        done = cached_variants(p["patient_id"], weights_path)
        out.append({"id": i, "label": p["patient_id"], "done": len(done),
                    "total": len(ORDER), "variants": done})
    return jsonify(out)


@inftest_bp.route("/api/weights")
def api_weights():
    out = []
    for e in list_weight_files():
        label = (e["timestamp"].strftime("%Y-%m-%d %H:%M") if e["timestamp"]
                 else e["filename"])
        parts = []
        if e["val_loss"] is not None:
            parts.append(f"val_loss {e['val_loss']:.4f}")
        if e["test_dice"] is not None:
            parts.append(f"test_dice {e['test_dice']:.3f}")
        if parts:
            label += " · " + " · ".join(parts)
        out.append({"filename": e["filename"], "label": label})
    return jsonify(out)


@inftest_bp.route("/api/run/<int:idx>")
def api_run(idx):
    """Run (or load) **one** variant. `?variant=C2&force=1`."""
    variant_id = request.args.get("variant", REFERENCE)
    if variant_id not in VARIANTS:
        abort(404)
    run = _run(idx, request.args.get("weights"))
    force = request.args.get("force", "").lower() in ("1", "true")
    try:
        result, _ = run.run_variant(variant_id, force=force)
    except FileNotFoundError as e:
        return jsonify({"error": str(e)}), 500
    except Exception as e:  # noqa: BLE001 - surface model/CUDA failures in the UI
        return jsonify({"error": f"{type(e).__name__}: {e}"}), 500
    result = dict(result)
    result["best_slice_index"] = _best_slice(result)
    return jsonify(result)


def _best_slice(result):
    rows = [r for r in result.get("slices") or [] if r["gt_voxels"] > 0]
    if not rows:
        return int(result.get("depth", 2)) // 2
    return max(rows, key=lambda r: r["gt_voxels"])["z"]


@inftest_bp.route("/api/probe/<int:idx>")
def api_probe(idx):
    """YOLO-only pre-flight: confidence against YOLO's own per-slice Dice."""
    run = _run(idx, request.args.get("weights"))
    conf = float(request.args.get("conf", 0.25))
    force = request.args.get("force", "").lower() in ("1", "true")
    try:
        return jsonify(run.confidence_probe(conf=conf, force=force))
    except Exception as e:  # noqa: BLE001
        return jsonify({"error": f"{type(e).__name__}: {e}"}), 500


@inftest_bp.route("/api/aggregate")
def api_aggregate():
    """Every cached patient, collapsed to one row per variant."""
    weights_path = _resolve_weights_path(request.args.get("weights"))
    return jsonify(aggregate(weights_path=weights_path))


@inftest_bp.route("/segment.png")
def segment_png():
    """Ground truth | reference | selected variant (+ the prompt, on an anchor slice)."""
    idx = int(request.args.get("id", -1))
    variant_id = request.args.get("variant", REFERENCE)
    if variant_id not in VARIANTS:
        abort(404)
    run = _run(idx, request.args.get("weights"))

    cached = set(cached_variants(run.patient_id, run.weights_path))
    if variant_id not in cached:
        return send_file(
            R.render_message(f"Variant {variant_id} has not been run for this patient."),
            mimetype="image/png")

    masks = {}
    for vid in {REFERENCE, variant_id} & cached:
        masks[vid] = run.load(vid)[1]

    z = max(0, min(int(request.args.get("z", 0)), run.depth - 1))

    prompt = None
    variant = VARIANTS[variant_id]
    if is_anchor_slice(run, variant, z):
        from .variants import build_prompt
        import numpy as np
        prompt = build_prompt(variant, run.prompt_source(variant, z))
        # An all-zero prompt was never registered, so the panel should say "no prompt"
        # rather than draw an empty heatmap and imply one was handed over.
        if prompt is not None and not np.any(prompt):
            prompt = None

    return send_file(R.render(run, z, variant_id, masks, prompt=prompt),
                     mimetype="image/png")


@inftest_bp.route("/api/status/<int:idx>")
def api_status(idx):
    """Which variants already have a result on disk for this (patient, weights)."""
    weights_path = _resolve_weights_path(request.args.get("weights"))
    if idx < 0 or idx >= len(PATIENTS):
        abort(404)
    pid = PATIENTS[idx]["patient_id"]
    return jsonify({
        "patient_id": pid,
        "cached": cached_variants(pid, weights_path),
        "cache_dir": os.path.relpath(patient_cache_dir(pid, weights_path)),
    })
