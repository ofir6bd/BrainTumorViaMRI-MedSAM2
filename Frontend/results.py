"""Results page — every scored YOLO and MedSAM2 run, patient by patient, in one place.

A Flask blueprint at /results/. It only *reads* what the runs already saved
(`YOLO_finetune/runs/<id>/eval/{val,test}.json`, `MedSAM2_Finetune/runs/<id>/eval/...`), so
opening it never touches the GPU. The page joins these scores with the dataset statistics
(`/api/dashboard/data`) by patient id, which is what lets it answer "where does each model
fail?" — by tumour size, tumour parts, location, pool.

Pools: `test` is scored by both models. `val` means a different pool for each model —
YOLO's is `yolo_val`, MedSAM2's is `medsam2_val` — so the page names them explicitly.
"""
import os
from collections import OrderedDict

from flask import Blueprint, abort, jsonify, render_template, request
from werkzeug.exceptions import HTTPException

from MedSAM2_Finetune import routes as med
from YOLO_finetune import routes as yolo
from YOLO_finetune.common import read_json

results_bp = Blueprint("results", __name__, url_prefix="/results", template_folder="templates")

# the pool behind each (model, split) pair
POOL = {("yolo", "val"): "yolo_val", ("yolo", "test"): "test",
        ("medsam2", "val"): "medsam2_val", ("medsam2", "test"): "test"}


@results_bp.errorhandler(HTTPException)
def _json_error(e):
    return jsonify({"error": e.description}), e.code


@results_bp.route("/")
def index():
    return render_template("results.html")


# ---------------------------------------------------------------------------- cached reads
_files = OrderedDict()


def _read(path):
    """A run's eval JSON, parsed once per file version."""
    if not os.path.exists(path):
        return None
    key = (path, os.path.getmtime(path))
    if key not in _files:
        _files[key] = read_json(path)
        while len(_files) > 16:
            _files.popitem(last=False)
    _files.move_to_end(key)
    return _files[key]


def _dice(inter, pred, gt):
    return 1.0 if pred + gt == 0 else 2.0 * inter / float(pred + gt)


def _run_dir(model, run_id):
    root = yolo.RUNS_DIR if model == "yolo" else med.RUNS_DIR
    d = os.path.join(root, os.path.basename(run_id))
    if not os.path.isdir(d):
        abort(404, f"No {model} run {run_id}.")
    return d


# ---------------------------------------------------------------------------- sources
def _yolo_source(run_id):
    s = yolo.run_summary(run_id)
    d = os.path.join(yolo.RUNS_DIR, run_id)
    return {"id": run_id, "model": "yolo", "created": s["created"], "state": s["state"], "smoke": s["smoke"],
            "label": f"{run_id} · {s['model']}", "arch": s["model"], "epochs_done": s["epochs_done"],
            "splits": [sp for sp in ("val", "test") if os.path.exists(os.path.join(d, "eval", f"{sp}.json"))],
            "val": _brief(s.get("val")), "test": _brief(s.get("test"))}


def _med_source(run_id):
    s = med.run_summary(run_id)
    d = os.path.join(med.RUNS_DIR, run_id)
    cfg = med.run_config(run_id)
    return {"id": run_id, "model": "medsam2", "created": s["created"], "state": s["state"], "smoke": s["smoke"],
            "label": f"{run_id} · {s['unfreeze']}", "yolo_run": cfg["prompt"].get("yolo_run"),
            "epochs_done": s["epochs_done"], "train_patients": cfg["data"].get("max_train_patients"),
            "splits": [sp for sp in ("val", "test") if os.path.exists(os.path.join(d, "eval", f"{sp}.json"))],
            "val": _brief(s.get("val")), "test": _brief(s.get("test"))}


def _brief(ev):
    if not ev:
        return None
    keep = ("patients", "dice3d_mean", "dice3d_median", "yolo_dice3d_mean", "delta_mean", "helped", "hurt",
            "conf", "threshold", "tumour_slice_dice", "sensitivity", "specificity")
    return {k: ev.get(k) for k in keep if k in ev}


@results_bp.route("/api/sources")
def api_sources():
    ys = [_yolo_source(r) for r in yolo.run_ids()]
    ms = [_med_source(r) for r in med.run_ids()]
    return jsonify({"yolo": [s for s in ys if s["splits"]], "medsam2": [s for s in ms if s["splits"]],
                    "pools": {f"{m}:{sp}": p for (m, sp), p in POOL.items()}})


# ---------------------------------------------------------------------------- per-patient scores
@results_bp.route("/api/scores/<model>/<run_id>/<split>")
def api_scores(model, run_id, split):
    """One run's per-patient 3D Dice on one pool, at every threshold it was scored at.

    YOLO: `dice[k]` is the 3D Dice at `thresholds[k]` (blob confidence); `at` is the index
    of the confidence the run reports. MedSAM2: the same over its logit cut-offs, plus the
    3D Dice of the YOLO hint it was given (`hint`) on the very same slices.
    """
    if model not in ("yolo", "medsam2") or split not in ("val", "test"):
        abort(404)
    d = _run_dir(model, run_id)
    data = _read(os.path.join(d, "eval", f"{split}.json"))
    if not data:
        abort(404, f"{run_id} has not been scored on {split}.")
    summ = read_json(os.path.join(d, "eval", "summary.json")) or {}
    ths = data.get("thresholds") or []
    if model == "yolo":
        conf = (summ.get(split) or {}).get("conf", summ.get("conf"))
        at = ths.index(conf) if conf in ths else 0
        rows = []
        for p in data["patients"]:
            sw = p["sweep"]
            sl = p.get("slices") or {}
            rows.append({"id": p["id"], "dice": [round(_dice(*t), 5) for t in sw],
                         "gt": sw[at][2], "pred": sw[at][1], "inter": sw[at][0],
                         "tumour_slices": sum(1 for g in sl.get("gt", []) if g),
                         "missed_slices": sum(1 for g, pr in zip(sl.get("gt", []), sl.get("pred", [])) if g and not pr),
                         "false_slices": sum(1 for g, pr in zip(sl.get("gt", []), sl.get("pred", [])) if pr and not g)})
        return jsonify({"model": model, "run": run_id, "split": split, "pool": POOL[(model, split)],
                        "thresholds": ths, "at": at, "unit": "confidence", "rows": rows})
    thr = float((summ.get(split) or {}).get("threshold", summ.get("threshold", 0.0)))
    at = ths.index(thr) if thr in ths else 0
    rows = []
    for p in data["patients"]:
        gt, pr, it = p.get("gt", []), p.get("pred", []), p.get("inter", [])
        rows.append({"id": p["id"], "dice": [round(_dice(*t), 5) for t in p["sweep"]], "hint": p.get("yolo_dice3d"),
                     "gt": sum(gt), "pred": sum(pr), "inter": sum(it), "anchor_z": (p.get("anchor_z") or [None])[0],
                     "tumour_slices": sum(1 for g in gt if g),
                     "missed_slices": sum(1 for g, q in zip(gt, pr) if g and not q),
                     "false_slices": sum(1 for g, q in zip(gt, pr) if q and not g)})
    return jsonify({"model": model, "run": run_id, "split": split, "pool": POOL[(model, split)],
                    "thresholds": ths, "at": at, "unit": "logit", "rows": rows,
                    "yolo_run": (data.get("settings") or {}).get("prompt", {}).get("yolo_run")})


@results_bp.route("/api/patient")
def api_patient():
    """Slice-by-slice numbers of one patient from the chosen runs (no GPU)."""
    pid = request.args.get("pid", "")
    out = {"id": pid, "series": []}
    for model in ("yolo", "medsam2"):
        run_id, split = request.args.get(model), request.args.get(f"{model}_split", "test")
        if not run_id:
            continue
        data = _read(os.path.join(_run_dir(model, run_id), "eval", f"{split}.json"))
        rec = next((p for p in (data or {}).get("patients", []) if p["id"] == pid), None)
        if rec is None:
            continue
        if model == "yolo":
            sl = rec["slices"]
            out["series"].append({"model": "yolo", "run": run_id, "z": sl["z"], "gt": sl["gt"], "pred": sl["pred"],
                                  "inter": sl["inter"], "score": sl.get("conf")})
        else:
            out["series"].append({"model": "medsam2", "run": run_id, "z": rec["z"], "gt": rec["gt"], "pred": rec["pred"],
                                  "inter": rec["inter"], "obj": rec.get("obj"), "anchor_z": rec.get("anchor_z")})
            out["series"].append({"model": "hint", "run": run_id, "z": rec["z"], "gt": rec["gt"], "pred": rec["yolo"],
                                  "inter": rec["yolo_inter"], "score": rec.get("score")})
    if not out["series"]:
        abort(404, "Neither run scored this patient.")
    return jsonify(out)
