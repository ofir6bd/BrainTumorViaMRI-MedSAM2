"""YOLO_finetune page — start fine-tune runs, follow them live, analyse the results.

A Flask blueprint, registered by the main viewer (Frontend/app.py) under /finetune/, so it
lives at http://localhost:5000/finetune/ next to Analytics.

A run is a separate process (`python -m YOLO_finetune.train --run <dir>`), so closing the
page or restarting the viewer never stops a training; the page picks it up again from
runs/<id>/status.json.
"""
import csv
import io
import os
import subprocess
import sys
from collections import OrderedDict
from datetime import datetime

import numpy as np
import psutil
import yaml
from flask import Blueprint, abort, jsonify, render_template, request, send_file
from werkzeug.exceptions import HTTPException
from PIL import Image

from .common import (CONFIG_PATH, DATASETS_DIR, ROOT, RUNS_DIR, brain_slices, list_patients,
                     load_config, load_volumes, read_json, rgb_slice, write_json)
from .train import new_run

MODELS = ["yolo11n-seg.pt", "yolo11s-seg.pt", "yolo11m-seg.pt", "yolo11l-seg.pt", "yolo11x-seg.pt",
          "yolo26m-seg.pt"]
# Settings a run may override from the UI: (type, min, max).
EDITABLE = {"train.epochs": (int, 1, 1000), "train.patience": (int, 0, 1000),
            "train.imgsz": (int, 64, 2048), "train.batch": (int, 1, 256)}
LIVE = ("queued", "running")
OVERLAY = {"tp": (12, 163, 12), "fn": (250, 178, 25), "fp": (208, 59, 59)}  # found / missed / false

finetune_bp = Blueprint("finetune", __name__, url_prefix="/finetune", template_folder="templates",
                        static_folder="static", static_url_path="/static")


@finetune_bp.errorhandler(HTTPException)
def _json_error(e):
    return jsonify({"error": e.description}), e.code


# ---------------------------------------------------------------------------- runs
def run_dir(run_id):
    d = os.path.join(RUNS_DIR, os.path.basename(run_id))
    if not os.path.isfile(os.path.join(d, "run_config.yaml")):
        abort(404)
    return d


def run_ids():
    if not os.path.isdir(RUNS_DIR):
        return []
    return sorted((r for r in os.listdir(RUNS_DIR)
                   if os.path.isfile(os.path.join(RUNS_DIR, r, "run_config.yaml"))), reverse=True)


def _alive(pid):
    try:
        return bool(pid) and psutil.Process(pid).is_running() and \
            psutil.Process(pid).status() != psutil.STATUS_ZOMBIE
    except psutil.Error:
        return False


def status_of(run_id):
    """status.json, with a run whose process is gone reported as `interrupted`."""
    st = read_json(os.path.join(RUNS_DIR, run_id, "status.json"), {}) or {}
    if st.get("state") in LIVE and not (_alive(st.get("pid")) or _alive(st.get("launcher_pid"))):
        st["state"] = "interrupted"
    return st


def active_run():
    return next((r for r in run_ids() if status_of(r).get("state") in LIVE), None)


def run_config(run_id):
    with open(os.path.join(RUNS_DIR, run_id, "run_config.yaml"), "r", encoding="utf-8") as f:
        return yaml.safe_load(f)


def results(run_id):
    """Ultralytics' per-epoch results.csv as {column: [values]}."""
    path = os.path.join(RUNS_DIR, run_id, "train", "results.csv")
    if not os.path.exists(path):
        return {}
    with open(path, "r", encoding="utf-8") as f:
        rows = [{k.strip(): v for k, v in r.items()} for r in csv.DictReader(f)]
    cols = OrderedDict((k, []) for k in (rows[0] if rows else {}))
    for r in rows:
        for k in cols:
            try:
                cols[k].append(float(r[k]))
            except (TypeError, ValueError):
                cols[k].append(None)
    return cols


def run_summary(run_id):
    cfg, st, res = run_config(run_id), status_of(run_id), results(run_id)
    ev = read_json(os.path.join(RUNS_DIR, run_id, "eval", "summary.json"))
    best = None
    fit = res.get("metrics/mAP50-95(M)")
    if fit:
        i = int(np.nanargmax([v if v is not None else np.nan for v in fit]))
        best = {"epoch": int(res["epoch"][i]), "map50_m": res.get("metrics/mAP50(M)", [None])[i],
                "map5095_m": fit[i]}
    return {
        "id": run_id, "created": cfg["run"]["created"], "smoke": cfg["run"].get("smoke"),
        "model": cfg["model"], "epochs_cfg": cfg["train"]["epochs"], "imgsz": cfg["train"]["imgsz"],
        "batch": cfg["train"]["batch"], "state": st.get("state"), "stage": st.get("stage"),
        "epoch": st.get("epoch"), "epochs": st.get("epochs"), "started": st.get("started"),
        "finished": st.get("finished"), "error": st.get("error"),
        "epochs_done": len(res.get("epoch", [])),
        "train_seconds": res.get("time", [None])[-1] if res.get("time") else None,
        "best": best,
        "test": ev.get("test") if ev else None, "val": ev.get("val") if ev else None,
        "has_best": os.path.exists(os.path.join(RUNS_DIR, run_id, "train", "weights", "best.pt")),
        "has_last": os.path.exists(os.path.join(RUNS_DIR, run_id, "train", "weights", "last.pt")),
    }


def spawn(rdir, *extra):
    """Start the run as its own process, output appended to log.txt."""
    if active_run():
        abort(409, "Another run is still going — stop it first.")
    log = open(os.path.join(rdir, "log.txt"), "ab")
    log.write(f"\n===== {datetime.now():%Y-%m-%d %H:%M:%S} YOLO_finetune.train {' '.join(extra)} =====\n".encode())
    log.flush()
    # Run as a module from inside the run folder; PYTHONPATH makes the package importable.
    env = {**os.environ, "PYTHONIOENCODING": "utf-8", "PYTHONUNBUFFERED": "1",
           "PYTHONPATH": os.pathsep.join(filter(None, [ROOT, os.environ.get("PYTHONPATH")]))}
    flags = subprocess.CREATE_NO_WINDOW if os.name == "nt" else 0
    proc = subprocess.Popen([sys.executable, "-m", "YOLO_finetune.train", "--run", rdir, *extra], cwd=rdir,
                            stdout=log, stderr=subprocess.STDOUT, env=env, creationflags=flags)
    st = read_json(os.path.join(rdir, "status.json"), {}) or {}
    st.update(state="queued", launcher_pid=proc.pid, pid=None, error=None,
              updated=datetime.now().isoformat(timespec="seconds"))
    write_json(os.path.join(rdir, "status.json"), st)


# ---------------------------------------------------------------------------- pages / API
@finetune_bp.route("/")
def index():
    return render_template("finetune.html")


@finetune_bp.route("/api/overview")
def api_overview():
    with open(CONFIG_PATH, "r", encoding="utf-8") as f:
        text = f.read()
    cfg = load_config()
    pools = {k: len(list_patients(cfg["data"][f"{k}_pool"])) for k in ("train", "val", "test")}
    return jsonify({"config_text": text, "config": cfg, "models": MODELS,
                    "editable": {k: {"min": v[1], "max": v[2]} for k, v in EDITABLE.items()},
                    "pools": pools, "active": active_run(),
                    "runs": [run_summary(r) for r in run_ids()]})


@finetune_bp.route("/api/runs", methods=["POST"])
def api_start():
    body = request.get_json(force=True) or {}
    overrides = []
    for key, (typ, lo, hi) in EDITABLE.items():
        if key in (body.get("overrides") or {}):
            try:
                v = typ(body["overrides"][key])
            except (TypeError, ValueError):
                abort(400, f"{key} must be a whole number")
            if not lo <= v <= hi:
                abort(400, f"{key} must be between {lo} and {hi}")
            if key == "train.imgsz" and v % 32:
                abort(400, "train.imgsz must be a multiple of 32")
            overrides.append(f"{key}={v}")
    model = body.get("model")
    if model:
        if model not in MODELS:
            abort(400, "unknown model")
        overrides.append(f"model={model}")
    if active_run():
        abort(409, "Another run is still going — stop it first.")
    rdir = new_run(overrides, smoke=bool(body.get("smoke")))
    spawn(rdir)
    return jsonify({"id": os.path.basename(rdir)})


@finetune_bp.route("/api/runs/<run_id>")
def api_run(run_id):
    d = run_dir(run_id)
    st = status_of(run_id)
    meta = read_json(os.path.join(DATASETS_DIR, st.get("dataset_key") or "-", "meta.json"))
    plots = sorted(f for f in os.listdir(os.path.join(d, "train"))
                   if f.lower().endswith((".png", ".jpg"))) if os.path.isdir(os.path.join(d, "train")) else []
    with open(os.path.join(d, "run_config.yaml"), "r", encoding="utf-8") as f:
        cfg_text = f.read()
    return jsonify({**run_summary(run_id), "status": st, "config": run_config(run_id),
                    "config_text": cfg_text, "results": results(run_id), "dataset": meta,
                    "eval": read_json(os.path.join(d, "eval", "summary.json")), "plots": plots})


@finetune_bp.route("/api/runs/<run_id>/eval/<split>")
def api_eval(run_id, split):
    if split not in ("val", "test"):
        abort(404)
    data = read_json(os.path.join(run_dir(run_id), "eval", f"{split}.json"))
    return jsonify(data) if data else abort(404)


@finetune_bp.route("/api/runs/<run_id>/log")
def api_log(run_id):
    path = os.path.join(run_dir(run_id), "log.txt")
    offset = int(request.args.get("offset", 0))
    if not os.path.exists(path):
        return jsonify({"text": "", "offset": 0})
    size = os.path.getsize(path)
    if offset > size:
        offset = 0
    offset = max(offset, size - 200_000)  # never ship more than the last ~200 kB
    with open(path, "rb") as f:
        f.seek(offset)
        text = f.read().decode("utf-8", "replace")
    return jsonify({"text": text, "offset": size})


@finetune_bp.route("/api/runs/<run_id>/stop", methods=["POST"])
def api_stop(run_id):
    d = run_dir(run_id)
    st = read_json(os.path.join(d, "status.json"), {}) or {}
    for pid in {st.get("pid"), st.get("launcher_pid")} - {None}:
        try:
            proc = psutil.Process(pid)
            for p in proc.children(recursive=True) + [proc]:  # dataloader workers too
                p.kill()
        except psutil.Error:
            pass
    st.update(state="stopped", finished=datetime.now().isoformat(timespec="seconds"),
              updated=datetime.now().isoformat(timespec="seconds"))
    write_json(os.path.join(d, "status.json"), st)
    return jsonify({"ok": True})


@finetune_bp.route("/api/runs/<run_id>/resume", methods=["POST"])
def api_resume(run_id):
    d = run_dir(run_id)
    if not os.path.exists(os.path.join(d, "train", "weights", "last.pt")):
        abort(400, "This run has no last.pt to resume from.")
    spawn(d, "--resume")
    return jsonify({"ok": True})


@finetune_bp.route("/api/runs/<run_id>/evaluate", methods=["POST"])
def api_evaluate(run_id):
    d = run_dir(run_id)
    if not os.path.exists(os.path.join(d, "train", "weights", "best.pt")):
        abort(400, "This run has no best.pt yet.")
    spawn(d, "--evaluate-only")
    return jsonify({"ok": True})


@finetune_bp.route("/api/runs/<run_id>/file/<path:name>")
def api_file(run_id, name):
    """results.csv, run_config.yaml, and the plots Ultralytics saved under train/."""
    d = run_dir(run_id)
    allowed = {"results.csv": os.path.join(d, "train", "results.csv"),
               "run_config.yaml": os.path.join(d, "run_config.yaml")}
    train_dir = os.path.join(d, "train")
    if os.path.isdir(train_dir):
        allowed.update({f: os.path.join(train_dir, f) for f in os.listdir(train_dir)
                        if f.lower().endswith((".png", ".jpg"))})
    if name not in allowed or not os.path.exists(allowed[name]):
        abort(404)
    return send_file(allowed[name], as_attachment=request.args.get("download") == "1")


@finetune_bp.route("/api/gpu")
def api_gpu():
    q = "name,utilization.gpu,memory.used,memory.total,temperature.gpu,power.draw"
    try:
        out = subprocess.run(["nvidia-smi", f"--query-gpu={q}", "--format=csv,noheader,nounits"],
                             capture_output=True, text=True, timeout=5,
                             creationflags=subprocess.CREATE_NO_WINDOW if os.name == "nt" else 0).stdout
        name, util, used, total, temp, power = [x.strip() for x in out.splitlines()[0].split(",")]
    except (OSError, subprocess.SubprocessError, ValueError, IndexError):
        return jsonify({"available": False})
    num = lambda v: float(v) if v.replace(".", "", 1).isdigit() else None  # noqa: E731
    return jsonify({"available": True, "name": name, "util": num(util), "mem_used": num(used),
                    "mem_total": num(total), "temp": num(temp), "power": num(power)})


# ---------------------------------------------------------------------------- prediction viewer
_models = OrderedDict()
_volumes = OrderedDict()
_profiles = OrderedDict()   # (run, split, patient, conf) -> per-slice numbers


def _cached(store, key, make, size):
    if key not in store:
        store[key] = make()
        while len(store) > size:
            store.popitem(last=False)
    store.move_to_end(key)
    return store[key]


def _model_of(run_id):
    """The run's best checkpoint, loaded once and kept."""
    weights = os.path.join(run_dir(run_id), "train", "weights", "best.pt")
    if not os.path.exists(weights):
        abort(404, "This run has no trained model (best.pt) yet.")
    from ultralytics import YOLO
    return _cached(_models, weights, lambda: YOLO(weights), 2)


def _patient_of(cfg, split, patient_id):
    if split not in ("val", "test"):
        abort(400, "split must be val or test")
    patient = next((p for p in list_patients(cfg["data"][f"{split}_pool"]) if p["id"] == patient_id), None)
    if patient is None:
        abort(404, "No such patient in that pool.")
    return patient


def _predict(run_id, split, patient_id, z, conf):
    cfg = run_config(run_id)
    patient = _patient_of(cfg, split, patient_id)
    vols = _cached(_volumes, patient["dir"], lambda: load_volumes(patient), 3)
    zs = brain_slices(vols["FLAIR"], cfg["data"]["min_fg_voxels"])
    z = min(zs, key=lambda v: abs(v - z)) if zs else z
    rgb = rgb_slice(vols, z, cfg["data"]["min_fg_voxels"])

    model = _model_of(run_id)
    r = model.predict(Image.fromarray(rgb), conf=conf, imgsz=cfg["train"]["imgsz"],
                      retina_masks=True, verbose=False, device=cfg["train"]["device"])[0]
    gt = vols["SEG"][:, :, z] > 0
    masks = r.masks.data.cpu().numpy().astype(bool) if r.masks is not None else np.zeros((0, *gt.shape), bool)
    confs = r.boxes.conf.cpu().numpy().tolist() if r.boxes is not None else []
    pred = masks.any(axis=0) if len(masks) else np.zeros_like(gt)
    inter = int((pred & gt).sum())
    return {"z": z, "brain_slices": zs, "rgb": rgb, "gt": gt, "pred": pred,
            "stats": {"z": z, "gt": int(gt.sum()), "pred": int(pred.sum()), "inter": inter,
                      "dice": 1.0 if not (pred.sum() + gt.sum()) else 2 * inter / float(pred.sum() + gt.sum()),
                      "confs": [round(c, 3) for c in confs]}}


def _args():
    """split, patient, z, conf from the query; an empty z means "pick the first slice"."""
    try:
        return (request.args["split"], request.args["patient"], int(request.args.get("z") or 0),
                float(request.args.get("conf") or 0.25))
    except (KeyError, ValueError):
        abort(400, "need split, patient, and numeric z / conf")


@finetune_bp.route("/api/pool_patients")
def api_pool_patients():
    """Patient ids of a pool, so a patient can be picked before any evaluation exists."""
    split = request.args.get("split", "test")
    if split not in ("val", "test"):
        abort(400, "split must be val or test")
    cfg = load_config()
    return jsonify({"split": split, "pool": cfg["data"][f"{split}_pool"],
                    "patients": [p["id"] for p in list_patients(cfg["data"][f"{split}_pool"])]})


@finetune_bp.route("/api/runs/<run_id>/profile.json")
def api_profile(run_id):
    """Run this run's model over every brain slice of one patient (a second or two)."""
    from .evaluate import patient_profile

    split = request.args.get("split", "test")
    patient_id = request.args.get("patient", "")
    try:
        conf = float(request.args.get("conf", 0.25))
    except ValueError:
        abort(400, "conf must be a number")
    cfg = run_config(run_id)
    patient = _patient_of(cfg, split, patient_id)
    key = (run_id, split, patient_id, round(conf, 3))
    return jsonify(_cached(_profiles, key, lambda: patient_profile(_model_of(run_id), patient, cfg, conf), 12))


@finetune_bp.route("/api/runs/<run_id>/predict.json")
def api_predict_json(run_id):
    res = _predict(run_id, *_args())
    return jsonify({**res["stats"], "brain_slices": res["brain_slices"]})


@finetune_bp.route("/api/runs/<run_id>/predict.png")
def api_predict_png(run_id):
    """Left: the RGB frame the model sees. Right: FLAIR (blue channel) with found / missed /
    false tumour pixels. Radiological view (patient's right on the left)."""
    res = _predict(run_id, *_args())
    rgb, gt, pred = res["rgb"], res["gt"], res["pred"]
    base = np.repeat(rgb[:, :, 2:3], 3, axis=2).astype(np.float32)
    for key, m in (("tp", gt & pred), ("fn", gt & ~pred), ("fp", pred & ~gt)):
        base[m] = base[m] * 0.35 + np.array(OVERLAY[key]) * 0.65
    left = np.rot90(rgb)
    right = np.rot90(base.astype(np.uint8))
    gap = np.full((left.shape[0], 6, 3), 18, np.uint8)
    img = Image.fromarray(np.concatenate([left, gap, right], axis=1))
    img = img.resize((img.width * 2, img.height * 2), Image.NEAREST)
    buf = io.BytesIO()
    img.save(buf, format="PNG")
    buf.seek(0)
    return send_file(buf, mimetype="image/png")

