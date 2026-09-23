"""MedSAM2_Finetune page — start runs, follow them live, and analyse what MedSAM2 added.

A Flask blueprint, registered by the main viewer (Frontend/app.py) under /medsam2/, so it
lives at http://localhost:5000/medsam2/ next to Analytics and YOLO_finetune.

A run is a separate process (`python -m MedSAM2_Finetune.train --run <dir>`), so closing
the page or restarting the viewer never stops a training; the page picks it up again from
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
from PIL import Image
from flask import Blueprint, abort, jsonify, render_template, request, send_file
from werkzeug.exceptions import HTTPException

from .common import (CONFIG_PATH, PROMPT_VARIANTS, ROOT, RUNS_DIR, UNFREEZE, checkpoints,
                     list_patients, load_config, read_json, write_json, yolo_runs)
from .train import new_run

LIVE = ("queued", "running")
OVERLAY = {"tp": (12, 163, 12), "fn": (250, 178, 25), "fp": (208, 59, 59)}  # found / missed / false

# Settings the page may override for a run: (kind, min, max). "int?" allows "all" (null).
EDITABLE = {
    "prompt.yolo_run": ("choice", "yolo_runs"),
    "prompt.variant": ("choice", "variants"),
    "prompt.yolo_conf": ("float", 0.005, 0.9),
    "prompt.score_min": ("float", 0.0, 0.95),
    "prompt.logit_scale": ("float", 0.05, 20.0),
    "prompt.empty_logit": ("float", -30.0, 0.0),
    "anchors.pick": ("choice", "anchor_rules"),
    "anchors.count": ("int", 1, 20),
    "anchors.min_gap": ("int", 1, 60),
    "hitl.rounds": ("int", 1, 10),
    "hitl.min_improvement": ("float", 0.0, 0.5),
    "model.checkpoint": ("choice", "checkpoints"),
    "data.max_train_patients": ("int?", 1, 5000),
    "data.val_patients": ("int?", 1, 5000),
    "data.clips_per_patient": ("int", 1, 64),
    "data.tumour_clip_fraction": ("float", 0.0, 1.0),
    "data.min_fg_voxels": ("int", 0, 50000),
    "video.num_frames": ("int", 2, 32),
    "video.reverse_fraction": ("float", 0.0, 1.0),
    "video.batch": ("int", 1, 8),
    "train.epochs": ("int", 1, 500),
    "train.patience": ("int", 0, 500),
    "train.accum": ("int", 1, 64),
    "train.lr": ("float", 1e-7, 1e-2),
    "train.memory_lr": ("float", 1e-9, 1e-2),
    "train.vision_lr": ("float", 1e-8, 1e-3),
    "train.unfreeze": ("choice", "unfreeze"),
    "train.dice_weight": ("float", 0.0, 10.0),
    "train.bce_weight": ("float", 0.0, 10.0),
    "train.iou_weight": ("float", 0.0, 10.0),
    "train.obj_weight": ("float", 0.0, 10.0),
    "train.fliplr": ("float", 0.0, 1.0),
    "train.workers": ("int", 0, 16),
    "evaluate.mask_threshold": ("float", -10.0, 10.0),
}
ANCHOR_RULES = {
    "yolo_peak": "the slice where YOLO is most sure and the tumour looks biggest",
    "yolo_score": "the slice with YOLO's single highest-scoring blob",
    "spread": "evenly spaced over the slices YOLO marked",
}
REBUILDS = ("prompt.yolo_run", "prompt.yolo_conf", "prompt.score_min", "data.min_fg_voxels")

medsam2_bp = Blueprint("medsam2", __name__, url_prefix="/medsam2", template_folder="templates",
                       static_folder="static", static_url_path="/static")


@medsam2_bp.errorhandler(HTTPException)
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
    """results.csv (one row per round) as {column: [values]}."""
    path = os.path.join(RUNS_DIR, run_id, "results.csv")
    if not os.path.exists(path):
        return {}
    with open(path, "r", encoding="utf-8") as f:
        rows = list(csv.DictReader(f))
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
    best = read_json(os.path.join(RUNS_DIR, run_id, "best.json"))
    d = os.path.join(RUNS_DIR, run_id)
    return {
        "id": run_id, "created": cfg["run"]["created"], "smoke": cfg["run"].get("smoke"),
        "checkpoint": cfg["model"]["checkpoint"], "variant": cfg["prompt"]["variant"],
        "yolo_run": st.get("yolo_run") or cfg["prompt"]["yolo_run"] or "(best)",
        "unfreeze": cfg["train"]["unfreeze"], "epochs_cfg": cfg["train"]["epochs"],
        "anchors": cfg.get("anchors", {}).get("pick"), "rounds": cfg.get("hitl", {}).get("rounds"),
        "state": st.get("state"), "stage": st.get("stage"), "epoch": st.get("epoch"),
        "epochs": st.get("epochs"), "started": st.get("started"), "finished": st.get("finished"),
        "error": st.get("error"), "epochs_done": len(res.get("epoch", [])),
        "train_seconds": sum(v for v in res.get("time", []) if v) or None,
        "best": best,
        "test": ev.get("test") if ev else None, "val": ev.get("val") if ev else None,
        "has_best": os.path.exists(os.path.join(d, "weights", "best.pt")),
        "has_last": os.path.exists(os.path.join(d, "weights", "last.pt")),
    }


def spawn(rdir, *extra):
    """Start the run as its own process, output appended to log.txt."""
    if active_run():
        abort(409, "Another run is still going — stop it first.")
    log = open(os.path.join(rdir, "log.txt"), "ab")
    log.write(f"\n===== {datetime.now():%Y-%m-%d %H:%M:%S} MedSAM2_Finetune.train "
              f"{' '.join(extra)} =====\n".encode())
    log.flush()
    env = {**os.environ, "PYTHONIOENCODING": "utf-8", "PYTHONUNBUFFERED": "1",
           "PYTHONPATH": os.pathsep.join(filter(None, [ROOT, os.environ.get("PYTHONPATH")]))}
    flags = subprocess.CREATE_NO_WINDOW if os.name == "nt" else 0
    proc = subprocess.Popen([sys.executable, "-m", "MedSAM2_Finetune.train", "--run", rdir, *extra],
                            cwd=ROOT, stdout=log, stderr=subprocess.STDOUT, env=env,
                            creationflags=flags)
    st = read_json(os.path.join(rdir, "status.json"), {}) or {}
    st.update(state="queued", launcher_pid=proc.pid, pid=None, error=None,
              updated=datetime.now().isoformat(timespec="seconds"))
    write_json(os.path.join(rdir, "status.json"), st)


def _choices():
    return {"yolo_runs": [r["id"] for r in yolo_runs()] + [""],
            "variants": list(PROMPT_VARIANTS), "unfreeze": list(UNFREEZE),
            "anchor_rules": list(ANCHOR_RULES),
            "checkpoints": checkpoints()}


# ---------------------------------------------------------------------------- pages / API
@medsam2_bp.route("/")
def index():
    return render_template("medsam2.html")


@medsam2_bp.route("/api/overview")
def api_overview():
    with open(CONFIG_PATH, "r", encoding="utf-8") as f:
        text = f.read()
    cfg = load_config()
    pools = {}
    for k in ("train", "val", "test"):
        try:
            pools[k] = len(list_patients(cfg["data"][f"{k}_pool"]))
        except OSError:
            pools[k] = None
    editable = {k: ({"kind": v[0], "choices": _choices()[v[1]]} if v[0] == "choice"
                    else {"kind": v[0], "min": v[1], "max": v[2]}) for k, v in EDITABLE.items()}
    return jsonify({"config_text": text, "config": cfg, "editable": editable, "rebuilds": REBUILDS,
                    "variants": PROMPT_VARIANTS, "unfreeze": UNFREEZE, "pools": pools,
                    "anchor_rules": ANCHOR_RULES,
                    "yolo_runs": yolo_runs(), "checkpoints": checkpoints(),
                    "active": active_run(), "runs": [run_summary(r) for r in run_ids()]})


@medsam2_bp.route("/api/runs", methods=["POST"])
def api_start():
    body = request.get_json(force=True) or {}
    given = body.get("overrides") or {}
    choices = _choices()
    overrides = []
    for key, spec in EDITABLE.items():
        if key not in given:
            continue
        raw = given[key]
        kind = spec[0]
        if kind == "choice":
            if raw not in choices[spec[1]]:
                abort(400, f"{key}: {raw!r} is not one of the offered values")
            overrides.append(f"{key}={raw!r}" if raw != "" else f"{key}=''")
            continue
        if kind == "int?" and (raw in ("", None, "all")):
            overrides.append(f"{key}=null")
            continue
        try:
            v = int(raw) if kind.startswith("int") else float(raw)
        except (TypeError, ValueError):
            abort(400, f"{key} must be a number")
        if not spec[1] <= v <= spec[2]:
            abort(400, f"{key} must be between {spec[1]} and {spec[2]}")
        overrides.append(f"{key}={v}")
    if active_run():
        abort(409, "Another run is still going — stop it first.")
    if not yolo_runs():
        abort(400, "No YOLO_finetune run has a checkpoint yet — there is nothing to prompt with.")
    rdir = new_run(overrides, smoke=bool(body.get("smoke")))
    spawn(rdir)
    return jsonify({"id": os.path.basename(rdir)})


@medsam2_bp.route("/api/runs/<run_id>")
def api_run(run_id):
    d = run_dir(run_id)
    st = status_of(run_id)
    with open(os.path.join(d, "run_config.yaml"), "r", encoding="utf-8") as f:
        cfg_text = f.read()
    prompt_meta = None
    if st.get("prompt_key"):
        prompt_meta = read_json(os.path.join(ROOT, "MedSAM2_Finetune", "cache",
                                             st["prompt_key"], "meta.json"))
    return jsonify({**run_summary(run_id), "status": st, "config": run_config(run_id),
                    "config_text": cfg_text, "results": results(run_id), "prompt": prompt_meta,
                    "eval": read_json(os.path.join(d, "eval", "summary.json"))})


_eval_files = OrderedDict()


@medsam2_bp.route("/api/runs/<run_id>/eval/<split>")
def api_eval(run_id, split):
    """The stored per-patient record of a scoring pass.

    With `?patient=`, one patient's numbers — the page uses this to chart a patient that has
    already been scored without asking the GPU to do the work again.
    """
    if split not in ("val", "test"):
        abort(404)
    path = os.path.join(run_dir(run_id), "eval", f"{split}.json")
    data = _cached(_eval_files, path, lambda: read_json(path), 2)
    if not data:
        abort(404, "This run has not been scored on that pool yet.")
    patient_id = request.args.get("patient")
    if not patient_id:
        return jsonify(data)
    rec = next((r for r in data["patients"] if r["id"] == patient_id), None)
    if rec is None:
        abort(404, "That patient is not in this pool's results.")
    from .common import dice
    return jsonify({**rec, "stored": True,
                    "dice": [round(dice(i, p, g), 4)
                             for i, p, g in zip(rec["inter"], rec["pred"], rec["gt"])],
                    "yolo_dice": [round(dice(i, p, g), 4)
                                  for i, p, g in zip(rec["yolo_inter"], rec["yolo"], rec["gt"])]})


@medsam2_bp.route("/api/runs/<run_id>/log")
def api_log(run_id):
    path = os.path.join(run_dir(run_id), "log.txt")
    offset = int(request.args.get("offset", 0))
    if not os.path.exists(path):
        return jsonify({"text": "", "offset": 0})
    size = os.path.getsize(path)
    if offset > size:
        offset = 0
    offset = max(offset, size - 200_000)
    with open(path, "rb") as f:
        f.seek(offset)
        text = f.read().decode("utf-8", "replace")
    return jsonify({"text": text, "offset": size})


@medsam2_bp.route("/api/runs/<run_id>/stop", methods=["POST"])
def api_stop(run_id):
    d = run_dir(run_id)
    st = read_json(os.path.join(d, "status.json"), {}) or {}
    for pid in {st.get("pid"), st.get("launcher_pid")} - {None}:
        try:
            proc = psutil.Process(pid)
            for p in proc.children(recursive=True) + [proc]:   # dataloader workers too
                p.kill()
        except psutil.Error:
            pass
    st.update(state="stopped", finished=datetime.now().isoformat(timespec="seconds"),
              updated=datetime.now().isoformat(timespec="seconds"))
    write_json(os.path.join(d, "status.json"), st)
    return jsonify({"ok": True})


@medsam2_bp.route("/api/runs/<run_id>/resume", methods=["POST"])
def api_resume(run_id):
    d = run_dir(run_id)
    if not os.path.exists(os.path.join(d, "weights", "last.pt")):
        abort(400, "This run has no last.pt to resume from.")
    spawn(d, "--resume")
    return jsonify({"ok": True})


@medsam2_bp.route("/api/runs/<run_id>/evaluate", methods=["POST"])
def api_evaluate(run_id):
    """Score this run's best.pt on the whole val and test pools, as its own process."""
    d = run_dir(run_id)
    if not os.path.exists(os.path.join(d, "weights", "best.pt")):
        abort(400, "This run has no best.pt yet.")
    spawn(d, "--evaluate-only")
    return jsonify({"ok": True})


@medsam2_bp.route("/api/gpu")
def api_gpu():
    q = "name,utilization.gpu,memory.used,memory.total,temperature.gpu,power.draw"
    try:
        out = subprocess.run(["nvidia-smi", f"--query-gpu={q}", "--format=csv,noheader,nounits"],
                             capture_output=True, text=True, timeout=5,
                             creationflags=subprocess.CREATE_NO_WINDOW if os.name == "nt" else 0).stdout
        name, util, used, total, temp, power = [x.strip() for x in out.splitlines()[0].split(",")]
    except (OSError, subprocess.SubprocessError, ValueError, IndexError):
        return jsonify({"available": False})
    num = lambda v: float(v) if v.replace(".", "", 1).isdigit() else None   # noqa: E731
    return jsonify({"available": True, "name": name, "util": num(util), "mem_used": num(used),
                    "mem_total": num(total), "temp": num(temp), "power": num(power)})


@medsam2_bp.route("/api/pool_patients")
def api_pool_patients():
    split = request.args.get("split", "test")
    if split not in ("train", "val", "test"):
        abort(400, "split must be train, val or test")
    cfg = load_config()
    pool = cfg["data"][f"{split}_pool"]
    return jsonify({"split": split, "pool": pool,
                    "patients": [p["id"] for p in list_patients(pool)]})


# ---------------------------------------------------------------------------- slice viewer
_models = OrderedDict()
_entries = OrderedDict()
_runs = OrderedDict()      # (run, patient, which) -> one propagation, reused by every panel


def _cached(store, key, make, size):
    if key not in store:
        store[key] = make()
        while len(store) > size:
            store.popitem(last=False)
    store.move_to_end(key)
    return store[key]


def _device(cfg):
    import torch
    return f"cuda:{cfg['train']['device']}" if torch.cuda.is_available() else "cpu"


def _model_of(run_id, which="best.pt"):
    from .evaluate import load_run_model

    d = run_dir(run_id)
    if not os.path.exists(os.path.join(d, "weights", which)):
        abort(404, f"This run has no {which} yet.")
    cfg = run_config(run_id)
    return _cached(_models, (d, which), lambda: load_run_model(d, cfg, _device(cfg), which)[0], 1)


def _entry_of(run_id, patient_id):
    """One patient's prompt cache; built on demand if this patient was never scored."""
    from .yolo_prompts import ensure_cache, load_prompt, patient_file

    cfg = run_config(run_id)
    if not os.path.exists(patient_file(cfg, patient_id)):
        patient = None
        for split in ("test", "val", "train"):
            patient = next((p for p in list_patients(cfg["data"][f"{split}_pool"])
                            if p["id"] == patient_id), None)
            if patient:
                break
        if patient is None:
            abort(404, "No such patient in any pool.")
        ensure_cache(cfg, [patient])
    return _cached(_entries, (run_id, patient_id), lambda: load_prompt(cfg, patient_id), 2)


def _prediction(run_id, patient_id, which="best.pt"):
    """Propagate this patient once (a few seconds) and keep the result for every panel."""
    from .evaluate import run_patient

    if which not in ("best.pt", "last.pt"):
        abort(400, "which must be best.pt or last.pt")
    cfg = run_config(run_id)
    entry = _entry_of(run_id, patient_id)
    key = (run_id, patient_id, which)
    return entry, _cached(_runs, key, lambda: run_patient(
        _model_of(run_id, which), cfg, patient_id, _device(cfg), entry), 2)


def _mask_at(cfg, out, k):
    """The MedSAM2 mask on one slice, from the propagated logits."""
    m = out["logits"][k] > float(cfg["evaluate"]["mask_threshold"])
    if cfg["evaluate"]["use_obj_score"] and out["objs"][k] <= 0:
        m = np.zeros_like(m)
    return m


@medsam2_bp.route("/api/runs/<run_id>/profile.json")
def api_profile(run_id):
    """Propagate one patient through this run's model and chart it slice by slice."""
    from .evaluate import patient_profile

    patient_id = request.args.get("patient", "")
    which = request.args.get("which", "best.pt")
    cfg = run_config(run_id)
    entry, out = _prediction(run_id, patient_id, which)
    return jsonify(patient_profile(_model_of(run_id, which), cfg, patient_id,
                                   _device(cfg), entry, out))


def _heat(prob, base):
    """The probability map laid over the FLAIR, so the brain stays visible underneath.

    Where YOLO says nothing you see the slice; where it is sure you see red.
    """
    p = (prob.astype(np.float32) / 255.0)[:, :, None]
    colour = np.concatenate([np.clip(p * 2.2, 0, 1),
                             np.clip(p * 1.7 - 0.2, 0, 1) * (1 - np.clip(p * 1.6 - 0.6, 0, 1)),
                             np.clip(0.95 - p * 2.2, 0, 1)], axis=-1) * 255
    alpha = np.power(p, 0.7) * 0.85
    return (base * (1 - alpha) + colour * alpha).astype(np.uint8)


def _slice_index(entry, z):
    zs = [int(v) for v in entry["z"]]
    try:
        z = int(z or zs[0])
    except ValueError:
        abort(400, "z must be a whole number")
    return zs, int(np.argmin([abs(v - z) for v in zs]))


@medsam2_bp.route("/api/runs/<run_id>/slice.png")
def api_slice_png(run_id):
    """Panels for one slice: the frame, YOLO's probability, and what MedSAM2 made of it."""
    patient_id = request.args.get("patient", "")
    which = request.args.get("which", "best.pt")
    panels = (request.args.get("panels") or "frame,prompt,medsam2,yolo").split(",")
    cfg = run_config(run_id)
    entry, out = _prediction(run_id, patient_id, which)
    zs, k = _slice_index(entry, request.args.get("z"))

    rgb = entry["rgb"][k]
    gt = entry["gt"][k] > 0
    yolo = entry["prob_f"][k] > 127
    flair = np.repeat(rgb[:, :, 2:3], 3, axis=2).astype(np.float32)
    pics = []
    for name in panels:
        if name == "frame":
            pics.append(rgb)
        elif name == "prompt":
            pics.append(_heat(entry["prob"][k], flair))
        elif name in ("yolo", "medsam2"):
            mask = _mask_at(cfg, out, k) if name == "medsam2" else yolo
            base = flair.copy()
            for key, m in (("tp", gt & mask), ("fn", gt & ~mask), ("fp", mask & ~gt)):
                base[m] = base[m] * 0.35 + np.array(OVERLAY[key]) * 0.65
            pics.append(base.astype(np.uint8))
    if not pics:
        abort(400, "no panels asked for")

    pics = [np.rot90(p) for p in pics]
    gap = np.full((pics[0].shape[0], 6, 3), 18, np.uint8)
    joined = pics[0]
    for p in pics[1:]:
        joined = np.concatenate([joined, gap, p], axis=1)
    img = Image.fromarray(joined)
    img = img.resize((img.width * 2, img.height * 2), Image.NEAREST)
    buf = io.BytesIO()
    img.save(buf, format="PNG")
    buf.seek(0)
    return send_file(buf, mimetype="image/png")


@medsam2_bp.route("/api/runs/<run_id>/slice.json")
def api_slice_json(run_id):
    """The numbers under the picture, for one slice."""
    from .common import dice

    patient_id = request.args.get("patient", "")
    which = request.args.get("which", "best.pt")
    cfg = run_config(run_id)
    entry, out = _prediction(run_id, patient_id, which)
    zs, k = _slice_index(entry, request.args.get("z"))
    gt = entry["gt"][k] > 0
    yolo = entry["prob_f"][k] > 127
    mask = _mask_at(cfg, out, k)
    stat = lambda m: {"px": int(m.sum()), "inter": int((m & gt).sum()),      # noqa: E731
                      "dice": round(dice(int((m & gt).sum()), int(m.sum()), int(gt.sum())), 4)}
    return jsonify({"z": zs[k], "brain_slices": zs, "gt": int(gt.sum()),
                    "yolo": stat(yolo), "medsam2": stat(mask),
                    "obj_score": round(float(out["objs"][k]), 3),
                    "yolo_score": round(float(entry["scores"][k]), 3),
                    "blobs": int(entry["nblobs"][k]),
                    "is_anchor": k in out["anchors"],
                    "anchor_z": [int(entry["z"][a]) for a in out["anchors"]],
                    "rounds": out["rounds"]})
