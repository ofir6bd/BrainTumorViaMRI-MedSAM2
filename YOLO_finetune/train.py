"""Run one fine-tune: build the dataset -> train YOLO11m-seg -> evaluate on val + test.

From the repo root:

    python -m YOLO_finetune.train                          # new run from config.yaml
    python -m YOLO_finetune.train --set train.epochs=5     # override any key (dotted, repeatable)
    python -m YOLO_finetune.train --smoke                  # 2 patients per pool, 2 epochs: end-to-end check
    python -m YOLO_finetune.train --run <dir>              # run a prepared run folder (the page does this)
    python -m YOLO_finetune.train --run <dir> --resume     # continue a stopped run from its last.pt
    python -m YOLO_finetune.train --run <dir> --evaluate-only

Each run lives in runs/<YYYYMMDD-HHMMSS>/: run_config.yaml (the exact settings), status.json
(progress, read by the UI), train/ (Ultralytics output: results.csv, weights/best.pt, plots)
and eval/ (val.json, test.json, summary.json).
"""
import argparse
import copy
import os
import time
import traceback
from datetime import datetime

import yaml

from .build_dataset import build
from .common import PRETRAINED_DIR, RUNS_DIR, load_config, read_json, write_json
from .evaluate import evaluate

TRAIN_KEYS = ("epochs", "patience", "imgsz", "batch", "optimizer", "close_mosaic", "mosaic",
              "fliplr", "hsv_h", "hsv_s", "hsv_v", "seed", "workers", "device", "amp")
SMOKE = {"max_patients": 2, "epochs": 2}


class Status:
    """runs/<run>/status.json — the only channel from this process to the UI."""

    def __init__(self, run_dir):
        self.path = os.path.join(run_dir, "status.json")
        self.data = read_json(self.path, {}) or {}

    def update(self, **kw):
        self.data.update(kw, updated=datetime.now().isoformat(timespec="seconds"))
        write_json(self.path, self.data)


def _set(cfg, dotted, value):
    node = cfg
    *path, last = dotted.split(".")
    for k in path:
        node = node[k]
    if last not in node:
        raise KeyError(f"unknown setting {dotted}")
    node[last] = yaml.safe_load(value)


def new_run(overrides=(), smoke=False):
    """Create runs/<id>/ with its own copy of the settings; return its path."""
    cfg = copy.deepcopy(load_config())
    for item in overrides:
        key, _, value = item.partition("=")
        _set(cfg, key.strip(), value.strip())
    if smoke:
        cfg["train"]["epochs"] = SMOKE["epochs"]
    run_id = datetime.now().strftime("%Y%m%d-%H%M%S")
    run_dir = os.path.join(RUNS_DIR, run_id)
    os.makedirs(run_dir)
    cfg["run"] = {"id": run_id, "created": datetime.now().isoformat(timespec="seconds"),
                  "smoke": smoke, "max_patients": SMOKE["max_patients"] if smoke else None}
    with open(os.path.join(run_dir, "run_config.yaml"), "w", encoding="utf-8") as f:
        yaml.safe_dump(cfg, f, sort_keys=False)
    Status(run_dir).update(state="queued", created=cfg["run"]["created"])
    return run_dir


def _model_source(name):
    """A .pt name is fetched into pretrained/ (Ultralytics downloads known weights to the
    path it is given); a .yaml builds the architecture untrained."""
    return os.path.join(PRETRAINED_DIR, name) if name.endswith(".pt") else name


def _train(run_dir, cfg, dataset_dir, status, resume):
    from ultralytics import YOLO

    os.makedirs(PRETRAINED_DIR, exist_ok=True)
    last = os.path.join(run_dir, "train", "weights", "last.pt")
    model = YOLO(last if resume else _model_source(cfg["model"]))
    progress = {"batch": 0, "t": 0.0}

    def epoch_start(trainer):
        progress["batch"] = 0

    def batch_end(trainer):
        progress["batch"] += 1
        if time.time() - progress["t"] > 2:  # no need to hit the disk every batch
            progress["t"] = time.time()
            status.update(epoch=trainer.epoch + 1, epochs=trainer.epochs,
                          batch=progress["batch"], batches=len(trainer.train_loader))

    def epoch_end(trainer):
        # Ultralytics' final validation fires this once more after the last epoch.
        status.update(epoch=min(trainer.epoch + 1, trainer.epochs), epochs=trainer.epochs, batch=progress["batch"],
                      batches=len(trainer.train_loader), epoch_time=trainer.epoch_time,
                      best_fitness=float(trainer.best_fitness or 0))

    model.add_callback("on_train_epoch_start", epoch_start)
    model.add_callback("on_train_batch_end", batch_end)
    model.add_callback("on_fit_epoch_end", epoch_end)
    if resume:
        model.train(resume=True)
    else:
        model.train(data=os.path.join(dataset_dir, "data.yaml"), project=run_dir, name="train",
                    exist_ok=True, plots=True, **{k: cfg["train"][k] for k in TRAIN_KEYS})
    return os.path.join(run_dir, "train", "weights", "best.pt")


def run(run_dir, resume=False, evaluate_only=False):
    with open(os.path.join(run_dir, "run_config.yaml"), "r", encoding="utf-8") as f:
        cfg = yaml.safe_load(f)
    status = Status(run_dir)
    max_patients = cfg["run"].get("max_patients")
    status.update(state="running", pid=os.getpid(), started=datetime.now().isoformat(timespec="seconds"),
                  error=None, stage="dataset")
    try:
        best = os.path.join(run_dir, "train", "weights", "best.pt")
        if not evaluate_only:
            dataset_dir, meta = build(cfg, max_patients=max_patients,
                                      progress=lambda d, t: status.update(dataset_done=d, dataset_total=t))
            status.update(stage="train", dataset_key=meta["key"])
            best = _train(run_dir, cfg, dataset_dir, status, resume)
        if not os.path.exists(best):
            raise FileNotFoundError(f"no trained weights at {best}")
        status.update(stage="evaluate", eval_done=0, eval_total=None)
        evaluate(best, cfg, os.path.join(run_dir, "eval"), max_patients=max_patients,
                 progress=lambda d, t, split: status.update(eval_done=d, eval_total=t, eval_split=split))
        status.update(state="done", stage="done", finished=datetime.now().isoformat(timespec="seconds"))
    except Exception as e:
        traceback.print_exc()
        status.update(state="failed", error=f"{type(e).__name__}: {e}",
                      finished=datetime.now().isoformat(timespec="seconds"))
        raise


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--run", help="existing run folder (created by the UI or a previous call)")
    ap.add_argument("--set", action="append", default=[], metavar="KEY=VALUE",
                    help="override a config.yaml setting for a new run, e.g. train.epochs=5")
    ap.add_argument("--smoke", action="store_true", help="2 patients per pool, 2 epochs")
    ap.add_argument("--resume", action="store_true", help="continue from the run's last.pt")
    ap.add_argument("--evaluate-only", action="store_true", help="re-run evaluation of best.pt")
    args = ap.parse_args()
    run_dir = os.path.abspath(args.run or new_run(args.set, args.smoke))
    print(f"[run] {run_dir}", flush=True)
    # Ultralytics drops side files (e.g. the model it downloads for its AMP check) into the
    # working directory; working inside the run keeps them in this folder.
    os.chdir(run_dir)
    run(run_dir, resume=args.resume, evaluate_only=args.evaluate_only)


if __name__ == "__main__":
    main()
