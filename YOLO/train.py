"""Stage D — one-shot YOLO-seg fine-tuning wrapper (README.md §3 Stage D).

Run once, from the repo root, after activating .venv:

    .\\.venv\\Scripts\\Activate.ps1
    python YOLO\\train.py

Builds `YOLO/dataset/` first (Stages A-C) if it doesn't already exist (train 60% / val
20% / test 20%, all labeled), fine-tunes a pretrained `yolo11n-seg.pt` on it, evaluates
the result's Dice on the held-out test split (`pipeline.evaluate_test_split`), then
saves the checkpoint as `YOLO/weights/best_<timestamp>_valloss<v>_testdice<v>.pt` plus a
sidecar `.json` with the full per-epoch training history — a uniquely named file per
run, so a new training run never silently overwrites a previously deployed checkpoint.
The web UI lets the user pick which saved weights file to use and inspect its training
data (see `YOLO/pipeline.py:list_weight_files` and the "Model" dropdown / "Show training
data" button in the YOLO tab).

Smoke test (quick, small subset — verify the pipeline end-to-end before a full run):

    python YOLO\\train.py --smoke-test

This rebuilds `YOLO/dataset/` from only a few patients per split, with 2 epochs and a
small image size, then trains and saves a new `weights/best_*.pt` as usual.

Full training on a fraction of the data (keeps the 60/20/20 split ratio, keeps the
full epochs/imgsz/batch defaults — only the patient count is reduced):

    python YOLO\train.py --rebuild-dataset --data-fraction 0.5
"""
import argparse
import csv
import os
import shutil
import sys
from datetime import datetime

_HERE = os.path.dirname(os.path.abspath(__file__))
if _HERE not in sys.path:
    sys.path.insert(0, _HERE)

from pipeline import (  # noqa: E402
    PARAMS, build_dataset, evaluate_test_split, save_weights_metadata, split_patients,
)

_DATASET_DIR = os.path.join(_HERE, "dataset")
_WEIGHTS_DIR = os.path.join(_HERE, "weights")
_RUNS_DIR = os.path.join(_HERE, "runs")


def _best_val_loss(run_dir):
    """Read `results.csv` from a completed run and return the lowest total val loss.

    Total val loss = sum of every `val/*_loss` column (box/seg/cls/dfl[/sem]), taken
    from whichever epoch minimises that sum. Returns `None` if `results.csv` is
    missing/unreadable (e.g. unexpected Ultralytics version) — callers must handle
    that by omitting the metric from the filename rather than guessing a value.
    """
    csv_path = os.path.join(run_dir, "results.csv")
    if not os.path.exists(csv_path):
        return None
    try:
        with open(csv_path, "r", newline="", encoding="utf-8") as f:
            rows = list(csv.DictReader(f))
        loss_cols = [c for c in (rows[0].keys() if rows else []) if c.startswith("val/") and c.endswith("_loss")]
        if not rows or not loss_cols:
            return None
        best = min(sum(float(row[c]) for c in loss_cols) for row in rows)
        return best
    except (OSError, ValueError, KeyError, IndexError):
        return None


def _metrics_history(run_dir):
    """Full per-epoch metrics history from a run's `results.csv`, as a list of
    `{column: float}` dicts — embedded in the weights sidecar JSON so the UI's "Show
    training data" button works even after `YOLO/runs/` is deleted. Returns `[]` if
    `results.csv` is missing/unreadable.
    """
    csv_path = os.path.join(run_dir, "results.csv")
    if not os.path.exists(csv_path):
        return []
    try:
        with open(csv_path, "r", newline="", encoding="utf-8") as f:
            rows = list(csv.DictReader(f))
    except OSError:
        return []
    history = []
    for row in rows:
        parsed = {}
        for k, v in row.items():
            try:
                parsed[k.strip()] = float(v)
            except (TypeError, ValueError):
                continue
        if parsed:
            history.append(parsed)
    return history


def _weights_filename(stamp_dt, val_loss, test_dice=None):
    """`best_<YYYYMMDD-HHMMSS>[_valloss<value>][_testdice<value>].pt` — unique per run,
    sortable by name. `stamp_dt` is reused for the sidecar JSON's `timestamp` field so
    both stay perfectly in sync.
    """
    stamp = stamp_dt.strftime("%Y%m%d-%H%M%S")
    name = f"best_{stamp}"
    if val_loss is not None:
        name += f"_valloss{f'{val_loss:.4f}'.replace('.', 'p')}"
    if test_dice is not None:
        name += f"_testdice{f'{test_dice:.4f}'.replace('.', 'p')}"
    return name + ".pt"


def _parse_args():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--smoke-test", action="store_true",
                     help="Quick end-to-end run: few patients, 2 epochs, imgsz=256.")
    ap.add_argument("--epochs", type=int, default=None, help="Override PARAMS['epochs'].")
    ap.add_argument("--imgsz", type=int, default=None, help="Override PARAMS['imgsz'].")
    ap.add_argument("--batch", type=int, default=None, help="Override PARAMS['batch'].")
    ap.add_argument("--max-patients", type=int, default=None,
                     help="Cap patients per split (smoke-test-only knob).")
    ap.add_argument("--data-fraction", type=float, default=None,
                     help="Proportionally subsample each split (e.g. 0.5 = half the data, "
                          "same train/val ratio preserved). For full-but-smaller training runs.")
    ap.add_argument("--rebuild-dataset", action="store_true",
                     help="Rebuild YOLO/dataset/ even if data.yaml already exists.")
    return ap.parse_args()


def main():
    args = _parse_args()

    overrides = {}
    max_patients = args.max_patients
    if args.smoke_test:
        overrides.update({"epochs": 2, "imgsz": 256, "batch": 2})
        if max_patients is None:
            max_patients = 3
    if args.epochs is not None:
        overrides["epochs"] = args.epochs
    if args.imgsz is not None:
        overrides["imgsz"] = args.imgsz
    if args.batch is not None:
        overrides["batch"] = args.batch

    p = dict(PARAMS)
    p.update(overrides)

    data_yaml = os.path.join(_DATASET_DIR, "data.yaml")
    if args.rebuild_dataset or not os.path.exists(data_yaml):
        print(f"[train] Building dataset (max_patients={max_patients}, "
              f"fraction={args.data_fraction}) -> {_DATASET_DIR} ...")
        build_dataset(params=p, max_patients=max_patients, fraction=args.data_fraction)

    from ultralytics import YOLO

    # Shared timestamp for the run folder name (`YOLO/runs/<stamp>/`), the saved
    # weights filename, and the sidecar JSON's `timestamp` field, so all three always
    # correlate to the exact same training run instead of Ultralytics' default
    # `train`/`train2`/`train3`... run-folder naming.
    stamp_dt = datetime.now()
    stamp = stamp_dt.strftime("%Y%m%d-%H%M%S")

    model = YOLO("yolo11n-seg.pt")
    model.train(
        data=data_yaml,
        epochs=p["epochs"],
        imgsz=p["imgsz"],
        batch=p["batch"],
        project=_RUNS_DIR,
        name=stamp,
    )

    run_dir = model.trainer.save_dir
    best_path = os.path.join(run_dir, "weights", "best.pt")
    if not os.path.exists(best_path):
        raise FileNotFoundError(f"[train] Expected best weights at {best_path}, not found")

    val_loss = _best_val_loss(run_dir)

    print("[train] Evaluating Dice on the held-out test split...")
    eval_result = evaluate_test_split(
        weights_path=best_path, params=p, max_patients=max_patients, fraction=args.data_fraction,
    )
    test_dice = eval_result["overall_mean_dice"]

    # Same split call `build_dataset`/`evaluate_test_split` used (same params/max_patients/
    # fraction), just to report patient counts per split in the sidecar metadata.
    splits = split_patients(p, max_patients=max_patients, fraction=args.data_fraction)

    os.makedirs(_WEIGHTS_DIR, exist_ok=True)
    dest = os.path.join(_WEIGHTS_DIR, _weights_filename(stamp_dt, val_loss, test_dice))
    shutil.copy2(best_path, dest)

    metadata = {
        "timestamp": stamp_dt.isoformat(timespec="seconds"),
        "val_loss": val_loss,
        "test_dice": test_dice,
        "n_train_patients": len(splits["train"]),
        "n_val_patients": len(splits["val"]),
        "n_test_patients": eval_result["n_patients"],
        "per_patient_test_dice": eval_result["per_patient"],
        "epochs": p["epochs"],
        "imgsz": p["imgsz"],
        "batch": p["batch"],
        "max_patients": max_patients,
        "data_fraction": args.data_fraction,
        "run_dir": os.path.relpath(run_dir, _HERE),
        "metrics_history": _metrics_history(run_dir),
    }
    save_weights_metadata(dest, metadata)

    print(f"[train] Best weights saved to {dest}"
          + (f" (val_loss={val_loss:.4f})" if val_loss is not None else " (val_loss unknown)")
          + (f" (test_dice={test_dice:.4f})" if test_dice is not None else " (test_dice unknown)"))


if __name__ == "__main__":
    main()
