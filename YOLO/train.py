"""Stage D — one-shot YOLO-seg fine-tuning wrapper (README.md §3 Stage D).

Run once, from the repo root, after activating .venv:

    .\\.venv\\Scripts\\Activate.ps1
    python YOLO\\train.py

Builds `YOLO/dataset/` first (Stages A-C) if it doesn't already exist, fine-tunes a
pretrained `yolo11n-seg.pt` on it, then saves the run's best weights to
`YOLO/weights/best_<timestamp>_valloss<value>.pt` — a uniquely named file per run, so a
new training run never silently overwrites a previously deployed checkpoint. The web UI
lets the user pick which saved weights file to use (see `YOLO/pipeline.py:list_weight_files`
and the "Model" dropdown in the YOLO tab).

Smoke test (quick, small subset — verify the pipeline end-to-end before a full run):

    python YOLO\\train.py --smoke-test

This rebuilds `YOLO/dataset/` from only a few patients per split, with 2 epochs and a
small image size, then trains and saves a new `weights/best_*.pt` as usual.

Full training on a fraction of the data (keeps the 80/20 train/val ratio, keeps the
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

from pipeline import PARAMS, build_dataset  # noqa: E402

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


def _weights_filename(val_loss):
    """`best_<YYYYMMDD-HHMMSS>[_valloss<value>].pt` — unique per run, sortable by name."""
    stamp = datetime.now().strftime("%Y%m%d-%H%M%S")
    if val_loss is None:
        return f"best_{stamp}.pt"
    tag = f"{val_loss:.4f}".replace(".", "p")
    return f"best_{stamp}_valloss{tag}.pt"


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

    model = YOLO("yolo11n-seg.pt")
    model.train(
        data=data_yaml,
        epochs=p["epochs"],
        imgsz=p["imgsz"],
        batch=p["batch"],
        project=_RUNS_DIR,
    )

    run_dir = model.trainer.save_dir
    best_path = os.path.join(run_dir, "weights", "best.pt")
    if not os.path.exists(best_path):
        raise FileNotFoundError(f"[train] Expected best weights at {best_path}, not found")

    val_loss = _best_val_loss(run_dir)
    os.makedirs(_WEIGHTS_DIR, exist_ok=True)
    dest = os.path.join(_WEIGHTS_DIR, _weights_filename(val_loss))
    shutil.copy2(best_path, dest)
    print(f"[train] Best weights saved to {dest}"
          + (f" (val_loss={val_loss:.4f})" if val_loss is not None else " (val_loss unknown)"))


if __name__ == "__main__":
    main()
