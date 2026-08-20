"""Standalone Dice quality summary on the held-out test split (README.md addendum).

Run any time after weights exist in `YOLO/weights/` (i.e. after `YOLO/train.py`), from
the repo root, after activating .venv:

    .\\.venv\\Scripts\\Activate.ps1
    python YOLO\\evaluate.py                        # best available checkpoint
    python YOLO\\evaluate.py --weights best_....pt   # a specific checkpoint

Runs inference with the chosen checkpoint over every patient in the labeled, held-out
test split (`pipeline.list_test_patients` / `pipeline.split_patients`) and prints
per-patient + overall mean Dice — the same computation `YOLO/train.py` runs
automatically right after training to tag the saved `.pt` filename and sidecar
`.json` (see `pipeline.evaluate_test_split`).
"""
import argparse
import os
import sys

_HERE = os.path.dirname(os.path.abspath(__file__))
if _HERE not in sys.path:
    sys.path.insert(0, _HERE)

from pipeline import evaluate_test_split, list_weight_files  # noqa: E402

_WEIGHTS_DIR = os.path.join(_HERE, "weights")


def _parse_args():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--weights", default=None,
                     help="Weights filename in YOLO/weights/ (default: best available).")
    return ap.parse_args()


def main():
    args = _parse_args()

    weights_path = None
    if args.weights:
        weights_path = next(
            (e["path"] for e in list_weight_files() if e["filename"] == args.weights),
            os.path.join(_WEIGHTS_DIR, args.weights),
        )

    result = evaluate_test_split(weights_path=weights_path)

    print(f"\n[evaluate] weights: {result['weights_path']}")
    overall = result["overall_mean_dice"]
    print(f"[evaluate] {result['n_patients']} test patients, overall mean Dice = "
          + (f"{overall:.4f}" if overall is not None else "n/a"))


if __name__ == "__main__":
    main()
