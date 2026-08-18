"""YOLO tumour segmentation package.

Builds a YOLO-seg dataset from BraTS volumes (Stages A-C), wraps Ultralytics
fine-tuning (Stage D, `train.py`), and exposes a per-patient inference pipeline plus
a Flask blueprint for the web UI (Stages E-F). See README.md for the full spec.
"""
from .pipeline import PARAMS, YoloPipeline, build_dataset, split_patients

__all__ = ["YoloPipeline", "PARAMS", "build_dataset", "split_patients"]
