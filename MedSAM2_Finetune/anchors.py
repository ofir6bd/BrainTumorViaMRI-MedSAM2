"""Where a pass starts, and what YOLO itself would have answered.

Every slice is prompted with YOLO's own map, so the start slice (the **anchor**) is not the
only place the model hears from YOLO — it is only the one slice segmented *without* memory,
the one whose memory every other slice keeps attending to. The two passes run away from it,
up and down the stack.

The anchor is the slice with the most confident tumour area (area x score) — the automatic
stand-in for "the slice a radiologist would start on". Nothing in this file ever looks at
the expert mask.

The old HITL loop (extra anchors placed where the answer disagreed with YOLO) is gone: with
a hint on every slice there is nothing left for it to add, and each round cost a full
propagation.
"""
import numpy as np


def yolo_mask(entry, cfg):
    """What YOLO itself would publish: blobs at `score_min`, probability cut at 0.5."""
    return entry["prob_f"] > 127


def strength(entry, cfg):
    """How good a starting slice each slice would make: big and confident."""
    area = (entry["prob"] > 127).reshape(len(entry["z"]), -1).sum(axis=1).astype(np.float32)
    return area * entry["scores"]


def anchor(entry, cfg):
    """The slice both passes start from. With no YOLO tumour anywhere, the middle of the
    brain — the passes still run, and every slice still gets its (empty) hint."""
    s = strength(entry, cfg)
    return int(np.argmax(s)) if s.any() else len(entry["z"]) // 2
