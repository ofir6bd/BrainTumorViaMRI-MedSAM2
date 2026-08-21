"""One-off script: generate presentation assets (segmentation render + training
curves PNG) from the current best YOLO checkpoint. Not part of the pipeline;
safe to delete after use.
"""
import json
import os
import sys

_HERE = os.path.dirname(os.path.abspath(__file__))
if _HERE not in sys.path:
    sys.path.insert(0, _HERE)

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

import pipeline as P
import render as R

OUT_DIR = os.path.join(_HERE, "_slide_assets")
os.makedirs(OUT_DIR, exist_ok=True)

WEIGHTS = os.path.join(_HERE, "weights", "best_20260821-013417_valloss2p9192_testdice0p8662.pt")
META_PATH = os.path.join(_HERE, "weights", "best_20260821-013417_valloss2p9192_testdice0p8662.json")

with open(META_PATH, "r", encoding="utf-8") as f:
    meta = json.load(f)

overall = meta["test_dice"]
per_patient = meta["per_patient_test_dice"]
# pick the patient whose mean dice is closest to the overall mean -> representative case
best_match = min(per_patient, key=lambda r: abs(r["mean_dice"] - overall))
patient_id = best_match["patient_id"]
print(f"[assets] representative patient: {patient_id} (mean_dice={best_match['mean_dice']:.4f}, overall={overall:.4f})")

# locate patient dir among the two pools
patient_dir = None
for root in P.train_pool_dirs():
    cand = os.path.join(root, patient_id)
    if os.path.isdir(cand):
        patient_dir = cand
        break
if patient_dir is None:
    raise SystemExit(f"Could not locate directory for {patient_id}")

pl = P.YoloPipeline(patient_dir, weights_path=WEIGHTS)

# Pick the slice whose Dice is closest to this patient's mean Dice (a representative
# slice showing both correct overlap and typical error, rather than a lucky perfect
# match) among slices that actually contain some tumour.
target_mean = best_match["mean_dice"]
scored = []
for zi in pl.slice_indices:
    s = pl.process_slice(zi)
    if s["gt_wt"].any():
        scored.append((zi, s["dice"]))
z, d = min(scored, key=lambda t: abs(t[1] - target_mean))
slice_no = pl.slice_indices.index(z) + 1
print(f"[assets] representative slice index z={z} (slice #{slice_no}) of depth={pl.depth}, "
      f"dice={d:.4f} (target mean={target_mean:.4f})")

buf = R.render(pl, z, slice_no=slice_no)
seg_png_path = os.path.join(OUT_DIR, "segmentation_result.png")
with open(seg_png_path, "wb") as f:
    f.write(buf.read())
print(f"[assets] wrote {seg_png_path}")

s = pl.process_slice(z)
print(f"[assets] rendered slice dice = {s['dice']:.4f}")

# ---------------------------------------------------------------------------
# Training curves chart
# ---------------------------------------------------------------------------
history = meta["metrics_history"]
epochs = [row["epoch"] for row in history]
train_loss = [row.get("train/box_loss", 0) + row.get("train/seg_loss", 0)
              + row.get("train/cls_loss", 0) + row.get("train/dfl_loss", 0) for row in history]
val_loss = [row.get("val/box_loss", 0) + row.get("val/seg_loss", 0)
            + row.get("val/cls_loss", 0) + row.get("val/dfl_loss", 0) for row in history]
mask_map50 = [row.get("metrics/mAP50(M)", None) for row in history]

fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(11, 4.2))
ax1.plot(epochs, train_loss, label="train loss", color="#3b82f6", linewidth=2)
ax1.plot(epochs, val_loss, label="val loss", color="#ef4444", linewidth=2)
ax1.set_xlabel("epoch")
ax1.set_ylabel("total loss (box+seg+cls+dfl)")
ax1.set_title("Loss")
ax1.legend()
ax1.grid(alpha=0.3)

ax2.plot(epochs, mask_map50, color="#10b981", linewidth=2)
ax2.set_xlabel("epoch")
ax2.set_ylabel("mask mAP@50")
ax2.set_title("Segmentation mAP@50")
ax2.grid(alpha=0.3)

fig.suptitle(f"YOLO11n-seg training — {meta['epochs']} epochs, imgsz={meta['imgsz']}, batch={meta['batch']}")
fig.tight_layout(rect=(0, 0, 1, 0.94))
curves_path = os.path.join(OUT_DIR, "training_curves.png")
fig.savefig(curves_path, dpi=150)
print(f"[assets] wrote {curves_path}")

print("[assets] done.")
print(json.dumps({
    "patient_id": patient_id,
    "slice_no": slice_no,
    "rendered_dice": s["dice"],
    "overall_test_dice": overall,
    "n_test_patients": meta["n_test_patients"],
    "epochs": meta["epochs"],
    "imgsz": meta["imgsz"],
    "batch": meta["batch"],
    "val_loss": meta["val_loss"],
}, indent=2))
