"""YOLO's hint (for MedSAM2) against YOLO's own mask (its final answer), on one slice.

Run from the repo root (CPU is enough):
    .venv\\Scripts\\python.exe Presentation\\make_hint_vs_mask.py
Hint = cached `prob` (every shape >= 0.05, soft, max per pixel); own mask = cached `prob_f` > 0.5
(only shapes >= 0.25) — the same two maps the MedSAM2 page draws. Writes images/hvm_*.png.
"""
import os

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
from PIL import Image

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
OUT = os.path.join(ROOT, "Presentation", "images")
PATIENT, Z = "BraTS-GLI-00046-101", 120
YOLO_RUN, PROMPT_KEY = "20260929-074550", "980735f3de"
BG, TEXT, YOLO_C, EXTRA, GT_C = "#0f1220", "#e8eaf2", "#3987e5", "#f0a030", "#2fb35c"


def main():
    e = np.load(os.path.join(ROOT, "MedSAM2_Finetune", "cache", PROMPT_KEY, f"{PATIENT}.npz"))
    k = list(e["z"]).index(Z)
    rgb, prob, prob_f, gt = e["rgb"][k], e["prob"][k] / 255.0, e["prob_f"][k] / 255.0, e["gt"][k] > 0
    own = prob_f > 0.5
    extra = (prob > 0) & ~own

    from ultralytics import YOLO
    model = YOLO(os.path.join(ROOT, "YOLO_finetune", "runs", YOLO_RUN, "train", "weights", "best.pt"))
    r = model.predict(Image.fromarray(rgb), conf=0.05, imgsz=512, retina_masks=True, verbose=False, device="cpu")[0]
    boxes = [(float(c), b) for c, b in zip(r.boxes.conf.cpu().numpy(), r.boxes.xyxy.cpu().numpy())]

    rot = lambda a: np.rot90(a)
    W = rgb.shape[1]
    flair = rot(rgb[:, :, 2])
    ys, xs = np.nonzero(rot((prob > 0) | gt))
    pad = 18
    y0, y1, x0, x1 = max(ys.min() - pad, 0), min(ys.max() + pad, flair.shape[0]), max(xs.min() - pad, 0), min(xs.max() + pad, flair.shape[1])

    def frame(name, draw):
        fig, ax = plt.subplots(figsize=(3.4, 3.4 * (y1 - y0) / (x1 - x0)), dpi=220)
        ax.imshow(flair, cmap="gray")
        draw(ax)
        ax.set_xlim(x0, x1); ax.set_ylim(y1, y0); ax.axis("off")
        fig.subplots_adjust(0, 0, 1, 1)
        fig.savefig(os.path.join(OUT, f"hvm_{name}.png"), facecolor="black")
        plt.close(fig)

    def overlay(ax, m, color, alpha):
        rgba = np.zeros(m.shape + (4,)); rgba[..., :3] = matplotlib.colors.to_rgb(color); rgba[..., 3] = m * alpha
        ax.imshow(rot(rgba), interpolation="nearest")

    def hint(ax):
        a = np.ma.masked_where(rot(prob) == 0, rot(prob))
        ax.imshow(a, cmap="magma", vmin=0, vmax=1, interpolation="nearest")
        for c, (bx1, by1, bx2, by2) in sorted(boxes, key=lambda t: t[0]):
            strong = c >= 0.25
            ax.add_patch(plt.Rectangle((by1, W - bx2), by2 - by1, bx2 - bx1, fill=False, lw=1.1,
                                       ec="white" if strong else "#9aa0b8", ls="-" if strong else "--"))
            ax.text(by1 + 1, W - bx2 + 1, f"{c:.2f}", fontsize=6.5, va="top", color="white",
                    bbox=dict(fc=YOLO_C if strong else "#555b73", ec="none", pad=0.6))

    def own_mask(ax):
        overlay(ax, own, YOLO_C, 0.75)
        ax.contour(rot(gt.astype(float)), levels=[0.5], colors=[GT_C], linewidths=1.2)

    def only_hint(ax):
        overlay(ax, own, YOLO_C, 0.25)
        overlay(ax, extra, EXTRA, 0.9)
        ax.contour(rot(gt.astype(float)), levels=[0.5], colors=[GT_C], linewidths=1.2)

    frame("hint", hint)
    frame("own", own_mask)
    frame("extra", only_hint)
    strong = sum(c >= 0.25 for c, _ in boxes)
    print(f"z{Z}: {len(boxes)} shapes ({strong} >= 0.25), confidences {sorted(round(c, 2) for c, _ in boxes)}; "
          f"hint px {int((prob > 0).sum())}, own mask px {int(own.sum())}, only in hint {int(extra.sum())}, "
          f"of which expert tumour {int((extra & gt).sum())}; expert px {int(gt.sum())}")


if __name__ == "__main__":
    main()
