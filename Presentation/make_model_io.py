"""Input/output pictures for the two model slides (YOLO and MedSAM2), one real slice.

Run from the repo root (CPU is enough):
    .venv\\Scripts\\python.exe Presentation\\make_model_io.py
Uses the trained YOLO run and the saved MedSAM2 hint cache; MedSAM2's answers come from the
running web app (http://localhost:5000). Writes Presentation/images/io_*.png.
"""
import io
import os
import urllib.request

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
from PIL import Image

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
OUT = os.path.join(ROOT, "Presentation", "images")
PATIENT, Z = "BraTS-GLI-00008-101", 76
YOLO_RUN, MS_RUN, PROMPT_KEY = "20260929-074550", "20260930-234255", "980735f3de"
YOLO_C, MS_C, GT_C = "#3987e5", "#e06a2e", "#8fe08f"


def save(img, name):
    Image.fromarray(img).save(os.path.join(OUT, name))


def show(ax, img):
    ax.imshow(img, interpolation="nearest")
    ax.axis("off")


def main():
    entry = np.load(os.path.join(ROOT, "MedSAM2_Finetune", "cache", PROMPT_KEY, f"{PATIENT}.npz"))
    zs = list(entry["z"]); k = zs.index(Z)
    rgb = entry["rgb"][k]                       # exactly the picture YOLO and MedSAM2 see
    rot = lambda a: np.ascontiguousarray(np.rot90(a))   # the app's orientation

    # --- YOLO input
    save(rot(rgb), "io_yolo_input.png")

    # --- YOLO output: boxes, confidences and masks on this slice
    from ultralytics import YOLO
    model = YOLO(os.path.join(ROOT, "YOLO_finetune", "runs", YOLO_RUN, "train", "weights", "best.pt"))
    r = model.predict(Image.fromarray(rgb), conf=0.05, imgsz=512, retina_masks=True, verbose=False, device="cpu")[0]
    flair = np.repeat(rgb[:, :, 2:3], 3, axis=2)
    fig, ax = plt.subplots(figsize=(4, 4 * rgb.shape[1] / rgb.shape[0]), dpi=220)   # rotated: rows = old width
    show(ax, rot(flair))
    h = rgb.shape[0]
    order = np.argsort(-r.boxes.conf.cpu().numpy())
    for n, i in enumerate(order):
        c = float(r.boxes.conf[i])
        m = r.masks.data[i].cpu().numpy() > 0.5
        rgba = np.zeros(m.shape + (4,)); rgba[..., :3] = matplotlib.colors.to_rgb(YOLO_C); rgba[..., 3] = m * 0.55
        ax.imshow(rot(rgba), interpolation="nearest")
        x1, y1, x2, y2 = r.boxes.xyxy[i].cpu().numpy()        # image coords (x = column, y = row)
        # after np.rot90 (counter-clockwise): new_row = (W-1) - col, new_col = row
        W = rgb.shape[1]
        ax.add_patch(plt.Rectangle((y1, W - x2), y2 - y1, x2 - x1, fill=False, ec="white", lw=1.2))
        # most confident label above its box, the others below theirs, so labels never stack
        ty, va = (W - x2 - 2, "bottom") if n == 0 else (W - x1 + 2, "top")
        ax.text(y1, ty, f"tumour {c:.2f}", color="white", fontsize=8, fontweight="bold", va=va,
                bbox=dict(fc=YOLO_C, ec="none", pad=1.2))
    fig.subplots_adjust(0, 0, 1, 1)
    fig.savefig(os.path.join(OUT, "io_yolo_output.png"), facecolor="black")
    plt.close(fig)
    print("YOLO shapes on this slice:", [round(float(r.boxes.conf[i]), 3) for i in order])

    # --- MedSAM2 inputs/outputs from the web app (same renderer as the MedSAM2 page)
    def panel(z, which):
        url = (f"http://localhost:5000/medsam2/api/runs/{MS_RUN}/slice.png?patient={PATIENT}"
               f"&which=best.pt&z={z}&panels={which}")
        return np.array(Image.open(io.BytesIO(urllib.request.urlopen(url, timeout=120).read())).convert("RGB"))
    save(panel(Z, "prompt"), "io_ms_hint.png")
    save(panel(Z, "medsam2"), "io_ms_output.png")
    for j, dz in enumerate((3, 2, 1)):                          # memory: the slices it just did
        save(panel(Z - dz, "medsam2"), f"io_ms_mem{j}.png")
    print("written io_*.png")


if __name__ == "__main__":
    main()
