"""One patient's tumour in each MRI sequence (zoomed), for slide 3.

Run from the repo root:
    .venv\\Scripts\\python.exe Presentation\\make_modalities.py
Writes Presentation/images/mod_<name>.png. Same orientation as the web app (np.rot90: the
patient's right on the viewer's left) and the app's label colours.
"""
import glob
import os

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import nibabel as nib
import numpy as np

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
OUT = os.path.join(ROOT, "Presentation", "images")
PATIENT, Z = "BraTS-GLI-00046-101", 115          # slice 115 holds all four labelled parts
LABELS = {1: ("NETC", "#e0534c"), 2: ("SNFH", "#2fb35c"), 3: ("ET", "#9b6bff"), 4: ("RC", "#e8b400")}
# which part each sequence shows best (outlined on that panel)
SEQ = [("t1n", "T1", [3]), ("t1c", "T1C", [3, 1]), ("t2w", "T2", [2]), ("t2f", "FLAIR", [2])]


def load(folder, key):
    return np.asarray(nib.load(glob.glob(os.path.join(folder, f"*{key}.nii*"))[0]).dataobj, dtype=np.float32)


def window(img, brain):
    lo, hi = np.percentile(img[brain], [1, 99.5])
    return np.clip((img - lo) / max(hi - lo, 1e-6), 0, 1)


def main():
    folder = glob.glob(os.path.join(ROOT, "data", "dataset", "*", PATIENT))[0]
    seg = np.rint(load(folder, "seg")[:, :, Z]).astype(int)
    imgs = {k: load(folder, k)[:, :, Z] for k, _, _ in SEQ}
    brain = imgs["t2f"] > 0
    rot = np.rot90
    seg_r = rot(seg)
    ys, xs = np.nonzero(seg_r)
    pad = 22
    y0, y1 = max(0, ys.min() - pad), min(seg_r.shape[0], ys.max() + pad)
    x0, x1 = max(0, xs.min() - pad), min(seg_r.shape[1], xs.max() + pad)
    crop = lambda a: a[y0:y1, x0:x1]

    def panel(name, gray, parts=(), fill=False):
        fig, ax = plt.subplots(figsize=(3.2, 3.2 * (y1 - y0) / (x1 - x0)), dpi=200)
        ax.imshow(crop(gray), cmap="gray", vmin=0, vmax=1, interpolation="nearest")
        s = crop(seg_r)
        for k in parts:
            m = (s == k).astype(float)
            if fill:
                rgba = np.zeros(m.shape + (4,))
                rgba[..., :3] = matplotlib.colors.to_rgb(LABELS[k][1]); rgba[..., 3] = m * 0.6
                ax.imshow(rgba, interpolation="nearest")
            else:
                ax.contour(m, levels=[0.5], colors=[LABELS[k][1]], linewidths=1.6)
        ax.axis("off")
        fig.subplots_adjust(0, 0, 1, 1)
        fig.savefig(os.path.join(OUT, f"mod_{name}.png"), facecolor="black")
        plt.close(fig)

    for key, name, parts in SEQ:
        panel(name.lower(), rot(window(imgs[key], brain)), parts)
    panel("labels", rot(window(imgs["t2f"], brain)), [2, 4, 1, 3], fill=True)
    print("written", [f"mod_{n.lower()}.png" for _, n, _ in SEQ] + ["mod_labels.png"],
          {LABELS[k][0]: int((seg == k).sum()) for k in LABELS})


if __name__ == "__main__":
    main()
