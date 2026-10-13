"""The mirror check (flip TTA) on one test patient, for the 'mirror check' slide.

Run from the repo root (uses the GPU if free, else CPU — slower):
    .venv\\Scripts\\python.exe Presentation\\make_mirror.py
Runs the best MedSAM2 run's own infer(), which returns both views (normal and mirrored,
already flipped back), then scores each view alone and their average against the expert.
Writes Presentation/images/mirror_*.png and prints the numbers used on the slide.
"""
import os
import sys

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import torch
import yaml

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, ROOT)
from MedSAM2_Finetune.evaluate import infer, postprocess  # noqa: E402
from MedSAM2_Finetune.model import build_train_model  # noqa: E402
from MedSAM2_Finetune.yolo_prompts import load_prompt  # noqa: E402

OUT = os.path.join(ROOT, "Presentation", "images")
PATIENT, MS_RUN = "BraTS-GLI-00008-101", "20260930-234255"
BG, TEXT, MUTED = "#0f1220", "#e8eaf2", "#9aa0b8"
TP, FN, FP = (0.07, 0.64, 0.07), (0.98, 0.70, 0.10), (0.82, 0.23, 0.23)


def dice(a, b):
    s = a.sum() + b.sum()
    return 1.0 if s == 0 else 2 * np.logical_and(a, b).sum() / s


def main():
    run = os.path.join(ROOT, "MedSAM2_Finetune", "runs", MS_RUN)
    cfg = yaml.safe_load(open(os.path.join(run, "run_config.yaml")))
    cfg["evaluate"]["tta_flip"] = True
    device = "cuda" if torch.cuda.is_available() else "cpu"
    model, _ = build_train_model(cfg, device, os.path.join(run, "weights", "best.pt"))
    model.eval()
    entry = load_prompt(cfg, PATIENT)
    out = infer(model, cfg, entry, device)
    t = float(cfg["evaluate"]["mask_threshold"])
    gt = np.asarray(entry["gt"]) > 0
    normal, mirror = out["views"][0], out["views"][1]
    masks = {"normal": postprocess(normal > t, cfg), "mirror": postprocess(mirror > t, cfg),
             "average": postprocess(out["logits"] > t, cfg)}
    scores = {k: dice(m, gt) for k, m in masks.items()}
    # the slice where the two views disagree most (pixels on one side of the cut-off only)
    dis = ((normal > t) != (mirror > t)).reshape(len(gt), -1).sum(1)
    k = int(np.argmax(dis)); z = int(entry["z"][k])
    print(f"device {device}; 3D Dice normal {scores['normal']:.4f}, mirror {scores['mirror']:.4f}, "
          f"average {scores['average']:.4f}; disagreeing px whole patient {int(dis.sum())}; slice z{z}: {int(dis[k])} px")

    rot = np.rot90
    sig = lambda L: 1 / (1 + np.exp(-np.clip(L, -40, 40)))
    flair = rot(entry["rgb"][k][:, :, 2])
    region = rot((normal[k] > t) | (mirror[k] > t) | gt[k])
    ys, xs = np.nonzero(region); pad = 20
    y0, y1 = max(ys.min() - pad, 0), min(ys.max() + pad, flair.shape[0])
    x0, x1 = max(xs.min() - pad, 0), min(xs.max() + pad, flair.shape[1])

    def save(name, draw, mirrored_view=False):
        fig, ax = plt.subplots(figsize=(3.0, 3.0 * (y1 - y0) / (x1 - x0)), dpi=220)
        img = flair[:, ::-1] if mirrored_view else flair
        ax.imshow(img, cmap="gray")
        draw(ax)
        xa, xb = ((flair.shape[1] - x1, flair.shape[1] - x0) if mirrored_view else (x0, x1))
        ax.set_xlim(xa, xb); ax.set_ylim(y1, y0); ax.axis("off")
        fig.subplots_adjust(0, 0, 1, 1)
        fig.savefig(os.path.join(OUT, f"mirror_{name}.png"), facecolor="black")
        plt.close(fig)

    def prob(L, flip=False):
        def f(ax):
            p = rot(sig(L))
            p = p[:, ::-1] if flip else p
            ax.imshow(np.ma.masked_where(p < 0.05, p), cmap="magma", vmin=0, vmax=1, interpolation="nearest")
        return f

    def result(ax):
        m = rot(masks["average"][k]); g = rot(gt[k])
        rgba = np.zeros(m.shape + (4,))
        for sel, c in ((m & g, TP), (g & ~m, FN), (m & ~g, FP)):
            rgba[sel, :3] = c; rgba[sel, 3] = 0.8
        ax.imshow(rgba, interpolation="nearest")

    def disagree(ax):
        d = rot((normal[k] > t) != (mirror[k] > t))
        rgba = np.zeros(d.shape + (4,)); rgba[d, :3] = (1.0, 0.85, 0.2); rgba[d, 3] = 0.95
        ax.imshow(np.ma.masked_where(rot(sig(out["logits"][k])) < 0.05, rot(sig(out["logits"][k]))),
                  cmap="magma", vmin=0, vmax=1, alpha=0.6, interpolation="nearest")
        ax.imshow(rgba, interpolation="nearest")

    # the two inputs, whole slice, with YOLO's hint in red: the normal patient and its mirror image
    hint = rot(np.asarray(entry["prob"][k]) / 255.0)
    for name, flip in (("in_normal", False), ("in_mirrored", True)):
        img, h = (flair[:, ::-1], hint[:, ::-1]) if flip else (flair, hint)
        fig, ax = plt.subplots(figsize=(2.6, 2.6 * img.shape[0] / img.shape[1]), dpi=220)
        ax.imshow(img, cmap="gray")
        rgba = np.zeros(h.shape + (4,)); rgba[..., 0] = 0.95; rgba[..., 1] = 0.2; rgba[..., 3] = np.clip(h, 0, 1) * 0.85
        ax.imshow(rgba, interpolation="nearest"); ax.axis("off")
        fig.subplots_adjust(0, 0, 1, 1)
        fig.savefig(os.path.join(OUT, f"mirror_{name}.png"), facecolor="black")
        plt.close(fig)
    save("normal", prob(normal[k]))
    save("mirror", prob(mirror[k]))                 # already flipped back by infer()
    save("disagree", disagree)
    save("result", result)
    yaml.safe_dump({"patient": PATIENT, "z": z, "dice": {k_: round(float(v), 4) for k_, v in scores.items()},
                    "disagree_px_slice": int(dis[k]), "disagree_px_patient": int(dis.sum())},
                   open(os.path.join(OUT, "mirror_numbers.yaml"), "w"))


if __name__ == "__main__":
    main()
