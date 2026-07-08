import io
import os
from functools import lru_cache

import yaml
import numpy as np
import nibabel as nib
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
from matplotlib.colors import ListedColormap
from flask import Flask, jsonify, send_file, abort, render_template, request

with open("config.yaml", "r") as _f:
    _CFG = yaml.safe_load(_f)
DATASET_DIR = _CFG["paths"]["dataset"]

BRATS_COLORS = ["none", "red", "green", "blue", "yellow"]
BRATS_CMAP = ListedColormap(BRATS_COLORS)

app = Flask(__name__)
app.config["TEMPLATES_AUTO_RELOAD"] = True
app.config["SEND_FILE_MAX_AGE_DEFAULT"] = 0


def find_patients():
    patients = []
    if not os.path.exists(DATASET_DIR):
        return patients
    for root, _, files in os.walk(DATASET_DIR):
        nii = [f for f in files if f.endswith((".nii", ".nii.gz"))]
        if len(nii) < 2:
            continue
        brain, seg = [], []
        for f in nii:
            (seg if "seg" in f.lower() else brain).append(os.path.join(root, f))
        if not (brain and seg):
            continue
        pid = os.path.basename(root)
        if pid == "training_data":
            continue
        for bf in brain:
            name = os.path.basename(bf).replace(".nii.gz", "").replace(".nii", "")
            patients.append({"patient_id": pid, "brain_file": bf,
                             "seg_file": seg[0], "brain_name": name})
    patients.sort(key=lambda p: (p["patient_id"], p["brain_name"]))
    return patients


PATIENTS = find_patients()


@lru_cache(maxsize=8)
def load_volume(path):
    return nib.load(path).get_fdata()


def get_patient(idx):
    if idx < 0 or idx >= len(PATIENTS):
        abort(404)
    return PATIENTS[idx]


def best_slice(seg):
    sums = [int(np.sum(seg[:, :, z] > 0)) for z in range(seg.shape[2])]
    return int(np.argmax(sums)) if max(sums) > 0 else seg.shape[2] // 2


def figure_to_png(fig):
    buf = io.BytesIO()
    fig.savefig(buf, format="png", bbox_inches="tight", facecolor="white")
    plt.close(fig)
    buf.seek(0)
    return buf


@app.route("/")
def index():
    return render_template("index.html")


@app.route("/api/patients")
def api_patients():
    return jsonify([
        {"id": i, "label": f'{p["patient_id"]} - {p["brain_name"]}'}
        for i, p in enumerate(PATIENTS)
    ])


@app.route("/api/patient/<int:idx>")
def api_patient(idx):
    p = get_patient(idx)
    seg = load_volume(p["seg_file"])
    brain = load_volume(p["brain_file"])
    return jsonify({
        "patient_id": p["patient_id"],
        "brain_name": p["brain_name"],
        "depth": int(seg.shape[2]),
        "best_slice": best_slice(seg),
        "shape": [int(x) for x in brain.shape],
    })


@app.route("/slice.png")
def slice_png():
    idx = int(request.args.get("id", -1))
    p = get_patient(idx)
    brain = load_volume(p["brain_file"])
    seg = load_volume(p["seg_file"])
    z = int(request.args.get("z", best_slice(seg)))
    z = max(0, min(z, seg.shape[2] - 1))

    b, s = brain[:, :, z], seg[:, :, z]
    fig, axes = plt.subplots(1, 4, figsize=(20, 5))
    fig.suptitle(f'{p["patient_id"]} - {p["brain_name"]} - slice {z}',
                 fontsize=14, fontweight="bold")
    axes[0].imshow(b, cmap="gray")
    axes[0].set_title("Brain")
    axes[1].imshow(s, cmap=BRATS_CMAP, vmin=0, vmax=4)
    axes[1].set_title("BraTS Labels (1:R 2:G 3:B 4:Y)")
    axes[2].imshow(b, cmap="gray")
    axes[2].imshow(s, cmap=BRATS_CMAP, alpha=0.5, vmin=0, vmax=4)
    axes[2].set_title("Overlay")
    axes[3].imshow(b, cmap="gray")
    axes[3].imshow(s > 0, cmap="Reds", alpha=0.4)
    axes[3].set_title("Binary Mask")
    for ax in axes:
        ax.axis("off")
    fig.tight_layout()
    return send_file(figure_to_png(fig), mimetype="image/png")


@app.route("/scatter.png")
def scatter_png():
    idx = int(request.args.get("id", -1))
    p = get_patient(idx)
    seg = load_volume(p["seg_file"])
    max_points = 3000

    fig = plt.figure(figsize=(9, 8))
    ax = fig.add_subplot(111, projection="3d")
    values = np.unique(seg)
    values = values[values > 0]
    for v in values:
        xs, ys, zs = np.where(seg == v)
        n = len(xs)
        if n == 0:
            continue
        if n > max_points:
            step = max(1, n // max_points)
            xs, ys, zs = xs[::step], ys[::step], zs[::step]
        color = BRATS_COLORS[int(v)] if int(v) < len(BRATS_COLORS) else "purple"
        ax.scatter(xs, ys, zs, c=color, s=2, alpha=0.7,
                   label=f"Label {int(v)} ({n} vox)")
    ax.set_xlabel("X")
    ax.set_ylabel("Y")
    ax.set_zlabel("Z")
    ax.set_title(f'{p["patient_id"]} - {p["brain_name"]}\n'
                 f"NETC=R  SNFH=G  ET=B  RC=Y")
    if values.size:
        ax.legend(loc="upper left", bbox_to_anchor=(1.02, 1))
    fig.tight_layout()
    return send_file(figure_to_png(fig), mimetype="image/png")
