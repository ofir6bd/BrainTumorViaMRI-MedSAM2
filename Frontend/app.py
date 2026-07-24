import io
import os
import sys
from functools import lru_cache

# Repo root on sys.path so the Assymetry package (at repo root) is importable.
_REPO_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if _REPO_ROOT not in sys.path:
    sys.path.insert(0, _REPO_ROOT)

import yaml
import numpy as np
import nibabel as nib
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
from mpl_toolkits.mplot3d import Axes3D
from matplotlib.colors import ListedColormap
from flask import Flask, jsonify, send_file, abort, render_template, request

with open("config.yaml", "r") as _f:
    _CFG = yaml.safe_load(_f)
DATASET_DIR = _CFG["paths"]["dataset"]

LABEL_NAMES = {1: "NETC", 2: "SNFH", 3: "ET", 4: "RC"}
BRATS_COLORS = ["none", "red", "green", "blue", "yellow"]
BRATS_CMAP = ListedColormap(BRATS_COLORS)
LABEL_OVERLAY = {
    1: (1.00, 0.27, 0.00, 0.50),
    2: (0.00, 1.00, 0.50, 0.50),
    3: (0.00, 0.80, 1.00, 0.50),
    4: (1.00, 0.90, 0.00, 0.50),
}
LABEL_BBOX = {1: "#ff4500", 2: "#00ff80", 3: "#00ccff", 4: "#ffe600"}
MODALITY_ORDER = ["T1", "T1C", "T2", "FLAIR"]
BACKGROUND_ORDER = ["T1C", "T1", "T2", "FLAIR"]

app = Flask(__name__)
app.config["TEMPLATES_AUTO_RELOAD"] = True
app.config["SEND_FILE_MAX_AGE_DEFAULT"] = 0

# Asymmetry pipeline routes (served under /asym).
from Assymetry.routes import asym_bp  # noqa: E402
app.register_blueprint(asym_bp)


def find_patients():
    patients = []
    if not os.path.exists(DATASET_DIR):
        return patients
    for root, _, files in os.walk(DATASET_DIR):
        nii = [f for f in files if f.endswith((".nii", ".nii.gz"))]
        seg = [f for f in nii if "seg" in f.lower()]
        if not nii or not seg:
            continue
        pid = os.path.basename(root)
        if pid == "training_data":
            continue
        modalities = {}
        for f in nii:
            if "seg" in f.lower():
                continue
            name = f.lower()
            if "-t1c" in name:
                modalities["T1C"] = os.path.join(root, f)
            if "-t1n" in name:
                modalities["T1"] = os.path.join(root, f)
            if "-t2w" in name:
                modalities["T2"] = os.path.join(root, f)
            if "-t2f" in name:
                modalities["FLAIR"] = os.path.join(root, f)
        patients.append({
            "patient_id": pid,
            "seg_file": os.path.join(root, seg[0]),
            "modalities": modalities,
        })
    patients.sort(key=lambda p: p["patient_id"])
    return patients


PATIENTS = find_patients()


@lru_cache(maxsize=12)
def load_volume(path):
    return nib.load(path).get_fdata()


def get_patient(idx):
    if idx < 0 or idx >= len(PATIENTS):
        abort(404)
    return PATIENTS[idx]


def best_slice(seg):
    sums = [int(np.sum(seg[:, :, z] > 0)) for z in range(seg.shape[2])]
    return int(np.argmax(sums)) if max(sums) > 0 else seg.shape[2] // 2


def pick_background(modalities):
    for key in BACKGROUND_ORDER:
        if key in modalities:
            return load_volume(modalities[key]), key
    return None, None


def norm_bg(vol, z):
    sl = vol[:, :, z].astype(np.float32)
    nz = sl[sl != 0]
    lo = float(np.percentile(nz, 1)) if nz.size else 0.0
    hi = float(np.percentile(nz, 99)) if nz.size else 1.0
    return np.clip((sl - lo) / max(hi - lo, 1e-8), 0, 1)


def peak_slice_per_label(seg):
    result = {}
    for label_id in LABEL_NAMES:
        mask = (seg == label_id)
        if not np.any(mask):
            continue
        counts = np.array([mask[:, :, z].sum() for z in range(seg.shape[2])])
        peak_z = int(np.argmax(counts))
        result[label_id] = (peak_z, int(counts[peak_z]))
    return result


def compute_bbox(mask_2d):
    rows = np.any(mask_2d, axis=1)
    cols = np.any(mask_2d, axis=0)
    if not rows.any():
        return None
    r_min = int(np.argmax(rows))
    r_max = int(len(rows) - 1 - np.argmax(rows[::-1]))
    c_min = int(np.argmax(cols))
    c_max = int(len(cols) - 1 - np.argmax(cols[::-1]))
    return r_min, r_max, c_min, c_max


def figure_to_png(fig):
    buf = io.BytesIO()
    fig.savefig(buf, format="png", bbox_inches="tight",
                facecolor=fig.get_facecolor())
    plt.close(fig)
    buf.seek(0)
    return buf


def label_legend_handles(present):
    return [
        plt.Line2D([0], [0], marker="o", color="w", markersize=8,
                   markerfacecolor=BRATS_COLORS[i],
                   label=f"{i}: {LABEL_NAMES[i]}")
        for i in present
    ]


@app.route("/")
def index():
    return render_template("index.html")


@app.route("/api/patients")
def api_patients():
    return jsonify([
        {"id": i, "label": p["patient_id"]}
        for i, p in enumerate(PATIENTS)
    ])


@app.route("/api/patient/<int:idx>")
def api_patient(idx):
    p = get_patient(idx)
    seg = load_volume(p["seg_file"])
    return jsonify({
        "patient_id": p["patient_id"],
        "modalities": [m for m in MODALITY_ORDER if m in p["modalities"]],
        "depth": int(seg.shape[2]),
        "best_slice": best_slice(seg),
    })


@app.route("/panels.png")
def panels_png():
    idx = int(request.args.get("id", -1))
    p = get_patient(idx)
    seg = load_volume(p["seg_file"])
    z = max(0, min(int(request.args.get("z", best_slice(seg))), seg.shape[2] - 1))
    bg_vol, bg_name = pick_background(p["modalities"])
    if bg_vol is None:
        abort(404)

    b, s = bg_vol[:, :, z], seg[:, :, z]
    fig, axes = plt.subplots(1, 4, figsize=(20, 5))
    fig.suptitle(f'{p["patient_id"]}  -  slice {z}  (background: {bg_name})',
                 fontsize=14, fontweight="bold")
    axes[0].imshow(b, cmap="gray")
    axes[0].set_title(bg_name)
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


@app.route("/modalities.png")
def modalities_png():
    idx = int(request.args.get("id", -1))
    p = get_patient(idx)
    seg = load_volume(p["seg_file"])
    z = max(0, min(int(request.args.get("z", best_slice(seg))), seg.shape[2] - 1))

    fig, axes = plt.subplots(2, 3, figsize=(15, 11))
    fig.suptitle(f'{p["patient_id"]}  -  slice z={z}', fontsize=16)
    flat = axes.flatten()
    display = ["T1", "T1C", "T2", "FLAIR", "SEG"]
    for i, item in enumerate(display):
        ax = flat[i]
        if item == "SEG":
            sl = seg[:, :, z]
            ax.imshow(sl, cmap=BRATS_CMAP, vmin=0, vmax=4)
            ax.set_title("Segmentation (BraTS labels)")
            present = [lab for lab in LABEL_NAMES if lab in np.unique(sl)]
            if present:
                ax.legend(handles=label_legend_handles(present),
                          loc="lower left", fontsize="x-small", frameon=True)
        elif item in p["modalities"]:
            vol = load_volume(p["modalities"][item])
            ax.imshow(vol[:, :, z], cmap="gray")
            ax.set_title(item)
        else:
            ax.set_title(f"{item} (missing)")
        ax.axis("off")
    flat[-1].axis("off")
    fig.tight_layout()
    return send_file(figure_to_png(fig), mimetype="image/png")


@app.route("/bbox.png")
def bbox_png():
    idx = int(request.args.get("id", -1))
    p = get_patient(idx)
    seg = load_volume(p["seg_file"])
    peaks = peak_slice_per_label(seg)
    bg_vol, bg_name = pick_background(p["modalities"])

    if not peaks:
        fig = plt.figure(figsize=(8, 6), facecolor="#1a1a2e")
        fig.text(0.5, 0.5, "No tumour labels in this segmentation",
                 ha="center", va="center", color="white", fontsize=14)
        return send_file(figure_to_png(fig), mimetype="image/png")

    n = len(peaks)
    n_cols = int(np.ceil(np.sqrt(n)))
    n_rows = int(np.ceil(n / n_cols))
    fig, axes = plt.subplots(n_rows, n_cols, figsize=(7 * n_cols, 7 * n_rows),
                             squeeze=False, facecolor="#1a1a2e")
    fig.suptitle(f'BBox per label  -  {p["patient_id"]}  (background: {bg_name})',
                 fontsize=14, fontweight="bold", color="white")

    for idx_p, (label_id, (peak_z, px)) in enumerate(sorted(peaks.items())):
        ax = axes[idx_p // n_cols][idx_p % n_cols]
        ax.set_facecolor("black")
        bg = norm_bg(bg_vol, peak_z) if bg_vol is not None else np.zeros(seg.shape[:2])
        ax.imshow(bg, cmap="gray", vmin=0, vmax=1)

        mask = (seg[:, :, peak_z] == label_id)
        overlay = np.zeros((*mask.shape, 4), dtype=float)
        overlay[mask] = LABEL_OVERLAY[label_id]
        ax.imshow(overlay, interpolation="none")

        bbox = compute_bbox(mask)
        if bbox:
            r_min, r_max, c_min, c_max = bbox
            w, h = c_max - c_min, r_max - r_min
            ax.add_patch(mpatches.Rectangle(
                (c_min, r_min), w, h, linewidth=2.5,
                edgecolor=LABEL_BBOX[label_id], facecolor="none", linestyle="--"))
            ax.text(c_min, max(r_min - 6, 0),
                    f"z={peak_z}  px={px:,}  W={w}  H={h}",
                    color="white", fontsize=8, fontweight="bold",
                    bbox=dict(facecolor="black", alpha=0.55, pad=2, edgecolor="none"))

        ax.set_title(f"Label {label_id}: {LABEL_NAMES[label_id]}",
                     color=LABEL_BBOX[label_id], fontsize=11, fontweight="bold")
        ax.axis("off")

    for j in range(n, n_rows * n_cols):
        axes[j // n_cols][j % n_cols].set_visible(False)
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
                   label=f"{int(v)}: {LABEL_NAMES.get(int(v), '?')} ({n} vox)")
    ax.set_xlabel("X")
    ax.set_ylabel("Y")
    ax.set_zlabel("Z")
    ax.set_title(f'{p["patient_id"]}\nNETC=R  SNFH=G  ET=B  RC=Y')
    if values.size:
        ax.legend(loc="upper left", bbox_to_anchor=(1.02, 1))
    fig.tight_layout()
    return send_file(figure_to_png(fig), mimetype="image/png")
