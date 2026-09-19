import io
import json
import os
from functools import lru_cache

import yaml
import numpy as np
import nibabel as nib
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
from mpl_toolkits.mplot3d import Axes3D  # noqa: F401  (registers the 3d projection)
from matplotlib.colors import ListedColormap
from flask import Flask, jsonify, send_file, abort, render_template, request

from dataset_stats import StatsStore

with open("config.yaml", "r") as _f:
    _CFG = yaml.safe_load(_f)
_USE_SAMPLE = bool(_CFG.get("inference", {}).get("use_sample", False))
DATASET_DIR = _CFG["paths"]["sample"] if _USE_SAMPLE else _CFG["paths"]["dataset"]
MANIFEST_PATH = os.path.join(_CFG["paths"]["dataset"], "split_manifest.json")

# Physical dataset pools, in display order — paths come from config.yaml.
POOLS = [
    {"key": "yolo_train", "family": "yolo", "role": "train", "use": "YOLO trains on these"},
    {"key": "yolo_val", "family": "yolo", "role": "val", "use": "YOLO picks its best epoch"},
    {"key": "medsam2_train", "family": "medsam2", "role": "train", "use": "MedSAM2 trains on these"},
    {"key": "medsam2_val", "family": "medsam2", "role": "val", "use": "MedSAM2 picks its best epoch"},
    {"key": "test", "family": "test", "role": "test", "use": "Final test for both, never trained on"},
]
for _pool in POOLS:
    _pool["path"] = _CFG["paths"][_pool["key"]]

LABEL_NAMES = {1: "NETC", 2: "SNFH", 3: "ET", 4: "RC"}
# One colour per label, shared by every image here and the Analytics charts.
LABEL_COLORS = {1: "#e66767", 2: "#008300", 3: "#9085e9", 4: "#c98500"}
LABEL_CMAP = ListedColormap(["none"] + [LABEL_COLORS[i] for i in (1, 2, 3, 4)])
LABEL_KEY = "  ".join(f"{i} {LABEL_NAMES[i]}" for i in LABEL_NAMES)
MODALITY_ORDER = ["T1", "T1C", "T2", "FLAIR"]
BACKGROUND_ORDER = ["T1C", "T1", "T2", "FLAIR"]
MIN_FG_VOXELS = 100  # a slice with fewer non-zero FLAIR voxels renders black in RGB

app = Flask(__name__)
app.config["TEMPLATES_AUTO_RELOAD"] = True
app.config["SEND_FILE_MAX_AGE_DEFAULT"] = 0


def _pool_of(root):
    for pool in POOLS:
        if os.path.normcase(os.path.abspath(os.path.dirname(root))) == \
                os.path.normcase(os.path.abspath(pool["path"])):
            return pool["key"]
    return None


def find_patients():
    patients = []
    if not os.path.exists(DATASET_DIR):
        return patients
    for root, _, files in os.walk(DATASET_DIR):
        nii = [f for f in files if f.endswith((".nii", ".nii.gz"))]
        seg = [f for f in nii if "seg" in f.lower()]
        if not nii or not seg:
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
            "patient_id": os.path.basename(root),
            "pool": _pool_of(root),
            "seg_file": os.path.join(root, seg[0]),
            "modalities": modalities,
        })
    patients.sort(key=lambda p: p["patient_id"])
    return patients


PATIENTS = find_patients()
PATIENT_INDEX = {p["patient_id"]: i for i, p in enumerate(PATIENTS)}
stats_store = StatsStore(os.path.join(_CFG["paths"]["outputs"], "dashboard", "dataset_stats.json"))


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


def _norm_slice_uint8(slice_2d):
    """Per-slice p0.5-p99.5 normalisation to uint8; black if too little foreground."""
    s = slice_2d.astype(np.float32)
    nonzero = s[s != 0]
    if nonzero.size < MIN_FG_VOXELS:
        return np.zeros(s.shape, dtype=np.uint8)
    lo = float(np.percentile(nonzero, 0.5))
    hi = float(np.percentile(nonzero, 99.5))
    if hi - lo < 1e-8:
        return np.zeros(s.shape, dtype=np.uint8)
    return np.clip((s - lo) / (hi - lo) * 255, 0, 255).astype(np.uint8)


def build_rgb_slice(t1c, t1, t2w, flair):
    """R = clip(T1C - T1, 0), G = T2, B = FLAIR -> HxWx3 uint8 (the YOLO input frame)."""
    diff = np.clip(t1c.astype(np.float32) - t1.astype(np.float32), 0, None)
    return np.stack([_norm_slice_uint8(diff), _norm_slice_uint8(t2w),
                     _norm_slice_uint8(flair)], axis=-1)


def build_rgb_slice_raw(t1c, t2w, flair):
    """R = raw T1C (no subtraction), G = T2, B = FLAIR — for comparison only."""
    return np.stack([_norm_slice_uint8(t1c), _norm_slice_uint8(t2w),
                     _norm_slice_uint8(flair)], axis=-1)


def label_overlay(seg_2d, alpha=0.55):
    overlay = np.zeros((*seg_2d.shape, 4), dtype=float)
    for label_id, hexcol in LABEL_COLORS.items():
        overlay[seg_2d == label_id] = (*matplotlib.colors.to_rgb(hexcol), alpha)
    return overlay


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
                   markerfacecolor=LABEL_COLORS[i],
                   label=f"{i}: {LABEL_NAMES[i]}")
        for i in present
    ]


@app.route("/")
def index():
    return render_template("index.html")


@app.route("/api/patients")
def api_patients():
    return jsonify([
        {"id": i, "label": p["patient_id"], "pool": p["pool"]}
        for i, p in enumerate(PATIENTS)
    ])


@app.route("/api/patient/<int:idx>")
def api_patient(idx):
    p = get_patient(idx)
    seg = load_volume(p["seg_file"])
    return jsonify({
        "patient_id": p["patient_id"],
        "pool": p["pool"],
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
    axes[1].imshow(s, cmap=LABEL_CMAP, vmin=0, vmax=4, interpolation="none")
    axes[1].set_title(f"BraTS labels ({LABEL_KEY})")
    axes[2].imshow(b, cmap="gray")
    axes[2].imshow(label_overlay(s), interpolation="none")
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
            ax.imshow(sl, cmap=LABEL_CMAP, vmin=0, vmax=4, interpolation="none")
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


@app.route("/rgb.png")
def rgb_png():
    """Side-by-side comparison of two RGB channel-construction schemes.

    Left  = R: T1C-T1 subtraction, G: T2, B: FLAIR (the YOLO input frame).
    Right = R: raw T1C,            G: T2, B: FLAIR (no subtraction, for comparison).
    """
    idx = int(request.args.get("id", -1))
    p = get_patient(idx)
    seg = load_volume(p["seg_file"])
    z = max(0, min(int(request.args.get("z", best_slice(seg))), seg.shape[2] - 1))

    required = ("T1C", "T1", "T2", "FLAIR")
    if not all(k in p["modalities"] for k in required):
        abort(404)
    t1c = load_volume(p["modalities"]["T1C"])[:, :, z]
    t1 = load_volume(p["modalities"]["T1"])[:, :, z]
    t2w = load_volume(p["modalities"]["T2"])[:, :, z]
    flair = load_volume(p["modalities"]["FLAIR"])[:, :, z]

    fig, axes = plt.subplots(1, 2, figsize=(13, 7), facecolor="#141428")
    fig.suptitle(f'{p["patient_id"]}  -  slice z={z}  -  RGB channel construction',
                 color="white", fontsize=15, fontweight="bold", y=1.02)
    axes[0].imshow(build_rgb_slice(t1c, t1, t2w, flair))
    axes[0].set_title("R = T1C − T1 (subtraction)\nG = T2      B = FLAIR",
                      color="white", fontsize=11.5, fontweight="bold")
    axes[1].imshow(build_rgb_slice_raw(t1c, t2w, flair))
    axes[1].set_title("R = T1C (raw)\nG = T2      B = FLAIR",
                      color="white", fontsize=11.5, fontweight="bold")
    for ax in axes:
        ax.axis("off")
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
        ax.imshow(label_overlay(np.where(mask, label_id, 0)), interpolation="none")

        bbox = compute_bbox(mask)
        if bbox:
            r_min, r_max, c_min, c_max = bbox
            w, h = c_max - c_min, r_max - r_min
            ax.add_patch(mpatches.Rectangle(
                (c_min, r_min), w, h, linewidth=2.5,
                edgecolor=LABEL_COLORS[label_id], facecolor="none", linestyle="--"))
            ax.text(c_min, max(r_min - 6, 0),
                    f"z={peak_z}  px={px:,}  W={w}  H={h}",
                    color="white", fontsize=8, fontweight="bold",
                    bbox=dict(facecolor="black", alpha=0.55, pad=2, edgecolor="none"))

        ax.set_title(f"Label {label_id}: {LABEL_NAMES[label_id]}",
                     color=LABEL_COLORS[label_id], fontsize=11, fontweight="bold")
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
        ax.scatter(xs, ys, zs, c=LABEL_COLORS.get(int(v), "gray"), s=2, alpha=0.7,
                   label=f"{int(v)}: {LABEL_NAMES.get(int(v), '?')} ({n} vox)")
    ax.set_xlabel("X")
    ax.set_ylabel("Y")
    ax.set_zlabel("Z")
    ax.set_title(f'{p["patient_id"]}\n{LABEL_KEY}')
    if values.size:
        ax.legend(loc="upper left", bbox_to_anchor=(1.02, 1))
    fig.tight_layout()
    return send_file(figure_to_png(fig), mimetype="image/png")


# ---------------------------------------------------------------------------
# Analytics
# ---------------------------------------------------------------------------
@app.route("/thumb.png")
def thumb_png():
    """Small FLAIR + label overlay for the Analytics detail drawer."""
    idx = int(request.args.get("id", -1))
    p = get_patient(idx)
    if "FLAIR" not in p["modalities"]:
        abort(404)
    seg = load_volume(p["seg_file"])
    z = max(0, min(int(request.args.get("z", best_slice(seg))), seg.shape[2] - 1))
    flair = load_volume(p["modalities"]["FLAIR"])
    fig, ax = plt.subplots(figsize=(3.2, 3.8), facecolor="#12122a")
    # Axial slice shown in radiological orientation: patient's right on the left.
    ax.imshow(np.rot90(norm_bg(flair, z)), cmap="gray", vmin=0, vmax=1)
    ax.imshow(np.rot90(label_overlay(seg[:, :, z])), interpolation="none")
    ax.axis("off")
    fig.subplots_adjust(0, 0, 1, 1)
    return send_file(figure_to_png(fig), mimetype="image/png")


def _manifest_report():
    """Compare split_manifest.json with where each patient folder actually is."""
    if not os.path.exists(MANIFEST_PATH):
        return {"found": False}
    with open(MANIFEST_PATH, "r", encoding="utf-8") as f:
        manifest = json.load(f)
    assignment = manifest.get("assignment", {})
    on_disk = {p["patient_id"]: p["pool"] for p in PATIENTS}
    moved = [{"id": pid, "manifest": assignment[pid], "disk": pool}
             for pid, pool in sorted(on_disk.items())
             if pid in assignment and assignment[pid] != pool]
    return {
        "found": True,
        "note": manifest.get("note"),
        "seed": manifest.get("seed"),
        "counts": manifest.get("counts"),
        "moved": moved,
        "not_in_manifest": sorted(set(on_disk) - set(assignment)),
        "missing_on_disk": sorted(set(assignment) - set(on_disk)),
    }


@app.route("/api/dashboard/status")
def api_dashboard_status():
    # Opening the dashboard fills in whatever is not cached yet.
    stats_store.start(PATIENTS, log=print)
    return jsonify(stats_store.status(PATIENTS))


@app.route("/api/dashboard/recompute", methods=["POST"])
def api_dashboard_recompute():
    started = stats_store.start(PATIENTS, force=True, log=print)
    return jsonify({"started": started, **stats_store.status(PATIENTS)})


@app.route("/api/dashboard/data")
def api_dashboard_data():
    records = []
    for i, p in enumerate(PATIENTS):
        rec = stats_store.records.get(p["patient_id"])
        if rec is None:
            continue
        rec = {k: v for k, v in rec.items() if k != "_key"}
        rec["idx"] = i
        rec["pool"] = p["pool"]
        records.append(rec)
    return jsonify({
        "records": records,
        "pools": POOLS,
        "labels": [{"id": i, "name": LABEL_NAMES[i], "color": LABEL_COLORS[i]}
                   for i in LABEL_NAMES],
        "dataset_dir": DATASET_DIR,
        "n_patients": len(PATIENTS),
        "manifest": _manifest_report(),
        **stats_store.status(PATIENTS),
    })
