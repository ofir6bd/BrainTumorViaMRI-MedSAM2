import os
import sys
import csv
import gc
import time
import shutil
import warnings
from datetime import datetime

import nibabel as nib
import numpy as np
from PIL import Image
from scipy.ndimage import distance_transform_edt, binary_erosion
import torch
import yaml
import matplotlib
matplotlib.use('TkAgg')
import matplotlib.pyplot as plt
from matplotlib.gridspec import GridSpec
from matplotlib.patches import Patch
import matplotlib.colors as mcolors
from MedSAM2.sam2.build_sam import build_sam2_video_predictor

warnings.filterwarnings("ignore")

with open("config.yaml", "r") as _f:
    _CFG = yaml.safe_load(_f)

MEDSAM2_PATH    = os.path.abspath(_CFG["paths"]["medsam2"])

SAM2_CHECKPOINT = "./checkpoints/MedSAM2_latest.pt"
SAM2_CFG        = "sam2/configs/sam2.1_hiera_t512.yaml"

DATASET_DIR     = _CFG["paths"]["sample"] if _CFG["inference"]["use_sample"] else _CFG["paths"]["dataset"]

OUTPUT_BASE_DIR = os.path.abspath(_CFG["paths"]["outputs"])

TEMP_VIDEO_DIR  = os.path.abspath(_CFG["paths"]["tmp"])

NUM_PATIENTS    = _CFG["inference"]["num_patients"]

DEBUG           = False

SHOW_PLOTS      = False

LABEL_NAMES    = {1: "NETC", 2: "SNFH", 3: "ET", 4: "RC"}

TARGET_LABELS  = [1, 2, 3, 4]

MERGE_PRIORITY = [2, 3, 4, 1]

WT_LABELS = [1, 2, 3, 4]
TC_LABELS = [1, 3, 4]

ANCHOR_PERCENTILES  = [5, 25, 50, 75, 95]

COMPACT_SLICE_LIMIT = 10

MASK_THRESHOLD  = 0

if MEDSAM2_PATH not in sys.path:
    sys.path.insert(0, MEDSAM2_PATH)
    print(f"[INIT] Added MedSAM2 to sys.path: {MEDSAM2_PATH}")

import subprocess
subprocess.run(
    [sys.executable, "-m", "pip", "install", "portalocker==2.7.0", "-q"],
    check=False,
)

try:
    print("[INIT] sam2.build_sam import OK")
except ImportError as e:
    raise ImportError(
        f"Could not import sam2.\n"
        f"Expected path: {MEDSAM2_PATH}\n"
        f"Run: cd MedSAM2 && pip install -e \".[dev]\"\n"
        f"Error: {e}"
    )

def dprint(*args, **kwargs):
    if DEBUG:
        print("[DEBUG]", *args, **kwargs)

def clear_dir(path):
    if os.path.exists(path):
        shutil.rmtree(path)
    os.makedirs(path)

def normalize_percentile_uint8(vol):
    lo, hi = np.percentile(vol, 0.5), np.percentile(vol, 99.5)
    vol = np.clip(vol, lo, hi)
    if hi - lo < 1e-8:
        return np.zeros(vol.shape, dtype=np.uint8)
    return ((vol - lo) / (hi - lo) * 255).astype(np.uint8)

def load_nifti(path):
    dprint(f"  Loading: {path}")
    img  = nib.load(path)
    data = img.get_fdata(dtype=np.float32)
    dprint(f"    shape={data.shape}  min={data.min():.1f}  max={data.max():.1f}")
    return data, img.affine, img.header, img

def calculate_metrics(gt_bin, pred_bin):
    gt_bin   = gt_bin.astype(bool)
    pred_bin = pred_bin.astype(bool)

    if not np.any(gt_bin) and not np.any(pred_bin):
        return 1.0, 0.0
    if not np.any(gt_bin) or not np.any(pred_bin):
        return 0.0, 373.13

    intersection = np.logical_and(gt_bin, pred_bin).sum()
    dice = (2.0 * intersection) / (gt_bin.sum() + pred_bin.sum())

    gt_surface   = gt_bin   & ~binary_erosion(gt_bin)
    pred_surface = pred_bin & ~binary_erosion(pred_bin)

    dt_pred = distance_transform_edt(~pred_bin)
    dt_gt   = distance_transform_edt(~gt_bin)

    surf_distances = np.hstack([
        dt_pred[gt_surface],
        dt_gt[pred_surface],
    ])
    hd95 = float(np.percentile(surf_distances, 95))
    return float(dice), hd95

def prepare_rgb_frames(t1c_vol, t2w_vol, t2f_vol, video_name):
    frame_dir = os.path.join(TEMP_VIDEO_DIR, video_name)
    os.makedirs(frame_dir, exist_ok=True)

    r = normalize_percentile_uint8(t1c_vol)
    g = normalize_percentile_uint8(t2w_vol)
    b = normalize_percentile_uint8(t2f_vol)

    H, W, D = r.shape
    dprint(f"  Saving {D} RGB JPEG frames (R=T1c · G=T2w · B=T2f) → {frame_dir}")

    for z in range(D):
        rgb = np.stack([r[:, :, z], g[:, :, z], b[:, :, z]], axis=-1)
        Image.fromarray(rgb, mode="RGB").save(
            os.path.join(frame_dir, f"{z:05d}.jpg"), quality=95
        )
    dprint(f"  RGB frames saved: {D} files")

def _compute_anchors_for_areas(areas, z_indices):
    occupied = z_indices[areas > 0]
    n_slices = len(occupied)

    if n_slices == 0:
        return {}

    if n_slices == 1:
        return {p: int(occupied[0]) for p in ANCHOR_PERCENTILES}

    if n_slices < COMPACT_SLICE_LIMIT:
        indices = np.linspace(0, n_slices - 1, min(5, n_slices)).astype(int)
        sampled = occupied[indices]
        pct_z   = {}
        for i, p in enumerate(ANCHOR_PERCENTILES):
            pct_z[p] = int(sampled[min(i, len(sampled) - 1)])
        return pct_z

    weighted_z = np.repeat(z_indices, areas.astype(int))
    return {p: int(np.percentile(weighted_z, p)) for p in ANCHOR_PERCENTILES}

def find_penta_anchors_per_label(seg_data, present_labels):
    anchors   = {}
    D         = seg_data.shape[2]
    z_indices = np.arange(D)

    pct_labels = " ".join(f"p{p:02d}_z" for p in ANCHOR_PERCENTILES)
    print(f"\n  Per-label PENTA anchor slices ({' · '.join(f'P{p}' for p in ANCHOR_PERCENTILES)}):")
    print(f"  {'Label':<20} {pct_labels}   {'Unique':>20}  {'Peak px':>10}  {'Strategy':>12}")
    print(f"  {'-'*100}")

    for label_id in present_labels:
        areas = np.array([
            np.sum(seg_data[:, :, z] == label_id)
            for z in range(D)
        ])

        if areas.max() == 0:
            dprint(f"  Label {label_id} absent — skipped")
            continue

        peak_z         = int(np.argmax(areas))
        occupied       = z_indices[areas > 0]
        strategy       = "geometric" if len(occupied) < COMPACT_SLICE_LIMIT else "weighted-pct"
        pct_z          = _compute_anchors_for_areas(areas, z_indices)
        unique_anchors = sorted(set(pct_z.values()))

        anchors[label_id] = {
            'pct_z'    : pct_z,
            'peak_z'   : peak_z,
            'unique'   : unique_anchors,
            'strategy' : strategy,
        }

        pct_str = "  ".join(f"{pct_z[p]:>4}" for p in ANCHOR_PERCENTILES)
        print(f"  Label {label_id} ({LABEL_NAMES[label_id]:<6})  {pct_str}  "
              f"  {str(unique_anchors):>20}  {areas[peak_z]:>8} px  {strategy:>12}")

    if DEBUG:
        print("\n  [DEBUG] Pixel counts at each label's peak anchor:")
        header = f"  {'':22}"
        for lid in anchors:
            header += f"  peak({LABEL_NAMES[lid]},z={anchors[lid]['peak_z']:>3})"
        print(header)
        for lid_row in anchors:
            row = f"  Label {lid_row} ({LABEL_NAMES[lid_row]:<6})      "
            for lid_col in anchors:
                cnt    = int(np.sum(
                    seg_data[:, :, anchors[lid_col]['peak_z']] == lid_row
                ))
                marker = " ←" if lid_row == lid_col else "  "
                row   += f"  {cnt:>8}{marker}"
            print(row)

    return anchors

def find_composite_anchors(seg_data, label_set, track_name):
    D         = seg_data.shape[2]
    z_indices = np.arange(D)

    combined_bin = np.isin(seg_data, label_set)
    areas        = np.array([combined_bin[:, :, z].sum() for z in range(D)])

    if areas.max() == 0:
        print(f"  [WARNING] {track_name}: no voxels found for labels {label_set}")
        return None

    peak_z         = int(np.argmax(areas))
    occupied       = z_indices[areas > 0]
    strategy       = "geometric" if len(occupied) < COMPACT_SLICE_LIMIT else "weighted-pct"
    pct_z          = _compute_anchors_for_areas(areas, z_indices)
    unique_anchors = sorted(set(pct_z.values()))

    pct_str = "  ".join(f"{pct_z[p]:>4}" for p in ANCHOR_PERCENTILES)
    print(f"  {track_name:<8} labels={label_set}  {pct_str}  "
          f"  unique={unique_anchors}  peak_px={areas[peak_z]:,}  [{strategy}]")

    return {
        'pct_z'    : pct_z,
        'peak_z'   : peak_z,
        'unique'   : unique_anchors,
        'strategy' : strategy,
        'mask_3d'  : combined_bin,
    }

_predictor = None

def get_predictor():
    global _predictor
    if _predictor is not None:
        return _predictor

    print("\n  [MODEL] Loading MedSAM2 predictor...")

    ckpt_full = os.path.join(MEDSAM2_PATH, SAM2_CHECKPOINT)
    cfg_full  = os.path.join(MEDSAM2_PATH, SAM2_CFG)
    cfg_hydra = "configs/sam2.1_hiera_t512.yaml"

    dprint(f"  [MODEL] Checkpoint : {ckpt_full}")
    dprint(f"  [MODEL] Config     : {cfg_full}")

    if not os.path.exists(ckpt_full):
        raise FileNotFoundError(f"Checkpoint not found: {ckpt_full}")
    if not os.path.exists(cfg_full):
        raise FileNotFoundError(f"Config not found: {cfg_full}")

    _predictor = build_sam2_video_predictor(
        config_file=cfg_hydra,
        ckpt_path=ckpt_full,
        apply_postprocessing=False,
    )
    _predictor.eval()
    print("  [MODEL] Predictor loaded")
    return _predictor

@torch.inference_mode()
@torch.autocast(device_type="cuda", dtype=torch.bfloat16)
def run_all_tracks_shared_encoder(video_name, track_configs, shape):
    H, W, D   = shape
    predictor = get_predictor()
    video_dir = os.path.join(TEMP_VIDEO_DIR, video_name)
    OBJ_ID    = 1

    if not os.path.exists(video_dir):
        print(f"  [ERROR] Video dir not found: {video_dir}")
        return {name: np.zeros((H, W, D), dtype=bool) for name in track_configs}

    frame_files = sorted([
        f for f in os.listdir(video_dir)
        if os.path.splitext(f)[-1].lower() in (".jpg", ".jpeg")
    ])
    n_frames = len(frame_files)

    if n_frames == 0:
        print(f"  [ERROR] No JPEG frames in {video_dir}")
        return {name: np.zeros((H, W, D), dtype=bool) for name in track_configs}

    try:
        inference_state = predictor.init_state(
            video_path=video_dir,
            async_loading_frames=False,
        )
    except Exception as e:
        print(f"  [ERROR] init_state failed: {e}")
        return {name: np.zeros((H, W, D), dtype=bool) for name in track_configs}

    results = {}

    for track_name, anchor_masks_by_z in track_configs.items():

        if not anchor_masks_by_z:
            dprint(f"  [SKIP] {track_name}: no anchors")
            results[track_name] = np.zeros((H, W, D), dtype=bool)
            continue

        unique_anchors = sorted(anchor_masks_by_z.keys())

        for z in unique_anchors:
            m = anchor_masks_by_z[z]
            dprint(f"  [INFER-{track_name}] anchor z={z:>4}  prompt_px={m.sum():>6}")

        dprint(f"  [INFER-{track_name}] unique anchors={unique_anchors}  frames={n_frames}")

        predictor.reset_state(inference_state)

        def _add_prompts():
            for z, mask_hw in anchor_masks_by_z.items():
                predictor.add_new_mask(
                    inference_state=inference_state,
                    frame_idx=z,
                    obj_id=OBJ_ID,
                    mask=mask_hw,
                )

        _add_prompts()
        fwd_start     = min(unique_anchors)
        output_scores = {}

        dprint(f"  [INFER-{track_name}] Forward  z={fwd_start} → z={n_frames - 1}")
        for out_frame_idx, out_obj_ids, out_mask_logits in predictor.propagate_in_video(
            inference_state, start_frame_idx=fwd_start, reverse=False,
        ):
            for i, oid in enumerate(out_obj_ids):
                if oid == OBJ_ID:
                    output_scores[out_frame_idx] = out_mask_logits[i][0].cpu().numpy()

        dprint(f"  [INFER-{track_name}] Forward done — {len(output_scores)} frames")

        predictor.reset_state(inference_state)
        _add_prompts()
        bwd_start = max(unique_anchors)

        dprint(f"  [INFER-{track_name}] Backward z={bwd_start} → z=0")
        for out_frame_idx, out_obj_ids, out_mask_logits in predictor.propagate_in_video(
            inference_state, start_frame_idx=bwd_start, reverse=True,
        ):
            for i, oid in enumerate(out_obj_ids):
                if oid == OBJ_ID and out_frame_idx not in output_scores:
                    output_scores[out_frame_idx] = out_mask_logits[i][0].cpu().numpy()

        dprint(f"  [INFER-{track_name}] Backward done — {len(output_scores)}/{n_frames} frames total")

        mask_3d = np.zeros((H, W, D), dtype=bool)

        if SHOW_PLOTS:
            if track_name in ("WT", "TC"):
                all_logits = np.concatenate([v.flatten() for v in output_scores.values()])
                all_logits = all_logits[all_logits > -500]
                plt.figure(); plt.hist(all_logits, bins=int(all_logits.max()-all_logits.min())); plt.axvline(MASK_THRESHOLD, color='r', label='threshold'); plt.xlim(all_logits.min(), all_logits.max()+1); plt.yscale('log'); plt.legend(); plt.title(f"{track_name} logits"); plt.show()

        for z in range(D):
            if z in output_scores:
                mask_3d[:, :, z] = output_scores[z] > MASK_THRESHOLD

        n_vox = mask_3d.sum()
        dprint(f"  [INFER-{track_name}] {n_vox:,} predicted voxels")

        if n_vox == 0:
            print(f"  [WARNING] Empty prediction for {track_name}")

        results[track_name] = mask_3d
        torch.cuda.empty_cache()

    predictor.reset_state(inference_state)
    gc.collect()
    return results

def merge_masks_to_label_map(binary_masks, shape):
    print("\n  Merging binary masks → label map...")
    label_map = np.zeros(shape, dtype=np.uint8)

    for label_id in MERGE_PRIORITY:
        if label_id not in binary_masks:
            dprint(f"  Label {label_id} not in binary_masks — skip")
            continue
        mask = binary_masks[label_id]
        label_map[mask] = label_id
        dprint(f"  Painted label {label_id} ({LABEL_NAMES[label_id]}): {mask.sum():,} voxels")

    print("  Final label distribution:")
    for uid, cnt in zip(*np.unique(label_map, return_counts=True)):
        name = LABEL_NAMES.get(int(uid), "background")
        print(f"    Label {uid} ({name}): {cnt:,} voxels")

    return label_map

def show_visualization(bg_vol, gt_bin, pred_bin,
                       patient_id, label_id, dice_raw, hd95_raw,
                       dice_merged, hd95_merged, anchor_info, vis_dir,
                       track_label=None):
    os.makedirs(vis_dir, exist_ok=True)

    D          = bg_vol.shape[2]
    label_name = track_label if track_label else LABEL_NAMES[label_id]
    pct_z      = anchor_info['pct_z']
    peak_z     = anchor_info['peak_z']
    unique     = anchor_info['unique']
    strategy   = anchor_info.get('strategy', 'weighted-pct')

    print(f"  [VIZ] {label_name} | anchors={unique} | strategy={strategy}")

    def _draw_row(fig, gs, row_offset, z, row_label):
        bg       = bg_vol[:, :, z]
        gt_slice = gt_bin[:, :, z].astype(bool)
        pr_slice = pred_bin[:, :, z].astype(bool)

        tp = gt_slice & pr_slice
        fn = gt_slice & ~pr_slice
        fp = ~gt_slice & pr_slice

        b_min, b_max = np.percentile(bg, 1), np.percentile(bg, 99)
        bg_norm = np.clip(
            (bg.astype(float) - b_min) / max(b_max - b_min, 1e-8), 0, 1
        )

        overlay = np.zeros((*bg.shape, 4), dtype=float)
        overlay[tp] = [1.0, 1.0, 0.0, 0.55]
        overlay[fn] = [0.0, 1.0, 0.0, 0.55]
        overlay[fp] = [1.0, 0.0, 0.0, 0.55]

        configs = [
            ("Ground Truth",       'cyan',    gt_slice, 'cyan',    None),
            ("MedSAM2 Prediction", 'magenta', pr_slice, 'magenta', None),
            ("Overlap",            '#f0a500', None,     None,      overlay),
        ]

        for col, (title, border, contour_arr, cont_col, extra) in enumerate(configs):
            ax = fig.add_subplot(gs[row_offset, col])
            ax.imshow(bg_norm, cmap='gray', vmin=0, vmax=1, interpolation='bilinear')
            if contour_arr is not None and np.any(contour_arr):
                ax.contour(contour_arr, colors=cont_col, linewidths=2.0, alpha=0.9)
                fill = np.zeros((*bg.shape, 4), dtype=float)
                fill[contour_arr] = [*mcolors.to_rgb(cont_col), 0.20]
                ax.imshow(fill, interpolation='none')
            if extra is not None:
                ax.imshow(extra, interpolation='none')
            ax.set_title(f"{row_label}  z={z}\n{title}", color='white', fontsize=9, pad=4)
            ax.axis('off')
            for spine in ax.spines.values():
                spine.set_edgecolor(border)
                spine.set_linewidth(2)
            ax.set_facecolor('#0d0d1a')

    n_rows = len(ANCHOR_PERCENTILES)

    fig = plt.figure(figsize=(24, 6 * n_rows + 2), facecolor='#1a1a2e')
    gs  = GridSpec(
        n_rows + 1, 3,
        figure=fig,
        height_ratios=[10] * n_rows + [1],
        hspace=0.08, wspace=0.04,
        left=0.02, right=0.98, top=0.95, bottom=0.01
    )

    fig.text(0.5, 0.975,
             f"Patient: {patient_id}  |  RGB: R=T1c · G=T2w · B=T2f  |  "
             f"Track: {label_name}  |  Anchor strategy: {strategy}",
             ha='center', fontsize=16, fontweight='bold', color='white')
    fig.text(0.5, 0.960,
             f"Dice raw: {dice_raw:.4f}   HD95 raw: {hd95_raw:.2f}   |   "
             f"Dice merged: {dice_merged:.4f}   HD95 merged: {hd95_merged:.2f}   |   "
             f"Anchors: {' · '.join(f'P{p}={pct_z[p]}' for p in ANCHOR_PERCENTILES)}",
             ha='center', fontsize=10, color='#cccccc')

    for row_idx, pct in enumerate(ANCHOR_PERCENTILES):
        z         = pct_z[pct]
        star      = " ★" if z == peak_z else ""
        row_label = f"P{pct:02d} ANCHOR (z={z}){star}"
        _draw_row(fig, gs, row_idx, z, row_label)

    ax_leg = fig.add_subplot(gs[n_rows, :])
    ax_leg.axis('off')
    ax_leg.legend(handles=[
        Patch(facecolor='cyan',    edgecolor='white', alpha=0.7, label='GT contour'),
        Patch(facecolor='magenta', edgecolor='white', alpha=0.7, label='Pred contour'),
        Patch(facecolor='yellow',  edgecolor='white', alpha=0.7, label='True Positive'),
        Patch(facecolor='green',   edgecolor='white', alpha=0.7, label='False Negative (missed)'),
        Patch(facecolor='red',     edgecolor='white', alpha=0.7, label='False Positive (extra)'),
    ], loc='center', ncol=5, fontsize=11,
       framealpha=0.15, edgecolor='white', facecolor='#1a1a2e', labelcolor='white')

    z_vals   = [pct_z[p] for p in ANCHOR_PERCENTILES]
    pct_tag  = f"z{min(z_vals)}-{max(z_vals)}_n{len(set(z_vals))}"
    fname    = f"{patient_id}_{label_name}_{pct_tag}.png"
    out_path = os.path.join(vis_dir, fname)
    plt.savefig(out_path, dpi=100, bbox_inches='tight', facecolor=fig.get_facecolor())
    dprint(f"  [VIZ] Saved: {out_path}")

    if SHOW_PLOTS:
        plt.show(block=True)
    plt.close(fig)

def process_patient(patient_id, p_path, writer, results_accumulator):
    print(f"\n{'='*60}")
    print(f"  PATIENT: {patient_id}")
    print(f"{'='*60}")
    t0 = time.time()

    all_files = os.listdir(p_path)
    dprint(f"  Files: {all_files}")

    def find_file(suffix):
        matches = [
            f for f in all_files
            if f.endswith(('.nii', '.nii.gz'))
            and (f'-{suffix}.' in f.lower() or f'-{suffix}_' in f.lower()
                 or f.lower().endswith(f'-{suffix}.nii.gz')
                 or f.lower().endswith(f'-{suffix}.nii'))
        ]
        return os.path.join(p_path, matches[0]) if matches else None

    seg_path = find_file('seg')
    t1c_path = find_file('t1c')
    t2w_path = find_file('t2w')
    t2f_path = find_file('t2f')
    t1n_path = find_file('t1n')

    if seg_path is None:
        print(f"  [SKIP] No segmentation file for {patient_id}")
        return []

    print(f"  Seg  : {os.path.basename(seg_path)}")
    for name, path in [('T1c', t1c_path), ('T2w', t2w_path),
                       ('T2f', t2f_path), ('T1n', t1n_path)]:
        status = f"FOUND ({os.path.basename(path)})" if path else "MISSING"
        print(f"  {name:<4}: {status}")

    seg_data, affine, header, _ = load_nifti(seg_path)
    seg_data = seg_data.astype(np.uint8)
    H, W, D  = seg_data.shape

    unique_labels  = np.unique(seg_data).astype(int)
    present_labels = [L for L in TARGET_LABELS if L in unique_labels]
    print(f"\n  GT labels present: {present_labels}  "
          f"(all unique: {unique_labels.tolist()})")

    if not present_labels:
        print(f"  [SKIP] No tumor labels found in {patient_id}")
        return []

    def _load(path):
        return load_nifti(path)[0] if path and os.path.exists(path) else None

    t1c_raw = _load(t1c_path)
    t2w_raw = _load(t2w_path)
    t2f_raw = _load(t2f_path)
    t1n_raw = _load(t1n_path)

    def _first(*args):
        return next((a for a in args if a is not None), None)

    t1c = _first(t1c_raw, t2w_raw, t1n_raw, t2f_raw)
    t2w = _first(t2w_raw, t1n_raw, t1c_raw, t2f_raw)
    t2f = _first(t2f_raw, t2w_raw, t1c_raw)

    if t1c is None or t2w is None or t2f is None:
        print(f"  [SKIP] Cannot construct RGB channels for {patient_id}")
        return []

    if t2w_raw is None:
        print("  [FALLBACK] T2w missing → using T1n for G channel")
    if t2f_raw is None:
        print("  [FALLBACK] T2f missing → using T2w for B channel")

    video_name = f"{patient_id}_rgb"
    clear_dir(os.path.join(TEMP_VIDEO_DIR, video_name))
    prepare_rgb_frames(t1c, t2w, t2f, video_name)

    anchors = find_penta_anchors_per_label(seg_data, present_labels)
    if not anchors:
        print(f"  [SKIP] No anchors found for {patient_id}")
        return []

    valid_anchors = {}
    for label_id, info in anchors.items():
        peak_mask = (seg_data[:, :, info['peak_z']] == label_id)
        if not np.any(peak_mask):
            print(f"  [WARNING] Empty peak mask for label {label_id} — skip")
            continue
        valid_anchors[label_id] = info

    if not valid_anchors:
        print(f"  [SKIP] No valid anchors for {patient_id}")
        return []

    wt_present = [L for L in WT_LABELS if L in present_labels]
    tc_present = [L for L in TC_LABELS if L in present_labels]

    print(f"\n  Composite anchors:")
    wt_info = find_composite_anchors(seg_data, wt_present, "WT") if wt_present else None
    tc_info = find_composite_anchors(seg_data, tc_present, "TC") if tc_present else None

    print(f"\n  Building track configs for shared-encoder inference...")
    track_configs = {}

    for label_id, info in valid_anchors.items():
        track_name        = LABEL_NAMES[label_id]
        anchor_masks_by_z = {
            z: (seg_data[:, :, z] == label_id).astype(bool)
            for z in info['unique']
        }
        track_configs[track_name] = anchor_masks_by_z
        print(f"    {track_name}: anchors={info['unique']}  [{info.get('strategy','?')}]  peak_px="
              f"{int(np.sum(seg_data[:, :, info['peak_z']] == label_id))}")

    if wt_info is not None:
        wt_masks = {
            z: np.isin(seg_data[:, :, z], wt_present).astype(bool)
            for z in wt_info['unique']
        }
        track_configs["WT"] = wt_masks
        print(f"    WT: anchors={wt_info['unique']}  [{wt_info.get('strategy','?')}]")

    if tc_info is not None:
        tc_masks = {
            z: np.isin(seg_data[:, :, z], tc_present).astype(bool)
            for z in tc_info['unique']
        }
        track_configs["TC"] = tc_masks
        print(f"    TC: anchors={tc_info['unique']}  [{tc_info.get('strategy','?')}]")

    print(f"\n  Running {len(track_configs)} tracks in SHARED encoder session "
          f"({len(valid_anchors)} labels + WT + TC)...")
    t_infer = time.time()

    all_track_results = run_all_tracks_shared_encoder(
        video_name, track_configs, (H, W, D)
    )

    print(f"  Inference done in {time.time() - t_infer:.1f}s  "
          f"(all {len(track_configs)} tracks, single encoder)")

    binary_masks = {
        label_id: all_track_results[LABEL_NAMES[label_id]]
        for label_id in valid_anchors
        if LABEL_NAMES[label_id] in all_track_results
    }
    wt_mask_pred = all_track_results.get("WT")
    wt_ok        = wt_mask_pred is not None
    tc_mask_pred = all_track_results.get("TC")
    tc_ok        = tc_mask_pred is not None

    if not binary_masks:
        print(f"  [ERROR] All per-label inferences failed for {patient_id}")
        return []

    pred_label_map = merge_masks_to_label_map(binary_masks, (H, W, D))

    nifti_dir = os.path.join(OUTPUT_BASE_DIR, "segs_nifti")
    os.makedirs(nifti_dir, exist_ok=True)

    out_nifti = os.path.join(nifti_dir, f"{patient_id}_pred.nii.gz")
    nib.save(nib.Nifti1Image(pred_label_map, affine, header), out_nifti)
    print(f"  Saved NIfTI: {out_nifti}")

    if wt_ok and np.any(wt_mask_pred):
        wt_nifti_path = os.path.join(nifti_dir, f"{patient_id}_pred_WT.nii.gz")
        nib.save(nib.Nifti1Image(wt_mask_pred.astype(np.uint8), affine, header), wt_nifti_path)
        print(f"  Saved WT NIfTI: {wt_nifti_path}")

    if tc_ok and np.any(tc_mask_pred):
        tc_nifti_path = os.path.join(nifti_dir, f"{patient_id}_pred_TC.nii.gz")
        nib.save(nib.Nifti1Image(tc_mask_pred.astype(np.uint8), affine, header), tc_nifti_path)
        print(f"  Saved TC NIfTI: {tc_nifti_path}")

    patient_video_dir = os.path.join(TEMP_VIDEO_DIR, video_name)
    if os.path.exists(patient_video_dir):
        shutil.rmtree(patient_video_dir)
        dprint(f"  Cleaned temp frames: {patient_video_dir}")

    bg_vol = (
        normalize_percentile_uint8(t1c).astype(float) +
        normalize_percentile_uint8(t2w).astype(float) +
        normalize_percentile_uint8(t2f).astype(float)
    ) / 3.0

    patient_rows = []
    labels_str   = f"[{','.join(map(str, present_labels))}]"

    pct_header = " ".join(f"p{p:02d}" for p in ANCHOR_PERCENTILES)
    print(f"\n  {'Track':<22} {pct_header}  "
          f"{'Dice_raw':>10} {'Dice_mrg':>10} "
          f"{'HD95_raw':>10} {'HD95_mrg':>10} "
          f"{'GT vox':>10} {'Pred vox':>10} {'Strategy':>12}")
    print(f"  {'-'*130}")

    for label_id in present_labels:
        if label_id not in binary_masks:
            print(f"  [SKIP] No output mask for label {label_id}")
            continue

        label_name   = LABEL_NAMES[label_id]
        info         = valid_anchors[label_id]
        pred_raw     = binary_masks[label_id]
        pred_merged  = (pred_label_map == label_id)
        gt_bin       = (seg_data == label_id)

        dice_raw,    hd95_raw    = calculate_metrics(gt_bin, pred_raw)
        dice_merged, hd95_merged = calculate_metrics(gt_bin, pred_merged)

        pct_str  = " ".join(f"{info['pct_z'][p]:>4}" for p in ANCHOR_PERCENTILES)
        strategy = info.get('strategy', '?')
        print(f"  Label {label_id} ({label_name:<6})  {pct_str}  "
              f"  {dice_raw:.4f}    {dice_merged:.4f}  "
              f"  {hd95_raw:>8.2f}    {hd95_merged:>8.2f}  "
              f"  {gt_bin.sum():>8,}  {pred_raw.sum():>10,}  {strategy:>12}")

        vis_dir = os.path.join(OUTPUT_BASE_DIR, "visualizations", patient_id)
        show_visualization(
            bg_vol, gt_bin, pred_raw,
            patient_id, label_id,
            dice_raw, hd95_raw, dice_merged, hd95_merged,
            info, vis_dir
        )

        writer.writerow([
            patient_id, "RGB(T1c+T2w+T2f)", f"Label_{label_id}", label_name,
            labels_str,
            *[info['pct_z'][p] for p in ANCHOR_PERCENTILES],
            str(info['unique']),
            strategy,
            f"{dice_raw:.4f}",    f"{hd95_raw:.2f}",
            f"{dice_merged:.4f}", f"{hd95_merged:.2f}",
            gt_bin.sum(), pred_raw.sum(),
            "per-label",
        ])

        patient_rows.append({
            'patient_id':   patient_id,
            'label_id':     label_id,
            'label_name':   label_name,
            'peak_z':       info['peak_z'],
            'pct_z':        info['pct_z'],
            'strategy':     strategy,
            'dice':         dice_raw,
            'hd95':         hd95_raw,
            'dice_merged':  dice_merged,
            'hd95_merged':  hd95_merged,
        })

    if wt_ok and wt_info is not None:
        gt_wt            = np.isin(seg_data, wt_present)
        dice_wt, hd95_wt = calculate_metrics(gt_wt, wt_mask_pred)
        pct_str          = " ".join(f"{wt_info['pct_z'][p]:>4}" for p in ANCHOR_PERCENTILES)
        strategy         = wt_info.get('strategy', '?')
        print(f"  {'WT (indep.)':<22}  {pct_str}  "
              f"  {dice_wt:.4f}    {'N/A':>8}  "
              f"  {hd95_wt:>8.2f}    {'N/A':>8}  "
              f"  {gt_wt.sum():>8,}  {wt_mask_pred.sum():>10,}  {strategy:>12}")

        vis_dir = os.path.join(OUTPUT_BASE_DIR, "visualizations", patient_id)
        show_visualization(
            bg_vol, gt_wt, wt_mask_pred,
            patient_id, 0,
            dice_wt, hd95_wt, dice_wt, hd95_wt,
            wt_info, vis_dir,
            track_label="WT"
        )

        writer.writerow([
            patient_id, "RGB(T1c+T2w+T2f)", "WT",
            f"WholeTumour(labels={wt_present})",
            labels_str,
            *[wt_info['pct_z'][p] for p in ANCHOR_PERCENTILES],
            str(wt_info['unique']),
            strategy,
            f"{dice_wt:.4f}", f"{hd95_wt:.2f}",
            "N/A",            "N/A",
            gt_wt.sum(), wt_mask_pred.sum(),
            "independent-WT",
        ])

        patient_rows.append({
            'patient_id':   patient_id,
            'label_id':     "WT",
            'label_name':   "WT",
            'peak_z':       wt_info['peak_z'],
            'pct_z':        wt_info['pct_z'],
            'strategy':     strategy,
            'dice':         dice_wt,
            'hd95':         hd95_wt,
            'dice_merged':  dice_wt,
            'hd95_merged':  hd95_wt,
        })
    else:
        print("  [SKIP] WT inference failed or no WT labels present")

    if tc_ok and tc_info is not None:
        gt_tc            = np.isin(seg_data, tc_present)
        dice_tc, hd95_tc = calculate_metrics(gt_tc, tc_mask_pred)
        pct_str          = " ".join(f"{tc_info['pct_z'][p]:>4}" for p in ANCHOR_PERCENTILES)
        strategy         = tc_info.get('strategy', '?')
        print(f"  {'TC (indep.)':<22}  {pct_str}  "
              f"  {dice_tc:.4f}    {'N/A':>8}  "
              f"  {hd95_tc:>8.2f}    {'N/A':>8}  "
              f"  {gt_tc.sum():>8,}  {tc_mask_pred.sum():>10,}  {strategy:>12}")

        vis_dir = os.path.join(OUTPUT_BASE_DIR, "visualizations", patient_id)
        show_visualization(
            bg_vol, gt_tc, tc_mask_pred,
            patient_id, 0,
            dice_tc, hd95_tc, dice_tc, hd95_tc,
            tc_info, vis_dir,
            track_label="TC"
        )

        writer.writerow([
            patient_id, "RGB(T1c+T2w+T2f)", "TC",
            f"TumourCore(labels={tc_present})",
            labels_str,
            *[tc_info['pct_z'][p] for p in ANCHOR_PERCENTILES],
            str(tc_info['unique']),
            strategy,
            f"{dice_tc:.4f}", f"{hd95_tc:.2f}",
            "N/A",            "N/A",
            gt_tc.sum(), tc_mask_pred.sum(),
            "independent-TC",
        ])

        patient_rows.append({
            'patient_id':   patient_id,
            'label_id':     "TC",
            'label_name':   "TC",
            'peak_z':       tc_info['peak_z'],
            'pct_z':        tc_info['pct_z'],
            'strategy':     strategy,
            'dice':         dice_tc,
            'hd95':         hd95_tc,
            'dice_merged':  dice_tc,
            'hd95_merged':  hd95_tc,
        })
    else:
        print("  [SKIP] TC inference failed or no TC labels present")

    torch.cuda.empty_cache()
    gc.collect()

    elapsed = time.time() - t0
    print(f"\n  Patient {patient_id} done in {elapsed:.1f}s")
    return patient_rows

def run_pipeline():
    pct_str = " · ".join(f"P{p}" for p in ANCHOR_PERCENTILES)
    print(f"\n{'*'*60}")
    print(f"  BraTS 2024  ·  MedSAM2  ·  Per-Label PENTA-ANCHOR ({pct_str})")
    print(f"  + Whole Tumour (WT) and Tumour Core (TC) — shared encoder session")
    print(f"{'*'*60}")
    print(f"  RGB mapping    :  R=T1c  G=T2w  B=T2f(FLAIR)")
    print(f"  Anchor policy  :  adaptive (geometric < {COMPACT_SLICE_LIMIT} slices, else weighted-pct)")
    print(f"  Propagation    :  shared encoder · independent per-track reset")
    print(f"  Sessions/pt    :  1 init_state  ·  6 reset_state cycles")
    print(f"  Merge priority :  SNFH < NETC < ET < RC  (per-label only)")
    print(f"  WT labels      :  {WT_LABELS}  (Labels 1+2+3+4)")
    print(f"  TC labels      :  {TC_LABELS}  (Labels 1+3+4)")
    print(f"  WT/TC source   :  INDEPENDENT reset cycles (NOT derived from per-label)")
    print(f"  Metrics        :  raw + merged per label  |  direct WT/TC Dice/HD95")
    print(f"  Temp cleanup   :  per-patient RGB frames deleted after NIfTI save")
    print(f"  Dataset        :  {os.path.abspath(DATASET_DIR)}")
    print(f"  Output         :  {OUTPUT_BASE_DIR}")
    print(f"  MedSAM2        :  {MEDSAM2_PATH}")
    print(f"  Max patients   :  {NUM_PATIENTS}")
    print(f"{'*'*60}\n")

    if not os.path.isdir(DATASET_DIR):
        raise FileNotFoundError(f"Dataset not found: {DATASET_DIR}")
    if not os.path.isdir(MEDSAM2_PATH):
        raise FileNotFoundError(f"MedSAM2 not found: {MEDSAM2_PATH}")

    os.makedirs(OUTPUT_BASE_DIR, exist_ok=True)
    os.makedirs(TEMP_VIDEO_DIR,  exist_ok=True)

    all_patients = sorted([
        d for d in os.listdir(DATASET_DIR)
        if os.path.isdir(os.path.join(DATASET_DIR, d))
    ])
    if NUM_PATIENTS is not None:
        all_patients = all_patients[:NUM_PATIENTS]

    print(f"Patients to process: {len(all_patients)}\n")

    csv_path = os.path.join(
        OUTPUT_BASE_DIR,
        f"results_{datetime.now().strftime('%Y%m%d_%H%M')}.csv"
    )
    results_accumulator = []

    pct_cols = [f"p{p:02d}_z" for p in ANCHOR_PERCENTILES]

    with open(csv_path, 'w', newline='') as f:
        writer = csv.writer(f)
        writer.writerow([
            "PatientID", "Modality", "TrackID", "TrackName",
            "PresentLabels",
            *pct_cols, "UniqueAnchors",
            "AnchorStrategy",
            "Dice_raw",    "HD95_raw",
            "Dice_merged", "HD95_merged",
            "GT_voxels",   "Pred_voxels",
            "InferenceMode",
        ])

        failed = []
        for i, p_id in enumerate(all_patients):
            p_path = os.path.join(DATASET_DIR, p_id)
            print(f"\n[{i+1}/{len(all_patients)}] {p_id}")
            try:
                rows = process_patient(
                    p_id, p_path, writer, results_accumulator
                )
                f.flush()
                results_accumulator.extend(rows)
                if not rows:
                    failed.append(p_id)
            except Exception as e:
                print(f"  [FATAL] {p_id}: {e}")
                import traceback; traceback.print_exc()
                failed.append(p_id)

    print(f"\n\n{'#'*60}")
    print(f"  FINAL SUMMARY  —  Penta-Anchor ({pct_str})")
    print(f"{'#'*60}")
    print(f"  Processed:       {len(all_patients)} patients")
    print(f"  Failed/skipped:  {len(failed)}  {failed[:10]}")
    print(f"  Results CSV:     {csv_path}\n")

    if results_accumulator:

        for phase, key_dice, key_hd95 in [
            ("RAW (before merge)",            'dice',        'hd95'),
            ("MERGED (after priority paint)", 'dice_merged', 'hd95_merged'),
        ]:
            print(f"  --- {phase} ---")
            print(f"  {'Label':<22} {'N':>5} {'Dice Mean':>12} {'Dice Std':>10} "
                  f"{'HD95 Mean':>12} {'HD95 Std':>10}")
            print(f"  {'-'*73}")
            for lid in TARGET_LABELS:
                rows = [r for r in results_accumulator if r['label_id'] == lid]
                if not rows:
                    continue
                dices = [r[key_dice] for r in rows]
                hd95s = [r[key_hd95] for r in rows]
                print(f"  {f'Label {lid} ({LABEL_NAMES[lid]})':<22} "
                      f"{len(rows):>5} "
                      f"{np.mean(dices):>12.4f}  {np.std(dices):>10.4f} "
                      f"{np.mean(hd95s):>12.4f}  {np.std(hd95s):>10.4f}")
            print()

        print(f"  --- INDEPENDENT WT / TC TRACKS ---")
        print(f"  {'Track':<22} {'N':>5} {'Dice Mean':>12} {'Dice Std':>10} "
              f"{'HD95 Mean':>12} {'HD95 Std':>10}")
        print(f"  {'-'*73}")

        for track_name in ["WT", "TC"]:
            rows = [r for r in results_accumulator if r['label_id'] == track_name]
            if not rows:
                print(f"  {track_name:<22}  — no data —")
                continue
            dices = [r['dice'] for r in rows]
            hd95s = [r['hd95'] for r in rows]
            print(f"  {track_name:<22} "
                  f"{len(rows):>5} "
                  f"{np.mean(dices):>12.4f}  {np.std(dices):>10.4f} "
                  f"{np.mean(hd95s):>12.4f}  {np.std(hd95s):>10.4f}")
        print()

        print(f"  --- ANCHOR STRATEGY BREAKDOWN ---")
        for lid in TARGET_LABELS:
            rows = [r for r in results_accumulator if r['label_id'] == lid]
            if not rows:
                continue
            geo = [r for r in rows if r.get('strategy') == 'geometric']
            wpc = [r for r in rows if r.get('strategy') == 'weighted-pct']
            print(f"  Label {lid} ({LABEL_NAMES[lid]}): "
                  f"geometric={len(geo)}  weighted-pct={len(wpc)}")

    print(f"\nAll outputs saved to: {OUTPUT_BASE_DIR}")

if __name__ == "__main__":
    run_pipeline()
