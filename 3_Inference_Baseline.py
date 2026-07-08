"""
medsam2_brats_pipeline.py
=========================
End-to-end BraTS 2024 segmentation pipeline using MedSAM2.

Overview
--------
Each patient volume is processed in 7 logical phases:

  Phase 00 – Bootstrap:    add MedSAM2 to sys.path, install portalocker.
  Phase 01 – Data I/O:     discover NIfTI files by suffix, load all modalities
                            (seg + T1c/T2w/T2f/T1n), apply fallback chains for
                            missing modalities, validate label presence.
  Phase 02 – RGB Frames:   stack T1c/T2w/T2f into per-slice JPEG triplets
                            that the SAM2 video predictor treats as a "video".
  Phase 03 – Anchors:      for every label (and for the WT/TC composites)
                            pick 5 representative axial slices (penta-anchor).
  Phase 04 – Inference:    run ALL tracks inside ONE shared init_state session
                            to avoid re-encoding the volume per track.
  Phase 05 – Merge:        paint per-label binary masks onto a single label
                            map using a fixed priority order.
  Phase 06 – Metrics:      compute Dice and HD95 (raw and post-merge) and
                            write results to a CSV.
  Visualisation:           save a 5-row figure (one row per anchor percentile)
                            showing GT, prediction, and overlap.

BraTS 2024 label convention
----------------------------
  1 = NETC  (Non-Enhancing Tumour Core)
  2 = SNFH  (Surrounding Non-tumour FLAIR Hyperintensity)
  3 = ET    (Enhancing Tumour)
  4 = RC    (Resection Cavity)

Composite tracks
----------------
  WT (Whole Tumour)  = labels 1 + 2 + 3 + 4
  TC (Tumour Core)   = labels 1 + 3 + 4

WT and TC are tracked INDEPENDENTLY (not derived from per-label outputs),
which gives them their own clean propagation path through the volume.

Anchor strategy
---------------
For each label we select 5 axial "anchor" slices from which GT masks are
fed as prompts into the video predictor:

  - weighted-pct  (default): area-weighted percentile across occupied slices.
                              Works well for large structures like SNFH.
  - geometric     (fallback): evenly-spaced sample of occupied z-range.
                              Used when a label occupies fewer than
                              COMPACT_SLICE_LIMIT slices (e.g. small NETC).

Shared-encoder optimisation
----------------------------
SAM2 video inference normally re-encodes the entire volume for each call to
init_state.  By calling init_state ONCE and using reset_state between tracks
we keep the cached frame features and avoid N redundant encodes per patient
(N = number of tracks, typically 6: NETC, SNFH, ET, RC, WT, TC).
Typical wall-clock speedup: ~3-4×.

Propagation pattern (per track)
--------------------------------
  1. Add GT anchor masks as prompts (add_new_mask).
  2. Forward pass  : propagate from lowest anchor z → last frame.
  3. reset_state   : clear memory bank (frame features stay cached).
  4. Re-add prompts.
  5. Backward pass : propagate from highest anchor z → first frame.
  6. Union of forward and backward logits → binary 3-D mask.

Output files (per patient)
--------------------------
  segs_nifti/<id>_pred.nii.gz       – merged multi-label prediction
  segs_nifti/<id>_pred_WT.nii.gz    – independent WT binary mask
  segs_nifti/<id>_pred_TC.nii.gz    – independent TC binary mask
  visualizations/<id>/              – per-label/WT/TC PNG figures
  results_<timestamp>.csv           – Dice + HD95 for every track
"""

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
import matplotlib
matplotlib.use('TkAgg')
import matplotlib.pyplot as plt
from matplotlib.gridspec import GridSpec
from matplotlib.patches import Patch
import matplotlib.colors as mcolors
from MedSAM2.sam2.build_sam import build_sam2_video_predictor

warnings.filterwarnings("ignore")


# =============================================================================
# CONFIGURATION
# All paths and run-time knobs are centralised here so nothing is hard-coded
# elsewhere in the file.
# =============================================================================

# Path to the cloned MedSAM2 repository root (added to sys.path at boot).
MEDSAM2_PATH    = os.path.abspath("./MedSAM2")

# MedSAM2 checkpoint (.pt) and Hydra config (.yaml) paths.
# The checkpoint is relative to MEDSAM2_PATH; the yaml is relative to the
# sam2 package inside that repo.
SAM2_CHECKPOINT = "./checkpoints/MedSAM2_latest.pt"
SAM2_CFG        = "sam2/configs/sam2.1_hiera_t512.yaml"

# Root folder containing one sub-directory per BraTS patient.
DATASET_DIR     = r"./2_BraTS2024_dataset/training_data1_v2"
DATASET_DIR   = r"./One_Patient_test_MRI_DATA"   # single-patient smoke test

# Where all outputs (NIfTI, CSV, PNGs) are written.
OUTPUT_BASE_DIR = os.path.abspath("./medsam2_results")

# Temporary per-patient JPEG frames consumed by the SAM2 video predictor.
# Cleaned up after each patient to cap disk usage (~31 MB/patient).
TEMP_VIDEO_DIR  = os.path.abspath("./temp_video_frames")

# Maximum number of patients to process (set None to run the full dataset).
NUM_PATIENTS    = 1

# If True, extra per-step print statements are emitted.
DEBUG           = False

# If True, figures are displayed interactively (blocks execution).
# Keep False for headless / cluster runs.
SHOW_PLOTS      = False


# =============================================================================
# LABEL DEFINITIONS
# =============================================================================

# Human-readable names for the four BraTS 2024 tumour sub-regions.
LABEL_NAMES    = {1: "NETC", 2: "SNFH", 3: "ET", 4: "RC"}

# Sub-region label IDs we actually want to segment.
TARGET_LABELS  = [1, 2, 3, 4]

# Paint order for the priority-merge step (lowest priority first, so later
# labels overwrite earlier ones).  SNFH (2) is painted first so that the
# smaller, more clinically critical ET (3) and RC (4) are never occluded.
MERGE_PRIORITY = [2, 3, 4, 1]

# Label sets for the two standard BraTS composite regions.
WT_LABELS = [1, 2, 3, 4]   # Whole Tumour  — union of all sub-regions
TC_LABELS = [1, 3, 4]       # Tumour Core   — excludes the oedema ring (SNFH)

# The five percentile levels used for the penta-anchor strategy.
# Spread across the volume distribution to capture thin/thick ends plus median.

ANCHOR_PERCENTILES  = [5, 25, 50, 75, 95]
# ANCHOR_PERCENTILES  = [1, 5, 10, 15, 20, 25, 30, 35, 40, 45, 50, 55, 60, 65, 70, 75, 80, 85, 90, 95]



# Labels whose occupied-slice count falls below this threshold switch from
# area-weighted percentile to geometric spacing to prevent anchor collapse
# on small, localised structures (e.g. tiny NETC necrotic cores).
COMPACT_SLICE_LIMIT = 10

# Logit threshold for converting SAM2 float scores to a binary mask.
# 0.0 keeps everything above the decision boundary without extra tuning.
MASK_THRESHOLD  = 0


# =============================================================================
# PHASE 00 — Environment bootstrap
# Adds the MedSAM2 source tree to sys.path and installs portalocker, which
# is required by the SAM2 data-loader but is not always pre-installed.
# =============================================================================

if MEDSAM2_PATH not in sys.path:
    sys.path.insert(0, MEDSAM2_PATH)
    print(f"[INIT] Added MedSAM2 to sys.path: {MEDSAM2_PATH}")

import subprocess
subprocess.run(
    [sys.executable, "-m", "pip", "install", "portalocker==2.7.0", "-q"],
    check=False,
)

try:
    # Verify the sam2 package is importable before any GPU work starts.
    print("[INIT] sam2.build_sam import OK")
except ImportError as e:
    raise ImportError(
        f"Could not import sam2.\n"
        f"Expected path: {MEDSAM2_PATH}\n"
        f"Run: cd MedSAM2 && pip install -e \".[dev]\"\n"
        f"Error: {e}"
    )


# =============================================================================
# UTILITIES
# Small helpers used throughout the pipeline.
# =============================================================================

def dprint(*args, **kwargs):
    """Print only when DEBUG=True — zero-cost in production runs."""
    if DEBUG:
        print("[DEBUG]", *args, **kwargs)


def clear_dir(path):
    """Delete a directory tree and recreate it empty."""
    if os.path.exists(path):
        shutil.rmtree(path)
    os.makedirs(path)


def normalize_percentile_uint8(vol):
    """
    Robust intensity normalisation: clip at the 0.5th / 99.5th percentile to
    suppress extreme outliers, then rescale to [0, 255] uint8.

    Using percentile clipping (rather than global min/max) prevents a single
    hot or cold voxel from washing out the entire dynamic range.
    """
    lo, hi = np.percentile(vol, 0.5), np.percentile(vol, 99.5)
    vol = np.clip(vol, lo, hi)
    if hi - lo < 1e-8:
        # Flat volume (e.g. missing modality fallback) — return zeros safely.
        return np.zeros(vol.shape, dtype=np.uint8)
    return ((vol - lo) / (hi - lo) * 255).astype(np.uint8)


# =============================================================================
# PHASE 01 — Data I/O
# Responsible for discovering and loading all NIfTI files for a patient.
#
# BraTS directory layout (one folder per patient):
#   <patient_id>/
#     <patient_id>-seg.nii.gz      — ground-truth label volume
#     <patient_id>-t1c.nii.gz      — T1 post-contrast (enhancing tumour)
#     <patient_id>-t2w.nii.gz      — T2-weighted (oedema, vasogenic signal)
#     <patient_id>-t2f.nii.gz      — T2 FLAIR (suppresses free water)
#     <patient_id>-t1n.nii.gz      — T1 non-contrast (optional, fallback)
#
# Fallback chain (Phase 01 → Phase 02 handoff):
#   If a modality file is missing we substitute the nearest alternative so
#   that RGB frame generation (Phase 02) can still proceed:
#     R = T1c  → T2w  → T1n  → T2f
#     G = T2w  → T1n  → T1c  → T2f
#     B = T2f  → T2w  → T1c
#   Patients where even the fallback chain cannot fill all three channels
#   are skipped with a [SKIP] message.
#
# The load_nifti helper (used here and in Phase 06) centralises nibabel I/O
# so all downstream code works with plain numpy float32 arrays.
# =============================================================================

def load_nifti(path):
    """
    Load a NIfTI file and return its data as float32 plus spatial metadata.

    The affine and header are threaded through to the output NIfTI save in
    Phase 05 so that predicted segmentations are in the same physical space
    as the original scans (essential for challenge submission and viewer use).

    Returns
    -------
    data   : np.ndarray  float32  (H, W, D)
    affine : np.ndarray  (4, 4)   voxel-to-world transform
    header : nib.Nifti1Header     original header (preserved for output)
    img    : nib.Nifti1Image      full nibabel object (rarely needed directly)
    """
    dprint(f"  Loading: {path}")
    img  = nib.load(path)
    data = img.get_fdata(dtype=np.float32)
    dprint(f"    shape={data.shape}  min={data.min():.1f}  max={data.max():.1f}")
    return data, img.affine, img.header, img


# =============================================================================
# PHASE 06 — Metrics
# Dice Similarity Coefficient and 95th-percentile Hausdorff Distance (HD95).
# Both metrics are standard in BraTS challenge evaluation.
# =============================================================================

def calculate_metrics(gt_bin, pred_bin):
    """
    Compute Dice and HD95 between two binary 3-D masks.

    Special-case handling:
      - Both empty  → perfect agreement: Dice=1, HD95=0.
      - One empty   → total failure:     Dice=0, HD95=373 mm
                      (373 ≈ diagonal of a 240×240×155 mm BraTS volume).

    HD95 is computed on surface voxels only (boundary extracted via erosion)
    to match the standard BraTS evaluation protocol.

    Parameters
    ----------
    gt_bin   : bool ndarray  (H, W, D)
    pred_bin : bool ndarray  (H, W, D)

    Returns
    -------
    dice : float  [0, 1]
    hd95 : float  mm
    """
    gt_bin   = gt_bin.astype(bool)
    pred_bin = pred_bin.astype(bool)

    # Edge case: both masks empty → trivially perfect.
    if not np.any(gt_bin) and not np.any(pred_bin):
        return 1.0, 0.0
    # Edge case: one mask empty → worst-case distance.
    if not np.any(gt_bin) or not np.any(pred_bin):
        return 0.0, 373.13

    # Dice: 2|A∩B| / (|A| + |B|)
    intersection = np.logical_and(gt_bin, pred_bin).sum()
    dice = (2.0 * intersection) / (gt_bin.sum() + pred_bin.sum())

    # HD95: build surface masks via morphological boundary extraction,
    # then compute bidirectional distance transform and take the 95th pct.
    gt_surface   = gt_bin   & ~binary_erosion(gt_bin)
    pred_surface = pred_bin & ~binary_erosion(pred_bin)

    # dt_pred[x] = distance from voxel x to the nearest predicted surface voxel.
    dt_pred = distance_transform_edt(~pred_bin)
    # dt_gt[x]   = distance from voxel x to the nearest GT surface voxel.
    dt_gt   = distance_transform_edt(~gt_bin)

    # Collect distances in both directions for symmetry.
    surf_distances = np.hstack([
        dt_pred[gt_surface],   # GT surface → nearest pred surface
        dt_gt[pred_surface],   # pred surface → nearest GT surface
    ])
    hd95 = float(np.percentile(surf_distances, 95))
    return float(dice), hd95


# =============================================================================
# PHASE 02 — RGB Frame Generation
# SAM2 is a video model that expects a folder of JPEG images.  We map the
# three most informative MRI modalities to the R/G/B channels so that the
# model receives multi-contrast information in a single pass:
#   R = T1c   (contrast-enhancing structures pop against background)
#   G = T2w   (oedema and vasogenic signal)
#   B = T2f   (FLAIR — suppresses free water, highlights pathology)
# Each channel is independently normalised before stacking.
# =============================================================================

def prepare_rgb_frames(t1c_vol, t2w_vol, t2f_vol, video_name):
    """
    Convert a 3-D multi-contrast MRI volume into a folder of per-slice
    RGB JPEG images suitable for the SAM2 video predictor.

    Files are named 00000.jpg … ZZZZZ.jpg (zero-padded, axial order).

    Parameters
    ----------
    t1c_vol, t2w_vol, t2f_vol : float32 ndarray  (H, W, D)
    video_name                 : str  — subfolder under TEMP_VIDEO_DIR
    """
    frame_dir = os.path.join(TEMP_VIDEO_DIR, video_name)
    os.makedirs(frame_dir, exist_ok=True)

    # Independently normalise each channel to [0, 255].
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


# =============================================================================
# PHASE 03 — Per-Label PENTA Anchor
# The "penta-anchor" strategy selects 5 axial slices per label that are used
# as GT prompt frames for the video predictor.  Good anchor selection matters
# because SAM2 propagates state from prompted frames outward — a poor choice
# (e.g. all anchors clustered near the top of the tumour) leaves the opposite
# end under-prompted and typically under-segmented.
# =============================================================================

def _compute_anchors_for_areas(areas, z_indices):
    """
    Compute 5 anchor z-indices from a per-slice area (voxel-count) array.

    Two strategies depending on how many slices are occupied:

    geometric  (< COMPACT_SLICE_LIMIT occupied slices):
        Linearly spaced indices over the occupied z-range.
        Prevents all 5 anchors collapsing to the same slice when a label
        spans only a handful of slices (e.g. a small necrotic core).

    weighted-pct  (>= COMPACT_SLICE_LIMIT occupied slices):
        Repeat each z-index proportionally to its slice area, then take
        percentiles of the resulting distribution.  This naturally places
        anchors where the tumour is large and spreads them thin where it
        is small, giving the predictor the most informative prompts.

    Parameters
    ----------
    areas     : int ndarray  (D,)  per-slice voxel count for this label
    z_indices : int ndarray  (D,)  corresponding axial slice indices

    Returns
    -------
    dict  {percentile: z_index}  — may be empty if no occupied slices.
    """
    occupied = z_indices[areas > 0]
    n_slices = len(occupied)

    if n_slices == 0:
        return {}

    if n_slices == 1:
        # Only one occupied slice — all 5 anchors point to it.
        return {p: int(occupied[0]) for p in ANCHOR_PERCENTILES}

    if n_slices < COMPACT_SLICE_LIMIT:
        # Geometric spacing: evenly sample at most 5 points from the z-range.
        indices = np.linspace(0, n_slices - 1, min(5, n_slices)).astype(int)
        sampled = occupied[indices]
        pct_z   = {}
        for i, p in enumerate(ANCHOR_PERCENTILES):
            pct_z[p] = int(sampled[min(i, len(sampled) - 1)])
        return pct_z

    # Standard weighted-percentile: build a z-index array where each z is
    # repeated proportionally to its area, then compute percentiles directly.
    weighted_z = np.repeat(z_indices, areas.astype(int))
    return {p: int(np.percentile(weighted_z, p)) for p in ANCHOR_PERCENTILES}


def find_penta_anchors_per_label(seg_data, present_labels):
    """
    Compute 5 anchor slices for every label present in the GT segmentation.

    Prints a summary table showing the selected z-indices, the number of
    unique anchors (after deduplication), the peak slice pixel count, and
    which anchor strategy was chosen.

    Parameters
    ----------
    seg_data       : uint8 ndarray  (H, W, D)  GT label volume
    present_labels : list[int]                  labels actually in this patient

    Returns
    -------
    anchors : dict
        { label_id: {
            'pct_z'    : {percentile: z},
            'peak_z'   : int,
            'unique'   : sorted list of unique z-values,
            'strategy' : 'geometric' | 'weighted-pct',
          } }
    """
    anchors   = {}
    D         = seg_data.shape[2]
    z_indices = np.arange(D)

    pct_labels = " ".join(f"p{p:02d}_z" for p in ANCHOR_PERCENTILES)
    print(f"\n  Per-label PENTA anchor slices ({' · '.join(f'P{p}' for p in ANCHOR_PERCENTILES)}):")
    print(f"  {'Label':<20} {pct_labels}   {'Unique':>20}  {'Peak px':>10}  {'Strategy':>12}")
    print(f"  {'-'*100}")

    for label_id in present_labels:
        # Count how many GT voxels of this label exist in every axial slice.
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

    # Debug matrix: for each pair of labels, how many pixels does label A have
    # at label B's peak slice?  Useful for spotting label co-occurrence.
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


# =============================================================================
# PHASE 03W / 03T — Composite Anchor for WT and TC
# Same anchor computation as the per-label version, but applied to the union
# of multiple labels at once.  WT and TC are tracked independently so their
# predictions are not constrained by per-label boundary decisions.
# =============================================================================

def find_composite_anchors(seg_data, label_set, track_name):
    """
    Compute 5 penta-anchor slices for the UNION of a set of labels.

    This is used to build the WT (Whole Tumour) and TC (Tumour Core) tracks.
    The combined binary mask is treated exactly like a single-label mask for
    anchor selection.

    Parameters
    ----------
    seg_data   : uint8 ndarray  (H, W, D)
    label_set  : list[int]  labels to combine into one binary mask
    track_name : str        human-readable name for logging ('WT' or 'TC')

    Returns
    -------
    dict with 'pct_z', 'peak_z', 'unique', 'strategy', 'mask_3d'
    or None if no voxels of label_set are found.
    """
    D         = seg_data.shape[2]
    z_indices = np.arange(D)

    # Build combined binary mask and per-slice area counts.
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
        'mask_3d'  : combined_bin,   # full 3-D GT mask (not used in inference)
    }


# =============================================================================
# PHASE 04 — MedSAM2 Predictor (singleton)
# The predictor is expensive to load (~1-2 GB of GPU memory).  A module-level
# singleton ensures it is built only once regardless of how many patients
# are processed in a single run.
# =============================================================================

_predictor = None


def get_predictor():
    """
    Return the MedSAM2 video predictor, loading it on first call.

    Raises FileNotFoundError if the checkpoint or config are missing.
    Model is set to eval() and never returned to train mode during inference.
    """
    global _predictor
    if _predictor is not None:
        return _predictor

    print("\n  [MODEL] Loading MedSAM2 predictor...")

    # Resolve absolute paths for existence checks, then pass the Hydra-style
    # relative config key (cfg_hydra) to build_sam2_video_predictor which
    # resolves it inside the sam2 package.
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
        apply_postprocessing=False,   # keep raw logits; we threshold manually
    )
    _predictor.eval()
    print("  [MODEL] Predictor loaded")
    return _predictor


# =============================================================================
# PHASE 04 — SHARED ENCODER: all tracks in one init_state session
# =============================================================================

@torch.inference_mode()
@torch.autocast(device_type="cuda", dtype=torch.bfloat16)
def run_all_tracks_shared_encoder(video_name, track_configs, shape):
    """
    Segment ALL tracks (per-label + WT + TC) within a SINGLE init_state call.

    Shared-encoder design
    ---------------------
    predictor.init_state(video_path=...) encodes every JPEG frame in the
    folder into feature tensors and caches them in GPU memory.  This is by
    far the most expensive operation (~90% of runtime for large volumes).

    For each subsequent track we call predictor.reset_state(inference_state):
      - CLEARS  the memory bank (per-object propagation state).
      - RETAINS the cached per-frame image features.

    We then add fresh GT anchor masks and run bidirectional propagation.
    Result: one expensive encode shared across all N tracks instead of N
    separate encodes.

    Bidirectional propagation
    -------------------------
    SAM2 propagates state from a starting frame in one direction.  Running
    both forward (start → end) and backward (end → start) and taking the
    union ensures the entire volume is covered regardless of where the
    anchors happen to lie.  The backward pass only fills frames not already
    covered by the forward pass (to avoid double-writing logits).

    Parameters
    ----------
    video_name    : str
        Subfolder in TEMP_VIDEO_DIR containing per-slice JPEGs.
    track_configs : dict  { track_name: {z: bool ndarray (H,W)} }
        Maps each track to its anchor GT masks keyed by axial slice index.
    shape         : (H, W, D)
        Spatial dimensions of the volume.

    Returns
    -------
    dict  { track_name: bool ndarray (H, W, D) }
        Binary 3-D segmentation mask for every requested track.
        Tracks that fail return an all-False volume.
    """
    H, W, D   = shape
    predictor = get_predictor()
    video_dir = os.path.join(TEMP_VIDEO_DIR, video_name)
    OBJ_ID    = 1  # SAM2 object ID; single-object per track so always 1.

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

    # ── Single init_state: encodes ALL frames once ───────────────────────────
    try:
        inference_state = predictor.init_state(
            video_path=video_dir,
            async_loading_frames=False,  # synchronous load avoids race conditions
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

        # ── Reset: clears memory bank, keeps frame feature cache ─────────────
        predictor.reset_state(inference_state)

        def _add_prompts():
            """Register every anchor GT mask with the predictor for OBJ_ID."""
            for z, mask_hw in anchor_masks_by_z.items():
                predictor.add_new_mask(
                    inference_state=inference_state,
                    frame_idx=z,
                    obj_id=OBJ_ID,
                    mask=mask_hw,
                )

        # ── Forward pass: start from the lowest anchor, go to last frame ─────
        _add_prompts()
        fwd_start     = min(unique_anchors)
        output_scores = {}   # z → raw logit array (H, W)

        dprint(f"  [INFER-{track_name}] Forward  z={fwd_start} → z={n_frames - 1}")
        for out_frame_idx, out_obj_ids, out_mask_logits in predictor.propagate_in_video(
            inference_state, start_frame_idx=fwd_start, reverse=False,
        ):
            for i, oid in enumerate(out_obj_ids):
                if oid == OBJ_ID:
                    output_scores[out_frame_idx] = out_mask_logits[i][0].cpu().numpy()

        dprint(f"  [INFER-{track_name}] Forward done — {len(output_scores)} frames")

        # ── Backward pass: clear memory, re-prompt, go from top to start ─────
        # Only frames not already covered by the forward pass are added,
        # preventing logit overwriting for slices near the anchors.
        predictor.reset_state(inference_state)
        _add_prompts()
        bwd_start = max(unique_anchors)

        dprint(f"  [INFER-{track_name}] Backward z={bwd_start} → z=0")
        for out_frame_idx, out_obj_ids, out_mask_logits in predictor.propagate_in_video(
            inference_state, start_frame_idx=bwd_start, reverse=True,
        ):
            for i, oid in enumerate(out_obj_ids):
                # Only store if this frame was not already covered forward.
                if oid == OBJ_ID and out_frame_idx not in output_scores:
                    output_scores[out_frame_idx] = out_mask_logits[i][0].cpu().numpy()

        dprint(f"  [INFER-{track_name}] Backward done — {len(output_scores)}/{n_frames} frames total")

        # ── Threshold logits → boolean 3-D mask ──────────────────────────────
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
        torch.cuda.empty_cache()   # release intermediate tensors between tracks

    # ── Final cleanup: release inference state before moving to next patient ─
    predictor.reset_state(inference_state)
    gc.collect()
    return results


# =============================================================================
# PHASE 05 — Multi-Label Merge with Priority
# Individual per-label binary masks can overlap (e.g. ET inside SNFH).
# We resolve conflicts by painting labels in MERGE_PRIORITY order:
# the last label painted "wins" at any contested voxel.
# Priority (low → high): SNFH(2) < NETC(1) < RC(4) < ET(3)
# This matches the BraTS biological nesting hierarchy.
# =============================================================================

def merge_masks_to_label_map(binary_masks, shape):
    """
    Combine per-label binary masks into a single integer label map.

    Parameters
    ----------
    binary_masks : dict  { label_id: bool ndarray (H, W, D) }
    shape        : (H, W, D)

    Returns
    -------
    label_map : uint8 ndarray (H, W, D)
        Background = 0; tumour sub-regions = 1–4 per LABEL_NAMES.
    """
    print("\n  Merging binary masks → label map...")
    label_map = np.zeros(shape, dtype=np.uint8)

    # Paint in ascending priority; later labels overwrite earlier ones.
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


# =============================================================================
# VISUALISATION — 5-row anchor figure
# One row per percentile anchor (P05·P25·P50·P75·P95).
# Each row shows three panels: GT contour, prediction contour, and
# a colour-coded overlap map (TP=yellow, FN=green, FP=red).
# =============================================================================

def show_visualization(bg_vol, gt_bin, pred_bin,
                       patient_id, label_id, dice_raw, hd95_raw,
                       dice_merged, hd95_merged, anchor_info, vis_dir,
                       track_label=None):
    """
    Save a 5-row diagnostic figure for a single track and patient.

    Each row corresponds to one of the 5 percentile anchor slices.
    The ★ marker identifies the peak (largest) anchor slice.
    Supports both per-label tracks (label_id set) and composite tracks
    (track_label = 'WT' or 'TC').

    Parameters
    ----------
    bg_vol        : float ndarray (H,W,D)  greyscale background (average MRI)
    gt_bin        : bool ndarray  (H,W,D)  ground-truth binary mask
    pred_bin      : bool ndarray  (H,W,D)  predicted binary mask
    patient_id    : str
    label_id      : int  (used for file naming even for composite tracks)
    dice_raw      : float  Dice before merge
    hd95_raw      : float  HD95 before merge
    dice_merged   : float  Dice after merge
    hd95_merged   : float  HD95 after merge
    anchor_info   : dict   from find_penta_anchors_per_label or find_composite_anchors
    vis_dir       : str    output directory for PNG files
    track_label   : str | None   if set, overrides label name ('WT' or 'TC')
    """
    os.makedirs(vis_dir, exist_ok=True)

    D          = bg_vol.shape[2]
    label_name = track_label if track_label else LABEL_NAMES[label_id]
    pct_z      = anchor_info['pct_z']
    peak_z     = anchor_info['peak_z']
    unique     = anchor_info['unique']
    strategy   = anchor_info.get('strategy', 'weighted-pct')

    print(f"  [VIZ] {label_name} | anchors={unique} | strategy={strategy}")

    def _draw_row(fig, gs, row_offset, z, row_label):
        """Render one row of three panels for axial slice z."""
        bg       = bg_vol[:, :, z]
        gt_slice = gt_bin[:, :, z].astype(bool)
        pr_slice = pred_bin[:, :, z].astype(bool)

        # Decompose prediction into TP / FN / FP for colour-coded overlay.
        tp = gt_slice & pr_slice    # correct voxels (yellow)
        fn = gt_slice & ~pr_slice   # missed GT voxels (green)
        fp = ~gt_slice & pr_slice   # extra predicted voxels (red)

        # Normalise background to [0,1] for imshow.
        b_min, b_max = np.percentile(bg, 1), np.percentile(bg, 99)
        bg_norm = np.clip(
            (bg.astype(float) - b_min) / max(b_max - b_min, 1e-8), 0, 1
        )

        # Build RGBA overlay for the overlap panel.
        overlay = np.zeros((*bg.shape, 4), dtype=float)
        overlay[tp] = [1.0, 1.0, 0.0, 0.55]   # yellow
        overlay[fn] = [0.0, 1.0, 0.0, 0.55]   # green
        overlay[fp] = [1.0, 0.0, 0.0, 0.55]   # red

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
                # Semi-transparent fill inside the contour.
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

    # Figure: 5 anchor rows + 1 legend row; 3 columns.
    n_rows = len(ANCHOR_PERCENTILES)

    fig = plt.figure(figsize=(24, 6 * n_rows + 2), facecolor='#1a1a2e')
    gs  = GridSpec(
        n_rows + 1, 3,          # +1 for the legend row
        figure=fig,
        height_ratios=[10] * n_rows + [1],   # dynamic height ratios
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

    # One row per percentile; mark the peak slice with ★.
    for row_idx, pct in enumerate(ANCHOR_PERCENTILES):
        z         = pct_z[pct]
        star      = " ★" if z == peak_z else ""
        row_label = f"P{pct:02d} ANCHOR (z={z}){star}"
        _draw_row(fig, gs, row_idx, z, row_label)

    # Legend row.
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

    # File name encodes patient, label, and all 5 anchor z-positions for
    # easy identification without opening the image.
    z_vals   = [pct_z[p] for p in ANCHOR_PERCENTILES]
    pct_tag  = f"z{min(z_vals)}-{max(z_vals)}_n{len(set(z_vals))}"
    fname    = f"{patient_id}_{label_name}_{pct_tag}.png"
    out_path = os.path.join(vis_dir, fname)
    plt.savefig(out_path, dpi=100, bbox_inches='tight', facecolor=fig.get_facecolor())
    dprint(f"  [VIZ] Saved: {out_path}")

    if SHOW_PLOTS:
        plt.show(block=True)
    plt.close(fig)


# =============================================================================
# Per-patient pipeline
# Orchestrates all phases for a single BraTS patient directory.
# =============================================================================

def process_patient(patient_id, p_path, writer, results_accumulator):
    """
    Full segmentation pipeline for one patient.

    Steps (mirrors global phase numbering)
    ----------------------------------------
    Phase 01 – Discover NIfTI files by suffix; load seg + up to 4 modalities;
               apply per-channel fallback chains; validate label presence.
    Phase 02 – Write per-slice RGB JPEG frames (T1c=R, T2w=G, T2f=B).
    Phase 03 – Compute penta-anchors per label and for WT / TC composites.
    Phase 04 – Run shared-encoder inference for all tracks.
    Phase 05 – Merge per-label predictions into a label map; save NIfTI.
    Phase 06 – Compute Dice / HD95 (raw + merged); log CSV; save figures.
               Clean up per-patient temp frames after NIfTI save.

    Parameters
    ----------
    patient_id          : str
    p_path              : str  absolute path to patient directory
    writer              : csv.writer  open for the session CSV
    results_accumulator : list  appended with per-track metric dicts

    Returns
    -------
    list of metric dicts (one per track), or [] on skip / fatal error.
    """
    print(f"\n{'='*60}")
    print(f"  PATIENT: {patient_id}")
    print(f"{'='*60}")
    t0 = time.time()

    # ── Phase 01: file discovery and NIfTI loading ───────────────────────────
    all_files = os.listdir(p_path)
    dprint(f"  Files: {all_files}")

    # Strict suffix matching: BraTS filenames end with '-<modality>.nii.gz'.
    # The four guards below prevent false matches when a modality string
    # appears inside the patient ID prefix itself (e.g. a folder whose name
    # contains the substring "seg" or "t1c").
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
    t1n_path = find_file('t1n')   # non-contrast T1; kept as fallback for G channel

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
        """Load NIfTI data only if path exists; else return None."""
        return load_nifti(path)[0] if path and os.path.exists(path) else None

    # Load all four modalities independently so the fallback chain below
    # sees genuine None values rather than aliases to another channel.
    # (Independent loading is what prevents the circular-assignment bug
    # that appeared in earlier versions of this pipeline.)
    t1c_raw = _load(t1c_path)
    t2w_raw = _load(t2w_path)
    t2f_raw = _load(t2f_path)
    t1n_raw = _load(t1n_path)

    def _first(*args):
        """Return the first non-None argument."""
        return next((a for a in args if a is not None), None)

    # R channel: prefer T1c (best for enhancing tumour) → T2w → T1n → T2f
    t1c = _first(t1c_raw, t2w_raw, t1n_raw, t2f_raw)
    # G channel: prefer T2w → T1n (structurally similar) → T1c → T2f
    t2w = _first(t2w_raw, t1n_raw, t1c_raw, t2f_raw)
    # B channel: prefer T2f (FLAIR) → T2w → T1c
    t2f = _first(t2f_raw, t2w_raw, t1c_raw)

    if t1c is None or t2w is None or t2f is None:
        print(f"  [SKIP] Cannot construct RGB channels for {patient_id}")
        return []

    if t2w_raw is None:
        print("  [FALLBACK] T2w missing → using T1n for G channel")
    if t2f_raw is None:
        print("  [FALLBACK] T2f missing → using T2w for B channel")

    # ── Phase 02: write per-slice JPEG frames ────────────────────────────────
    video_name = f"{patient_id}_rgb"
    clear_dir(os.path.join(TEMP_VIDEO_DIR, video_name))
    prepare_rgb_frames(t1c, t2w, t2f, video_name)

    # ── Phase 03: per-label penta-anchors ────────────────────────────────────
    anchors = find_penta_anchors_per_label(seg_data, present_labels)
    if not anchors:
        print(f"  [SKIP] No anchors found for {patient_id}")
        return []

    # Guard: skip labels whose peak-slice GT mask is empty (degenerate case).
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

    # ── Phase 03W / 03T: composite anchors for WT and TC ─────────────────────
    wt_present = [L for L in WT_LABELS if L in present_labels]
    tc_present = [L for L in TC_LABELS if L in present_labels]

    print(f"\n  Composite anchors:")
    wt_info = find_composite_anchors(seg_data, wt_present, "WT") if wt_present else None
    tc_info = find_composite_anchors(seg_data, tc_present, "TC") if tc_present else None

    # ── Phase 04: build all track configs and run shared-encoder inference ───
    print(f"\n  Building track configs for shared-encoder inference...")
    track_configs = {}

    # Per-label tracks: prompt mask = GT slice for that label at each anchor z.
    for label_id, info in valid_anchors.items():
        track_name        = LABEL_NAMES[label_id]
        anchor_masks_by_z = {
            z: (seg_data[:, :, z] == label_id).astype(bool)
            for z in info['unique']
        }
        track_configs[track_name] = anchor_masks_by_z
        print(f"    {track_name}: anchors={info['unique']}  [{info.get('strategy','?')}]  peak_px="
              f"{int(np.sum(seg_data[:, :, info['peak_z']] == label_id))}")

    # WT track: prompt mask = union of all WT labels at each anchor z.
    if wt_info is not None:
        wt_masks = {
            z: np.isin(seg_data[:, :, z], wt_present).astype(bool)
            for z in wt_info['unique']
        }
        track_configs["WT"] = wt_masks
        print(f"    WT: anchors={wt_info['unique']}  [{wt_info.get('strategy','?')}]")

    # TC track: prompt mask = union of TC labels at each anchor z.
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

    # Unpack per-label results into a {label_id: mask} dict for the merge step.
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

    # ── Phase 05: merge → NIfTI ───────────────────────────────────────────────
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

    # ── Temp dir cleanup ──────────────────────────────────────────────────────
    # Each patient produces ~155 JPEGs × ~200 KB = ~31 MB of temp data.
    # Deleting immediately after NIfTI save keeps total disk usage under
    # ~31 MB at any point instead of accumulating ~1.5 GB over 50 patients.
    patient_video_dir = os.path.join(TEMP_VIDEO_DIR, video_name)
    if os.path.exists(patient_video_dir):
        shutil.rmtree(patient_video_dir)
        dprint(f"  Cleaned temp frames: {patient_video_dir}")

    # ── Phase 06: metrics ─────────────────────────────────────────────────────
    # Background image for visualisation: average of the three normalised
    # modality channels gives a balanced greyscale reference.
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

    # ── Per-label metrics ─────────────────────────────────────────────────────
    for label_id in present_labels:
        if label_id not in binary_masks:
            print(f"  [SKIP] No output mask for label {label_id}")
            continue

        label_name   = LABEL_NAMES[label_id]
        info         = valid_anchors[label_id]
        pred_raw     = binary_masks[label_id]       # direct predictor output
        pred_merged  = (pred_label_map == label_id) # after priority-merge
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

    # ── WT metrics ────────────────────────────────────────────────────────────
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

    # ── TC metrics ────────────────────────────────────────────────────────────
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


# =============================================================================
# Main loop + final summary
# Iterates over the patient list, calls process_patient(), and prints
# aggregated Dice / HD95 statistics broken down by label and anchor strategy.
# =============================================================================

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

    # Timestamp the CSV so re-runs never overwrite previous results.
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
                f.flush()   # persist rows immediately in case of later crash
                results_accumulator.extend(rows)
                if not rows:
                    failed.append(p_id)
            except Exception as e:
                print(f"  [FATAL] {p_id}: {e}")
                import traceback; traceback.print_exc()
                failed.append(p_id)

    # ── Final summary ─────────────────────────────────────────────────────────
    print(f"\n\n{'#'*60}")
    print(f"  FINAL SUMMARY  —  Penta-Anchor ({pct_str})")
    print(f"{'#'*60}")
    print(f"  Processed:       {len(all_patients)} patients")
    print(f"  Failed/skipped:  {len(failed)}  {failed[:10]}")
    print(f"  Results CSV:     {csv_path}\n")

    if results_accumulator:

        # ── Per-label aggregate statistics (raw and post-merge) ───────────────
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

        # ── WT / TC aggregate statistics ──────────────────────────────────────
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

        # ── Per-label anchor strategy breakdown ───────────────────────────────
        # Shows how often the geometric vs weighted-pct strategy was invoked,
        # which is useful for understanding the distribution of small labels
        # (e.g. if NETC is almost always "geometric" the COMPACT_SLICE_LIMIT
        # may need adjustment).
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
