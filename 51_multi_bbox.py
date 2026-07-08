"""
medsam2_brats_pipeline_hitl.py
===============================
End-to-end BraTS 2024 segmentation pipeline using MedSAM2
with a **Human-in-the-Loop (HITL) iterative anchor refinement** loop.

Overview
--------
Each patient volume is processed in 7 logical phases, plus a new
iterative HITL refinement phase that runs between initial inference
and the final merge:

  Phase 00 – Bootstrap:    add MedSAM2 to sys.path, install portalocker.
  Phase 01 – Data I/O:     discover NIfTI files by suffix, load all modalities
                            (seg + T1c/T2w/T2f/T1n), apply fallback chains for
                            missing modalities, validate label presence.
  Phase 02 – RGB Frames:   stack T1c/T2w/T2f into per-slice JPEG triplets
                            that the SAM2 video predictor treats as a "video".
                            Normalization is performed PER SLICE independently.
                            Returns cached uint8 RGB array (H,W,D,3) so that
                            the background volume and histogram re-use the same
                            normalised values without extra computation.
  Phase 03 – Anchors:      seed with the single densest slice per label/track
                            (peak-slice only — one anchor to start).
  Phase 04 – HITL Loop:    iterative refinement:
                              Iteration 0 : single peak-slice anchor -> infer -> Dice.
                              Iteration 1+: if Dice improvement < HITL_MIN_IMPROVEMENT
                                            or Dice is already >= HITL_DICE_TARGET -> stop.
                                            Otherwise add the slice with the largest
                                            per-slice error (max |GT area - pred area|)
                                            and re-infer.
                              Hard cap at HITL_MAX_ITERATIONS total iterations.
  Phase 05 – Merge:        paint per-label binary masks onto a single label
                            map using a fixed priority order.
  Phase 06 – Metrics:      compute Dice and HD95 (raw and post-merge) and
                            write results to a CSV.
  Visualisation:           save a figure showing GT, prediction, and overlap
                            at every anchor slice used in the FINAL iteration.

NEW FEATURES
------------
  1. MULTI-BBOX PROMPTING
       When a GT mask slice contains disconnected (separated) components,
       the pipeline now detects each connected component via scipy.ndimage.label
       and issues one bounding-box prompt per component instead of a single
       box enclosing everything.  A minimum-area threshold
       (BBOX_MIN_COMPONENT_AREA) suppresses tiny noise islands.

       The SAM2 API receives each box as a separate add_new_points_or_box()
       call on the same frame before propagation.

  2. BBOX INPUT VISUALISATION
       For every anchor slice (in both the per-round HITL figures and the
       final summary figure) a 5th panel is now saved alongside the existing
       4 columns:
           Col 4 — "Bbox Input" : greyscale background with each component
                   bounding box drawn as a coloured rectangle, labelled with
                   its component index and pixel area.
       The function save_bbox_input_image() can also be called standalone to
       write a dedicated PNG for every anchor slice x every track x every round
       into visualizations/<id>/bbox_inputs/.

OPTIMIZATIONS (logic/results unchanged)
-----------------------------------------
  1. prepare_rgb_frames_cached()
       Computes _norm_slice_uint8 ONCE per slice per modality, writes JPEGs,
       AND returns the uint8 RGB ndarray (H,W,D,3).  The background volume
       and histogram both consume this cache — no redundant normalisation.

  2. _render_segmentation_panel()
       Single axes-drawing helper shared by save_hitl_round_slice_figure()
       and show_visualization().  Eliminates the ~80% code duplication
       between those two functions.

  3. save_normalization_histogram() accepts rgb_cache
       Histogram rows are derived from the already-computed cache rather
       than re-running _norm_slice_uint8 on the raw volumes a second time.
       Raw / clipped stages are reconstructed cheaply from the raw volumes
       only when needed for the diagnostic rows.

  4. Peak-slice passed into run_all_tracks_hitl()
       find_peak_slice() is called once in Phase 03 and the result is
       forwarded into the HITL runner, eliminating the duplicate argmax
       inside run_all_tracks_hitl().

Human-in-the-Loop (HITL) design
---------------------------------
The pipeline starts from the *minimum viable prompt* — a single GT mask at
the slice with the most foreground pixels (peak slice).  After inference it
checks whether the volumetric Dice meets the target or if adding more anchors
is still improving results:

  stop if:
    * Dice >= HITL_DICE_TARGET                     (good enough)
    * Dice improvement D < HITL_MIN_IMPROVEMENT    (adding anchors stopped helping)
    * total anchor count == HITL_MAX_ITERATIONS    (budget exhausted)

  otherwise:
    * Identify the axial slice with the greatest segmentation error.
      Error metric: |GT_pixels(z) - pred_pixels(z)| (combined FP + FN).
    * Add that slice's GT mask as an additional anchor.
    * Re-run inference (shared encoder, reset only the memory bank).

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

Both are tracked INDEPENDENTLY (not derived from per-label outputs).

Shared-encoder optimisation
----------------------------
SAM2 video inference re-encodes the entire volume on each init_state call.
By calling init_state ONCE and using reset_state between tracks (and between
HITL iterations for the same track) we keep the cached frame features and
avoid redundant encodes.
Typical wall-clock speedup: ~3-4x vs naive per-track init_state.

Propagation pattern (per HITL iteration, per track)
-----------------------------------------------------
  1. reset_state            : clear memory bank (frame features stay cached).
  2. add_new_points_or_box  : register all currently accumulated GT bboxes
                              (one call per connected component per anchor slice).
  3. Forward pass           : propagate from lowest anchor z -> last frame.
  4. reset_state + re-add   : reset memory bank again.
  5. Backward pass          : propagate from highest anchor z -> first frame.
  6. Union of logits        : binary 3-D mask.
  7. Compute Dice vs GT.
  8. Decide: stop or add next anchor.

Per-round slice figure layout (5 columns per anchor row)
---------------------------------------------------------
  Col 0 — GT mask only          : cyan contour + fill on greyscale background
  Col 1 — Prediction only       : magenta contour + fill on greyscale background
  Col 2 — GT + Pred combined    : both contours overlaid together
  Col 3 — Delta map             : TP=yellow  FN=green  FP=red  on greyscale
  Col 4 — Bbox Input            : coloured rectangles per connected component

Normalization histogram layout (3 columns x 3 rows, linear Y axis)
-------------------------------------------------------------------
  Columns : T1c (R)  |  T2w (G)  |  T2f/FLAIR (B)
  Row 0   : Raw float values  (dashed lines = mean p0.5/p99.5 across slices;
            shaded bands = per-slice variability of clip boundaries)
  Row 1   : After per-slice percentile clipping
  Row 2   : After per-slice rescaling to uint8 [0..255]
  Y axis  : linear voxel count (not log-scale)

  NOTE: All three histogram rows reflect PER-SLICE normalization — each slice
  is normalised independently using its own p0.5/p99.5 clip boundaries.
  The histograms aggregate voxel distributions across all slices.

Output files (per patient)
--------------------------
  segs_nifti/<id>_pred.nii.gz                  - merged multi-label prediction
  segs_nifti/<id>_pred_WT.nii.gz               - independent WT binary mask
  segs_nifti/<id>_pred_TC.nii.gz               - independent TC binary mask
  visualizations/<id>/                          - final-anchor figures + histogram
  visualizations/<id>/hitl_rounds/              - per-round 5-column slice figures
  visualizations/<id>/bbox_inputs/              - standalone bbox prompt images
  results_hitl_<timestamp>.csv                  - Dice + HD95 + HITL audit trail
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
from scipy.ndimage import (
    distance_transform_edt,
    binary_erosion,
    label as nd_label,          # connected-component labelling
)
import torch
import matplotlib
matplotlib.use('Agg')   # headless-safe backend
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
from matplotlib.gridspec import GridSpec
from matplotlib.patches import Patch
import matplotlib.colors as mcolors
from MedSAM2.sam2.build_sam import build_sam2_video_predictor

warnings.filterwarnings("ignore")


# =============================================================================
# CONFIGURATION
# =============================================================================

MEDSAM2_PATH    = os.path.abspath("./MedSAM2")
SAM2_CHECKPOINT = "./checkpoints/MedSAM2_latest.pt"
SAM2_CFG        = "sam2/configs/sam2.1_hiera_t512.yaml"
DATASET_DIR     = r"./2_BraTS2024_dataset/training_data1_v2"
DATASET_DIR     = r"./One_Patient_test_MRI_DATA"   # single-patient smoke test
OUTPUT_BASE_DIR = os.path.abspath("./medsam2_results")
TEMP_VIDEO_DIR  = os.path.abspath("./temp_video_frames")
NUM_PATIENTS    = 1
DEBUG           = True
SHOW_PLOTS      = False


# =============================================================================
# HUMAN-IN-THE-LOOP (HITL) KNOBS
# =============================================================================
HITL_MAX_ITERATIONS  = 7
HITL_MIN_IMPROVEMENT = 0.005
HITL_DICE_TARGET     = 0.95


# =============================================================================
# MULTI-BBOX KNOBS  (NEW)
# =============================================================================

# Minimum number of foreground pixels a connected component must have to
# receive its own bounding-box prompt.  Smaller blobs are treated as noise
# and suppressed.  Tune this to your resolution / label size.
BBOX_MIN_COMPONENT_AREA = 50

# Colour palette used when drawing multiple bboxes on the same slice.
# Cycles if there are more components than colours.
BBOX_PALETTE = [
    "#FF4444",   # red
    "#44FF44",   # green
    "#4488FF",   # blue
    "#FFAA00",   # orange
    "#FF44FF",   # magenta
    "#00FFFF",   # cyan
    "#FFFF44",   # yellow
    "#FF8844",   # salmon
]


# =============================================================================
# LABEL DEFINITIONS
# =============================================================================

LABEL_NAMES    = {1: "NETC", 2: "SNFH", 3: "ET", 4: "RC"}
TARGET_LABELS  = [1, 2, 3, 4]
MERGE_PRIORITY = [2, 3, 4, 1]
WT_LABELS      = [1, 2, 3, 4]
TC_LABELS      = [1, 3, 4]
MASK_THRESHOLD = 0
MIN_FOREGROUND_VOXELS_PER_SLICE = 100


# =============================================================================
# PHASE 00 - Environment bootstrap
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
    print("[INIT] sam2.build_sam import OK")
except ImportError as import_error:
    raise ImportError(
        f"Could not import sam2.\n"
        f"Expected path: {MEDSAM2_PATH}\n"
        f"Run: cd MedSAM2 && pip install -e \".[dev]\"\n"
        f"Error: {import_error}"
    )


# =============================================================================
# UTILITIES
# =============================================================================

def dprint(*args, **kwargs):
    if DEBUG:
        print("[DEBUG]", *args, **kwargs)


def clear_dir(directory_path):
    if os.path.exists(directory_path):
        shutil.rmtree(directory_path)
    os.makedirs(directory_path)


def _norm_slice_uint8(slice_2d):
    """
    Normalize a single 2-D MRI slice to [0, 255] uint8.

    Clip bounds (p0.5 / p99.5) are computed from the slice's own non-zero
    voxels only.  Falls back to the whole slice when all voxels are zero.

    Parameters
    ----------
    slice_2d : float ndarray (H, W)

    Returns
    -------
    scaled_uint8 : uint8 ndarray  (H, W)
    lo           : float   lower clip boundary (p0.5 of non-zero voxels)
    hi           : float   upper clip boundary (p99.5 of non-zero voxels)
    """
    s       = slice_2d.astype(np.float32)
    nonzero = s[s != 0]

    if nonzero.size < MIN_FOREGROUND_VOXELS_PER_SLICE:
        return np.zeros(s.shape, dtype=np.uint8), 0.0, 0.0

    lo  = float(np.percentile(nonzero, 0.5))
    hi  = float(np.percentile(nonzero, 99.5))
    rng = hi - lo
    if rng < 1e-8:
        return np.zeros(s.shape, dtype=np.uint8), lo, hi
    scaled = np.clip((s - lo) / rng * 255, 0, 255).astype(np.uint8)
    return scaled, lo, hi


# =============================================================================
# NEW FEATURE 1 — Connected-component bbox extraction
# =============================================================================

def get_component_bboxes(mask_2d):
    """
    Decompose a binary 2-D mask into its connected components and return
    one axis-aligned bounding box per component that is large enough
    (>= BBOX_MIN_COMPONENT_AREA pixels).

    Parameters
    ----------
    mask_2d : bool ndarray (H, W)

    Returns
    -------
    bboxes : list of (row_min, col_min, row_max, col_max)
             Empty list if the mask has no foreground.

    The bbox coordinates are INCLUSIVE on both ends and follow the
    (y0, x0, y1, x1) = (row_min, col_min, row_max, col_max) convention
    that matches SAM2's box_coords input after conversion to (x0, y0, x1, y1).
    """
    if not np.any(mask_2d):
        return []

    labelled_array, num_components = nd_label(mask_2d)
    bboxes = []

    for component_id in range(1, num_components + 1):
        component_mask = labelled_array == component_id
        area = int(component_mask.sum())

        if area < BBOX_MIN_COMPONENT_AREA:
            dprint(f"    Component {component_id}: area={area} < "
                   f"{BBOX_MIN_COMPONENT_AREA} → suppressed")
            continue

        rows = np.any(component_mask, axis=1)
        cols = np.any(component_mask, axis=0)
        row_min, row_max = int(np.argmax(rows)), int(len(rows) - 1 - np.argmax(rows[::-1]))
        col_min, col_max = int(np.argmax(cols)), int(len(cols) - 1 - np.argmax(cols[::-1]))

        bboxes.append((row_min, col_min, row_max, col_max))
        dprint(f"    Component {component_id}: area={area}  "
               f"bbox=({row_min},{col_min},{row_max},{col_max})")

    return bboxes


def bboxes_to_sam2_boxes(bboxes):
    """
    Convert (row_min, col_min, row_max, col_max) bbox list to SAM2's
    expected (x0, y0, x1, y1) = (col_min, row_min, col_max, row_max) format.

    Returns a float32 ndarray of shape (N, 4).
    """
    if not bboxes:
        return np.zeros((0, 4), dtype=np.float32)
    converted = []
    for (r0, c0, r1, c1) in bboxes:
        converted.append([c0, r0, c1, r1])   # x0, y0, x1, y1
    return np.array(converted, dtype=np.float32)


# =============================================================================
# NEW FEATURE 2 — Bbox input visualisation panel + standalone image saver
# =============================================================================

def _render_bbox_panel(ax, bg_norm, bboxes, title):
    """
    Draw one "Bbox Input" panel onto axes `ax`.

    Shows the greyscale background with each bounding box drawn as a
    coloured rectangle.  Each box is labelled with its component index
    and approximate pixel area (box area as proxy).

    Parameters
    ----------
    ax       : matplotlib Axes
    bg_norm  : float ndarray (H, W)  greyscale background in [0, 1]
    bboxes   : list of (row_min, col_min, row_max, col_max)
    title    : str
    """
    ax.imshow(bg_norm, cmap='gray', vmin=0, vmax=1, interpolation='bilinear')
    ax.set_facecolor('#0d0d1a')
    ax.axis('off')
    ax.set_title(title, color='#ffdd00', fontsize=8, pad=4)

    # Border style — gold to distinguish this column
    for sp in ax.spines.values():
        sp.set_edgecolor('#ffdd00')
        sp.set_linewidth(2)

    if not bboxes:
        ax.text(
            0.5, 0.5, "No bboxes\n(empty mask)",
            transform=ax.transAxes, ha='center', va='center',
            color='white', fontsize=9, alpha=0.7,
        )
        return

    for idx, (r0, c0, r1, c1) in enumerate(bboxes):
        color = BBOX_PALETTE[idx % len(BBOX_PALETTE)]
        box_h = r1 - r0
        box_w = c1 - c0
        rect  = mpatches.Rectangle(
            (c0, r0), box_w, box_h,
            linewidth=2, edgecolor=color, facecolor='none', alpha=0.9,
        )
        ax.add_patch(rect)

        # Label: component index + box area (pixels)
        box_area = box_h * box_w
        ax.text(
            c0 + 2, r0 + 2,
            f"C{idx + 1}\n{box_area:,}px",
            color=color, fontsize=7, va='top', ha='left',
            fontweight='bold',
            bbox=dict(facecolor='black', alpha=0.45, pad=1, edgecolor='none'),
        )

    # Legend summary
    n_boxes = len(bboxes)
    ax.text(
        0.01, 0.99,
        f"{n_boxes} bbox{'es' if n_boxes != 1 else ''}",
        transform=ax.transAxes, va='top', ha='left',
        color='#ffdd00', fontsize=8, fontweight='bold',
        bbox=dict(facecolor='black', alpha=0.5, pad=2, edgecolor='none'),
    )


def save_bbox_input_image(
    bg_norm,
    bboxes,
    z_index,
    track_name,
    round_index,
    patient_id,
    bbox_output_directory,
    is_new_anchor=False,
):
    """
    Save a standalone PNG showing the bbox prompts for one anchor slice.

    Useful for debugging and auditing which components were prompted at
    each HITL round.

    Parameters
    ----------
    bg_norm              : float ndarray (H, W)  greyscale in [0, 1]
    bboxes               : list of (row_min, col_min, row_max, col_max)
    z_index              : int   axial slice index
    track_name           : str
    round_index          : int
    patient_id           : str
    bbox_output_directory: str   target directory for the PNG
    is_new_anchor        : bool  whether this slice was just added this round
    """
    os.makedirs(bbox_output_directory, exist_ok=True)

    fig, ax = plt.subplots(1, 1, figsize=(6, 6), facecolor='#1a1a2e')

    new_tag   = " ★ NEW" if is_new_anchor else ""
    n_comp    = len(bboxes)
    panel_ttl = (
        f"Bbox Input  z={z_index}{new_tag}\n"
        f"{n_comp} component{'s' if n_comp != 1 else ''}  "
        f"(min_area={BBOX_MIN_COMPONENT_AREA})"
    )
    _render_bbox_panel(ax, bg_norm, bboxes, panel_ttl)

    fig.suptitle(
        f"Patient: {patient_id}  |  Track: {track_name}  |  "
        f"Round: {round_index}  |  z={z_index}",
        color='white', fontsize=10, fontweight='bold', y=0.99,
    )

    plt.tight_layout(rect=[0, 0, 1, 0.96])

    anchor_tag = "new" if is_new_anchor else "prior"
    output_path = os.path.join(
        bbox_output_directory,
        f"{patient_id}_{track_name}_r{round_index:02d}"
        f"_z{z_index:03d}_{anchor_tag}_bbox.png",
    )
    plt.savefig(
        output_path, dpi=90, bbox_inches='tight',
        facecolor=fig.get_facecolor(),
    )
    dprint(f"  [BBOX-VIZ] Saved: {output_path}")
    plt.close(fig)

    return output_path


# =============================================================================
# OPTIMIZATION 1 — prepare_rgb_frames_cached
# =============================================================================

def prepare_rgb_frames_cached(t1c_volume, t2w_volume, t2f_volume,
                               video_subfolder_name):
    """
    Convert multi-contrast MRI volume -> per-slice RGB JPEG folder.
    R = T1c, G = T2w, B = T2f.  Normalization is PER SLICE.

    Returns the uint8 RGB cache (H, W, D, 3).
    """
    frame_output_directory = os.path.join(TEMP_VIDEO_DIR, video_subfolder_name)
    os.makedirs(frame_output_directory, exist_ok=True)

    H, W, D = t1c_volume.shape
    rgb_cache = np.zeros((H, W, D, 3), dtype=np.uint8)

    dprint(f"  Saving {D} RGB JPEG frames (per-slice norm) -> "
           f"{frame_output_directory}")

    for z in range(D):
        r_slice, _, _ = _norm_slice_uint8(t1c_volume[:, :, z])
        g_slice, _, _ = _norm_slice_uint8(t2w_volume[:, :, z])
        b_slice, _, _ = _norm_slice_uint8(t2f_volume[:, :, z])

        rgb_cache[:, :, z, 0] = r_slice
        rgb_cache[:, :, z, 1] = g_slice
        rgb_cache[:, :, z, 2] = b_slice

        Image.fromarray(rgb_cache[:, :, z], mode="RGB").save(
            os.path.join(frame_output_directory, f"{z:05d}.jpg"),
            quality=95,
        )

    dprint(f"  RGB frames saved: {D} files")
    return rgb_cache


# =============================================================================
# OPTIMIZATION 2 — _render_segmentation_panel
# =============================================================================

def _render_segmentation_panel(ax, bg_norm, gt_mask, pred_mask, mode,
                                title, title_color, border_color,
                                border_width=2):
    """
    Draw one segmentation panel onto axes `ax`.

    mode : 'gt' | 'pred' | 'both' | 'delta'
    """
    ax.imshow(bg_norm, cmap='gray', vmin=0, vmax=1, interpolation='bilinear')
    ax.set_facecolor('#0d0d1a')
    ax.axis('off')
    ax.set_title(title, color=title_color, fontsize=8, pad=4)
    for sp in ax.spines.values():
        sp.set_edgecolor(border_color)
        sp.set_linewidth(border_width)

    def _contour_fill(mask, color):
        if not np.any(mask):
            return
        ax.contour(mask, colors=color, linewidths=2.0, alpha=0.9)
        fill = np.zeros((*mask.shape, 4), dtype=float)
        fill[mask] = [*mcolors.to_rgb(color), 0.22]
        ax.imshow(fill, interpolation='none')

    if mode in ('gt', 'both'):
        _contour_fill(gt_mask, 'cyan')
    if mode in ('pred', 'both'):
        _contour_fill(pred_mask, 'magenta')
    if mode == 'delta':
        tp = gt_mask & pred_mask
        fn = gt_mask & ~pred_mask
        fp = ~gt_mask & pred_mask
        overlay = np.zeros((*gt_mask.shape, 4), dtype=float)
        overlay[tp] = [1.0, 1.0, 0.0, 0.60]
        overlay[fn] = [0.0, 1.0, 0.0, 0.60]
        overlay[fp] = [1.0, 0.0, 0.0, 0.60]
        ax.imshow(overlay, interpolation='none')


def _make_bg_norm(background_volume, z):
    """
    Return a [0,1] float greyscale array for the given axial slice,
    clipped at the 1st/99th percentile of that slice.
    """
    sl  = background_volume[:, :, z]
    lo  = np.percentile(sl, 1)
    hi  = np.percentile(sl, 99)
    rng = max(hi - lo, 1e-8)
    return np.clip((sl.astype(float) - lo) / rng, 0, 1)


# =============================================================================
# OPTIMIZATION 3 — save_normalization_histogram (accepts rgb_cache)
# =============================================================================

def save_normalization_histogram(
    t1c_volume_raw,
    t2w_volume_raw,
    t2f_volume_raw,
    rgb_cache,
    patient_id,
    visualization_output_directory,
):
    """Save a 3-column x 3-row histogram figure (LINEAR Y axis)."""
    os.makedirs(visualization_output_directory, exist_ok=True)

    modality_triples = [
        (t1c_volume_raw, rgb_cache[:, :, :, 0], "T1c  (R channel)",       "#e05252"),
        (t2w_volume_raw, rgb_cache[:, :, :, 1], "T2w  (G channel)",       "#52b052"),
        (t2f_volume_raw, rgb_cache[:, :, :, 2], "T2f/FLAIR  (B channel)", "#5278e0"),
    ]

    number_of_histogram_bins = 128

    figure_object, axes_grid = plt.subplots(
        3, 3, figsize=(18, 13), facecolor='#1a1a2e',
    )
    figure_object.suptitle(
        f"Normalization Histogram  |  Patient: {patient_id}\n"
        f"PER-SLICE normalization  —  each slice normalised independently  "
        f"|  Background-only slices excluded from clip-boundary statistics  "
        f"(MIN_FOREGROUND_VOXELS_PER_SLICE={MIN_FOREGROUND_VOXELS_PER_SLICE})\n"
        f"Rows: Raw → Clipped [per-slice p0.5–p99.5] → Rescaled uint8 [0–255]  "
        f"|  Y axis: linear voxel count",
        color='white', fontsize=11, fontweight='bold', y=0.98,
    )

    for column_index, (raw_volume, uint8_channel, modality_label, bar_color) \
            in enumerate(modality_triples):

        _, _, num_axial_slices = raw_volume.shape

        all_raw_flat      = []
        all_clipped_flat  = []
        all_uint8_flat    = []
        per_slice_lo_list = []
        per_slice_hi_list = []
        number_of_background_slices = 0

        for z in range(num_axial_slices):
            s            = raw_volume[:, :, z].astype(np.float32)
            nonzero_mask = s != 0
            nonzero_vals = s[nonzero_mask]

            if nonzero_vals.size < MIN_FOREGROUND_VOXELS_PER_SLICE:
                all_raw_flat.append(nonzero_vals)
                all_clipped_flat.append(nonzero_vals)
                all_uint8_flat.append(
                    uint8_channel[:, :, z][nonzero_mask].astype(np.float32)
                )
                number_of_background_slices += 1
                continue

            lo = float(np.percentile(nonzero_vals, 0.5))
            hi = float(np.percentile(nonzero_vals, 99.5))
            all_raw_flat.append(nonzero_vals)
            all_clipped_flat.append(np.clip(s, lo, hi)[nonzero_mask])
            all_uint8_flat.append(
                uint8_channel[:, :, z][nonzero_mask].astype(np.float32)
            )
            per_slice_lo_list.append(lo)
            per_slice_hi_list.append(hi)

        number_of_tissue_slices = len(per_slice_lo_list)

        raw_flat_nonzero     = np.concatenate(all_raw_flat)
        clipped_flat_nonzero = np.concatenate(all_clipped_flat)
        uint8_flat_nonzero   = np.concatenate(all_uint8_flat)

        mean_lo = float(np.mean(per_slice_lo_list))
        mean_hi = float(np.mean(per_slice_hi_list))
        min_lo  = float(np.min(per_slice_lo_list))
        max_lo  = float(np.max(per_slice_lo_list))
        min_hi  = float(np.min(per_slice_hi_list))
        max_hi  = float(np.max(per_slice_hi_list))

        total_voxel_count         = raw_volume.size
        total_nonzero_voxel_count = raw_flat_nonzero.size
        number_of_zero_voxels     = total_voxel_count - total_nonzero_voxel_count

        number_of_clamped_voxels = int(np.sum(
            (raw_flat_nonzero < mean_lo) | (raw_flat_nonzero > mean_hi)
        ))
        clamped_fraction_percent = (
            100.0 * number_of_clamped_voxels / total_nonzero_voxel_count
            if total_nonzero_voxel_count > 0 else 0.0
        )

        stage_triplets = [
            (
                raw_flat_nonzero,
                f"Raw float  |  per-slice norm\n"
                f"{total_nonzero_voxel_count:,} fg voxels  "
                f"({number_of_zero_voxels:,} zeros removed)\n"
                f"Tissue slices: {number_of_tissue_slices}  "
                f"Background slices excluded: {number_of_background_slices}",
                f"Value range: [{raw_flat_nonzero.min():.1f}, "
                f"{raw_flat_nonzero.max():.1f}]",
                True,
            ),
            (
                clipped_flat_nonzero,
                f"After per-slice clipping\n"
                f"~{number_of_clamped_voxels:,} voxels clamped  "
                f"({clamped_fraction_percent:.2f}%)",
                f"Mean clip bounds: [{mean_lo:.1f}, {mean_hi:.1f}]  "
                f"(vary per slice)",
                False,
            ),
            (
                uint8_flat_nonzero,
                f"After per-slice uint8 rescale\n"
                f"{uint8_flat_nonzero.size:,} fg voxels",
                "Range: [0, 255]  (each slice independently scaled)",
                False,
            ),
        ]

        for row_index, (
            flat_values, y_label_string, x_label_string, show_clip_annotations,
        ) in enumerate(stage_triplets):

            axes_object = axes_grid[row_index, column_index]
            axes_object.set_facecolor('#0d0d1a')

            axes_object.hist(
                flat_values,
                bins=number_of_histogram_bins,
                color=bar_color, alpha=0.80, edgecolor='none',
            )

            if show_clip_annotations:
                axes_object.axvspan(
                    min_lo, max_lo, alpha=0.18, color='yellow',
                    label=f'p0.5 range  [{min_lo:.1f} – {max_lo:.1f}]',
                )
                axes_object.axvspan(
                    min_hi, max_hi, alpha=0.18, color='orange',
                    label=f'p99.5 range [{min_hi:.1f} – {max_hi:.1f}]',
                )
                axes_object.axvline(
                    mean_lo, color='yellow', linewidth=1.8, linestyle='--',
                    label=f'mean p0.5  = {mean_lo:.1f}  '
                          f'(tissue slices only, {number_of_background_slices} bg excluded)',
                )
                axes_object.axvline(
                    mean_hi, color='orange', linewidth=1.8, linestyle='--',
                    label=f'mean p99.5 = {mean_hi:.1f}  (tissue slices only)',
                )
                axes_object.legend(
                    fontsize=7.5, facecolor='#1a1a2e',
                    labelcolor='white', edgecolor='grey',
                )
                axes_object.set_title(
                    modality_label, color='white',
                    fontsize=11, fontweight='bold', pad=6,
                )

            axes_object.set_ylabel(y_label_string, color='white', fontsize=8)
            axes_object.set_xlabel(x_label_string, color='#aaaaaa', fontsize=8)
            axes_object.tick_params(colors='white', labelsize=8)
            axes_object.yaxis.set_major_formatter(
                matplotlib.ticker.FuncFormatter(
                    lambda tick_value, _: f"{int(tick_value):,}"
                )
            )
            for spine_object in axes_object.spines.values():
                spine_object.set_edgecolor('#444466')

    plt.tight_layout(rect=[0, 0, 1, 0.96])

    output_histogram_path = os.path.join(
        visualization_output_directory,
        f"{patient_id}_normalization_histogram.png",
    )
    plt.savefig(
        output_histogram_path, dpi=100, bbox_inches='tight',
        facecolor=figure_object.get_facecolor(),
    )
    plt.close(figure_object)
    print(f"  [NORM-HIST] Saved normalization histogram "
          f"(per-slice): {output_histogram_path}")


# =============================================================================
# DIAGNOSTIC PLOT 2 — Per-round HITL slice figure
# Now 5 columns: GT | Pred | GT+Pred | Delta | Bbox Input
# =============================================================================

def save_hitl_round_slice_figure(
    background_volume,
    ground_truth_mask_3d,
    current_predicted_mask_3d,
    current_anchor_z_set_sorted,
    anchor_bboxes_by_z,          # NEW: dict { z: list of (r0,c0,r1,c1) }
    round_index,
    current_round_dice_score,
    track_name,
    patient_id,
    visualization_output_directory,
):
    """
    Save a 5-column per-round diagnostic figure for one HITL inference round.

    One row per anchor slice.  Five columns:
        Col 0 — GT mask only
        Col 1 — Prediction only
        Col 2 — GT + Prediction combined
        Col 3 — Delta map (TP / FN / FP colour-coded)
        Col 4 — Bbox Input (connected-component bboxes)  ← NEW

    The newest anchor row is highlighted with a yellow border.
    """
    hitl_rounds_output_directory = os.path.join(
        visualization_output_directory, "hitl_rounds"
    )
    os.makedirs(hitl_rounds_output_directory, exist_ok=True)

    number_of_anchor_rows = len(current_anchor_z_set_sorted)
    if number_of_anchor_rows == 0:
        return

    newest_anchor_row_index = number_of_anchor_rows - 1

    # 5 columns now
    figure_object, axes_grid = plt.subplots(
        number_of_anchor_rows, 5,
        figsize=(27, 5 * number_of_anchor_rows + 1),
        facecolor='#1a1a2e',
        squeeze=False,
    )
    figure_object.suptitle(
        f"HITL Round {round_index}  |  Track: {track_name}  |  "
        f"Patient: {patient_id}\n"
        f"Dice = {current_round_dice_score:.4f}  |  "
        f"Anchors so far: {current_anchor_z_set_sorted}",
        color='white', fontsize=13, fontweight='bold', y=0.99,
    )

    for row_index, anchor_z_value in enumerate(current_anchor_z_set_sorted):

        is_newest    = (row_index == newest_anchor_row_index)
        border_color = '#ffdd00' if is_newest else '#888888'
        border_width = 3 if is_newest else 1
        age_label    = (
            f"z={anchor_z_value}  [NEW ★]" if is_newest
            else f"z={anchor_z_value}  [prior]"
        )

        bg_norm    = _make_bg_norm(background_volume, anchor_z_value)
        gt_slice   = ground_truth_mask_3d[:, :, anchor_z_value].astype(bool)
        pred_slice = current_predicted_mask_3d[:, :, anchor_z_value].astype(bool)
        bboxes     = anchor_bboxes_by_z.get(anchor_z_value, [])

        gt_px   = int(gt_slice.sum())
        pred_px = int(pred_slice.sum())
        tp = int((gt_slice & pred_slice).sum())
        fn = int((gt_slice & ~pred_slice).sum())
        fp = int((~gt_slice & pred_slice).sum())
        n_comp = len(bboxes)

        panel_specs = [
            ('gt',    f"GT only  {age_label}\nGT pixels = {gt_px:,}",
             'cyan',    border_color, border_width),
            ('pred',  f"Pred only  z={anchor_z_value}\nPred pixels = {pred_px:,}",
             'magenta', border_color, border_width),
            ('both',  f"GT + Pred  z={anchor_z_value}\ncyan=GT  magenta=Pred",
             'white',   border_color, border_width),
            ('delta', f"Delta  z={anchor_z_value}\nTP={tp:,}  FN={fn:,}  FP={fp:,}",
             '#ffdd00', border_color, border_width),
        ]

        for col_index, (mode, title, title_color, b_color, b_width) \
                in enumerate(panel_specs):
            _render_segmentation_panel(
                ax=axes_grid[row_index, col_index],
                bg_norm=bg_norm,
                gt_mask=gt_slice,
                pred_mask=pred_slice,
                mode=mode,
                title=title,
                title_color=title_color,
                border_color=b_color,
                border_width=b_width,
            )

        # Col 4 — Bbox Input (NEW)
        _render_bbox_panel(
            ax=axes_grid[row_index, 4],
            bg_norm=bg_norm,
            bboxes=bboxes,
            title=(
                f"Bbox Input  z={anchor_z_value}  {'[NEW ★]' if is_newest else '[prior]'}\n"
                f"{n_comp} component{'s' if n_comp != 1 else ''}  "
                f"(min_area={BBOX_MIN_COMPONENT_AREA})"
            ),
        )

    plt.tight_layout(rect=[0, 0, 1, 0.97])

    output_round_figure_path = os.path.join(
        hitl_rounds_output_directory,
        f"{patient_id}_{track_name}_round{round_index:02d}"
        f"_dice{current_round_dice_score:.4f}.png",
    )
    plt.savefig(
        output_round_figure_path, dpi=90, bbox_inches='tight',
        facecolor=figure_object.get_facecolor(),
    )
    dprint(f"  [HITL-ROUND-VIZ] Saved: {output_round_figure_path}")
    plt.close(figure_object)


# =============================================================================
# PHASE 01 - Data I/O
# =============================================================================

def load_nifti(file_path):
    """Load a NIfTI file; return (volume_data float32, affine, header, image)."""
    dprint(f"  Loading: {file_path}")
    nifti_image = nib.load(file_path)
    volume_data = nifti_image.get_fdata(dtype=np.float32)
    dprint(f"    shape={volume_data.shape}  "
           f"min={volume_data.min():.1f}  max={volume_data.max():.1f}")
    return volume_data, nifti_image.affine, nifti_image.header, nifti_image


# =============================================================================
# PHASE 06 - Metrics
# =============================================================================

def calculate_metrics(ground_truth_binary_mask, predicted_binary_mask):
    """
    Compute Dice and HD95 between two binary 3-D masks.

    Edge cases:
      both empty  -> Dice=1,   HD95=0
      one empty   -> Dice=0,   HD95=373 (diagonal of BraTS volume)
    """
    ground_truth_binary_mask = ground_truth_binary_mask.astype(bool)
    predicted_binary_mask    = predicted_binary_mask.astype(bool)

    if (not np.any(ground_truth_binary_mask)
            and not np.any(predicted_binary_mask)):
        return 1.0, 0.0
    if (not np.any(ground_truth_binary_mask)
            or not np.any(predicted_binary_mask)):
        return 0.0, 373.13

    intersection_voxel_count = np.logical_and(
        ground_truth_binary_mask, predicted_binary_mask
    ).sum()
    dice_score = (
        (2.0 * intersection_voxel_count)
        / (ground_truth_binary_mask.sum() + predicted_binary_mask.sum())
    )

    ground_truth_surface_voxels = (
        ground_truth_binary_mask & ~binary_erosion(ground_truth_binary_mask)
    )
    predicted_surface_voxels = (
        predicted_binary_mask & ~binary_erosion(predicted_binary_mask)
    )

    distance_transform_from_predicted_surface = distance_transform_edt(
        ~predicted_binary_mask
    )
    distance_transform_from_ground_truth_surface = distance_transform_edt(
        ~ground_truth_binary_mask
    )

    all_surface_distances = np.hstack([
        distance_transform_from_predicted_surface[ground_truth_surface_voxels],
        distance_transform_from_ground_truth_surface[predicted_surface_voxels],
    ])
    hausdorff_distance_95th_percentile = float(
        np.percentile(all_surface_distances, 95)
    )
    return float(dice_score), hausdorff_distance_95th_percentile


# =============================================================================
# PHASE 03 - Anchor utilities
# =============================================================================

def find_peak_slice(binary_mask_3d):
    """
    Return the axial slice index with the most foreground pixels.
    Returns None if mask is entirely empty.
    """
    per_slice_counts = np.array([
        binary_mask_3d[:, :, z].sum()
        for z in range(binary_mask_3d.shape[2])
    ])
    if per_slice_counts.max() == 0:
        return None
    return int(np.argmax(per_slice_counts))


def find_largest_error_slice(
    ground_truth_mask_3d,
    predicted_mask_3d,
    already_used_anchor_z_set,
):
    """
    Find the axial slice with the greatest segmentation error not already
    in the anchor set.  Error = |GT_pixels(z) - pred_pixels(z)|.
    Returns None if all occupied slices are already anchors.
    """
    D = ground_truth_mask_3d.shape[2]
    per_slice_gt_counts   = np.array([
        ground_truth_mask_3d[:, :, z].sum() for z in range(D)
    ])
    per_slice_pred_counts = np.array([
        predicted_mask_3d[:, :, z].sum() for z in range(D)
    ])
    per_slice_errors = np.abs(
        per_slice_gt_counts.astype(float) - per_slice_pred_counts.astype(float)
    )

    for used_z in already_used_anchor_z_set:
        per_slice_errors[used_z] = -1.0
    per_slice_errors[per_slice_gt_counts == 0] = -1.0

    if per_slice_errors.max() <= 0:
        return None
    return int(np.argmax(per_slice_errors))


# =============================================================================
# PHASE 04 - MedSAM2 Predictor singleton
# =============================================================================

_global_predictor_singleton = None


def get_predictor():
    """Return the MedSAM2 video predictor, loading it on first call."""
    global _global_predictor_singleton
    if _global_predictor_singleton is not None:
        return _global_predictor_singleton

    print("\n  [MODEL] Loading MedSAM2 predictor...")

    checkpoint_full_path      = os.path.join(MEDSAM2_PATH, SAM2_CHECKPOINT)
    config_full_path          = os.path.join(MEDSAM2_PATH, SAM2_CFG)
    config_hydra_relative_key = "configs/sam2.1_hiera_t512.yaml"

    if not os.path.exists(checkpoint_full_path):
        raise FileNotFoundError(f"Checkpoint not found: {checkpoint_full_path}")
    if not os.path.exists(config_full_path):
        raise FileNotFoundError(f"Config not found: {config_full_path}")

    _global_predictor_singleton = build_sam2_video_predictor(
        config_file=config_hydra_relative_key,
        ckpt_path=checkpoint_full_path,
        apply_postprocessing=False,
    )
    _global_predictor_singleton.eval()
    print("  [MODEL] Predictor loaded")
    return _global_predictor_singleton


# =============================================================================
# PHASE 04 - Single-track inference within a shared inference_state
# Now uses multi-bbox prompting (one box per connected component per slice).
# =============================================================================

@torch.inference_mode()
@torch.autocast(device_type="cuda", dtype=torch.bfloat16)
def _infer_one_track(
    predictor,
    shared_inference_state,
    anchor_masks_keyed_by_z,
    volume_shape,
):
    """
    Run bidirectional propagation for ONE track given a set of anchor masks.

    CHANGED: Instead of calling add_new_mask() with the full binary mask,
    we decompose each anchor slice into connected components and call
    add_new_points_or_box() once per component.  This gives SAM2 a tighter
    spatial prior and correctly handles masks with disconnected regions.

    Parameters
    ----------
    predictor               : SAM2VideoPredictor
    shared_inference_state  : opaque state dict from init_state
    anchor_masks_keyed_by_z : dict { z: bool ndarray (H, W) }
    volume_shape            : (H, W, D)

    Returns
    -------
    predicted_binary_mask_3d : bool ndarray (H, W, D)
    anchor_bboxes_by_z       : dict { z: list of (r0,c0,r1,c1) }
                               Bboxes actually submitted as prompts —
                               returned so callers can visualise them.
    """
    H, W, D = volume_shape

    if not anchor_masks_keyed_by_z:
        return np.zeros((H, W, D), dtype=bool), {}

    sorted_anchor_z_indices = sorted(anchor_masks_keyed_by_z.keys())

    # Pre-compute bboxes once so we can reuse across forward + backward passes
    # and return them to the caller for visualisation.
    anchor_bboxes_by_z = {}
    for anchor_z, anchor_mask_2d in anchor_masks_keyed_by_z.items():
        bboxes = get_component_bboxes(anchor_mask_2d)
        if not bboxes:
            # Fallback: whole-mask bbox if no component survives the threshold
            rows = np.any(anchor_mask_2d, axis=1)
            cols = np.any(anchor_mask_2d, axis=0)
            if rows.any():
                r0 = int(np.argmax(rows))
                r1 = int(len(rows) - 1 - np.argmax(rows[::-1]))
                c0 = int(np.argmax(cols))
                c1 = int(len(cols) - 1 - np.argmax(cols[::-1]))
                bboxes = [(r0, c0, r1, c1)]
        anchor_bboxes_by_z[anchor_z] = bboxes

    def _register_all_anchor_prompts():
        """Add one box prompt per component per anchor frame."""
        for anchor_z, bboxes in anchor_bboxes_by_z.items():
            sam2_boxes = bboxes_to_sam2_boxes(bboxes)  # (N, 4) x0y0x1y1
            if sam2_boxes.shape[0] == 0:
                continue
            for box_idx in range(sam2_boxes.shape[0]):
                single_box = sam2_boxes[box_idx]           # shape (4,)
                obj_id     = box_idx + 1                   # unique id per box
                predictor.add_new_points_or_box(
                    inference_state=shared_inference_state,
                    frame_idx=anchor_z,
                    obj_id=obj_id,
                    box=single_box,
                )
            dprint(f"    z={anchor_z}: registered {sam2_boxes.shape[0]} box(es)")

    # Forward pass ────────────────────────────────────────────────────────────
    _register_all_anchor_prompts()
    per_frame_logit_scores = {}   # frame_idx -> combined logit (H, W)

    for (
        output_frame_index, output_object_ids, output_mask_logits,
    ) in predictor.propagate_in_video(
        shared_inference_state,
        start_frame_idx=min(sorted_anchor_z_indices),
        reverse=False,
    ):
        # Union-merge logits from all object IDs at this frame
        merged_logit = None
        for object_position, object_id in enumerate(output_object_ids):
            logit = output_mask_logits[object_position][0].cpu().numpy()
            merged_logit = logit if merged_logit is None else np.maximum(merged_logit, logit)
        if merged_logit is not None:
            per_frame_logit_scores[output_frame_index] = merged_logit

    # Backward pass ───────────────────────────────────────────────────────────
    predictor.reset_state(shared_inference_state)
    _register_all_anchor_prompts()

    for (
        output_frame_index, output_object_ids, output_mask_logits,
    ) in predictor.propagate_in_video(
        shared_inference_state,
        start_frame_idx=max(sorted_anchor_z_indices),
        reverse=True,
    ):
        if output_frame_index in per_frame_logit_scores:
            continue   # forward pass already covered this frame
        merged_logit = None
        for object_position, object_id in enumerate(output_object_ids):
            logit = output_mask_logits[object_position][0].cpu().numpy()
            merged_logit = logit if merged_logit is None else np.maximum(merged_logit, logit)
        if merged_logit is not None:
            per_frame_logit_scores[output_frame_index] = merged_logit

    # Threshold -> binary ─────────────────────────────────────────────────────
    predicted_binary_mask_3d = np.zeros((H, W, D), dtype=bool)
    for z in range(D):
        if z in per_frame_logit_scores:
            predicted_binary_mask_3d[:, :, z] = (
                per_frame_logit_scores[z] > MASK_THRESHOLD
            )

    return predicted_binary_mask_3d, anchor_bboxes_by_z


# =============================================================================
# OPTIMIZATION 4 — run_all_tracks_hitl
# Updated to pass anchor_bboxes_by_z through to the visualisation functions.
# =============================================================================

@torch.inference_mode()
@torch.autocast(device_type="cuda", dtype=torch.bfloat16)
def run_all_tracks_hitl(
    video_subfolder_name,
    track_ground_truth_masks,
    track_peak_slices,
    volume_shape,
    background_volume,
    patient_id,
    patient_visualization_directory,
):
    """
    Segment ALL tracks using the HITL iterative anchor refinement strategy,
    all within a SINGLE init_state session (shared encoder).

    Returns
    -------
    per_track_final_predicted_masks : dict { track_name: bool ndarray (H,W,D) }
    per_track_hitl_round_logs       : dict { track_name: list[dict] }
    """
    H, W, D = volume_shape
    predictor = get_predictor()
    video_dir = os.path.join(TEMP_VIDEO_DIR, video_subfolder_name)

    empty_masks = {
        name: np.zeros((H, W, D), dtype=bool)
        for name in track_ground_truth_masks
    }
    empty_logs = {name: [] for name in track_ground_truth_masks}

    if not os.path.exists(video_dir):
        print(f"  [ERROR] Video dir not found: {video_dir}")
        return empty_masks, empty_logs

    jpeg_frame_filenames = sorted([
        f for f in os.listdir(video_dir)
        if os.path.splitext(f)[-1].lower() in (".jpg", ".jpeg")
    ])
    if not jpeg_frame_filenames:
        print(f"  [ERROR] No JPEG frames in {video_dir}")
        return empty_masks, empty_logs

    # Single init_state: encode ALL frames ONCE
    try:
        shared_inference_state = predictor.init_state(
            video_path=video_dir,
            async_loading_frames=False,
        )
    except Exception as init_state_exception:
        print(f"  [ERROR] init_state failed: {init_state_exception}")
        return empty_masks, empty_logs

    # Standalone bbox image output directory
    bbox_vis_dir = os.path.join(patient_visualization_directory, "bbox_inputs")
    os.makedirs(bbox_vis_dir, exist_ok=True)

    per_track_final_predicted_masks = {}
    per_track_hitl_round_logs       = {}

    for track_name, ground_truth_mask_3d in track_ground_truth_masks.items():

        print(
            f"\n  -- HITL: {track_name} "
            f"(max_iter={HITL_MAX_ITERATIONS}, "
            f"min_improve={HITL_MIN_IMPROVEMENT}, "
            f"target={HITL_DICE_TARGET})"
        )
        print(f"  {'Rnd':>4} {'Anchors':<35} {'Dice':>8} {'Delta':>8} {'Status'}")
        print(f"  {'-'*80}")

        peak_axial_slice_index = track_peak_slices[track_name]

        current_anchor_z_set      = {peak_axial_slice_index}
        current_anchor_masks_by_z = {
            peak_axial_slice_index: (
                ground_truth_mask_3d[:, :, peak_axial_slice_index].copy()
            )
        }

        per_slice_gt_counts = np.array([
            ground_truth_mask_3d[:, :, z].sum() for z in range(D)
        ])

        hitl_round_log            = []
        previous_round_dice_score = -1.0
        best_predicted_mask_3d    = np.zeros((H, W, D), dtype=bool)
        best_round_dice_score     = -1.0
        best_round_anchor_list    = []
        stop_reason_string        = "budget"
        latest_anchor_bboxes_by_z = {}

        for round_index in range(HITL_MAX_ITERATIONS):

            predictor.reset_state(shared_inference_state)

            current_round_predicted_mask, anchor_bboxes_by_z = _infer_one_track(
                predictor,
                shared_inference_state,
                current_anchor_masks_by_z,
                volume_shape,
            )
            latest_anchor_bboxes_by_z = anchor_bboxes_by_z

            # ── Save standalone bbox images for every anchor this round ──
            sorted_anchors_this_round = sorted(current_anchor_z_set)
            newest_z                  = sorted_anchors_this_round[-1]
            for anchor_z in sorted_anchors_this_round:
                bg_norm = _make_bg_norm(background_volume, anchor_z)
                save_bbox_input_image(
                    bg_norm=bg_norm,
                    bboxes=anchor_bboxes_by_z.get(anchor_z, []),
                    z_index=anchor_z,
                    track_name=track_name,
                    round_index=round_index,
                    patient_id=patient_id,
                    bbox_output_directory=bbox_vis_dir,
                    is_new_anchor=(anchor_z == newest_z and round_index > 0),
                )

            current_round_dice_score, _ = calculate_metrics(
                ground_truth_mask_3d, current_round_predicted_mask
            )

            dice_improvement = (
                current_round_dice_score - previous_round_dice_score
                if round_index > 0
                else float('nan')
            )
            delta_display = (
                f"{dice_improvement:+.4f}" if round_index > 0 else "  init"
            )
            sorted_current_anchor_list = sorted(current_anchor_z_set)

            if round_index == 0:
                round_status = "seed"
            elif current_round_dice_score >= HITL_DICE_TARGET:
                round_status = f"stop: dice>={HITL_DICE_TARGET}"
            elif dice_improvement < HITL_MIN_IMPROVEMENT:
                round_status = f"stop: delta<{HITL_MIN_IMPROVEMENT}"
            elif round_index == HITL_MAX_ITERATIONS - 1:
                round_status = f"stop: max_iter={HITL_MAX_ITERATIONS}"
            else:
                round_status = "continue"

            print(
                f"  {round_index:>4} "
                f"{str(sorted_current_anchor_list):<35} "
                f"{current_round_dice_score:>8.4f} "
                f"{delta_display:>8} "
                f"{round_status}"
            )

            hitl_round_log.append({
                'round'       : round_index,
                'anchors'     : sorted_current_anchor_list[:],
                'dice'        : current_round_dice_score,
                'delta'       : dice_improvement,
                'stop_reason' : None,
            })

            if round_index == 0 or current_round_dice_score > best_round_dice_score:
                best_predicted_mask_3d = current_round_predicted_mask
                best_round_dice_score  = current_round_dice_score
                best_round_anchor_list = sorted_current_anchor_list[:]

            # Per-round 5-column figure (now includes bbox column)
            save_hitl_round_slice_figure(
                background_volume=background_volume,
                ground_truth_mask_3d=ground_truth_mask_3d,
                current_predicted_mask_3d=current_round_predicted_mask,
                current_anchor_z_set_sorted=sorted_current_anchor_list,
                anchor_bboxes_by_z=anchor_bboxes_by_z,     # NEW
                round_index=round_index,
                current_round_dice_score=current_round_dice_score,
                track_name=track_name,
                patient_id=patient_id,
                visualization_output_directory=patient_visualization_directory,
            )

            if current_round_predicted_mask.sum() == 0:
                print(
                    f"  [WARNING] Empty prediction for {track_name} "
                    f"at round {round_index}"
                )

            # Stopping criteria
            if current_round_dice_score >= HITL_DICE_TARGET:
                stop_reason_string                = f"dice_target ({HITL_DICE_TARGET})"
                hitl_round_log[-1]['stop_reason'] = stop_reason_string
                break

            if round_index > 0 and dice_improvement < HITL_MIN_IMPROVEMENT:
                stop_reason_string = (
                    f"no_improvement "
                    f"(delta={dice_improvement:.4f} "
                    f"< {HITL_MIN_IMPROVEMENT})"
                )
                hitl_round_log[-1]['stop_reason'] = stop_reason_string
                break

            if round_index == HITL_MAX_ITERATIONS - 1:
                stop_reason_string                = f"max_iterations ({HITL_MAX_ITERATIONS})"
                hitl_round_log[-1]['stop_reason'] = stop_reason_string
                break

            # Next anchor: largest error slice not yet prompted
            next_anchor_z = find_largest_error_slice(
                ground_truth_mask_3d,
                current_round_predicted_mask,
                current_anchor_z_set,
            )
            if next_anchor_z is None:
                stop_reason_string                = "no_eligible_slice"
                hitl_round_log[-1]['stop_reason'] = stop_reason_string
                break

            next_anchor_error = int(abs(
                per_slice_gt_counts[next_anchor_z]
                - current_round_predicted_mask[:, :, next_anchor_z].sum()
            ))
            print(
                f"  --> Adding anchor z={next_anchor_z}  "
                f"(error={next_anchor_error:,} px)"
            )
            current_anchor_z_set.add(next_anchor_z)
            current_anchor_masks_by_z[next_anchor_z] = (
                ground_truth_mask_3d[:, :, next_anchor_z].copy()
            )
            previous_round_dice_score = current_round_dice_score

        if hitl_round_log and hitl_round_log[-1]['stop_reason'] is None:
            hitl_round_log[-1]['stop_reason'] = stop_reason_string

        per_track_final_predicted_masks[track_name] = best_predicted_mask_3d
        per_track_hitl_round_logs[track_name]       = hitl_round_log

        # Stash the final bboxes on the log so show_visualization can use them
        for entry in hitl_round_log:
            entry['anchor_bboxes_by_z'] = latest_anchor_bboxes_by_z

        final_round_dice = hitl_round_log[-1]['dice'] if hitl_round_log else 0.0
        print(
            f"  FINAL: rounds={len(hitl_round_log)} "
            f"| best_anchors={best_round_anchor_list} "
            f"| best_Dice={best_round_dice_score:.4f} "
            f"| last_Dice={final_round_dice:.4f} "
            f"| stop={stop_reason_string}"
        )

        torch.cuda.empty_cache()

    predictor.reset_state(shared_inference_state)
    gc.collect()

    return per_track_final_predicted_masks, per_track_hitl_round_logs


# =============================================================================
# PHASE 05 - Multi-Label Merge with Priority
# =============================================================================

def merge_masks_to_label_map(per_label_binary_masks, volume_shape):
    """
    Combine per-label binary masks into a single integer label map.
    Paint order follows MERGE_PRIORITY (later labels overwrite earlier).
    """
    print("\n  Merging binary masks -> label map...")
    merged_label_map = np.zeros(volume_shape, dtype=np.uint8)

    for label_id in MERGE_PRIORITY:
        if label_id not in per_label_binary_masks:
            dprint(f"  Label {label_id} not in per_label_binary_masks -- skip")
            continue
        label_binary_mask = per_label_binary_masks[label_id]
        merged_label_map[label_binary_mask] = label_id
        dprint(
            f"  Painted label {label_id} ({LABEL_NAMES[label_id]}): "
            f"{label_binary_mask.sum():,} voxels"
        )

    print("  Final label distribution:")
    for unique_label_value, voxel_count in zip(
        *np.unique(merged_label_map, return_counts=True)
    ):
        label_display_name = LABEL_NAMES.get(int(unique_label_value), "background")
        print(
            f"    Label {unique_label_value} ({label_display_name}): "
            f"{voxel_count:,} voxels"
        )

    return merged_label_map


# =============================================================================
# VISUALISATION — final-anchor figure
# Now 4 columns: GT | Pred | Overlap | Bbox Input
# =============================================================================

def show_visualization(
    background_volume,
    ground_truth_binary_mask,
    predicted_binary_mask,
    patient_id,
    label_id,
    dice_score_raw,
    hausdorff_distance_95_raw,
    dice_score_merged,
    hausdorff_distance_95_merged,
    hitl_round_log,
    visualization_output_directory,
    track_label=None,
):
    """
    Save the final-anchor diagnostic figure for a single track.
    One row per anchor slice used in the FINAL HITL iteration.
    Four columns: GT | Prediction | Overlap (TP/FN/FP) | Bbox Input.
    """
    os.makedirs(visualization_output_directory, exist_ok=True)

    label_display_name = (
        track_label if track_label
        else LABEL_NAMES.get(label_id, str(label_id))
    )

    if not hitl_round_log:
        print(f"  [VIZ] {label_display_name}: empty HITL log, skipping figure.")
        return

    final_anchor_z_list    = hitl_round_log[-1]['anchors']
    final_bboxes_by_z      = hitl_round_log[-1].get('anchor_bboxes_by_z', {})
    total_rounds_completed = len(hitl_round_log)
    final_stop_reason      = hitl_round_log[-1].get('stop_reason', '?')

    print(
        f"  [VIZ] {label_display_name} "
        f"| final_anchors={final_anchor_z_list} "
        f"| rounds={total_rounds_completed}"
    )

    number_of_anchor_rows = max(len(final_anchor_z_list), 1)

    # 4 columns now (GT | Pred | Delta | Bbox)
    figure_object = plt.figure(
        figsize=(32, 6 * number_of_anchor_rows + 2),
        facecolor='#1a1a2e',
    )
    grid_spec_object = GridSpec(
        number_of_anchor_rows + 1, 4,
        figure=figure_object,
        height_ratios=[10] * number_of_anchor_rows + [1],
        hspace=0.08, wspace=0.04,
        left=0.02, right=0.98, top=0.95, bottom=0.01,
    )

    history_parts = [
        f"R{e['round']}:[{','.join(map(str, e['anchors']))}]->D={e['dice']:.3f}"
        for e in hitl_round_log
    ]
    history_string = "  ".join(history_parts)

    figure_object.text(
        0.5, 0.975,
        f"Patient: {patient_id}  |  RGB: R=T1c G=T2w B=T2f  |  "
        f"Track: {label_display_name}  |  "
        f"HITL rounds: {total_rounds_completed}  |  "
        f"Stop: {final_stop_reason}",
        ha='center', fontsize=15, fontweight='bold', color='white',
    )
    figure_object.text(
        0.5, 0.960,
        f"Dice raw: {dice_score_raw:.4f}   "
        f"HD95 raw: {hausdorff_distance_95_raw:.2f}   |   "
        f"Dice merged: {dice_score_merged:.4f}   "
        f"HD95 merged: {hausdorff_distance_95_merged:.2f}   |   "
        f"HITL: {history_string}",
        ha='center', fontsize=9, color='#cccccc',
    )

    border_colors = ['#00aaff', '#ff8800', '#ff00aa', '#00ffaa',
                     '#ffff00', '#aa00ff', '#00ffff']

    for row_index, anchor_z_value in enumerate(
        final_anchor_z_list[:number_of_anchor_rows]
    ):
        row_tag    = " [PEAK]" if row_index == 0 else f" [+R{row_index}]"
        b_color    = border_colors[row_index % len(border_colors)]
        bg_norm    = _make_bg_norm(background_volume, int(anchor_z_value))
        gt_slice   = ground_truth_binary_mask[:, :, int(anchor_z_value)].astype(bool)
        pred_slice = predicted_binary_mask[:, :, int(anchor_z_value)].astype(bool)
        bboxes     = final_bboxes_by_z.get(anchor_z_value, [])
        n_comp     = len(bboxes)

        panel_specs = [
            ('gt',    f"Anchor{row_tag}  z={anchor_z_value}\nGround Truth",
             'cyan',    b_color, 2),
            ('pred',  f"Anchor{row_tag}  z={anchor_z_value}\nMedSAM2 Prediction",
             'magenta', b_color, 2),
            ('delta', f"Anchor{row_tag}  z={anchor_z_value}\nOverlap",
             '#f0a500', b_color, 2),
        ]

        for col_index, (mode, title, title_color, bc, bw) in enumerate(panel_specs):
            ax = figure_object.add_subplot(grid_spec_object[row_index, col_index])
            _render_segmentation_panel(
                ax=ax,
                bg_norm=bg_norm,
                gt_mask=gt_slice,
                pred_mask=pred_slice,
                mode=mode,
                title=title,
                title_color=title_color,
                border_color=bc,
                border_width=bw,
            )

        # Col 3 — Bbox Input (NEW)
        bbox_ax = figure_object.add_subplot(grid_spec_object[row_index, 3])
        _render_bbox_panel(
            ax=bbox_ax,
            bg_norm=bg_norm,
            bboxes=bboxes,
            title=(
                f"Bbox Input{row_tag}  z={anchor_z_value}\n"
                f"{n_comp} component{'s' if n_comp != 1 else ''}  "
                f"(min_area={BBOX_MIN_COMPONENT_AREA})"
            ),
        )
        for sp in bbox_ax.spines.values():
            sp.set_edgecolor(b_color)
            sp.set_linewidth(2)

    legend_ax = figure_object.add_subplot(
        grid_spec_object[number_of_anchor_rows, :]
    )
    legend_ax.axis('off')
    legend_ax.legend(
        handles=[
            Patch(facecolor='cyan',    edgecolor='white', alpha=0.7,
                  label='GT contour'),
            Patch(facecolor='magenta', edgecolor='white', alpha=0.7,
                  label='Pred contour'),
            Patch(facecolor='yellow',  edgecolor='white', alpha=0.7,
                  label='True Positive'),
            Patch(facecolor='green',   edgecolor='white', alpha=0.7,
                  label='False Negative (missed)'),
            Patch(facecolor='red',     edgecolor='white', alpha=0.7,
                  label='False Positive (extra)'),
            Patch(facecolor='#ffdd00', edgecolor='white', alpha=0.7,
                  label=f'Bbox prompt (≥{BBOX_MIN_COMPONENT_AREA}px/comp)'),
        ],
        loc='center', ncol=6, fontsize=11,
        framealpha=0.15, edgecolor='white',
        facecolor='#1a1a2e', labelcolor='white',
    )

    output_figure_path = os.path.join(
        visualization_output_directory,
        f"{patient_id}_{label_display_name}_hitl_r{total_rounds_completed}.png",
    )
    plt.savefig(
        output_figure_path, dpi=100, bbox_inches='tight',
        facecolor=figure_object.get_facecolor(),
    )
    dprint(f"  [VIZ] Saved: {output_figure_path}")

    if SHOW_PLOTS:
        plt.show(block=True)
    plt.close(figure_object)


# =============================================================================
# Per-patient pipeline
# =============================================================================

def process_patient(
    patient_id,
    patient_directory_path,
    csv_writer,
    results_accumulator,
):
    """Full HITL segmentation pipeline for one patient."""
    print(f"\n{'='*60}")
    print(f"  PATIENT: {patient_id}")
    print(f"{'='*60}")
    patient_start_time = time.time()

    # -- Phase 01: file discovery ---------------------------------------------
    all_files = os.listdir(patient_directory_path)

    def find_modality_file(suffix):
        matches = [
            f for f in all_files
            if f.endswith(('.nii', '.nii.gz'))
            and (
                f'-{suffix}.' in f.lower()
                or f'-{suffix}_' in f.lower()
                or f.lower().endswith(f'-{suffix}.nii.gz')
                or f.lower().endswith(f'-{suffix}.nii')
            )
        ]
        return (
            os.path.join(patient_directory_path, matches[0])
            if matches else None
        )

    segmentation_file_path = find_modality_file('seg')
    t1c_modality_path      = find_modality_file('t1c')
    t2w_modality_path      = find_modality_file('t2w')
    t2f_modality_path      = find_modality_file('t2f')
    t1n_modality_path      = find_modality_file('t1n')

    if segmentation_file_path is None:
        print(f"  [SKIP] No segmentation file for {patient_id}")
        return []

    print(f"  Seg  : {os.path.basename(segmentation_file_path)}")
    for name, path in [
        ('T1c', t1c_modality_path), ('T2w', t2w_modality_path),
        ('T2f', t2f_modality_path), ('T1n', t1n_modality_path),
    ]:
        status = (
            f"FOUND ({os.path.basename(path)})" if path else "MISSING"
        )
        print(f"  {name:<4}: {status}")

    seg_volume_data, voxel_to_world_affine, nifti_header_object, _ = (
        load_nifti(segmentation_file_path)
    )
    seg_volume_data = seg_volume_data.astype(np.uint8)
    volume_height, volume_width, num_axial_slices = seg_volume_data.shape

    all_unique_labels       = np.unique(seg_volume_data).astype(int)
    present_tumor_label_ids = [
        l for l in TARGET_LABELS if l in all_unique_labels
    ]
    print(
        f"\n  GT labels present: {present_tumor_label_ids}  "
        f"(all unique: {all_unique_labels.tolist()})"
    )

    if not present_tumor_label_ids:
        print(f"  [SKIP] No tumor labels found in {patient_id}")
        return []

    def load_if_exists(path):
        return (
            load_nifti(path)[0]
            if path and os.path.exists(path) else None
        )

    t1c_raw_volume = load_if_exists(t1c_modality_path)
    t2w_raw_volume = load_if_exists(t2w_modality_path)
    t2f_raw_volume = load_if_exists(t2f_modality_path)
    t1n_raw_volume = load_if_exists(t1n_modality_path)

    def first_available(*vols):
        return next((v for v in vols if v is not None), None)

    red_channel_volume   = first_available(t1c_raw_volume, t2w_raw_volume,
                                           t1n_raw_volume, t2f_raw_volume)
    green_channel_volume = first_available(t2w_raw_volume, t1n_raw_volume,
                                           t1c_raw_volume, t2f_raw_volume)
    blue_channel_volume  = first_available(t2f_raw_volume, t2w_raw_volume,
                                           t1c_raw_volume)

    if any(v is None for v in (red_channel_volume,
                                green_channel_volume,
                                blue_channel_volume)):
        print(f"  [SKIP] Cannot construct RGB channels for {patient_id}")
        return []

    if t2w_raw_volume is None:
        print("  [FALLBACK] T2w missing -> using T1n for G channel")
    if t2f_raw_volume is None:
        print("  [FALLBACK] T2f missing -> using T2w for B channel")

    patient_visualization_dir = os.path.join(
        OUTPUT_BASE_DIR, "visualizations", patient_id
    )
    os.makedirs(patient_visualization_dir, exist_ok=True)

    # -- Phase 02: write JPEG frames AND cache uint8 RGB ----------------------
    video_subfolder_name = f"{patient_id}_rgb"
    clear_dir(os.path.join(TEMP_VIDEO_DIR, video_subfolder_name))

    print(f"\n  Phase 02 -- RGB frames + cache (per-slice normalization)...")
    rgb_cache = prepare_rgb_frames_cached(
        red_channel_volume,
        green_channel_volume,
        blue_channel_volume,
        video_subfolder_name,
    )

    background_visualization_volume = rgb_cache.mean(axis=-1).astype(np.float32)

    # -- Normalization histogram -----------------------------------------------
    print(f"\n  Saving normalization histogram (per-slice, from cache)...")
    save_normalization_histogram(
        t1c_volume_raw=red_channel_volume,
        t2w_volume_raw=green_channel_volume,
        t2f_volume_raw=blue_channel_volume,
        rgb_cache=rgb_cache,
        patient_id=patient_id,
        visualization_output_directory=patient_visualization_dir,
    )

    # -- Phase 03: build GT masks and compute peak slices ---------------------
    print(f"\n  Phase 03 -- Peak-slice seed (HITL round 0):")
    print(f"  {'Track':<22} {'Peak z':>8} {'GT vox':>12}")
    print(f"  {'-'*45}")

    track_ground_truth_masks = {}
    track_peak_slices        = {}

    for label_id in present_tumor_label_ids:
        label_name = LABEL_NAMES[label_id]
        label_gt   = (seg_volume_data == label_id)
        if not np.any(label_gt):
            continue
        peak_z = find_peak_slice(label_gt)
        print(
            f"  Label {label_id} ({label_name:<6})    "
            f"{peak_z:>8}    {label_gt.sum():>10,}"
        )
        track_ground_truth_masks[label_name] = label_gt
        track_peak_slices[label_name]        = peak_z

    wt_present = [l for l in WT_LABELS if l in present_tumor_label_ids]
    tc_present = [l for l in TC_LABELS if l in present_tumor_label_ids]

    if wt_present:
        wt_gt = np.isin(seg_volume_data, wt_present)
        if np.any(wt_gt):
            wt_peak = find_peak_slice(wt_gt)
            print(
                f"  {'WT':<22}    "
                f"{wt_peak:>8}    {wt_gt.sum():>10,}"
            )
            track_ground_truth_masks["WT"] = wt_gt
            track_peak_slices["WT"]        = wt_peak

    if tc_present:
        tc_gt = np.isin(seg_volume_data, tc_present)
        if np.any(tc_gt):
            tc_peak = find_peak_slice(tc_gt)
            print(
                f"  {'TC':<22}    "
                f"{tc_peak:>8}    {tc_gt.sum():>10,}"
            )
            track_ground_truth_masks["TC"] = tc_gt
            track_peak_slices["TC"]        = tc_peak

    if not track_ground_truth_masks:
        print(f"  [SKIP] No valid track GT masks for {patient_id}")
        return []

    # -- Phase 04 (HITL): run all tracks with iterative anchor refinement -----
    print(
        f"\n  Phase 04 -- HITL inference "
        f"[max_iter={HITL_MAX_ITERATIONS} | "
        f"min_improve={HITL_MIN_IMPROVEMENT} | "
        f"dice_target={HITL_DICE_TARGET}]"
    )
    inference_start = time.time()

    all_track_predicted_masks, all_track_hitl_round_logs = run_all_tracks_hitl(
        video_subfolder_name=video_subfolder_name,
        track_ground_truth_masks=track_ground_truth_masks,
        track_peak_slices=track_peak_slices,
        volume_shape=(volume_height, volume_width, num_axial_slices),
        background_volume=background_visualization_volume,
        patient_id=patient_id,
        patient_visualization_directory=patient_visualization_dir,
    )

    print(
        f"\n  Inference done in {time.time() - inference_start:.1f}s  "
        f"({len(track_ground_truth_masks)} tracks, shared encoder)"
    )

    per_label_predicted_binary_masks = {}
    for label_id in present_tumor_label_ids:
        label_name = LABEL_NAMES[label_id]
        if label_name in all_track_predicted_masks:
            per_label_predicted_binary_masks[label_id] = (
                all_track_predicted_masks[label_name]
            )

    wt_predicted_mask = all_track_predicted_masks.get("WT")
    tc_predicted_mask = all_track_predicted_masks.get("TC")
    wt_succeeded      = wt_predicted_mask is not None and np.any(wt_predicted_mask)
    tc_succeeded      = tc_predicted_mask is not None and np.any(tc_predicted_mask)

    if not per_label_predicted_binary_masks:
        print(f"  [ERROR] All per-label inferences failed for {patient_id}")
        return []

    # -- Phase 05: merge -> NIfTI ---------------------------------------------
    merged_label_map = merge_masks_to_label_map(
        per_label_predicted_binary_masks,
        (volume_height, volume_width, num_axial_slices),
    )

    nifti_out_dir = os.path.join(OUTPUT_BASE_DIR, "segs_nifti")
    os.makedirs(nifti_out_dir, exist_ok=True)

    merged_nifti_path = os.path.join(nifti_out_dir, f"{patient_id}_pred.nii.gz")
    nib.save(
        nib.Nifti1Image(merged_label_map,
                        voxel_to_world_affine, nifti_header_object),
        merged_nifti_path,
    )
    print(f"\n  Saved NIfTI: {merged_nifti_path}")

    if wt_succeeded:
        wt_nifti_path = os.path.join(
            nifti_out_dir, f"{patient_id}_pred_WT.nii.gz"
        )
        nib.save(
            nib.Nifti1Image(wt_predicted_mask.astype(np.uint8),
                            voxel_to_world_affine, nifti_header_object),
            wt_nifti_path,
        )
        print(f"  Saved WT NIfTI: {wt_nifti_path}")

    if tc_succeeded:
        tc_nifti_path = os.path.join(
            nifti_out_dir, f"{patient_id}_pred_TC.nii.gz"
        )
        nib.save(
            nib.Nifti1Image(tc_predicted_mask.astype(np.uint8),
                            voxel_to_world_affine, nifti_header_object),
            tc_nifti_path,
        )
        print(f"  Saved TC NIfTI: {tc_nifti_path}")

    # Temp frame cleanup
    temp_jpeg_dir = os.path.join(TEMP_VIDEO_DIR, video_subfolder_name)
    if os.path.exists(temp_jpeg_dir):
        shutil.rmtree(temp_jpeg_dir)
        dprint(f"  Cleaned temp frames: {temp_jpeg_dir}")

    # -- Phase 06: metrics ----------------------------------------------------
    patient_metric_rows   = []
    present_labels_string = f"[{','.join(map(str, present_tumor_label_ids))}]"

    print(
        f"\n  {'Track':<22} {'Rnds':>5} {'Final Anchors':<30} "
        f"{'Dice_raw':>10} {'Dice_mrg':>10} "
        f"{'HD95_raw':>10} {'HD95_mrg':>10} "
        f"{'GT vox':>10} {'Pred vox':>10} {'Stop':<35}"
    )
    print(f"  {'-'*165}")

    for label_id in present_tumor_label_ids:
        if label_id not in per_label_predicted_binary_masks:
            print(f"  [SKIP] No output mask for label {label_id}")
            continue

        label_name  = LABEL_NAMES[label_id]
        raw_pred    = per_label_predicted_binary_masks[label_id]
        merged_pred = (merged_label_map == label_id)
        label_gt    = (seg_volume_data == label_id)
        label_log   = all_track_hitl_round_logs.get(label_name, [])

        dice_raw,    hd95_raw    = calculate_metrics(label_gt, raw_pred)
        dice_merged, hd95_merged = calculate_metrics(label_gt, merged_pred)

        n_rounds      = len(label_log)
        final_anchors = label_log[-1]['anchors'] if label_log else []
        stop_reason   = label_log[-1].get('stop_reason', '?') if label_log else '?'

        print(
            f"  Label {label_id} ({label_name:<6}) "
            f"{n_rounds:>5}  {str(final_anchors):<30} "
            f"  {dice_raw:.4f}    {dice_merged:.4f}  "
            f"  {hd95_raw:>8.2f}    {hd95_merged:>8.2f}  "
            f"  {label_gt.sum():>8,}  {raw_pred.sum():>8,}  {stop_reason:<35}"
        )

        show_visualization(
            background_visualization_volume, label_gt, raw_pred,
            patient_id, label_id, dice_raw, hd95_raw, dice_merged, hd95_merged,
            label_log, patient_visualization_dir,
        )

        history_summary = "|".join(
            f"R{e['round']}:D={e['dice']:.4f}" for e in label_log
        )
        csv_writer.writerow([
            patient_id, "RGB(T1c+T2w+T2f)", f"Label_{label_id}", label_name,
            present_labels_string, n_rounds, str(final_anchors), stop_reason,
            f"{dice_raw:.4f}", f"{hd95_raw:.2f}",
            f"{dice_merged:.4f}", f"{hd95_merged:.2f}",
            label_gt.sum(), raw_pred.sum(), "HITL-per-label", history_summary,
        ])

        patient_metric_rows.append({
            'patient_id'   : patient_id,
            'label_id'     : label_id,
            'label_name'   : label_name,
            'n_rounds'     : n_rounds,
            'final_anchors': final_anchors,
            'stop_reason'  : stop_reason,
            'dice'         : dice_raw,
            'hd95'         : hd95_raw,
            'dice_merged'  : dice_merged,
            'hd95_merged'  : hd95_merged,
        })

    # WT metrics
    if wt_succeeded:
        wt_gt         = np.isin(seg_volume_data, wt_present)
        dice_wt, hd95_wt = calculate_metrics(wt_gt, wt_predicted_mask)
        wt_log        = all_track_hitl_round_logs.get("WT", [])
        n_rounds_wt   = len(wt_log)
        wt_anchors    = wt_log[-1]['anchors'] if wt_log else []
        wt_stop       = wt_log[-1].get('stop_reason', '?') if wt_log else '?'

        print(
            f"  {'WT (indep.)':<22} {n_rounds_wt:>5}  "
            f"{str(wt_anchors):<30} "
            f"  {dice_wt:.4f}    {'N/A':>8}  "
            f"  {hd95_wt:>8.2f}    {'N/A':>8}  "
            f"  {wt_gt.sum():>8,}  {wt_predicted_mask.sum():>8,}  {wt_stop:<35}"
        )

        show_visualization(
            background_visualization_volume, wt_gt, wt_predicted_mask,
            patient_id, 0, dice_wt, hd95_wt, dice_wt, hd95_wt,
            wt_log, patient_visualization_dir, track_label="WT",
        )

        wt_history = "|".join(
            f"R{e['round']}:D={e['dice']:.4f}" for e in wt_log
        )
        csv_writer.writerow([
            patient_id, "RGB(T1c+T2w+T2f)", "WT",
            f"WholeTumour(labels={wt_present})", present_labels_string,
            n_rounds_wt, str(wt_anchors), wt_stop,
            f"{dice_wt:.4f}", f"{hd95_wt:.2f}", "N/A", "N/A",
            wt_gt.sum(), wt_predicted_mask.sum(),
            "HITL-independent-WT", wt_history,
        ])

        patient_metric_rows.append({
            'patient_id'   : patient_id,
            'label_id'     : "WT",
            'label_name'   : "WT",
            'n_rounds'     : n_rounds_wt,
            'final_anchors': wt_anchors,
            'stop_reason'  : wt_stop,
            'dice'         : dice_wt,
            'hd95'         : hd95_wt,
            'dice_merged'  : dice_wt,
            'hd95_merged'  : hd95_wt,
        })
    else:
        print("  [SKIP] WT inference failed or no WT labels present")

    # TC metrics
    if tc_succeeded:
        tc_gt         = np.isin(seg_volume_data, tc_present)
        dice_tc, hd95_tc = calculate_metrics(tc_gt, tc_predicted_mask)
        tc_log        = all_track_hitl_round_logs.get("TC", [])
        n_rounds_tc   = len(tc_log)
        tc_anchors    = tc_log[-1]['anchors'] if tc_log else []
        tc_stop       = tc_log[-1].get('stop_reason', '?') if tc_log else '?'

        print(
            f"  {'TC (indep.)':<22} {n_rounds_tc:>5}  "
            f"{str(tc_anchors):<30} "
            f"  {dice_tc:.4f}    {'N/A':>8}  "
            f"  {hd95_tc:>8.2f}    {'N/A':>8}  "
            f"  {tc_gt.sum():>8,}  {tc_predicted_mask.sum():>8,}  {tc_stop:<35}"
        )

        show_visualization(
            background_visualization_volume, tc_gt, tc_predicted_mask,
            patient_id, 0, dice_tc, hd95_tc, dice_tc, hd95_tc,
            tc_log, patient_visualization_dir, track_label="TC",
        )

        tc_history = "|".join(
            f"R{e['round']}:D={e['dice']:.4f}" for e in tc_log
        )
        csv_writer.writerow([
            patient_id, "RGB(T1c+T2w+T2f)", "TC",
            f"TumourCore(labels={tc_present})", present_labels_string,
            n_rounds_tc, str(tc_anchors), tc_stop,
            f"{dice_tc:.4f}", f"{hd95_tc:.2f}", "N/A", "N/A",
            tc_gt.sum(), tc_predicted_mask.sum(),
            "HITL-independent-TC", tc_history,
        ])

        patient_metric_rows.append({
            'patient_id'   : patient_id,
            'label_id'     : "TC",
            'label_name'   : "TC",
            'n_rounds'     : n_rounds_tc,
            'final_anchors': tc_anchors,
            'stop_reason'  : tc_stop,
            'dice'         : dice_tc,
            'hd95'         : hd95_tc,
            'dice_merged'  : dice_tc,
            'hd95_merged'  : hd95_tc,
        })
    else:
        print("  [SKIP] TC inference failed or no TC labels present")

    torch.cuda.empty_cache()
    gc.collect()

    print(
        f"\n  Patient {patient_id} done in "
        f"{time.time() - patient_start_time:.1f}s"
    )
    return patient_metric_rows


# =============================================================================
# Main loop + final summary
# =============================================================================

def run_pipeline():
    print(f"\n{'*'*65}")
    print(f"  BraTS 2024  *  MedSAM2  *  HUMAN-IN-THE-LOOP iterative anchoring")
    print(f"{'*'*65}")
    print(f"  RGB mapping             :  R=T1c  G=T2w  B=T2f(FLAIR)")
    print(f"  Normalization           :  PER SLICE  (each slice's own p0.5/p99.5)")
    print(f"  HITL seed               :  single peak (densest) slice per track")
    print(f"  HITL_MAX_ITERATIONS     :  {HITL_MAX_ITERATIONS}")
    print(f"  HITL_MIN_IMPROVEMENT    :  {HITL_MIN_IMPROVEMENT}")
    print(f"  HITL_DICE_TARGET        :  {HITL_DICE_TARGET}")
    print(f"  Next-anchor rule        :  argmax |GT_px(z) - pred_px(z)|")
    print(f"  Shared encoder          :  1 init_state  *  N reset_state cycles")
    print(f"  Propagation             :  bidirectional (forward + backward union)")
    print(f"  Merge priority          :  SNFH < NETC < ET < RC")
    print(f"  WT labels               :  {WT_LABELS}")
    print(f"  TC labels               :  {TC_LABELS}")
    print(f"  Metrics                 :  Dice + HD95  (raw + merged)")
    print(f"  Norm histogram          :  per-slice, linear Y, 3 stages x 3 modalities")
    print(f"  Prompting mode          :  MULTI-BBOX (one box per connected component)")
    print(f"  BBOX_MIN_COMPONENT_AREA :  {BBOX_MIN_COMPONENT_AREA} px")
    print(f"  Round slice plot        :  5 cols (GT|Pred|GT+Pred|Delta|BboxInput)")
    print(f"  Standalone bbox images  :  visualizations/<id>/bbox_inputs/")
    print(f"  Dataset                 :  {os.path.abspath(DATASET_DIR)}")
    print(f"  Output                  :  {OUTPUT_BASE_DIR}")
    print(f"  MedSAM2                 :  {MEDSAM2_PATH}")
    print(f"  Max patients            :  {NUM_PATIENTS}")
    print(f"{'*'*65}\n")

    if not os.path.isdir(DATASET_DIR):
        raise FileNotFoundError(f"Dataset not found: {DATASET_DIR}")
    if not os.path.isdir(MEDSAM2_PATH):
        raise FileNotFoundError(f"MedSAM2 not found: {MEDSAM2_PATH}")

    if os.path.exists(OUTPUT_BASE_DIR):
        print(f"  [CLEANUP] Deleting existing output directory: {OUTPUT_BASE_DIR}")
        shutil.rmtree(OUTPUT_BASE_DIR)
        print(f"  [CLEANUP] Deleted.")
    os.makedirs(OUTPUT_BASE_DIR, exist_ok=True)
    os.makedirs(TEMP_VIDEO_DIR,  exist_ok=True)

    all_patient_dirs = sorted([
        d for d in os.listdir(DATASET_DIR)
        if os.path.isdir(os.path.join(DATASET_DIR, d))
    ])
    if NUM_PATIENTS is not None:
        all_patient_dirs = all_patient_dirs[:NUM_PATIENTS]

    print(f"Patients to process: {len(all_patient_dirs)}\n")

    results_csv_path = os.path.join(
        OUTPUT_BASE_DIR,
        f"results_hitl_{datetime.now().strftime('%Y%m%d_%H%M')}.csv",
    )
    results_accumulator = []

    with open(results_csv_path, 'w', newline='') as csv_file:
        csv_writer = csv.writer(csv_file)
        csv_writer.writerow([
            "PatientID", "Modality", "TrackID", "TrackName",
            "PresentLabels", "HITL_Rounds", "FinalAnchors", "StopReason",
            "Dice_raw", "HD95_raw", "Dice_merged", "HD95_merged",
            "GT_voxels", "Pred_voxels", "InferenceMode", "HITL_RoundHistory",
        ])

        failed_patients = []
        for idx, patient_id in enumerate(all_patient_dirs):
            patient_dir = os.path.join(DATASET_DIR, patient_id)
            print(f"\n[{idx + 1}/{len(all_patient_dirs)}] {patient_id}")
            try:
                rows = process_patient(
                    patient_id, patient_dir, csv_writer, results_accumulator
                )
                csv_file.flush()
                results_accumulator.extend(rows)
                if not rows:
                    failed_patients.append(patient_id)
            except Exception as e:
                print(f"  [FATAL] {patient_id}: {e}")
                import traceback
                traceback.print_exc()
                failed_patients.append(patient_id)

    # -- Final summary --------------------------------------------------------
    print(f"\n\n{'#'*65}")
    print(f"  FINAL SUMMARY  --  HITL iterative anchoring  (multi-bbox)")
    print(
        f"  Settings: max_iter={HITL_MAX_ITERATIONS} | "
        f"min_improvement={HITL_MIN_IMPROVEMENT} | "
        f"dice_target={HITL_DICE_TARGET} | "
        f"bbox_min_area={BBOX_MIN_COMPONENT_AREA}"
    )
    print(f"{'#'*65}")
    print(f"  Processed:       {len(all_patient_dirs)} patients")
    print(f"  Failed/skipped:  {len(failed_patients)}  {failed_patients[:10]}")
    print(f"  Results CSV:     {results_csv_path}\n")

    if results_accumulator:

        for phase_desc, dice_key, hd95_key in [
            ("RAW (before merge)",            'dice',        'hd95'),
            ("MERGED (after priority paint)", 'dice_merged', 'hd95_merged'),
        ]:
            print(f"  --- {phase_desc} ---")
            print(
                f"  {'Label':<22} {'N':>5} {'Dice Mean':>12} {'Dice Std':>10} "
                f"{'HD95 Mean':>12} {'HD95 Std':>10} {'Avg Rounds':>12}"
            )
            print(f"  {'-'*85}")
            for label_id in TARGET_LABELS:
                rows = [
                    r for r in results_accumulator
                    if r['label_id'] == label_id
                ]
                if not rows:
                    continue
                dice_vals  = [r[dice_key]    for r in rows]
                hd95_vals  = [r[hd95_key]    for r in rows]
                round_vals = [r['n_rounds']  for r in rows]
                print(
                    f"  {f'Label {label_id} ({LABEL_NAMES[label_id]})':<22} "
                    f"{len(rows):>5} "
                    f"{np.mean(dice_vals):>12.4f}  {np.std(dice_vals):>10.4f} "
                    f"{np.mean(hd95_vals):>12.4f}  {np.std(hd95_vals):>10.4f} "
                    f"{np.mean(round_vals):>12.2f}"
                )
            print()

        print(f"  --- INDEPENDENT WT / TC TRACKS ---")
        print(
            f"  {'Track':<22} {'N':>5} {'Dice Mean':>12} {'Dice Std':>10} "
            f"{'HD95 Mean':>12} {'HD95 Std':>10} {'Avg Rounds':>12}"
        )
        print(f"  {'-'*85}")
        for track in ["WT", "TC"]:
            rows = [r for r in results_accumulator if r['label_id'] == track]
            if not rows:
                print(f"  {track:<22}  -- no data --")
                continue
            dice_vals  = [r['dice']    for r in rows]
            hd95_vals  = [r['hd95']    for r in rows]
            round_vals = [r['n_rounds'] for r in rows]
            print(
                f"  {track:<22} {len(rows):>5} "
                f"{np.mean(dice_vals):>12.4f}  {np.std(dice_vals):>10.4f} "
                f"{np.mean(hd95_vals):>12.4f}  {np.std(hd95_vals):>10.4f} "
                f"{np.mean(round_vals):>12.2f}"
            )
        print()

        # -- Tuning advisory --------------------------------------------------
        from collections import Counter

        all_round_counts = [r['n_rounds'] for r in results_accumulator]
        all_dice_vals    = [r['dice']     for r in results_accumulator]
        avg_rounds       = float(np.mean(all_round_counts))
        max_rounds_used  = int(np.max(all_round_counts))
        avg_dice         = float(np.mean(all_dice_vals))

        n_hit_budget = sum(
            1 for r in results_accumulator
            if (r.get('stop_reason') or '').startswith('max_iter')
        )
        n_converged = len(results_accumulator) - n_hit_budget

        heaviest = max(results_accumulator, key=lambda r: r['n_rounds'])

        peak_z_vals = [
            r['final_anchors'][0]
            for r in results_accumulator
            if r.get('final_anchors')
        ]

        print(f"  --- TUNING ADVISORY ---")
        print()
        print(f"  HITL_MAX_ITERATIONS  (currently {HITL_MAX_ITERATIONS})")
        print(f"    Tracks that hit the budget cap : {n_hit_budget}")
        print(f"    Tracks that converged early    : {n_converged}")
        print(f"    Avg rounds used / max used     : {avg_rounds:.1f} / {max_rounds_used}"
              f"  (heaviest: {heaviest['label_name']})")
        if n_hit_budget == 0:
            print(f"    -> No track was cut off.  Current cap ({HITL_MAX_ITERATIONS}) is sufficient.")
            suggested = max(max_rounds_used + 1, 3)
            if suggested < HITL_MAX_ITERATIONS:
                print(f"    -> Safe to reduce to {suggested} "
                      f"(max actually used = {max_rounds_used}).")
        else:
            print(f"    -> {n_hit_budget} track(s) cut off.  "
                  f"Increase HITL_MAX_ITERATIONS above {HITL_MAX_ITERATIONS}.")
        print()

        print(f"  BBOX_MIN_COMPONENT_AREA  (currently {BBOX_MIN_COMPONENT_AREA})")
        print(f"    Controls how small a disconnected region must be to get its")
        print(f"    own bounding-box prompt.  Raise to suppress noise islands;")
        print(f"    lower to capture tiny satellite lesions.")
        print()

        print(f"  MIN_FOREGROUND_VOXELS_PER_SLICE  (currently {MIN_FOREGROUND_VOXELS_PER_SLICE})")
        if peak_z_vals:
            print(f"    Peak-slice z range across all tracks : [{min(peak_z_vals)}, {max(peak_z_vals)}]")
        print(f"    Avg Dice across all tracks           : {avg_dice:.4f}")
        if avg_dice < 0.75:
            print(f"    -> Low Dice. Consider raising to 150-300 "
                  f"if histogram Row 0 shows noisy edge slices.")
        else:
            print(f"    -> Dice reasonable. Current value ({MIN_FOREGROUND_VOXELS_PER_SLICE}) adequate.")
            print(f"    -> If histogram Row 2 still shows artefacts, try 150-300.")
        print()

    print(f"All outputs saved to: {OUTPUT_BASE_DIR}")


if __name__ == "__main__":
    run_pipeline()