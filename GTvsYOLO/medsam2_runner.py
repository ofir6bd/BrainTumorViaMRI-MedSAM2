"""MedSAM2 video-predictor plumbing for the GT-vs-YOLO comparison.

Lifted from `src/05_infer_multibbox_hitl.py` (which stays untouched — it is a script,
not an importable module: at import time it pip-installs, reads `config.yaml` and
mutates `sys.path`). Same model, same config, same propagation scheme:

- `MedSAM2_latest.pt` + `sam2/configs/sam2.1_hiera_t512.yaml` via
  `build_sam2_video_predictor`.
- A patient's axial slices are the video's frames (`00000.jpg` .. `000ZZ.jpg`),
  RGB = R:T1c / G:T2w / B:T2f, each channel percentile-normalised per slice.
  NOTE this is *not* the YOLO RGB (which uses R = T1C - T1); MedSAM2 sees exactly the
  frames `05_infer_multibbox_hitl.py` feeds it, YOLO sees exactly what it trained on.
- **Mask** prompts are registered on anchor frames (`add_new_mask`, one object carrying
  the whole-tumour mask of that slice), then propagated forward from the first anchor and
  backward from the last, merging logits per frame.
"""
import os
import sys

import numpy as np
import yaml
from PIL import Image

MASK_THRESHOLD = 0
MIN_FOREGROUND_VOXELS_PER_SLICE = 100

SAM2_CHECKPOINT = "./checkpoints/MedSAM2_latest.pt"
SAM2_CFG = "sam2/configs/sam2.1_hiera_t512.yaml"
_CFG_HYDRA_KEY = "configs/sam2.1_hiera_t512.yaml"


def _config():
    with open("config.yaml", "r") as f:
        return yaml.safe_load(f) or {}


def medsam2_path():
    return os.path.abspath(_config()["paths"]["medsam2"])


def frames_root():
    """`outputs/tmp_frames/gtvsyolo/` — JPEG frame dirs, one per patient."""
    return os.path.join(os.path.abspath(_config()["paths"]["tmp"]), "gtvsyolo")


# ---------------------------------------------------------------------------
# Frames
# ---------------------------------------------------------------------------
def norm_slice_uint8(slice_2d):
    """Per-slice 0.5-99.5 percentile window -> uint8. Near-empty slices go to all-zero."""
    s = slice_2d.astype(np.float32)
    nonzero = s[s != 0]
    if nonzero.size < MIN_FOREGROUND_VOXELS_PER_SLICE:
        return np.zeros(s.shape, dtype=np.uint8)
    lo = float(np.percentile(nonzero, 0.5))
    hi = float(np.percentile(nonzero, 99.5))
    rng = hi - lo
    if rng < 1e-8:
        return np.zeros(s.shape, dtype=np.uint8)
    return np.clip((s - lo) / rng * 255, 0, 255).astype(np.uint8)


def build_rgb_volume(t1c, t2w, t2f):
    """(H,W,D,3) uint8 — R:T1c, G:T2w, B:T2f, per-slice normalised."""
    H, W, D = t1c.shape
    rgb = np.zeros((H, W, D, 3), dtype=np.uint8)
    for z in range(D):
        rgb[:, :, z, 0] = norm_slice_uint8(t1c[:, :, z])
        rgb[:, :, z, 1] = norm_slice_uint8(t2w[:, :, z])
        rgb[:, :, z, 2] = norm_slice_uint8(t2f[:, :, z])
    return rgb


def write_frames(rgb_volume, patient_id):
    """Write the JPEG frame sequence `init_state` reads. Cached: an existing dir with the
    right number of frames is reused."""
    out_dir = os.path.join(frames_root(), patient_id)
    D = rgb_volume.shape[2]
    if os.path.isdir(out_dir):
        existing = [f for f in os.listdir(out_dir) if f.lower().endswith(".jpg")]
        if len(existing) == D:
            return out_dir
    os.makedirs(out_dir, exist_ok=True)
    for z in range(D):
        Image.fromarray(rgb_volume[:, :, z], mode="RGB").save(
            os.path.join(out_dir, f"{z:05d}.jpg"), quality=95)
    return out_dir


# ---------------------------------------------------------------------------
# Predictor
# ---------------------------------------------------------------------------
_PREDICTOR = None


def get_predictor():
    """Load `MedSAM2_latest.pt` once per process. Imported lazily so the Flask app still
    starts when MedSAM2/torch aren't importable."""
    global _PREDICTOR
    if _PREDICTOR is not None:
        return _PREDICTOR

    root = medsam2_path()
    if root not in sys.path:
        sys.path.insert(0, root)
    from sam2.build_sam import build_sam2_video_predictor

    ckpt = os.path.join(root, SAM2_CHECKPOINT)
    cfg = os.path.join(root, SAM2_CFG)
    if not os.path.exists(ckpt):
        raise FileNotFoundError(f"Checkpoint not found: {ckpt}")
    if not os.path.exists(cfg):
        raise FileNotFoundError(f"Config not found: {cfg}")

    _PREDICTOR = build_sam2_video_predictor(
        config_file=_CFG_HYDRA_KEY, ckpt_path=ckpt, apply_postprocessing=False)
    _PREDICTOR.eval()
    return _PREDICTOR


def infer_track(predictor, state, anchor_masks_by_z, volume_shape):
    """Register each anchor slice's mask as a prompt, then propagate through the volume.

    One object (`obj_id=1`) carrying the whole-tumour mask of the slice, per anchor —
    `add_new_mask`, not a box, so the model is handed the segmentation itself.
    Propagates forward from the first anchor, then backward from the last (backward fills
    only frames the forward pass did not reach). Per-frame logits are merged with `max`.

    Returns `(mask_3d, prompts_by_z)`.
    """
    import torch

    H, W, D = volume_shape
    if not anchor_masks_by_z:
        return np.zeros((H, W, D), dtype=bool), {}

    anchor_zs = sorted(anchor_masks_by_z.keys())
    prompts = {z: np.asarray(anchor_masks_by_z[z]).astype(bool) for z in anchor_zs}

    def register():
        for z, mask in prompts.items():
            predictor.add_new_mask(
                inference_state=state, frame_idx=z, obj_id=1, mask=mask)

    def collect(start, reverse, into):
        for frame_idx, obj_ids, logits in predictor.propagate_in_video(
                state, start_frame_idx=start, reverse=reverse):
            if reverse and frame_idx in into:
                continue
            merged = None
            for pos, _ in enumerate(obj_ids):
                lg = logits[pos][0].cpu().numpy()
                merged = lg if merged is None else np.maximum(merged, lg)
            if merged is not None:
                into[frame_idx] = merged

    with torch.inference_mode(), torch.autocast(device_type="cuda", dtype=torch.bfloat16):
        per_frame = {}
        register()
        collect(min(anchor_zs), False, per_frame)
        predictor.reset_state(state)
        register()
        collect(max(anchor_zs), True, per_frame)

    mask = np.zeros((H, W, D), dtype=bool)
    for z in range(D):
        if z in per_frame:
            mask[:, :, z] = per_frame[z] > MASK_THRESHOLD
    return mask, prompts
