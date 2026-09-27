"""MedSAM2 propagation that accepts graded (non-binary) mask prompts.

`GTvsYOLO/medsam2_runner.py` is reused wholesale for the predictor singleton, the frame
writing and the RGB construction — the frames are byte-identical to the reference run,
which is the point. Only prompt *registration* is different, because SAM2 binarises a
soft mask twice on the way in:

1. `add_new_mask` opens with `torch.tensor(mask, dtype=torch.bool)` when it is handed a
   numpy array, so float values die immediately. Passing a `torch.Tensor` skips that.
2. When the mask is not already at the model's `image_size`, it resizes and then applies
   `(mask_inputs >= 0.5).float()`. BraTS slices are 240x240 and the t512 model wants 512,
   so this always fires. Pre-resizing to 512 ourselves skips it.

Past those two, the graded values are used as-is: `use_mask_input_as_output_without_sam`
is true in the t512 config, so the prompt becomes the logit `20p - 10` directly
(`sam2_base._use_mask_as_output`) and is written to the memory bank as a float, which is
what carries the gradation to the non-anchor slices.

A prompt that is already 0/1 is passed through as a numpy bool array — SAM2's own path,
unchanged — so variant O reproduces the reference exactly rather than merely closely.
"""
import numpy as np

from GTvsYOLO import medsam2_runner as M

# Re-exported so callers do not need to import both modules.
build_rgb_volume = M.build_rgb_volume
write_frames = M.write_frames
get_predictor = M.get_predictor
MASK_THRESHOLD = M.MASK_THRESHOLD


def is_binary(mask):
    """True when every value is exactly 0 or 1 — i.e. nothing would be lost to bool."""
    a = np.asarray(mask)
    return bool(np.all((a == 0) | (a == 1)))


def _prompt_tensor(mask, image_size):
    """Graded mask as a float tensor already at the model's resolution.

    Bilinear + antialias to match what SAM2 would have done itself, minus the `>= 0.5`
    it applies afterwards.
    """
    import torch
    import torch.nn.functional as F

    a = np.ascontiguousarray(np.asarray(mask, dtype=np.float32))
    t = torch.from_numpy(a)[None, None]
    if t.shape[-2:] != (image_size, image_size):
        t = F.interpolate(t, size=(image_size, image_size), mode="bilinear",
                          align_corners=False, antialias=True)
    return t[0, 0].clamp_(0.0, 1.0)


def register_prompts(predictor, state, prompts):
    """Register one object (`obj_id=1`) per anchor slice, binary or graded."""
    image_size = predictor.image_size
    for z, mask in prompts.items():
        if is_binary(mask):
            predictor.add_new_mask(inference_state=state, frame_idx=z, obj_id=1,
                                   mask=np.asarray(mask).astype(bool))
        else:
            predictor.add_new_mask(inference_state=state, frame_idx=z, obj_id=1,
                                   mask=_prompt_tensor(mask, image_size))


def infer_track(predictor, state, anchor_masks_by_z, volume_shape):
    """Propagate forward from the first anchor, then backward from the last.

    Identical to `GTvsYOLO.medsam2_runner.infer_track` apart from `register_prompts`;
    per-frame logits are merged with `max` and thresholded at `MASK_THRESHOLD`.
    """
    import torch

    H, W, D = volume_shape
    if not anchor_masks_by_z:
        return np.zeros((H, W, D), dtype=bool)

    anchor_zs = sorted(anchor_masks_by_z)
    prompts = {z: anchor_masks_by_z[z] for z in anchor_zs}

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

    with torch.inference_mode(), torch.autocast(device_type="cuda",
                                                dtype=torch.bfloat16):
        per_frame = {}
        register_prompts(predictor, state, prompts)
        collect(min(anchor_zs), False, per_frame)
        predictor.reset_state(state)
        register_prompts(predictor, state, prompts)
        collect(max(anchor_zs), True, per_frame)

    mask = np.zeros((H, W, D), dtype=bool)
    for z in range(D):
        if z in per_frame:
            mask[:, :, z] = per_frame[z] > MASK_THRESHOLD
    return mask
