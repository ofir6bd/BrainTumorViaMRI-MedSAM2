"""YOLO inference that keeps the confidence Ultralytics normally throws away.

`YOLO/pipeline.py:process_slice` returns only the binary union of the predicted
instances — the per-instance scores are computed and dropped, and the per-pixel mask
probabilities never leave Ultralytics at all. Both are needed here, so this module runs
the same prediction and hands back the full picture.

Two levels of confidence come out:

- **per instance** — `results[0].boxes.conf`, one score per detected blob. Free.
- **per pixel** — Ultralytics binarises the mask inside `ops.process_mask`, which ends
  `return crop_mask(masks.gt_(0.0).byte(), bboxes)`. The values it thresholds are
  logits, so `sigmoid(raw)` is the per-pixel probability, and `sigmoid(x) > 0.5` is
  exactly `x > 0` — the binarisation Ultralytics does anyway. `soft_masks()` swaps in a
  `process_mask` that applies the sigmoid instead of the threshold, so the probability
  survives and thresholding it at 0.5 reproduces the stock mask. Everything else — NMS,
  bbox cropping, letterboxing — is untouched stock code.

Masks come back at the letterboxed model resolution and are resampled to the volume's
native size here: nearest for binary (byte-identical to `YOLO/pipeline.py`), bilinear for
probabilities.
"""
import contextlib

import numpy as np
from scipy import ndimage

# Ultralytics reads a numpy array as BGR and would silently swap R and B; a PIL image is
# honoured as RGB. Same reason as YOLO/pipeline.py:594 — getting this wrong feeds the
# model FLAIR where it learned enhancement and destroys the Dice.
from PIL import Image


@contextlib.contextmanager
def soft_masks(enabled=True):
    """Make `ops.process_mask` return per-pixel probabilities instead of 0/1.

    Stock tail is `crop_mask(masks.gt_(0.0).byte(), bboxes)`; this replaces the threshold
    with a sigmoid and keeps the crop, so pixels outside a detection's own box stay 0
    exactly as before.
    """
    from ultralytics.utils import ops

    if not enabled:
        yield
        return

    original = ops.process_mask

    def process_mask_soft(protos, masks_in, bboxes, shape, upsample=False):
        import torch.nn.functional as F

        c, mh, mw = protos.shape
        if masks_in.shape[0] == 0:
            out_shape = shape if upsample else (mh, mw)
            return protos.new_zeros((0, *out_shape))
        masks = (masks_in @ protos.float().view(c, -1)).view(-1, mh, mw)
        if upsample:
            masks = F.interpolate(masks[None], shape, mode="bilinear")[0]
        else:
            ratios = masks.new_tensor(
                [[mw / shape[1], mh / shape[0], mw / shape[1], mh / shape[0]]])
            bboxes = bboxes * ratios
        return ops.crop_mask(masks.sigmoid(), bboxes)

    ops.process_mask = process_mask_soft
    try:
        yield
    finally:
        ops.process_mask = original


def _to_native_binary(mask, shape):
    """Resample a mask to the volume's in-plane size — same call YOLO/pipeline.py makes."""
    m = np.asarray(mask, dtype=np.float32)
    if m.shape == shape:
        return m >= 0.5
    zoom = (shape[0] / m.shape[0], shape[1] / m.shape[1])
    return ndimage.zoom(m, zoom, order=0) >= 0.5


def _to_native_prob(prob, shape):
    """Bilinear, because a probability field has meaningful in-between values."""
    p = np.asarray(prob, dtype=np.float32)
    if p.shape == shape:
        return np.clip(p, 0.0, 1.0)
    zoom = (shape[0] / p.shape[0], shape[1] / p.shape[1])
    return np.clip(ndimage.zoom(p, zoom, order=1), 0.0, 1.0)


def predict_slice(model, rgb, conf_threshold, shape, want_prob=False):
    """Run YOLO on one RGB slice.

    Returns `{"instances": [{"conf", "mask", "prob"}], "union": bool2d, "error": str|None}`
    with every array at the volume's native `shape`. `prob` is `None` unless `want_prob`.
    """
    out = {"instances": [], "union": np.zeros(shape, dtype=bool), "error": None}
    if model is None:
        return out

    try:
        with soft_masks(want_prob):
            results = model.predict(Image.fromarray(rgb), conf=conf_threshold,
                                    verbose=False)
    except Exception as e:  # noqa: BLE001 - surfaced in the run log, not fatal
        out["error"] = f"{type(e).__name__}: {e}"
        return out

    r0 = results[0]
    if r0.masks is None:
        return out

    confs = []
    if r0.boxes is not None and r0.boxes.conf is not None:
        confs = [float(c) for c in r0.boxes.conf.cpu().numpy()]
    data = r0.masks.data.cpu().numpy().astype(np.float32)

    for i, raw in enumerate(data):
        # Stock Ultralytics gives 0/1 here; under `soft_masks` it gives probabilities.
        # `>= 0.5` is the same cut in both cases, so `mask` matches the stock output.
        mask = _to_native_binary(raw, shape)
        if not mask.any():
            continue
        out["instances"].append({
            "conf": float(confs[i]) if i < len(confs) else 1.0,
            "mask": mask,
            "prob": _to_native_prob(raw, shape) if want_prob else None,
        })
        out["union"] |= mask
    return out


def slice_confidence(instances):
    """One confidence for the whole slice: instance scores weighted by mask area.

    A plain mean lets one tiny doubtful blob drag down a slice dominated by a confident
    one. Area weighting makes the number describe the mask that is actually handed over.
    """
    if not instances:
        return None
    areas = np.array([float(np.count_nonzero(i["mask"])) for i in instances])
    confs = np.array([i["conf"] for i in instances], dtype=np.float64)
    total = areas.sum()
    if total <= 0:
        return float(confs.mean())
    return float((confs * areas).sum() / total)
