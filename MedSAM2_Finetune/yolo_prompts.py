"""Turn a fine-tuned YOLO into the per-voxel prompt MedSAM2 is trained on.

YOLO-seg gives two things per slice: a **confidence score per blob** and, behind the
binary mask it normally shows, a **probability per pixel**. Ultralytics throws the second
one away — `process_mask_native` ends with `.gt_(0.0)`, i.e. it keeps only "is this pixel
above probability 0.5". The soft value is what MedSAM2 wants, because SAM2's dense prompt
is a *logit* map, exactly the same kind of number YOLO produced before the cut.

So we let Ultralytics do its own (letterboxing, NMS, native-size mask) work untouched and
keep the soft masks it was about to throw away (`YOLO_finetune.common.keep_soft_masks`,
shared with YOLO's own scorer). Nothing in the installed package is edited.

Per slice the cache keeps three uint8 probability maps, so that changing the *variant*
never needs a rebuild:
  prob    max over all blobs                     -> variants `max`, `binary`
  prob_w  max over blobs of probability x score  -> variant  `max_weighted`
  prob_f  max over blobs whose score >= score_min -> variant `conf_filtered`
plus the best score and the blob count per slice, which the page shows.

The frame and the expert mask are stored next to them, so one patient file holds
everything a training step needs and training never touches the NIfTI files again.
"""
import os

import numpy as np
from PIL import Image

from YOLO_finetune.common import keep_soft_masks

from .common import (CACHE_DIR, CHANNELS, brain_slices, load_volumes, prob_to_logit, read_json,
                     rgb_slice, short_hash, write_json, yolo_run)

BATCH = 24  # slices per YOLO forward pass


def prompt_key(cfg):
    """Cache identity: everything that changes the numbers in the cache, and nothing else.

    The patient list is deliberately *not* part of it — one file per patient means a later
    run can add patients to the same cache instead of rebuilding it.
    """
    p, run = cfg["prompt"], yolo_run(cfg["prompt"]["yolo_run"])
    return short_hash({"yolo_run": run["id"], "which": run["which"], "imgsz": run["imgsz"],
                       "channels": CHANNELS, "yolo_conf": p["yolo_conf"], "score_min": p["score_min"],
                       "min_fg_voxels": cfg["data"]["min_fg_voxels"]})


def cache_dir(cfg):
    return os.path.join(CACHE_DIR, prompt_key(cfg))


def patient_file(cfg, patient_id):
    return os.path.join(cache_dir(cfg), f"{patient_id}.npz")


def _u8(prob):
    return np.clip(prob * 255.0 + 0.5, 0, 255).astype(np.uint8)


def predict_patient(model, patient, cfg):
    """Every brain slice of one patient through YOLO; returns the cache arrays."""
    p = cfg["prompt"]
    run = yolo_run(p["yolo_run"])
    vols = load_volumes(patient)
    zs = brain_slices(vols["FLAIR"], cfg["data"]["min_fg_voxels"])
    h, w = vols["FLAIR"].shape[:2]
    out = {k: np.zeros((len(zs), h, w), np.uint8) for k in ("prob", "prob_w", "prob_f")}
    rgb = np.zeros((len(zs), h, w, 3), np.uint8)
    gt = np.zeros((len(zs), h, w), np.uint8)
    scores, nblobs = np.zeros(len(zs), np.float32), np.zeros(len(zs), np.int16)

    for i in range(0, len(zs), BATCH):
        chunk = zs[i:i + BATCH]
        # PIL images, not numpy: Ultralytics reads a numpy array as BGR and would swap
        # the T1C and FLAIR channels relative to the pictures the model was trained on.
        frames = [rgb_slice(vols, z, cfg["data"]["min_fg_voxels"]) for z in chunk]
        for j, z in enumerate(chunk):
            rgb[i + j] = frames[j]
            gt[i + j] = (vols["SEG"][:, :, z] > 0).astype(np.uint8)
        imgs = [Image.fromarray(f) for f in frames]
        with keep_soft_masks() as soft:
            results = model.predict(imgs, conf=p["yolo_conf"], imgsz=run["imgsz"], retina_masks=True,
                                    verbose=False, device=cfg["train"]["device"])
        taken = 0
        for j, r in enumerate(results):
            k = i + j
            conf = r.boxes.conf.cpu().numpy() if r.boxes is not None else np.zeros(0, np.float32)
            # A slice with no detection at all never reaches the mask step, so it consumed
            # nothing from the capture.
            if r.masks is None:
                masks = np.zeros((0, h, w), np.float32)
            else:
                masks = soft[taken].cpu().numpy()
                taken += 1
            if len(masks) != len(conf):
                raise RuntimeError(f"{patient['id']} z={chunk[j]}: "
                                   f"{len(masks)} soft masks but {len(conf)} blobs")
            nblobs[k] = len(conf)
            scores[k] = float(conf.max()) if conf.size else 0.0
            if len(masks):
                out["prob"][k] = _u8(masks.max(axis=0))
                out["prob_w"][k] = _u8((masks * conf[:, None, None]).max(axis=0))
                keep = conf >= p["score_min"]
                if keep.any():
                    out["prob_f"][k] = _u8(masks[keep].max(axis=0))

    return {"z": np.asarray(zs, np.int16), "shape": np.asarray([h, w], np.int16),
            "scores": scores, "nblobs": nblobs, "rgb": rgb, "gt": gt,
            "gt_px": gt.reshape(len(zs), -1).sum(axis=1).astype(np.int32),
            **out}


def ensure_cache(cfg, patients, progress=None, log=print, model=None):
    """Make sure every given patient has a prompt file; returns (dir, how many were built)."""
    d = cache_dir(cfg)
    os.makedirs(d, exist_ok=True)
    missing = [p for p in patients if not os.path.exists(patient_file(cfg, p["id"]))]
    if not read_json(os.path.join(d, "meta.json")):
        run = yolo_run(cfg["prompt"]["yolo_run"])
        write_json(os.path.join(d, "meta.json"),
                   {"key": prompt_key(cfg), "yolo_run": run["id"], "yolo_weights": run["which"],
                    "imgsz": run["imgsz"], "yolo_conf": cfg["prompt"]["yolo_conf"],
                    "score_min": cfg["prompt"]["score_min"],
                    "min_fg_voxels": cfg["data"]["min_fg_voxels"], "channels": list(CHANNELS)})
    if not missing:
        log(f"[prompts] cache {d} already has all {len(patients)} patients")
        return d, 0

    if model is None:
        from ultralytics import YOLO
        model = YOLO(yolo_run(cfg["prompt"]["yolo_run"])["weights"])
    log(f"[prompts] building {len(missing)} of {len(patients)} patients in {d}")
    for i, patient in enumerate(missing, 1):
        arrays = predict_patient(model, patient, cfg)
        tmp = patient_file(cfg, patient["id"]) + ".tmp.npz"
        np.savez_compressed(tmp, **arrays)
        os.replace(tmp, patient_file(cfg, patient["id"]))
        if progress:
            progress(i, len(missing))
    log(f"[prompts] built {len(missing)} patients")
    return d, len(missing)


# ------------------------------------------------------------------ reading it back
def load_prompt(cfg, patient_id, keys=None):
    """One patient's cache file. `keys` reads only those arrays (the small ones are enough
    to decide which slices to train on, without unpacking the pictures)."""
    path = patient_file(cfg, patient_id)
    if not os.path.exists(path):
        raise FileNotFoundError(f"no prompt cache for {patient_id} — build it first ({path})")
    with np.load(path) as f:
        return {k: f[k] for k in (keys or f.files)}


FIELDS = {"max": "prob", "max_weighted": "prob_w", "conf_filtered": "prob_f"}


def prompt_logits(entry, cfg, index=None, fields=None):
    """The hint MedSAM2 is given, as logits, for one slice (`index`) or all of them.

    Every variant is a soft map: which of the three cached probability maps it reads is the
    only difference. `fields` lets the training augmentation pick a different map per slice.
    """
    n = len(entry["z"])
    idx = np.arange(n) if index is None else np.atleast_1d(index)
    field = FIELDS[cfg["prompt"]["variant"]]
    prob = np.stack([entry[(fields[j] if fields else field)][k] for j, k in enumerate(idx)])
    return hint_logits(prob, head_of(entry, idx, cfg), cfg)


def head_of(entry, idx, cfg):
    """Where the frame has a head in it (None when `prompt.clip_to_head` is off)."""
    if not cfg["prompt"].get("clip_to_head", True):
        return None
    if "rgb" not in entry:
        raise KeyError("clip_to_head needs the frames: load the patient without a `keys` filter")
    return entry["rgb"][idx].max(axis=-1) > 0


def hint_logits(prob, head, cfg):
    """uint8 probability maps [K, H, W] -> the logits handed to MedSAM2.

    A slice YOLO left empty becomes a confident "nothing" (`prompt.empty_logit`), and so does
    anywhere outside the head (`prompt.clip_to_head`): YOLO is asked for blobs down to
    `yolo_conf` (0.05), where boxes are close to random — on BraTS-GLI-00008-101 z=70 a 0.050
    blob covered 1669 px, 1184 of them outside the skull. A voxel counts as head if *any*
    channel is non-zero, which excludes 4 of 734,193 expert tumour voxels over 10 val patients.
    """
    p = cfg["prompt"]
    k = float(p["empty_logit"])
    logits = prob_to_logit(prob) * float(p["logit_scale"])
    logits[prob.max(axis=(1, 2)) == 0] = k
    if head is not None:
        logits = np.where(head, logits, k)
    return logits.astype(np.float32)
