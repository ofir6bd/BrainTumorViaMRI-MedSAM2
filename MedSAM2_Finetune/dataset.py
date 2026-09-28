"""Training samples: short clips of consecutive slices, read from the prompt cache.

A sample is one clip — `video.num_frames` consecutive slices, starting on a slice that looks
like an anchor. **Every** frame carries its own YOLO hint; frame 0 is segmented without
memory, the rest with the memory of the frames before them. That is exactly one pass of
inference, only shorter.

Half the clips (`video.reverse_fraction`) are fed in descending z. To the model that is
still "first frame, then the next ones", which is what the downward pass does at inference.

**Hint augmentation** (`augment`). If the hint in training is always YOLO's real answer,
the cheapest thing to learn is to copy it — which is what the anchor-only version ended up
doing. So some frames get a *worse* hint than YOLO gave: a blob dropped, the whole slice
blanked, the outline grown or shrunk, the map shifted a few pixels, the confidence cut moved.
The target stays the expert mask, so the model is rewarded for fixing the hint, not for
trusting it. Only training clips are augmented.

Clips are drawn patient by patient so each cache file is read once.
"""
from collections import OrderedDict

import numpy as np
import torch
import torch.nn.functional as F
from scipy import ndimage
from torch.utils.data import Dataset

from . import anchors as anchor_rules
from .common import to_model_image
from .yolo_prompts import FIELDS, head_of, hint_logits, load_prompt


def clip_sources(cfg, patients):
    """Read each patient's cache once: how many slices it has, and how good an anchor each
    slice would make. Kept in memory so the clips can be drawn again every round without
    touching the disk — the arrays are a few hundred floats per patient."""
    n_frames = cfg["video"]["num_frames"]
    out = []
    for p in patients:
        entry = load_prompt(cfg, p["id"], keys=("z", "prob", "scores"))
        if len(entry["z"]) < n_frames:
            continue
        out.append({"id": p["id"], "n": len(entry["z"]),
                    "strength": anchor_rules.strength(entry, cfg)})
    return out


def sample_clips(cfg, sources, seed=0):
    """Which clips to train on: (patient id, first slice index, step).

    `step` is +1 or -1 — a clip read backwards is the same slices in the other order.
    Most clips start on a slice YOLO is confident about, because that is what an anchor
    looks like at inference; the rest start anywhere, so empty stretches are seen too.
    Drawn again with a new seed every round.
    """
    d, v = cfg["data"], cfg["video"]
    rng = np.random.default_rng(seed)
    n_frames = v["num_frames"]
    items = []
    for src in sources:
        n = src["n"]
        strength = src["strength"]
        strong = np.nonzero(strength > 0)[0]
        want = int(d["clips_per_patient"])
        n_tumour = int(round(want * float(d["tumour_clip_fraction"])))
        starts = []
        if len(strong):
            weights = strength[strong] / strength[strong].sum()
            starts += list(rng.choice(strong, size=min(n_tumour, len(strong) * 3),
                                      replace=True, p=weights))
        starts += list(rng.integers(0, n, size=want - len(starts)))
        for start in starts[:want]:
            step = -1 if rng.random() < float(v["reverse_fraction"]) else 1
            start = int(start)
            # keep the whole clip inside the volume, in the direction it is read
            start = min(start, n - n_frames) if step == 1 else max(start, n_frames - 1)
            items.append((src["id"], start, step))
    return items


def augment_hint(prob, rng, a, other=None):
    """One slice's uint8 probability map, made worse in one random way (or left alone).

    `other` is the same slice under the other confidence cut (prob vs prob_f), so that
    "YOLO drew a low-confidence blob it should not have" and the reverse are both seen.
    """
    if rng.random() >= float(a["p"]):
        return prob
    kinds = [k for k in ("drop_blob", "drop_slice", "grow", "shrink", "shift", "cut")
             if float(a.get(k, 0)) > 0]
    weights = np.array([float(a[k]) for k in kinds])
    kind = rng.choice(kinds, p=weights / weights.sum())
    if kind == "cut" and other is not None:
        return other
    if not prob.any():
        return prob
    if kind == "drop_slice":
        return np.zeros_like(prob)
    if kind == "drop_blob":
        lab, n = ndimage.label(prob > 127)
        if n < 1:
            return prob
        # drop one blob, dilated a little so its soft edge goes with it
        gone = ndimage.binary_dilation(lab == rng.integers(1, n + 1), iterations=3)
        return np.where(gone, 0, prob).astype(np.uint8)
    if kind in ("grow", "shrink"):
        size = int(rng.integers(3, int(a.get("morph_max", 7)) + 1)) | 1
        op = ndimage.grey_dilation if kind == "grow" else ndimage.grey_erosion
        return op(prob, size=(size, size))
    if kind == "shift":
        m = int(a.get("shift_max", 4))
        dy, dx = rng.integers(-m, m + 1, size=2)
        return np.roll(prob, (int(dy), int(dx)), axis=(0, 1))
    return prob


class ClipDataset(Dataset):
    def __init__(self, cfg, items, train=True, cache_size=2):
        self.cfg = cfg
        self.items = items
        self.train = train
        self.size = cfg["model"]["image_size"]
        self.n_frames = cfg["video"]["num_frames"]
        self.cache_size = cache_size
        self._store = OrderedDict()

    def __len__(self):
        return len(self.items)

    def _entry(self, patient_id):
        if patient_id not in self._store:
            self._store[patient_id] = load_prompt(self.cfg, patient_id)
            while len(self._store) > self.cache_size:
                self._store.popitem(last=False)
        self._store.move_to_end(patient_id)
        return self._store[patient_id]

    def hints(self, entry, idx, rng):
        """uint8 probability maps for the clip's frames, augmented when training."""
        field = FIELDS[self.cfg["prompt"]["variant"]]
        other = "prob" if field == "prob_f" else "prob_f"
        a = self.cfg.get("augment") or {}
        out = []
        for k in idx:
            prob = entry[field][k]
            if self.train and a.get("p"):
                prob = augment_hint(prob, rng, a, entry[other][k])
            out.append(prob)
        return np.stack(out)

    def __getitem__(self, i):
        patient_id, start, step = self.items[i]
        entry = self._entry(patient_id)
        idx = [start + step * k for k in range(self.n_frames)]
        rng = np.random.default_rng()
        rgb = entry["rgb"][idx]
        gt = entry["gt"][idx]
        prompt = hint_logits(self.hints(entry, idx, rng), head_of(entry, idx, self.cfg), self.cfg)

        if self.train and rng.random() < float(self.cfg["train"]["fliplr"]):
            # axis 0 of these arrays is the left-right axis of the head, so a flip here is
            # still a plausible brain; tumours occur on both sides.
            rgb, gt, prompt = rgb[:, ::-1], gt[:, ::-1], prompt[:, ::-1]

        size = self.size
        images = torch.stack([to_model_image(np.ascontiguousarray(f), size) for f in rgb])
        masks = F.interpolate(torch.from_numpy(np.ascontiguousarray(gt)).float()[:, None],
                              size=(size, size), mode="nearest")[:, 0] > 0.5
        return {"images": images, "masks": masks, "prompt": to_prompt(prompt, size),
                "patient": patient_id, "z": [int(entry["z"][k]) for k in idx]}


def to_prompt(logits, size):
    """[K, H, W] logits -> [K, 1, size, size]. Handed to the model already at image_size,
    because SAM2 binarises a mask prompt it has to resize itself."""
    pr = torch.from_numpy(np.ascontiguousarray(logits)).float()[:, None]
    return F.interpolate(pr, size=(size, size), mode="bilinear", align_corners=False)


def collate(batch):
    """Into the shapes the vendored tracking code expects: frames first, then clips."""
    return {"images": torch.stack([b["images"] for b in batch], dim=1),     # [T, B, 3, S, S]
            "masks": torch.stack([b["masks"] for b in batch], dim=1),       # [T, B, S, S]
            "prompts": torch.stack([b["prompt"] for b in batch], dim=1),    # [T, B, 1, S, S]
            "patient": [b["patient"] for b in batch], "z": [b["z"] for b in batch]}


class PatientShuffle(torch.utils.data.Sampler):
    """Shuffle the patients, and the clips inside each patient, but keep a patient's clips
    together so its cache file is read once. A new one is built every round with a new seed,
    alongside the freshly drawn clips."""

    def __init__(self, items, seed=0):
        self.items = items
        self.seed = seed
        self.groups = OrderedDict()
        for i, (patient_id, _start, _step) in enumerate(items):
            self.groups.setdefault(patient_id, []).append(i)

    def __len__(self):
        return len(self.items)

    def __iter__(self):
        rng = np.random.default_rng(self.seed)
        keys = list(self.groups)
        rng.shuffle(keys)
        for key in keys:
            idx = list(self.groups[key])
            rng.shuffle(idx)
            yield from idx
