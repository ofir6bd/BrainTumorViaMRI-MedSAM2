"""Training samples: short clips of consecutive slices, read from the prompt cache.

A sample is one clip — the **anchor** slice plus the `video.num_frames - 1` slices after it.
Only the anchor is prompted (with YOLO's map); the rest are what the model has to reach
through memory, and they are exactly where the loss has something to teach.

Half the clips (`video.reverse_fraction`) are fed in descending z. To the model that is
still "first frame, then the next ones", so it learns to carry a prediction in both
directions — which is what the two propagation passes do at inference.

Clips are drawn patient by patient so each cache file is read once.
"""
from collections import OrderedDict

import numpy as np
import torch
import torch.nn.functional as F
from torch.utils.data import Dataset

from . import anchors as anchor_rules
from .common import to_model_image
from .yolo_prompts import load_prompt, prompt_logits


def clip_sources(cfg, patients):
    """Read each patient's cache once: how many slices it has, and how good an anchor each
    slice would make. Kept in memory so the clips can be drawn again every round without
    touching the disk — the arrays are a few hundred floats per patient."""
    n_frames = cfg["video"]["num_frames"]
    out = []
    for p in patients:
        entry = load_prompt(cfg, p["id"], keys=("z", "gt_px", "prob", "scores"))
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

    Called again with a new seed every round. Drawing the clips once and reusing them would
    show the model the same ~1,200 positions over and over, when a 300-patient pool holds
    tens of thousands of them.
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
            if step == 1:
                start = min(start, n - n_frames)
            else:
                start = max(start, n_frames - 1)
            items.append((src["id"], start, step))
    return items


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

    def __getitem__(self, i):
        patient_id, start, step = self.items[i]
        entry = self._entry(patient_id)
        idx = [start + step * k for k in range(self.n_frames)]
        rgb = entry["rgb"][idx]
        gt = entry["gt"][idx]
        prompt = prompt_logits(entry, self.cfg, idx[0])[0]        # the anchor is frame 0

        if self.train and np.random.rand() < float(self.cfg["train"]["fliplr"]):
            # axis 0 of these arrays is the left-right axis of the head, so a flip here is
            # still a plausible brain; tumours occur on both sides.
            rgb, gt, prompt = rgb[:, ::-1], gt[:, ::-1], prompt[::-1]

        size = self.size
        images = torch.stack([to_model_image(np.ascontiguousarray(f), size) for f in rgb])
        masks = F.interpolate(torch.from_numpy(np.ascontiguousarray(gt)).float()[:, None],
                              size=(size, size), mode="nearest")[:, 0] > 0.5
        pr = torch.from_numpy(np.ascontiguousarray(prompt)).float()[None, None]
        pr = F.interpolate(pr, size=(size, size), mode="bilinear", align_corners=False)[0]
        return {"images": images, "masks": masks, "prompt": pr,
                "patient": patient_id, "z": [int(entry["z"][k]) for k in idx]}


def collate(batch):
    """Into the shapes the vendored tracking code expects: frames first, then clips."""
    images = torch.stack([b["images"] for b in batch], dim=1)        # [T, B, 3, S, S]
    masks = torch.stack([b["masks"] for b in batch], dim=1)          # [T, B, S, S]
    prompts = torch.stack([b["prompt"] for b in batch])[None]        # [1, B, 1, S, S]
    t, b = images.shape[0], images.shape[1]
    obj_to_frame_idx = torch.stack([
        torch.arange(t)[:, None].expand(t, b),                       # frame index
        torch.arange(b)[None, :].expand(t, b),                       # clip index
    ], dim=-1)
    return {"images": images, "masks": masks, "prompts": prompts,
            "obj_to_frame_idx": obj_to_frame_idx,
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
