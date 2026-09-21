"""Training samples: one axial slice each, read straight from the prompt cache.

A sample is (frame, YOLO prompt, expert mask). All three come from the same patient file,
so training never opens a NIfTI and never re-runs YOLO.

Slices are handed out patient by patient (shuffled inside each patient, patients shuffled
every round). That keeps one patient's file hot in memory while its slices are used, and
costs nothing in correctness: SAM2 normalises per sample (LayerNorm), so a batch drawn
from one patient is not a batch-statistics problem the way it would be with BatchNorm.
"""
from collections import OrderedDict

import numpy as np
import torch
import torch.nn.functional as F
from torch.utils.data import Dataset

from .common import to_model_image
from .yolo_prompts import load_prompt, prompt_logits


def sample_slices(cfg, patients, seed=0, train=True):
    """Which slices to use: every tumour slice, plus a share of the empty ones.

    Empty slices are what teach "draw nothing here", so they are kept — but all of them
    would swamp the tumour ones, hence `neg_fraction`. Evaluation always uses every brain
    slice, so the reported 3D Dice is over the whole patient.
    """
    d = cfg["data"]
    rng = np.random.default_rng(seed)
    items = []
    for p in patients:
        entry = load_prompt(cfg, p["id"], keys=("z", "gt_px"))
        idx = np.arange(len(entry["z"]))
        if not train:
            items += [(p["id"], int(i)) for i in idx]
            continue
        pos = idx[entry["gt_px"] > 0]
        neg = idx[entry["gt_px"] == 0]
        take = int(round(len(neg) * float(d["neg_fraction"])))
        chosen = np.concatenate([pos, rng.choice(neg, size=take, replace=False)]) if take else pos
        if d.get("max_slices_per_patient") and len(chosen) > d["max_slices_per_patient"]:
            chosen = rng.choice(chosen, size=d["max_slices_per_patient"], replace=False)
        items += [(p["id"], int(i)) for i in sorted(chosen.tolist())]
    return items


class SliceDataset(Dataset):
    def __init__(self, cfg, items, train=True, cache_size=2):
        self.cfg = cfg
        self.items = items
        self.train = train
        self.size = cfg["model"]["image_size"]
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
        patient_id, k = self.items[i]
        entry = self._entry(patient_id)
        rgb = entry["rgb"][k]
        gt = entry["gt"][k]
        prompt = prompt_logits(entry, self.cfg, k)[0]

        if self.train and np.random.rand() < float(self.cfg["train"]["fliplr"]):
            # axis 0 of these arrays is the left-right axis of the head, so a flip here is
            # a plausible brain; tumours occur on both sides.
            rgb, gt, prompt = rgb[::-1], gt[::-1], prompt[::-1]

        size = self.size
        image = to_model_image(np.ascontiguousarray(rgb), size)
        gt_t = torch.from_numpy(np.ascontiguousarray(gt)).float()[None, None]
        gt_t = F.interpolate(gt_t, size=(size, size), mode="nearest")[0]
        pr_t = torch.from_numpy(np.ascontiguousarray(prompt)).float()[None, None]
        pr_t = F.interpolate(pr_t, size=(size, size), mode="bilinear", align_corners=False)[0]
        return {"image": image, "prompt": pr_t, "gt": gt_t,
                "patient": patient_id, "k": k, "z": int(entry["z"][k])}


def collate(batch):
    out = {k: torch.stack([b[k] for b in batch]) for k in ("image", "prompt", "gt")}
    out["patient"] = [b["patient"] for b in batch]
    out["z"] = [b["z"] for b in batch]
    return out


class PatientShuffle(torch.utils.data.Sampler):
    """Shuffle patients each round, and the slices inside each patient, but keep a
    patient's slices together so its file is read once."""

    def __init__(self, items, seed=0):
        self.items = items
        self.seed = seed
        self.epoch = 0
        self.groups = OrderedDict()
        for i, (patient_id, _) in enumerate(items):
            self.groups.setdefault(patient_id, []).append(i)

    def set_epoch(self, epoch):
        self.epoch = epoch

    def __len__(self):
        return len(self.items)

    def __iter__(self):
        rng = np.random.default_rng(self.seed + self.epoch)
        keys = list(self.groups)
        rng.shuffle(keys)
        for key in keys:
            idx = list(self.groups[key])
            rng.shuffle(idx)
            yield from idx
