"""GT-prompted vs YOLO-prompted MedSAM2 — the controlled comparison.

Per patient, MedSAM2 is run twice over the same volume, with the same model
(`MedSAM2_latest.pt`), the same slices-as-video-frames propagation, and the same anchor
slices. The *only* difference is where the box prompts come from:

- **Arm GT**   — the expert mask itself, whole tumour (`seg > 0`, labels 1-4 merged).
- **Arm YOLO** — the YOLO-seg predicted mask (`YoloPipeline.process_slice`).

Prompts are **masks**, not boxes: each anchor slice's mask is handed to MedSAM2 via
`add_new_mask`, so the model receives the segmentation itself rather than a bounding
rectangle around it.

Both are scored against the same GT whole-tumour mask, so a difference in Dice is
attributable to prompt quality alone.

Keeping the arms comparable (see `anchor_schedule`):

- The anchor slices are fixed **before either arm runs** and derive from GT *geometry*
  only, never from any model output. Only slices carrying more than
  `MIN_ANCHOR_TUMOUR_PX` GT tumour pixels are eligible. Round 0 is the largest-GT-area
  slice; each further anchor bisects the widest un-anchored stretch of the tumour
  z-range. Both arms are handed the identical ordered list.
- Each arm adds one anchor per round and stops on its own criterion, so an arm's anchor
  set is always a *prefix* of that one list, never a different set of slices. "Arm GT
  converged in 3 anchors, Arm YOLO needed 6" is therefore a result, not a confound.
- Where YOLO predicts nothing on an anchor slice, that arm gets no prompt there for that
  round. No GT fallback (it would launder the miss into the GT arm's favour) and no
  substituting a different z (it would break the prefix property). Missing a slice costs
  the arm a round, which is exactly the thing being measured.

Head-to-head at equal anchor count is only meaningful up to `min(anchors_used)` rounds;
the summary reports that as `comparable_rounds`.
"""
import glob
import json
import os
import time

import nibabel as nib
import numpy as np
import yaml

from YOLO.pipeline import PARAMS as YOLO_PARAMS, YoloPipeline, _dice, list_test_patients
from . import medsam2_runner as M

HITL = {
    "max_anchors": 7,          # anchor budget == max HITL rounds per arm
    "min_improvement": 0.005,
    "dice_target": 0.95,
}

# A slice needs more than this many GT tumour pixels to be eligible as an anchor — a
# handful of pixels is too thin a prompt to seed propagation from.
MIN_ANCHOR_TUMOUR_PX = 100

# Part of the cache key: results computed with box prompts are not comparable with these.
PROMPT_MODE = "mask"

ARMS = ("gt", "yolo")
ARM_LABELS = {"gt": "MedSAM2 from GT mask", "yolo": "MedSAM2 from YOLO mask"}


def cache_root():
    """`outputs/gtvsyolo/` — one JSON summary + one packed-mask NPZ per (patient, weights)."""
    with open("config.yaml", "r") as f:
        cfg = yaml.safe_load(f) or {}
    return os.path.join(os.path.abspath(cfg["paths"]["outputs"]), "gtvsyolo")


def cache_stem(patient_id, weights_path=None):
    """Cache path prefix for a (patient, weights) pair, without loading the patient."""
    wname = (os.path.splitext(os.path.basename(weights_path))[0]
             if weights_path else "default")
    return os.path.join(cache_root(), f"{patient_id}__{wname}__{PROMPT_MODE}")


def is_cached(patient_id, weights_path=None):
    stem = cache_stem(patient_id, weights_path)
    return os.path.exists(stem + ".json") and os.path.exists(stem + ".npz")


# ---------------------------------------------------------------------------
# Anchor schedule — GT geometry only, identical for both arms
# ---------------------------------------------------------------------------
def anchor_schedule(gt_wt, max_anchors=None):
    """Ordered anchor z-list, derived from the GT whole-tumour mask alone.

    Only slices with more than `MIN_ANCHOR_TUMOUR_PX` GT tumour pixels are eligible, so
    the wisps at the top and bottom of the tumour never become anchors. Round 0 = the
    slice carrying the most tumour. Each subsequent anchor splits the widest stretch of
    the eligible z-range not yet covered, taking the eligible slice nearest that
    stretch's midpoint (ties broken by larger tumour area). Fully deterministic and
    independent of any model output, which is what lets both arms be handed the same
    slices.
    """
    n = max_anchors or HITL["max_anchors"]
    D = gt_wt.shape[2]
    areas = np.array([int(gt_wt[:, :, z].sum()) for z in range(D)])
    eligible = areas > MIN_ANCHOR_TUMOUR_PX
    tumour_z = np.flatnonzero(eligible)
    if tumour_z.size == 0:
        return []

    z_min, z_max = int(tumour_z[0]), int(tumour_z[-1])
    anchors = [int(np.argmax(areas))]

    while len(anchors) < n:
        bounds = sorted(set(anchors) | {z_min - 1, z_max + 1})
        best_width, best_z = 0, None
        for a, b in zip(bounds, bounds[1:]):
            lo, hi = a + 1, b - 1
            if hi < lo:
                continue
            candidates = [z for z in range(lo, hi + 1)
                          if eligible[z] and z not in anchors]
            if not candidates:
                continue
            width = hi - lo + 1
            if width <= best_width:
                continue
            mid = (lo + hi) // 2
            best_width = width
            best_z = min(candidates, key=lambda z: (abs(z - mid), -areas[z]))
        if best_z is None:
            break
        anchors.append(int(best_z))
    return anchors


# ---------------------------------------------------------------------------
# One arm
# ---------------------------------------------------------------------------
def _run_arm(arm, predictor, state, schedule, mask_for_z, gt_wt, shape, log=print):
    """Walk `schedule`, one more anchor each round, until this arm's stop rule fires.

    `mask_for_z(z)` supplies the prompt mask for an anchor (a GT slice, or YOLO's
    prediction on it). An empty/None mask means "no prompt here" — the z stays in the
    schedule, this arm just contributes nothing at it.
    """
    rounds = []
    best_mask = np.zeros(shape, dtype=bool)
    best_dice, prev_dice = -1.0, -1.0
    stop_reason = "budget"
    t_arm = time.time()

    for r, z in enumerate(schedule):
        used = schedule[:r + 1]
        prompts = {}
        for zz in used:
            m = mask_for_z(zz)
            if m is not None and m.any():
                prompts[zz] = m

        t0 = time.time()
        if not prompts:
            # No prompts registered yet (YOLO empty on every anchor so far) — there is
            # nothing to propagate, so skip the model call entirely and score empty.
            pred = np.zeros(shape, dtype=bool)
        else:
            predictor.reset_state(state)
            pred, _ = M.infer_track(predictor, state, prompts, shape)
        d = _dice(pred, gt_wt)
        secs = time.time() - t0

        rounds.append({
            "round": r,
            "anchors_used": r + 1,
            "z": z,
            "prompted_slices": sorted(prompts),
            "empty_prompt_slices": [zz for zz in used if zz not in prompts],
            "dice": d,
            "delta": None if r == 0 else d - prev_dice,
            "seconds": secs,
        })
        log(f"    [{arm}] round {r} (+z={z}, {len(prompts)}/{r + 1} prompted): "
            f"dice={d:.4f} ({secs:.1f}s)")

        if d > best_dice:
            best_mask, best_dice = pred, d

        if d >= HITL["dice_target"]:
            stop_reason = f"dice_target ({HITL['dice_target']})"
            break
        if r > 0 and (d - prev_dice) < HITL["min_improvement"]:
            stop_reason = f"no_improvement (delta < {HITL['min_improvement']})"
            break
        prev_dice = d

    if not rounds:
        stop_reason = "no_anchors"

    return {
        "arm": arm,
        "label": ARM_LABELS[arm],
        "anchors_used": len(rounds),
        "anchors_available": len(schedule),
        "anchor_z": [row["z"] for row in rounds],
        "stop_reason": stop_reason,
        "final_dice": best_dice if best_dice >= 0 else 0.0,
        "seconds": time.time() - t_arm,
        "rounds": rounds,
    }, best_mask


# ---------------------------------------------------------------------------
# Patient comparison
# ---------------------------------------------------------------------------
class Comparison:
    """Loads one patient, runs both arms, caches the result.

    Volumes stay in memory for rendering; masks are cached to disk bit-packed, so a
    re-selected patient comes back without re-running MedSAM2.
    """

    MODALITY_SUFFIX = {"T1C": "-t1c", "T1": "-t1n", "T2": "-t2w", "FLAIR": "-t2f"}

    def __init__(self, patient_dir, weights_path=None):
        self.dir = patient_dir
        self.patient_id = os.path.basename(os.path.normpath(patient_dir))
        self.weights_path = weights_path
        self.p = dict(YOLO_PARAMS)
        self._vols = {}
        self._result = None
        self._masks = None

        files = glob.glob(os.path.join(patient_dir, "*.nii*"))
        self.files = {}
        for role, suf in self.MODALITY_SUFFIX.items():
            self.files[role] = next(
                (f for f in files if suf in os.path.basename(f).lower()), None)
        self.files["SEG"] = next(
            (f for f in files if "-seg" in os.path.basename(f).lower()), None)

    # -- volumes -------------------------------------------------------------
    def _vol(self, role):
        if role not in self._vols:
            path = self.files.get(role)
            if path is None:
                raise FileNotFoundError(f"{role} missing for {self.patient_id}")
            self._vols[role] = nib.load(path).get_fdata()
        return self._vols[role]

    @property
    def gt_wt(self):
        """Whole tumour: every labelled voxel (`seg > 0`, BraTS labels 1-4 merged)."""
        if "gt_wt" not in self._vols:
            self._vols["gt_wt"] = self._vol("SEG") > 0
        return self._vols["gt_wt"]

    @property
    def rgb_volume(self):
        """The exact frames MedSAM2 sees (R:T1c / G:T2w / B:T2f)."""
        if "rgb" not in self._vols:
            self._vols["rgb"] = M.build_rgb_volume(
                self._vol("T1C"), self._vol("T2"), self._vol("FLAIR"))
        return self._vols["rgb"]

    @property
    def depth(self):
        return self.gt_wt.shape[2]

    # -- cache ---------------------------------------------------------------
    def _cache_stem(self):
        return cache_stem(self.patient_id, self.weights_path)

    def is_cached(self):
        return is_cached(self.patient_id, self.weights_path)

    def _load_cache(self):
        stem = self._cache_stem()
        with open(stem + ".json", "r", encoding="utf-8") as f:
            self._result = json.load(f)
        npz = np.load(stem + ".npz")
        shape = tuple(int(v) for v in npz["shape"])
        n = int(np.prod(shape))
        self._masks = {
            arm: np.unpackbits(npz[arm])[:n].astype(bool).reshape(shape) for arm in ARMS
        }
        return self._result

    def _save_cache(self):
        stem = self._cache_stem()
        os.makedirs(os.path.dirname(stem), exist_ok=True)
        with open(stem + ".json", "w", encoding="utf-8") as f:
            json.dump(self._result, f, indent=2)
        np.savez_compressed(
            stem + ".npz",
            shape=np.array(self._masks["gt"].shape),
            **{arm: np.packbits(self._masks[arm]) for arm in ARMS},
        )

    # -- run -----------------------------------------------------------------
    def result(self, log=print, force=False):
        """Cached result if there is one, otherwise run both arms now."""
        if self._result is not None and not force:
            return self._result
        if self.is_cached() and not force:
            return self._load_cache()
        return self.run(log=log)

    def masks(self, log=print):
        if self._masks is None:
            self.result(log=log)
        return self._masks

    def run(self, log=print):
        t_start = time.time()
        gt_wt = self.gt_wt
        shape = gt_wt.shape
        schedule = anchor_schedule(gt_wt)
        log(f"  [{self.patient_id}] anchor schedule (GT geometry, "
            f"slices with > {MIN_ANCHOR_TUMOUR_PX} GT px): {schedule}")
        if not schedule:
            log(f"  [{self.patient_id}] no slice carries more than "
                f"{MIN_ANCHOR_TUMOUR_PX} GT tumour pixels — nothing to prompt with")

        yolo = YoloPipeline(self.dir, weights_path=self.weights_path)
        yolo_anchor_masks = {}
        for z in schedule:
            info = yolo.process_slice(z)
            yolo_anchor_masks[z] = info["pred_wt"]
            if info.get("model_error"):
                log(f"  [warn] YOLO: {info['model_error']}")
        empty_yolo = [z for z, m in yolo_anchor_masks.items() if not m.any()]
        if empty_yolo:
            log(f"  [{self.patient_id}] YOLO found nothing on anchors {empty_yolo}")

        masks, arms = {}, {}
        if not schedule:
            for arm in ARMS:
                masks[arm] = np.zeros(shape, dtype=bool)
                arms[arm] = {
                    "arm": arm, "label": ARM_LABELS[arm], "anchors_used": 0,
                    "anchors_available": 0, "anchor_z": [], "stop_reason": "no_tumour",
                    "final_dice": 0.0, "seconds": 0.0, "rounds": [],
                }
        else:
            video_dir = M.write_frames(self.rgb_volume, self.patient_id)
            predictor = M.get_predictor()
            state = predictor.init_state(video_path=video_dir, async_loading_frames=False)
            sources = {
                "gt": lambda z: gt_wt[:, :, z],
                "yolo": lambda z: yolo_anchor_masks.get(z),
            }
            for arm in ARMS:
                arms[arm], masks[arm] = _run_arm(
                    arm, predictor, state, schedule, sources[arm], gt_wt, shape, log=log)
            predictor.reset_state(state)

        self._masks = masks
        used = [arms[a]["anchors_used"] for a in ARMS]
        self._result = {
            "patient_id": self.patient_id,
            "weights": os.path.basename(self.weights_path) if self.weights_path else None,
            "depth": int(shape[2]),
            "schedule": schedule,
            "yolo_empty_anchors": empty_yolo,
            "hitl": dict(HITL),
            "prompt_mode": PROMPT_MODE,
            "min_anchor_tumour_px": MIN_ANCHOR_TUMOUR_PX,
            "arms": arms,
            "comparable_rounds": int(min(used)) if used else 0,
            "slices": self._slice_rows(masks, gt_wt),
            "total_seconds": time.time() - t_start,
        }
        self._save_cache()
        log(f"  [{self.patient_id}] done in {self._result['total_seconds']:.1f}s — "
            f"GT {arms['gt']['final_dice']:.4f} ({arms['gt']['anchors_used']} anchors) vs "
            f"YOLO {arms['yolo']['final_dice']:.4f} ({arms['yolo']['anchors_used']})")
        return self._result

    def _slice_rows(self, masks, gt_wt):
        """Per-slice Dice for both arms, over every slice where GT or either arm has
        tumour — the rows behind the summary table and the two-line chart."""
        rows = []
        for z in range(gt_wt.shape[2]):
            g = gt_wt[:, :, z]
            a = masks["gt"][:, :, z]
            b = masks["yolo"][:, :, z]
            if not (g.any() or a.any() or b.any()):
                continue
            rows.append({
                "z": z,
                "gt_voxels": int(np.count_nonzero(g)),
                "dice_gt": _dice(a, g),
                "dice_yolo": _dice(b, g),
                "voxels_gt_arm": int(np.count_nonzero(a)),
                "voxels_yolo_arm": int(np.count_nonzero(b)),
            })
        return rows

    def best_slice_index(self):
        """Slice with the most ground-truth tumour — the UI's "Best" button."""
        rows = (self._result or {}).get("slices") or []
        if not rows:
            return self.depth // 2
        return max(rows, key=lambda r: r["gt_voxels"])["z"]


def test_patients():
    """Patients offered in the UI — the held-out test split only."""
    return list_test_patients()
