"""The harness: one patient, every variant, one thing different at a time.

What is held constant across every variant, deliberately:

- the patient and its volumes;
- the anchor schedule, derived from GT geometry alone by
  `GTvsYOLO.pipeline.anchor_schedule` (never from a model output). It is computed once at
  the largest budget any variant asks for; because the schedule is built by appending one
  anchor at a time, `schedule[:7]` is exactly what a 7-anchor run would have produced, so
  a 7-anchor and a 15-anchor variant share their first seven slices;
- the MedSAM2 checkpoint, config and JPEG frames (reused from `outputs/tmp_frames/gtvsyolo/`,
  so they are the same bytes the reference run saw);
- forward-then-backward propagation, logits merged with `max`, threshold 0;
- the HITL stop rules, and the scoring against GT whole tumour.

The only thing that changes is `variants.build_prompt`.

Results are cached per (patient, weights, variant) under `outputs/yolomedsam2test/`, so a
run can be stopped and resumed and the UI can show what is finished.

`confidence_probe` is the cheap pre-flight check: it runs YOLO alone over every slice and
records confidence against YOLO's own Dice. If confidence does not track quality there,
the confidence-driven variants (B, Bg, C1, C2) have nothing to work with, and that is
worth knowing before spending GPU hours.
"""
import glob
import json
import os
import time

import nibabel as nib
import numpy as np
import yaml

from GTvsYOLO.pipeline import HITL, MIN_ANCHOR_TUMOUR_PX, anchor_schedule
from YOLO.pipeline import (PARAMS as YOLO_PARAMS, _dice, default_weights_path,
                           list_test_patients)
from . import medsam2_soft as M
from . import yolo_infer
from .variants import ORDER, VARIANTS, build_prompt

MODALITY_SUFFIX = {"T1C": "-t1c", "T1": "-t1n", "T2": "-t2w", "FLAIR": "-t2f"}

# The largest anchor budget any variant asks for; the schedule is computed once at this
# depth and sliced per variant.
MAX_SCHEDULE = max(v.max_anchors for v in VARIANTS.values())

# Bumped whenever `confidence_probe` records something new, so a stale cached probe is
# recomputed instead of being drawn with fields missing.
PROBE_VERSION = 3

# Resolution of the per-pixel probability histogram: 50 bins of 0.02 across [0, 1].
PROB_BINS = 50


def cache_root():
    with open("config.yaml", "r") as f:
        cfg = yaml.safe_load(f) or {}
    return os.path.join(os.path.abspath(cfg["paths"]["outputs"]), "yolomedsam2test")


def _weights_name(weights_path):
    return (os.path.splitext(os.path.basename(weights_path))[0]
            if weights_path else "default")


def patient_cache_dir(patient_id, weights_path=None):
    return os.path.join(cache_root(), f"{patient_id}__{_weights_name(weights_path)}")


def is_cached(patient_id, variant_id, weights_path=None):
    d = patient_cache_dir(patient_id, weights_path)
    return (os.path.exists(os.path.join(d, f"{variant_id}.json"))
            and os.path.exists(os.path.join(d, f"{variant_id}.npz")))


def cached_variants(patient_id, weights_path=None):
    return [v for v in ORDER if is_cached(patient_id, v, weights_path)]


def test_patients():
    return list_test_patients()


def is_anchor_slice(run, variant, z):
    """Does this variant hand MedSAM2 a prompt on slice `z`?

    Shared by the route and the renderer so the fourth panel appears exactly where a
    prompt was actually offered — the GT-derived schedule for most variants, every brain
    slice for H, and nowhere at all for I, which never calls MedSAM2.
    """
    if variant.mode == "yolo_only":
        return False
    if variant.mode == "all_slices":
        return z in set(run.brain_slices())
    return z in run.schedule[:variant.max_anchors]


# ---------------------------------------------------------------------------
# Patient
# ---------------------------------------------------------------------------
class PatientRun:
    """One patient under one YOLO checkpoint; runs any subset of the variants."""

    def __init__(self, patient_dir, weights_path=None):
        self.dir = patient_dir
        self.patient_id = os.path.basename(os.path.normpath(patient_dir))
        self.weights_path = weights_path or default_weights_path()
        self.p = dict(YOLO_PARAMS)
        self._vols = {}
        self._yolo_model = None
        self._yolo_error = None
        self._slice_info = {}     # (conf, want_prob, z) -> predict_slice result
        self._state = None
        self._predictor = None

        files = glob.glob(os.path.join(patient_dir, "*.nii*"))
        self.files = {}
        for role, suf in MODALITY_SUFFIX.items():
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
            self._vols[role] = np.asarray(nib.load(path).get_fdata(), dtype=np.float32)
        return self._vols[role]

    @property
    def gt_wt(self):
        """Whole tumour: every labelled voxel (BraTS labels 1-4 merged)."""
        if "gt_wt" not in self._vols:
            self._vols["gt_wt"] = self._vol("SEG") > 0
        return self._vols["gt_wt"]

    @property
    def rgb_medsam2(self):
        """The frames MedSAM2 sees: R:T1c / G:T2w / B:T2f."""
        if "rgb_m" not in self._vols:
            self._vols["rgb_m"] = M.build_rgb_volume(
                self._vol("T1C"), self._vol("T2"), self._vol("FLAIR"))
        return self._vols["rgb_m"]

    def rgb_yolo(self, z):
        """The RGB YOLO trained on: R:T1C-T1 / G:T2 / B:FLAIR. Deliberately not the same
        composite MedSAM2 gets — each model is given the input it expects."""
        from YOLO.pipeline import build_rgb_slice
        return build_rgb_slice(self._vol("T1C")[:, :, z], self._vol("T1")[:, :, z],
                               self._vol("T2")[:, :, z], self._vol("FLAIR")[:, :, z],
                               self.p["min_fg_voxels"])

    @property
    def depth(self):
        return int(self.gt_wt.shape[2])

    @property
    def schedule(self):
        if "schedule" not in self._vols:
            self._vols["schedule"] = anchor_schedule(self.gt_wt, MAX_SCHEDULE)
        return self._vols["schedule"]

    # -- YOLO ----------------------------------------------------------------
    def _model(self):
        if self._yolo_model is not None or self._yolo_error is not None:
            return self._yolo_model
        if not os.path.exists(self.weights_path):
            self._yolo_error = f"weights not found: {self.weights_path}"
            return None
        try:
            from ultralytics import YOLO
            self._yolo_model = YOLO(self.weights_path)
        except Exception as e:  # noqa: BLE001
            self._yolo_error = str(e)
        return self._yolo_model

    def yolo_slice(self, z, conf, want_prob):
        key = (round(conf, 4), bool(want_prob), int(z))
        if key not in self._slice_info:
            info = yolo_infer.predict_slice(
                self._model(), self.rgb_yolo(z), conf,
                self.gt_wt.shape[:2], want_prob=want_prob)
            if info["error"]:
                self._yolo_error = info["error"]
            self._slice_info[key] = info
        return self._slice_info[key]

    def prompt_source(self, variant, z):
        """The `info` dict `build_prompt` consumes, for this variant on this slice."""
        if variant.source == "gt":
            return {"union": self.gt_wt[:, :, z], "instances": []}
        return self.yolo_slice(z, variant.conf, variant.wants_prob)

    # -- MedSAM2 -------------------------------------------------------------
    def _medsam2_state(self, log=print):
        """Frames + predictor + inference state, built once and shared by every variant."""
        if self._state is None:
            video_dir = M.write_frames(self.rgb_medsam2, self.patient_id)
            self._predictor = M.get_predictor()
            self._state = self._predictor.init_state(video_path=video_dir,
                                                     async_loading_frames=False)
            log(f"  [{self.patient_id}] frames ready ({self.depth} slices)")
        return self._predictor, self._state

    # -- one variant ---------------------------------------------------------
    def _cache_paths(self, variant_id):
        d = patient_cache_dir(self.patient_id, self.weights_path)
        return os.path.join(d, f"{variant_id}.json"), os.path.join(d, f"{variant_id}.npz")

    def load(self, variant_id):
        """Cached result for one variant, with per-slice rows rebuilt from the mask.

        The rows are derived rather than read back so their format can change without
        invalidating an expensive run.
        """
        jpath, npath = self._cache_paths(variant_id)
        with open(jpath, "r", encoding="utf-8") as f:
            result = json.load(f)
        npz = np.load(npath)
        shape = tuple(int(v) for v in npz["shape"])
        mask = np.unpackbits(npz["mask"])[:int(np.prod(shape))].astype(bool).reshape(shape)
        result["slices"] = self._slice_rows(mask)
        return result, mask

    def _save(self, variant_id, result, mask):
        """Write the summary and the bit-packed mask.

        `slices` is stripped: `load` rebuilds it from the mask anyway, so storing it would
        be dead weight — and at 182 rows per variant it is most of the file. Keeping the
        JSON small is what lets `aggregate` open thousands of them for the full sweep.
        """
        jpath, npath = self._cache_paths(variant_id)
        os.makedirs(os.path.dirname(jpath), exist_ok=True)
        with open(jpath, "w", encoding="utf-8") as f:
            json.dump({k: v for k, v in result.items() if k != "slices"}, f, indent=2)
        np.savez_compressed(npath, shape=np.array(mask.shape),
                            mask=np.packbits(mask))

    def _slice_rows(self, mask):
        """Per-slice Dice and voxel counts over every slice.

        Dice is `None` where GT and prediction are both empty: scoring those as a perfect
        1.0 would paint a flat line of success across the background. A slice predicting
        tumour that is not there still scores 0.0 — that is a real error worth seeing.
        """
        gt = self.gt_wt
        rows = []
        for z in range(gt.shape[2]):
            g, m = gt[:, :, z], mask[:, :, z]
            g_any, m_any = bool(g.any()), bool(m.any())
            rows.append({
                "z": z,
                "gt_voxels": int(np.count_nonzero(g)),
                "voxels": int(np.count_nonzero(m)),
                "dice": _dice(m, g) if (g_any or m_any) else None,
            })
        return rows

    def run_variant(self, variant_id, log=print, force=False):
        """Run (or load) one variant. Returns `(result_dict, mask_3d)`."""
        variant = VARIANTS[variant_id]
        if not force and is_cached(self.patient_id, variant_id, self.weights_path):
            return self.load(variant_id)

        gt = self.gt_wt
        shape = gt.shape
        schedule = self.schedule[:variant.max_anchors]
        t_start = time.time()

        if variant.mode == "yolo_only":
            return self._run_yolo_only(variant, log=log)
        if variant.mode == "all_slices":
            return self._run_all_slices(variant, log=log)

        if not schedule:
            mask = np.zeros(shape, dtype=bool)
            result = self._summary(variant, schedule, [], [], mask, "no_anchors",
                                   time.time() - t_start)
            self._save(variant_id, result, mask)
            return result, mask

        # Prompts for every anchor, built once — the model call is what costs, not this.
        prompts_all, empty = {}, []
        for z in schedule:
            p = build_prompt(variant, self.prompt_source(variant, z))
            if p is None or not np.any(p):
                empty.append(int(z))
            else:
                prompts_all[z] = p

        predictor, state = (None, None)
        rounds = []
        best_mask = np.zeros(shape, dtype=bool)
        best_dice, prev_dice = -1.0, -1.0
        stop_reason = "budget"

        for r, z in enumerate(schedule):
            used = schedule[:r + 1]
            prompts = {zz: prompts_all[zz] for zz in used if zz in prompts_all}

            t0 = time.time()
            if not prompts:
                # Nothing registered yet — there is nothing to propagate, so skip the
                # model entirely and score empty rather than invent a prompt.
                pred = np.zeros(shape, dtype=bool)
            else:
                if predictor is None:
                    predictor, state = self._medsam2_state(log=log)
                predictor.reset_state(state)
                pred = M.infer_track(predictor, state, prompts, shape)
            d = _dice(pred, gt)
            secs = time.time() - t0

            rounds.append({
                "round": r,
                "anchors_used": r + 1,
                "z": int(z),
                "prompted_slices": sorted(int(v) for v in prompts),
                "empty_prompt_slices": [int(zz) for zz in used if zz not in prompts],
                "dice": d,
                "delta": None if r == 0 else d - prev_dice,
                "seconds": secs,
            })
            log(f"    [{variant_id}] round {r} (+z={z}, {len(prompts)}/{r + 1}): "
                f"dice={d:.4f} ({secs:.1f}s)")

            if d > best_dice:
                best_mask, best_dice = pred, d

            if variant.early_stop:
                if d >= HITL["dice_target"]:
                    stop_reason = f"dice_target ({HITL['dice_target']})"
                    break
                if r > 0 and (d - prev_dice) < HITL["min_improvement"]:
                    stop_reason = f"no_improvement (delta < {HITL['min_improvement']})"
                    break
            prev_dice = d
        else:
            stop_reason = "forced_all" if not variant.early_stop else "budget"

        result = self._summary(variant, schedule, rounds, empty, best_mask, stop_reason,
                               time.time() - t_start)
        self._save(variant_id, result, best_mask)
        return result, best_mask

    def brain_slices(self):
        """Every z with enough FLAIR foreground to be an inference candidate.

        The same gate the YOLO tab and `confidence_probe` use, so "every slice of the
        brain" means the same thing here as it does there — not all 182 z, most of which
        are empty air above and below the head.
        """
        if "brain_z" not in self._vols:
            flair = self._vol("FLAIR")
            self._vols["brain_z"] = [z for z in range(self.depth)
                                     if int((flair[:, :, z] != 0).sum())
                                     >= self.p["min_fg_voxels"]]
        return self._vols["brain_z"]

    def _run_all_slices(self, variant, log=print):
        """Every brain slice prompted at once, in a single propagation.

        This is G with the budget removed, so there is no anchor schedule and no HITL
        ladder: adding anchors one at a time would mean ~70 full propagations per patient
        for a curve that ends where this one pass already lands. One pass, one number.

        Slices where YOLO found nothing are still left unprompted — MedSAM2's memory
        fills them from the neighbours, exactly as it does for a missed anchor anywhere
        else. That is the only work left for the propagation to do here.
        """
        t_start = time.time()
        brain = self.brain_slices()
        prompts, empty = {}, []
        for z in brain:
            p = build_prompt(variant, self.prompt_source(variant, z))
            if p is None or not np.any(p):
                empty.append(int(z))
            else:
                prompts[z] = p

        if not prompts:
            mask = np.zeros(self.gt_wt.shape, dtype=bool)
        else:
            predictor, state = self._medsam2_state(log=log)
            predictor.reset_state(state)
            mask = M.infer_track(predictor, state, prompts, self.gt_wt.shape)

        dice = _dice(mask, self.gt_wt)
        secs = time.time() - t_start
        result = self._summary(variant, brain, [], empty, mask,
                               f"all_slices ({len(prompts)}/{len(brain)} prompted)",
                               secs, dice=dice, anchors_used=len(prompts))
        self._save(variant.id, result, mask)
        log(f"    [{variant.id}] dice={dice:.4f} "
            f"({len(prompts)}/{len(brain)} brain slices prompted, {secs:.1f}s)")
        return result, mask

    def _run_yolo_only(self, variant, log=print):
        """YOLO stacked over every slice, no MedSAM2 anywhere. Returns `(result, mask)`.

        The mask is scored by the same volume Dice as every other variant, so H sits in
        the same column as O and GT — a *3D* number, not the mean of per-slice 2D Dice
        that the YOLO tab reports. Slices too empty to be inference candidates are left
        blank rather than run, matching `confidence_probe` and the YOLO tab.
        """
        t_start = time.time()
        brain = self.brain_slices()
        mask = np.zeros(self.gt_wt.shape, dtype=bool)
        for z in brain:
            mask[:, :, z] = self.yolo_slice(z, variant.conf, False)["union"]

        dice = _dice(mask, self.gt_wt)
        secs = time.time() - t_start
        result = self._summary(variant, [], [], [], mask,
                               f"no_medsam2 ({len(brain)}/{self.depth} brain slices)",
                               secs, dice=dice)
        self._save(variant.id, result, mask)
        log(f"    [{variant.id}] dice={dice:.4f} (no MedSAM2, {secs:.1f}s)")
        return result, mask

    def _summary(self, variant, schedule, rounds, empty, mask, stop_reason, secs,
                 dice=None, anchors_used=None):
        return {
            "patient_id": self.patient_id,
            "weights": os.path.basename(self.weights_path) if self.weights_path else None,
            "variant": variant.id,
            "label": variant.label,
            "changes": variant.changes,
            "why": variant.why,
            "prompt_mode": variant.mode,
            "conf_threshold": variant.conf,
            "tau": variant.tau,
            "early_stop": variant.early_stop,
            "soft_prompt": variant.is_soft,
            "depth": self.depth,
            "schedule": [int(z) for z in schedule],
            "anchors_available": len(schedule),
            # One round of HITL == one more anchor, except for the single-pass variants,
            # which prompt many slices in one go and say so explicitly.
            "anchors_used": len(rounds) if anchors_used is None else anchors_used,
            "anchor_z": [r["z"] for r in rounds],
            "empty_anchors": empty,
            "stop_reason": stop_reason,
            # `dice` is passed only by the variants that run no rounds at all (H), where
            # there is no best-round to take it from.
            "final_dice": best_of(rounds) if dice is None else dice,
            "last_dice": rounds[-1]["dice"] if rounds else (dice or 0.0),
            "rounds": rounds,
            "seconds": secs,
            "yolo_error": self._yolo_error,
            "min_anchor_tumour_px": MIN_ANCHOR_TUMOUR_PX,
            "hitl": dict(HITL),
            "slices": self._slice_rows(mask),
        }

    def release(self):
        """Drop the MedSAM2 inference state — volumes stay for rendering."""
        if self._predictor is not None and self._state is not None:
            self._predictor.reset_state(self._state)
        self._state = None

    # -- pre-flight: does confidence track quality? --------------------------
    def confidence_probe(self, conf=0.25, log=print, force=False):
        """YOLO alone on every slice: its confidence against its own Dice.

        No MedSAM2, so this is seconds rather than minutes. If the correlation here is
        flat, the confidence-driven variants cannot work and the expensive runs can be
        skipped.

        All three granularities are measured, one per confidence-driven variant, and each
        is scored against the Dice of the thing its number actually describes:

        - **B** — one point per slice: the area-weighted slice confidence, against the
          Dice of the whole slice's mask.
        - **C1** — one point per detected lump: that lump's own score, against the Dice of
          that lump alone. A slice contributes as many points as YOLO found blobs.
        - **C2** — one point per slice: the mean per-pixel probability inside the mask,
          against the Dice of the whole slice's mask.

        A granularity whose points tilt upwards is one worth prompting with; a flat cloud
        says that variant has nothing to work with, however fine its numbers are.
        """
        path = os.path.join(patient_cache_dir(self.patient_id, self.weights_path),
                            f"probe_conf{int(round(conf * 100)):03d}.json")
        if os.path.exists(path) and not force:
            with open(path, "r", encoding="utf-8") as f:
                cached = json.load(f)
            # A probe written before the C1/C2 granularities existed has only B's rows;
            # recompute rather than half-draw the chart from it.
            if cached.get("version") == PROBE_VERSION:
                return cached

        gt = self.gt_wt
        rows, inst_rows = [], []
        # C2's raw material: every per-pixel probability, not a per-slice average. Kept as
        # bin counts because the values themselves are ~40k floats per slice.
        edges = np.linspace(0.0, 1.0, PROB_BINS + 1)
        hist_in = np.zeros(PROB_BINS, dtype=np.int64)
        hist_out = np.zeros(PROB_BINS, dtype=np.int64)
        for z in self.brain_slices():
            info = self.yolo_slice(z, conf, True)
            g = gt[:, :, z]
            union = info["union"]
            if not (g.any() or union.any()):
                continue

            # C2's number: the mean of YOLO's own per-pixel probability over the pixels it
            # called tumour. Averaged over the mask, not the slice, or the background
            # would drown it.
            prob = np.zeros(union.shape, dtype=np.float32)
            for inst in info["instances"]:
                p = inst["prob"]
                np.maximum(prob, inst["mask"].astype(np.float32) if p is None else p,
                           out=prob)
            mean_prob = float(prob[union].mean()) if union.any() else None

            # Outside the mask only the fringe inside a detection's own box carries a
            # value at all; the rest of the slice is exactly 0 by cropping, and counting
            # those tens of thousands of zeros would flatten the chart.
            hist_in += np.histogram(prob[union], bins=edges)[0]
            fringe = prob[(~union) & (prob > 0)]
            hist_out += np.histogram(fringe, bins=edges)[0]

            for inst in info["instances"]:
                inst_rows.append({
                    "z": z,
                    "conf": float(inst["conf"]),
                    "dice": _dice(inst["mask"], g),
                    "voxels": int(np.count_nonzero(inst["mask"])),
                })

            rows.append({
                "z": z,
                "conf": yolo_infer.slice_confidence(info["instances"]),
                "mean_prob": mean_prob,
                "max_conf": (max(i["conf"] for i in info["instances"])
                             if info["instances"] else None),
                "instances": len(info["instances"]),
                "dice": _dice(union, g),
                "gt_voxels": int(np.count_nonzero(g)),
                "pred_voxels": int(np.count_nonzero(union)),
            })

        scored = [r for r in rows if r["conf"] is not None]
        probed = [r for r in rows if r["mean_prob"] is not None]

        def corr(pairs, key):
            xs = [r[key] for r in pairs]
            ys = [r["dice"] for r in pairs]
            return {"pearson": _pearson(xs, ys), "spearman": _spearman(xs, ys),
                    "n": len(pairs)}

        out = {
            "patient_id": self.patient_id,
            "version": PROBE_VERSION,
            "conf_threshold": conf,
            "rows": rows,
            "instance_rows": inst_rows,
            "n_scored": len(scored),
            "pearson_conf_dice": _pearson([r["conf"] for r in scored],
                                          [r["dice"] for r in scored]),
            "spearman_conf_dice": _spearman([r["conf"] for r in scored],
                                            [r["dice"] for r in scored]),
            # Keyed by the variant each granularity belongs to, so the UI can label them
            # without knowing what "mean_prob" means.
            "granularity": {
                "B": corr(scored, "conf"),
                "C1": corr(inst_rows, "conf"),
                "C2": corr(probed, "mean_prob"),
            },
            "prob_hist": {
                "edges": [round(float(e), 4) for e in edges],
                "inside": hist_in.tolist(),
                "fringe": hist_out.tolist(),
            },
            "yolo_error": self._yolo_error,
        }
        os.makedirs(os.path.dirname(path), exist_ok=True)
        with open(path, "w", encoding="utf-8") as f:
            json.dump(out, f, indent=2)
        log(f"  [{self.patient_id}] probe: {len(scored)} slices, "
            f"{len(inst_rows)} instances, pearson={out['pearson_conf_dice']}")
        return out


def best_of(rounds):
    return max((r["dice"] for r in rounds), default=0.0)


def _pearson(xs, ys):
    if len(xs) < 3:
        return None
    x, y = np.asarray(xs, dtype=float), np.asarray(ys, dtype=float)
    if x.std() < 1e-12 or y.std() < 1e-12:
        return None
    return float(np.corrcoef(x, y)[0, 1])


def _spearman(xs, ys):
    """Rank correlation — the question is "does more confidence mean better", not
    whether the relationship happens to be a straight line."""
    if len(xs) < 3:
        return None
    def rank(v):
        order = np.argsort(np.argsort(np.asarray(v, dtype=float)))
        return order.astype(float)
    return _pearson(rank(xs), rank(ys))


# ---------------------------------------------------------------------------
# Aggregate across patients
# ---------------------------------------------------------------------------
def load_summary(patient_id, variant_id, weights_path=None):
    """Cached JSON only — no volumes, no mask, and (since `_save` strips them) no
    per-slice rows, so this stays cheap across a 324-patient sweep."""
    jpath = os.path.join(patient_cache_dir(patient_id, weights_path), f"{variant_id}.json")
    if not os.path.exists(jpath):
        return None
    with open(jpath, "r", encoding="utf-8") as f:
        return json.load(f)


def aggregate(weights_path=None, patient_ids=None):
    """Variant-level table over every cached patient, plus the per-patient matrix.

    Only patients that have a result for **all** cached variants are counted in the
    per-variant means, so the columns are compared on the same patients rather than on
    whichever subset happened to finish.
    """
    if patient_ids is None:
        patient_ids = [p["patient_id"] for p in test_patients()]

    matrix, per_patient = {}, []
    for pid in patient_ids:
        row = {}
        for vid in ORDER:
            s = load_summary(pid, vid, weights_path)
            if s is not None:
                row[vid] = s
        if row:
            matrix[pid] = row

    complete = [pid for pid, row in matrix.items() if len(row) == len(ORDER)]

    variants = []
    for vid in ORDER:
        v = VARIANTS[vid]
        dices = [matrix[pid][vid]["final_dice"] for pid in complete if vid in matrix[pid]]
        ref = [matrix[pid]["O"]["final_dice"] for pid in complete if "O" in matrix[pid]]
        deltas = [d - r for d, r in zip(dices, ref)]
        anchors = [matrix[pid][vid]["anchors_used"] for pid in complete if vid in matrix[pid]]
        secs = [matrix[pid][vid]["seconds"] for pid in complete if vid in matrix[pid]]
        variants.append({
            "id": vid,
            "label": v.label,
            "changes": v.changes,
            "why": v.why,
            "n": len(dices),
            "mean_dice": float(np.mean(dices)) if dices else None,
            "median_dice": float(np.median(dices)) if dices else None,
            "std_dice": float(np.std(dices)) if len(dices) > 1 else None,
            "mean_delta_vs_ref": float(np.mean(deltas)) if deltas else None,
            "wins_vs_ref": int(sum(1 for d in deltas if d > 1e-6)),
            "losses_vs_ref": int(sum(1 for d in deltas if d < -1e-6)),
            "mean_anchors": float(np.mean(anchors)) if anchors else None,
            "mean_seconds": float(np.mean(secs)) if secs else None,
        })

    for pid in sorted(matrix):
        per_patient.append({
            "patient_id": pid,
            "dice": {vid: matrix[pid][vid]["final_dice"] for vid in matrix[pid]},
            "anchors": {vid: matrix[pid][vid]["anchors_used"] for vid in matrix[pid]},
            "complete": pid in complete,
        })

    return {
        "weights": _weights_name(weights_path),
        "n_patients_any": len(matrix),
        "n_patients_complete": len(complete),
        "variants": variants,
        "patients": per_patient,
    }
