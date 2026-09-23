"""Score a fine-tuned checkpoint the way it is meant to run: as a video, with HITL rounds.

For one patient:

  1. `anchors.initial` picks the slice(s) to prompt, from YOLO's map.
  2. Each anchor gets YOLO's per-voxel logits as a dense prompt.
  3. The volume is propagated **forward** from the lowest anchor and **backward** from the
     highest one, exactly as `05_infer_multibbox_hitl.py` did; the backward pass fills the
     frames the forward pass never reached.
  4. A HITL round then adds the anchor where the prediction disagrees most with YOLO,
     and the whole propagation runs again — up to `hitl.rounds` times.

Nothing in steps 1-4 ever looks at the expert mask, so the score at the end is always an
honest one: the model is never given a hint it would not have on a new patient.

Every number is measured at the slice's own size, so it is comparable with the
YOLO_finetune page. Alongside MedSAM2, the YOLO prompt itself is scored on the same
patients — that pairing is the point of the whole folder.
"""
import os
from datetime import datetime

import numpy as np
import torch

from . import anchors as anchor_rules
from .common import (CHECKPOINTS_DIR, dice, import_sam2, list_patients, read_json, to_model_image,
                     write_json)
from .yolo_prompts import load_prompt


def thresholds_of(cfg):
    e = cfg["evaluate"]
    return sorted(set([float(v) for v in e["sweep"]]) | {float(e["mask_threshold"])})


def build_predictor(cfg, device="cuda", weights=None):
    """The vendored video predictor, optionally carrying this run's fine-tuned weights."""
    import_sam2()
    import sam2.sam2_video_predictor_npz as npz
    from sam2.build_sam import build_sam2_video_predictor_npz

    # `propagate_in_video` draws a tqdm bar per pass, which is hundreds of lines per patient
    # in the run log the page shows. Replace the name the module calls; nothing is edited.
    npz.tqdm = lambda iterable, *a, **kw: iterable

    ckpt = os.path.join(CHECKPOINTS_DIR, cfg["model"]["checkpoint"])
    predictor = build_sam2_video_predictor_npz(cfg["model"]["config"], ckpt, device=device)
    if weights:
        state = torch.load(weights, map_location="cpu", weights_only=False)
        missing, unexpected = predictor.load_state_dict(state["model"], strict=False)
        if unexpected:
            raise RuntimeError(f"checkpoint does not fit the model: {unexpected[:3]}")
        return predictor.eval(), state
    return predictor.eval(), None


def load_run_model(run_dir, cfg, device="cuda", which="best.pt"):
    path = os.path.join(run_dir, "weights", which)
    if not os.path.exists(path):
        raise FileNotFoundError(f"no checkpoint at {path}")
    return build_predictor(cfg, device, weights=path)


def _frames(entry, cfg, device):
    """The whole patient as a "video": normalised frames the image encoder expects."""
    size = cfg["model"]["image_size"]
    return torch.stack([to_model_image(f, size) for f in entry["rgb"]]).to(device)


@torch.inference_mode()
def _propagate(predictor, state, prompts, size, device):
    """One forward + one backward pass from the anchors; returns logits and object scores.

    The two passes each start from a clean state with the same prompts, the way the old
    pipeline did it: forward covers everything from the lowest anchor up, backward fills in
    what is below the highest one.
    """
    frames = state["num_frames"]
    logits = [None] * frames
    objs = np.zeros(frames, np.float32)

    def register():
        predictor.reset_state(state)
        for k, prompt in prompts.items():
            # A 2D float tensor already at image_size keeps its soft values: `add_new_mask`
            # only binarises a mask it has to resize.
            predictor.add_new_mask(state, frame_idx=int(k), obj_id=1,
                                   mask=prompt.to(device))

    def collect():
        out = state["output_dict"]
        for store in ("cond_frame_outputs", "non_cond_frame_outputs"):
            for t, rec in out[store].items():
                if "object_score_logits" in rec:
                    objs[t] = float(rec["object_score_logits"].flatten()[0])

    order = sorted(prompts)
    register()
    for t, _ids, mask_logits in predictor.propagate_in_video(
            state, start_frame_idx=order[0], reverse=False):
        logits[t] = mask_logits[0, 0].float().cpu().numpy()
    collect()

    if order[-1] > 0:
        register()
        for t, _ids, mask_logits in predictor.propagate_in_video(
                state, start_frame_idx=order[-1], reverse=True):
            if logits[t] is None:
                logits[t] = mask_logits[0, 0].float().cpu().numpy()
        collect()

    blank = np.full((size[0], size[1]), -30.0, np.float32)
    return np.stack([blank if v is None else v for v in logits]), objs


def _counts(pred, gt):
    inter = int(np.logical_and(pred, gt).sum())
    return inter, int(pred.sum()), int(gt.sum())


def run_patient(predictor, cfg, patient_id, device="cuda", entry=None):
    """The HITL loop for one patient; returns the final logits and what each round did."""
    entry = entry if entry is not None else load_prompt(cfg, patient_id)
    h, w = [int(v) for v in entry["shape"]]
    size = cfg["model"]["image_size"]
    hitl, ev = cfg["hitl"], cfg["evaluate"]
    gt = entry["gt"] > 0
    threshold = float(ev["mask_threshold"])

    state = predictor.init_state(_frames(entry, cfg, device), h, w)
    picked = anchor_rules.initial(entry, cfg)
    prompts = {}
    rounds = []
    logits = objs = pred = None
    for round_index in range(max(1, int(hitl["rounds"]))):
        for k in picked:
            if k not in prompts:
                pr = torch.from_numpy(anchor_rules.prompt_for(entry, cfg, k)).float()[None, None]
                prompts[k] = torch.nn.functional.interpolate(
                    pr, size=(size, size), mode="bilinear", align_corners=False)[0, 0]
        logits, objs = _propagate(predictor, state, prompts, (h, w), device)
        gate = objs > 0 if ev["use_obj_score"] else np.ones(len(objs), bool)
        new_pred = (logits > threshold) & gate[:, None, None]
        d_gt = dice(*_counts(new_pred, gt))
        change = None if pred is None else dice(*_counts(new_pred, pred))
        rounds.append({"round": round_index + 1, "anchors": sorted(int(k) for k in picked),
                       "z": [int(entry["z"][k]) for k in sorted(picked)],
                       "dice3d": round(d_gt, 4),
                       "same_as_previous": None if change is None else round(change, 4)})
        pred = new_pred

        if round_index + 1 >= int(hitl["rounds"]):
            break
        if change is not None and 1.0 - change < float(hitl["min_improvement"]):
            break                       # the prediction stopped moving: more anchors change nothing
        nxt = anchor_rules.next_anchor(entry, cfg, pred, picked)
        if nxt is None:
            break
        picked.append(nxt)

    return {"logits": logits, "objs": objs, "rounds": rounds, "anchors": sorted(picked)}


def patient_record(predictor, cfg, patient_id, device="cuda", entry=None, out=None):
    """One patient: per-slice numbers for YOLO and MedSAM2, plus the threshold sweep.

    `entry` / `out` let a caller that has already propagated this patient (the page's slice
    viewer) reuse that work instead of paying for a second pass.
    """
    entry = entry if entry is not None else load_prompt(cfg, patient_id)
    out = out if out is not None else run_patient(predictor, cfg, patient_id, device, entry)
    logits, objs = out["logits"], out["objs"]
    gt = entry["gt"] > 0
    yolo = anchor_rules.yolo_mask(entry, cfg)
    ths = thresholds_of(cfg)
    k_at = ths.index(float(cfg["evaluate"]["mask_threshold"]))
    gate = objs > 0 if cfg["evaluate"]["use_obj_score"] else np.ones(len(objs), bool)

    sweep = np.zeros((len(ths), 3), np.int64)
    rec = {"id": patient_id, "z": entry["z"].tolist(), "gt": [], "pred": [], "inter": [],
           "yolo": [], "yolo_inter": [], "score": entry["scores"].round(3).tolist(),
           "nblobs": entry["nblobs"].tolist(), "obj": np.round(objs, 3).tolist(),
           "rounds": out["rounds"], "anchors": out["anchors"],
           "anchor_z": [int(entry["z"][k]) for k in out["anchors"]]}
    for k in range(len(entry["z"])):
        for j, t in enumerate(ths):
            pred = (logits[k] > t) & gate[k]
            sweep[j] += _counts(pred, gt[k])
            if j == k_at:
                i_, p_, g_ = _counts(pred, gt[k])
                rec["inter"].append(i_)
                rec["pred"].append(p_)
                rec["gt"].append(g_)
        yi, yp, _ = _counts(yolo[k], gt[k])
        rec["yolo"].append(yp)
        rec["yolo_inter"].append(yi)

    rec["sweep"] = sweep.tolist()
    rec["dice3d"] = round(dice(sum(rec["inter"]), sum(rec["pred"]), sum(rec["gt"])), 4)
    rec["yolo_dice3d"] = round(dice(sum(rec["yolo_inter"]), sum(rec["yolo"]), sum(rec["gt"])), 4)
    rec["delta"] = round(rec["dice3d"] - rec["yolo_dice3d"], 4)
    rec["gt_total"] = sum(rec["gt"])
    return rec


def patient_profile(predictor, cfg, patient_id, device="cuda", entry=None, out=None):
    """The per-slice view the page charts: Dice and pixel counts for both models."""
    rec = patient_record(predictor, cfg, patient_id, device, entry, out)
    rec["dice"] = [round(dice(i, p, g), 4) for i, p, g in zip(rec["inter"], rec["pred"], rec["gt"])]
    rec["yolo_dice"] = [round(dice(i, p, g), 4)
                        for i, p, g in zip(rec["yolo_inter"], rec["yolo"], rec["gt"])]
    return rec


def summarise(records, cfg):
    """Split-level numbers, including the paired MedSAM2-vs-YOLO comparison."""
    ths = thresholds_of(cfg)
    k_at = ths.index(float(cfg["evaluate"]["mask_threshold"]))
    sweep = [{"threshold": t,
              "dice3d_mean": float(np.mean([dice(*r["sweep"][j]) for r in records])) if records else None}
             for j, t in enumerate(ths)]
    slice_d, tumour_d, yolo_slice_d = [], [], []
    det = {"tp": 0, "fn": 0, "fp": 0, "tn": 0}
    for r in records:
        for g, p, i in zip(r["gt"], r["pred"], r["inter"]):
            d = dice(i, p, g)
            slice_d.append(d)
            if g:
                tumour_d.append(d)
            det["tp" if g and p else "fn" if g else "fp" if p else "tn"] += 1
        for g, p, i in zip(r["gt"], r["yolo"], r["yolo_inter"]):
            yolo_slice_d.append(dice(i, p, g))

    d3 = [dice(*r["sweep"][k_at]) for r in records]
    y3 = [r["yolo_dice3d"] for r in records]
    delta = [a - b for a, b in zip(d3, y3)]
    best = max(sweep, key=lambda s: s["dice3d_mean"] or 0) if records else {"threshold": None}
    mean = lambda v: float(np.mean(v)) if len(v) else None            # noqa: E731
    median = lambda v: float(np.median(v)) if len(v) else None        # noqa: E731
    # what the HITL rounds did, averaged over patients
    per_round = {}
    for r in records:
        for item in r["rounds"]:
            per_round.setdefault(item["round"], []).append(item["dice3d"])
    return {
        "patients": len(records),
        "threshold": float(cfg["evaluate"]["mask_threshold"]),
        "dice3d_mean": mean(d3), "dice3d_median": median(d3),
        "yolo_dice3d_mean": mean(y3), "yolo_dice3d_median": median(y3),
        "delta_mean": mean(delta), "delta_median": median(delta),
        "helped": int(sum(1 for d in delta if d > 0.01)),
        "hurt": int(sum(1 for d in delta if d < -0.01)),
        "unchanged": int(sum(1 for d in delta if abs(d) <= 0.01)),
        "tumour_slice_dice": mean(tumour_d),
        "legacy_slice_dice": mean(slice_d),
        "yolo_legacy_slice_dice": mean(yolo_slice_d),
        "detection": det,
        "sensitivity": det["tp"] / max(1, det["tp"] + det["fn"]),
        "specificity": det["tn"] / max(1, det["tn"] + det["fp"]),
        "sweep": sweep,
        "best_threshold": best["threshold"],
        "anchors_mean": mean([len(r["anchors"]) for r in records]),
        "rounds_mean": mean([len(r["rounds"]) for r in records]),
        "by_round": [{"round": k, "patients": len(v), "dice3d_mean": float(np.mean(v))}
                     for k, v in sorted(per_round.items())],
        "patient_scores": [{"id": r["id"], "dice3d": r["dice3d"], "yolo_dice3d": r["yolo_dice3d"],
                            "delta": r["delta"], "gt_total": r["gt_total"],
                            "anchors": len(r["anchors"]), "rounds": len(r["rounds"]),
                            "score_mean": round(float(np.mean(r["score"])), 3) if r["score"] else 0.0}
                           for r in records],
    }


def scoring_settings(cfg):
    """Everything that changes what a scoring pass produces."""
    return {"prompt": cfg["prompt"], "anchors": cfg["anchors"], "hitl": cfg["hitl"],
            "evaluate": cfg["evaluate"], "image_size": cfg["model"]["image_size"],
            "min_fg_voxels": cfg["data"]["min_fg_voxels"]}


def _already_scored(path, cfg, patients, weights):
    """A finished `<split>.json` this run can keep instead of scoring the pool again.

    Scoring a pool takes far longer than training a round, and a machine that goes down
    half way through should not cost the half that was already done. Reused only when the
    file covers the whole pool, was written *after* the checkpoint it is scoring, and its
    stored settings still match — so retraining or changing an anchor rule re-scores.
    """
    data = read_json(path)
    if not data or data.get("settings") != scoring_settings(cfg):
        return None
    if len(data.get("patients") or []) != len(patients):
        return None
    if not os.path.exists(weights) or os.path.getmtime(path) < os.path.getmtime(weights):
        return None
    return data["patients"]


def evaluate(run_dir, cfg, out_dir, splits=("val", "test"), max_patients=None, progress=None,
             log=print, device="cuda", predictor=None, which="best.pt"):
    """Score `which` checkpoint on the given pools; writes <split>.json and summary.json."""
    from .yolo_prompts import ensure_cache

    weights = os.path.join(run_dir, "weights", which)
    if predictor is None:
        predictor, state = load_run_model(run_dir, cfg, device, which)
        log(f"[eval] {which} from round {state.get('epoch')}")
    os.makedirs(out_dir, exist_ok=True)
    todo = {}
    for split in splits:
        pats = list_patients(cfg["data"][f"{split}_pool"])
        todo[split] = pats[:max_patients] if max_patients else pats
        ensure_cache(cfg, todo[split], log=log,
                     progress=lambda d, t, s=split: progress and progress(0, 0, f"prompts:{s} {d}/{t}"))
    total = sum(len(v) for v in todo.values())
    done = 0
    summary = {"created": datetime.now().isoformat(timespec="seconds"), "which": which,
               "threshold": float(cfg["evaluate"]["mask_threshold"])}
    for split, patients in todo.items():
        path = os.path.join(out_dir, f"{split}.json")
        records = _already_scored(path, cfg, patients, weights)
        if records is not None:
            log(f"[eval] {split}: keeping the {len(records)} patients already scored")
            done += len(patients)
        else:
            records = []
            for p in patients:
                records.append(patient_record(predictor, cfg, p["id"], device))
                done += 1
                if progress:
                    progress(done, total, split)
            write_json(path, {"thresholds": thresholds_of(cfg), "patients": records,
                              "settings": scoring_settings(cfg)})
        summary[split] = summarise(records, cfg)
        s = summary[split]
        log(f"[eval] {split}: MedSAM2 3D Dice {s['dice3d_mean']:.4f} (median {s['dice3d_median']:.4f}) "
            f"vs YOLO {s['yolo_dice3d_mean']:.4f} | helped {s['helped']} / hurt {s['hurt']} "
            f"| {s['anchors_mean']:.1f} anchors over {s['rounds_mean']:.1f} rounds "
            f"| best sweep threshold on {split}: {s['best_threshold']}")
    write_json(os.path.join(out_dir, "summary.json"), summary)
    return summary


def quick_dice(predictor, cfg, patients, device="cuda"):
    """Mean 3D Dice over some patients — the after-every-round check.

    One HITL round only: a check that ran the full loop would cost more than the round of
    training it is checking.
    """
    import copy

    one_round = copy.deepcopy(cfg)
    one_round["hitl"]["rounds"] = 1
    scores = []
    for p in patients:
        entry = load_prompt(one_round, p["id"])
        out = run_patient(predictor, one_round, p["id"], device, entry)
        gate = (out["objs"] > 0)[:, None, None] if cfg["evaluate"]["use_obj_score"] else True
        pred = (out["logits"] > float(cfg["evaluate"]["mask_threshold"])) & gate
        scores.append(dice(*_counts(pred, entry["gt"] > 0)))
    return float(np.mean(scores)) if scores else 0.0
