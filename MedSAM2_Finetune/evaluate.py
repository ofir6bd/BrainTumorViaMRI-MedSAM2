"""Score a fine-tuned checkpoint the way it runs: every slice prompted, memory in between.

For one patient:

  1. `anchors.anchor` picks the start slice (the most confident YOLO tumour area).
  2. Every slice gets YOLO's own map for that slice as its prompt.
  3. Two passes run away from the anchor — up to the top of the brain and down to the bottom —
     each through `model.track`, the very function training uses. The anchor is segmented
     without memory; every other slice with its own hint *and* the memory of the slices
     already passed.
  4. With `evaluate.tta_flip`, the same is done on the left-right mirrored patient (in the
     same batch) and the two answers are averaged.
  5. `evaluate.min_component` removes tiny 3D specks.

Nothing here ever looks at the expert mask, so every score is one the pipeline could
reproduce on a new patient. Alongside MedSAM2, the YOLO answer itself is scored on the same
patients — that pairing is the point of the whole folder. Every number is measured at the
slice's own size, so it is comparable with the YOLO_finetune page.
"""
import os
from datetime import datetime

import numpy as np
import torch
import torch.nn.functional as F
from scipy import ndimage

from . import anchors as anchor_rules
from .common import dice, list_patients, read_json, to_model_image, write_json
from .dataset import to_prompt
from .model import build_train_model, features, select, track
from .yolo_prompts import load_prompt, prompt_logits


def thresholds_of(cfg):
    e = cfg["evaluate"]
    return sorted(set([float(v) for v in e["sweep"]]) | {float(e["mask_threshold"])})


def load_run_model(run_dir, cfg, device="cuda", which="best.pt"):
    path = os.path.join(run_dir, "weights", which)
    if not os.path.exists(path):
        raise FileNotFoundError(f"no checkpoint at {path}")
    model, state = build_train_model(cfg, device, weights=path)
    return model.eval(), state


def _amp(cfg, device):
    return torch.autocast("cuda", dtype=torch.bfloat16,
                          enabled=bool(cfg["train"]["amp"]) and str(device).startswith("cuda"))


@torch.inference_mode()
def infer(model, cfg, entry, device="cuda"):
    """Mask logits [N, H, W] for one patient, the object score per slice, and the anchor."""
    sam2 = model.sam2
    size = cfg["model"]["image_size"]
    n = len(entry["z"])
    h, w = [int(v) for v in entry["shape"]]
    a = anchor_rules.anchor(entry, cfg)

    images = torch.stack([to_model_image(f, size) for f in entry["rgb"]]).to(device)
    prompts = to_prompt(prompt_logits(entry, cfg), size).to(device)          # [N, 1, S, S]
    flips = [False, True] if cfg["evaluate"].get("tta_flip") else [False]
    # array axis 0 of a slice (left-right of the head) is dim -2 of the image tensor
    images = torch.cat([images.flip(-2) if f else images for f in flips])
    prompts = torch.cat([prompts.flip(-2) if f else prompts for f in flips])
    v = len(flips)

    logits = np.zeros((v, n, h, w), np.float32)
    objs = np.zeros((v, n), np.float32)
    with _amp(cfg, device):
        feats = features(sam2, images)
        for order in (range(a, n), range(a, -1, -1)):
            order = torch.tensor(list(order), device=device)
            t = len(order)
            rows = (torch.arange(v, device=device)[:, None] * n + order[None, :]).flatten()
            p = torch.stack([prompts[k * n + order] for k in range(v)], dim=1)  # [T, V, 1, S, S]
            outs = track(sam2, select(feats, rows), p, t, v)
            for j, out in enumerate(outs):
                m = F.interpolate(out["pred_masks_high_res"].float(), size=(h, w),
                                  mode="bilinear", align_corners=False)[:, 0]
                k = int(order[j])
                logits[:, k] = m.cpu().numpy()
                objs[:, k] = out["multistep_object_score_logits"][-1].float().flatten().cpu().numpy()
            del outs
    for i, f in enumerate(flips):
        if f:
            logits[i] = logits[i][:, ::-1]
    return {"logits": logits.mean(axis=0), "objs": objs.mean(axis=0), "anchor": a,
            "views": logits}      # [V, N, H, W]: the plain and the mirrored answer, for tuning


def postprocess(mask, cfg):
    """Remove 3D specks smaller than `evaluate.min_component` voxels."""
    small = int(cfg["evaluate"].get("min_component") or 0)
    if small <= 0 or not mask.any():
        return mask
    lab, count = ndimage.label(mask)
    sizes = np.bincount(lab.ravel(), minlength=count + 1)
    keep = sizes >= small
    keep[0] = False
    return keep[lab]


def final_mask(out, cfg, threshold=None):
    t = float(cfg["evaluate"]["mask_threshold"]) if threshold is None else threshold
    m = out["logits"] > t
    if cfg["evaluate"].get("use_obj_score"):
        m &= (out["objs"] > 0)[:, None, None]
    return postprocess(m, cfg)


def run_patient(model, cfg, patient_id, device="cuda", entry=None):
    """One patient through the model; the page's slice viewer and the scorer both use this."""
    entry = entry if entry is not None else load_prompt(cfg, patient_id)
    out = infer(model, cfg, entry, device)
    out["mask"] = final_mask(out, cfg)
    out["anchors"] = [out["anchor"]]
    return out


def _counts(pred, gt):
    inter = int(np.logical_and(pred, gt).sum())
    return inter, int(pred.sum()), int(gt.sum())


def _per_slice(a, b):
    n = len(a)
    return np.logical_and(a, b).reshape(n, -1).sum(1), a.reshape(n, -1).sum(1)


def patient_record(model, cfg, patient_id, device="cuda", entry=None, out=None):
    """One patient: per-slice numbers for YOLO and MedSAM2, plus the threshold sweep.

    `entry` / `out` let a caller that has already run this patient (the page's slice
    viewer) reuse that work instead of paying for a second pass.
    """
    entry = entry if entry is not None else load_prompt(cfg, patient_id)
    out = out if out is not None else run_patient(model, cfg, patient_id, device, entry)
    gt = entry["gt"] > 0
    yolo = anchor_rules.yolo_mask(entry, cfg)
    sweep = [list(_counts(final_mask(out, cfg, t), gt)) for t in thresholds_of(cfg)]
    inter, pred = _per_slice(out["mask"], gt)
    y_inter, y_pred = _per_slice(yolo, gt)
    g = gt.reshape(len(gt), -1).sum(1)
    rec = {"id": patient_id, "z": entry["z"].tolist(), "gt": g.tolist(), "pred": pred.tolist(),
           "inter": inter.tolist(), "yolo": y_pred.tolist(), "yolo_inter": y_inter.tolist(),
           "score": entry["scores"].round(3).tolist(), "nblobs": entry["nblobs"].tolist(),
           "obj": np.round(out["objs"], 3).tolist(), "anchors": out["anchors"],
           "anchor_z": [int(entry["z"][k]) for k in out["anchors"]], "sweep": sweep}
    rec["dice3d"] = round(dice(int(inter.sum()), int(pred.sum()), int(g.sum())), 4)
    rec["yolo_dice3d"] = round(dice(int(y_inter.sum()), int(y_pred.sum()), int(g.sum())), 4)
    rec["delta"] = round(rec["dice3d"] - rec["yolo_dice3d"], 4)
    rec["gt_total"] = int(g.sum())
    return rec


def patient_profile(model, cfg, patient_id, device="cuda", entry=None, out=None):
    """The per-slice view the page charts: Dice and pixel counts for both models."""
    rec = patient_record(model, cfg, patient_id, device, entry, out)
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
        "patient_scores": [{"id": r["id"], "dice3d": r["dice3d"], "yolo_dice3d": r["yolo_dice3d"],
                            "delta": r["delta"], "gt_total": r["gt_total"],
                            "anchor_z": (r.get("anchor_z") or [None])[0],
                            "score_mean": round(float(np.mean(r["score"])), 3) if r["score"] else 0.0}
                           for r in records],
    }


def scoring_settings(cfg):
    """Everything that changes what a scoring pass produces."""
    return {"design": "dense-prompt", "prompt": cfg["prompt"], "evaluate": cfg["evaluate"],
            "image_size": cfg["model"]["image_size"],
            "min_fg_voxels": cfg["data"]["min_fg_voxels"],
            "mask_shortcut": cfg["model"].get("use_mask_input_as_output_without_sam", True)}


def _already_scored(path, cfg, patients, weights):
    """A finished `<split>.json` this run can keep instead of scoring the pool again.

    Reused only when the file covers the whole pool, was written *after* the checkpoint it
    is scoring, and its stored settings still match — so retraining or changing a setting
    re-scores.
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
             log=print, device="cuda", model=None, which="best.pt"):
    """Score `which` checkpoint on the given pools; writes <split>.json and summary.json."""
    from .yolo_prompts import ensure_cache

    weights = os.path.join(run_dir, "weights", which)
    if model is None:
        model, state = load_run_model(run_dir, cfg, device, which)
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
                records.append(patient_record(model, cfg, p["id"], device))
                done += 1
                if progress:
                    progress(done, total, split)
            write_json(path, {"thresholds": thresholds_of(cfg), "patients": records,
                              "settings": scoring_settings(cfg)})
        summary[split] = summarise(records, cfg)
        s = summary[split]
        log(f"[eval] {split}: MedSAM2 3D Dice {s['dice3d_mean']:.4f} (median {s['dice3d_median']:.4f}) "
            f"vs YOLO {s['yolo_dice3d_mean']:.4f} | helped {s['helped']} / hurt {s['hurt']} "
            f"| best sweep threshold on {split}: {s['best_threshold']}")
    write_json(os.path.join(out_dir, "summary.json"), summary)
    return summary


def quick_dice(model, cfg, patients, device="cuda"):
    """Mean 3D Dice over some patients — the after-every-round check. It is the full
    inference path (both passes, TTA, post-processing), so the round it picks as best is
    picked on the score that gets reported."""
    scores = []
    for p in patients:
        entry = load_prompt(cfg, p["id"])
        out = run_patient(model, cfg, p["id"], device, entry)
        scores.append(dice(*_counts(out["mask"], entry["gt"] > 0)))
    return float(np.mean(scores)) if scores else 0.0
