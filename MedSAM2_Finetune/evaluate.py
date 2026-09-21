"""Score a fine-tuned checkpoint, and — on the same slices — the YOLO prompt it started from.

Every number here is measured at the slice's own size, not at the 512 the model works in,
so a Dice is comparable with the one the YOLO_finetune page reports.

Two models are scored side by side on every patient:
  YOLO      the prompt as YOLO would publish it: blobs with score >= `prompt.score_min`,
            probability cut at 0.5 (this is exactly Ultralytics' own mask at that confidence)
  MedSAM2   this run's output for the same slices, prompted by the map in `prompt.variant`

That pairing is the point of the whole folder: it says whether MedSAM2 improved the guess
it was given, patient by patient.

Thresholds: one forward pass produces a logit map, so the whole `evaluate.sweep` costs
nothing extra — each threshold is just a different cut of the same numbers. As on the YOLO
page, a threshold may be chosen on val and is only ever *reported* on test.
"""
import os
from datetime import datetime

import numpy as np
import torch
import torch.nn.functional as F

from .common import build_model, dice, list_patients, to_model_image, write_json
from .model import SliceSAM2
from .yolo_prompts import load_prompt, prompt_logits


def thresholds_of(cfg):
    e = cfg["evaluate"]
    return sorted(set([float(v) for v in e["sweep"]]) | {float(e["mask_threshold"])})


def load_run_model(run_dir, cfg, device="cuda", which="best.pt"):
    """This run's fine-tuned weights on top of the vendored architecture."""
    path = os.path.join(run_dir, "weights", which)
    if not os.path.exists(path):
        raise FileNotFoundError(f"no checkpoint at {path}")
    state = torch.load(path, map_location="cpu", weights_only=False)
    sam2 = build_model(cfg, device=device)
    sam2.load_state_dict(state["model"])
    model = SliceSAM2(sam2).to(device).eval()
    return model, state


@torch.no_grad()
def predict_slices(model, entry, cfg, device="cuda", indices=None):
    """MedSAM2 logits at the slice's own size, plus the object score, for chosen slices."""
    e, size = cfg["evaluate"], cfg["model"]["image_size"]
    h, w = [int(v) for v in entry["shape"]]
    idx = np.arange(len(entry["z"])) if indices is None else np.asarray(indices)
    logits = np.zeros((len(idx), h, w), np.float32)
    objs = np.zeros(len(idx), np.float32)
    amp = torch.autocast("cuda", dtype=torch.bfloat16, enabled=bool(cfg["train"]["amp"]))
    for i in range(0, len(idx), e["batch"]):
        chunk = idx[i:i + e["batch"]]
        images = torch.stack([to_model_image(entry["rgb"][k], size) for k in chunk]).to(device)
        prompts = torch.from_numpy(prompt_logits(entry, cfg, chunk)).float()[:, None].to(device)
        prompts = F.interpolate(prompts, size=(size, size), mode="bilinear", align_corners=False)
        with amp:
            out, _ious, obj = model(images, prompts)
        out = F.interpolate(out.float(), size=(h, w), mode="bilinear", align_corners=False)
        logits[i:i + len(chunk)] = out[:, 0].cpu().numpy()
        objs[i:i + len(chunk)] = obj[:, 0].cpu().numpy()
    return logits, objs


def _counts(pred, gt):
    inter = int(np.logical_and(pred, gt).sum())
    return inter, int(pred.sum()), int(gt.sum())


def patient_record(model, cfg, patient_id, device="cuda"):
    """One patient: per-slice numbers for YOLO and MedSAM2, plus the threshold sweep."""
    entry = load_prompt(cfg, patient_id)
    logits, objs = predict_slices(model, entry, cfg, device)
    gt = entry["gt"] > 0
    yolo = entry["prob_f"] > 127            # YOLO's own mask at confidence `prompt.score_min`
    ths = thresholds_of(cfg)
    k_at = ths.index(float(cfg["evaluate"]["mask_threshold"]))
    gate = objs > 0 if cfg["evaluate"]["use_obj_score"] else np.ones(len(objs), bool)

    sweep = np.zeros((len(ths), 3), np.int64)
    rec = {"id": patient_id, "z": entry["z"].tolist(), "gt": [], "pred": [], "inter": [],
           "yolo": [], "yolo_inter": [], "score": entry["scores"].round(3).tolist(),
           "nblobs": entry["nblobs"].tolist(), "obj": np.round(objs, 3).tolist()}
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


def patient_profile(model, cfg, patient_id, device="cuda"):
    """The per-slice view the page charts: Dice and pixel counts for both models."""
    rec = patient_record(model, cfg, patient_id, device)
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
                            "score_mean": round(float(np.mean(r["score"])), 3) if r["score"] else 0.0}
                           for r in records],
    }


def evaluate(run_dir, cfg, out_dir, splits=("val", "test"), max_patients=None, progress=None,
             log=print, device="cuda", model=None, which="best.pt"):
    """Score `which` checkpoint on the given pools; writes <split>.json and summary.json."""
    from .yolo_prompts import ensure_cache

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
        records = []
        for p in patients:
            records.append(patient_record(model, cfg, p["id"], device))
            done += 1
            if progress:
                progress(done, total, split)
        write_json(os.path.join(out_dir, f"{split}.json"),
                   {"thresholds": thresholds_of(cfg), "patients": records})
        summary[split] = summarise(records, cfg)
        s = summary[split]
        log(f"[eval] {split}: MedSAM2 3D Dice {s['dice3d_mean']:.4f} (median {s['dice3d_median']:.4f}) "
            f"vs YOLO {s['yolo_dice3d_mean']:.4f} | helped {s['helped']} / hurt {s['hurt']} "
            f"| best sweep threshold on {split}: {s['best_threshold']}")
    write_json(os.path.join(out_dir, "summary.json"), summary)
    return summary


@torch.no_grad()
def quick_dice(model, cfg, patients, device="cuda"):
    """Mean 3D Dice over some patients — the after-every-round check, kept cheap."""
    scores = []
    for p in patients:
        entry = load_prompt(cfg, p["id"])
        logits, objs = predict_slices(model, entry, cfg, device)
        gate = (objs > 0)[:, None, None] if cfg["evaluate"]["use_obj_score"] else True
        pred = (logits > float(cfg["evaluate"]["mask_threshold"])) & gate
        gt = entry["gt"] > 0
        scores.append(dice(int(np.logical_and(pred, gt).sum()), int(pred.sum()), int(gt.sum())))
    return float(np.mean(scores)) if scores else 0.0
