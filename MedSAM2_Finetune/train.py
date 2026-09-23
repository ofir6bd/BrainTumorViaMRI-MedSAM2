"""Run one MedSAM2 fine-tune: build the YOLO prompts -> train on clips -> score val + test.

From the repo root:

    python -m MedSAM2_Finetune.train                           # new run from config.yaml
    python -m MedSAM2_Finetune.train --set train.epochs=5      # override any key (dotted, repeatable)
    python -m MedSAM2_Finetune.train --smoke                   # 2 patients, 1 round: end-to-end check
    python -m MedSAM2_Finetune.train --run <dir>               # run a prepared folder (the page does this)
    python -m MedSAM2_Finetune.train --run <dir> --resume      # continue from its last.pt
    python -m MedSAM2_Finetune.train --run <dir> --evaluate-only

Training works on clips: the anchor slice is prompted with YOLO's map and the rest of the
clip is reached through SAM2's memory, which is exactly what inference does over a whole
volume. The after-every-round check runs the real thing — a full forward + backward
propagation per patient — so the score that picks the best round is the score we report.

Each run lives in runs/<YYYYMMDD-HHMMSS>/: run_config.yaml (the exact settings), status.json
(progress, read by the page), results.csv (one row per round), weights/ and eval/.
"""
import argparse
import copy
import os
import time
import traceback
from datetime import datetime

import numpy as np
import torch
import yaml
from torch.utils.data import DataLoader

from .common import RUNS_DIR, list_patients, load_config, read_json, write_json
from .dataset import ClipDataset, PatientShuffle, clip_sources, collate, sample_clips
from .evaluate import build_predictor, evaluate, quick_dice
from .model import Clips, build_train_model, losses, param_groups

SMOKE = {"max_train_patients": 2, "val_patients": 2, "epochs": 1, "max_patients": 2}
CSV_COLUMNS = ["epoch", "time", "loss", "dice", "bce", "iou", "obj", "val_dice3d", "lr", "clips"]


class Status:
    """runs/<run>/status.json — the only channel from this process to the page."""

    def __init__(self, run_dir):
        self.path = os.path.join(run_dir, "status.json")
        self.data = read_json(self.path, {}) or {}

    def update(self, **kw):
        self.data.update(kw, updated=datetime.now().isoformat(timespec="seconds"))
        try:
            write_json(self.path, self.data)
        except OSError as e:
            # Progress reporting must never kill a run: the next update will catch up.
            print(f"[status] could not write status.json ({e}); continuing", flush=True)


def _set(cfg, dotted, value):
    node = cfg
    *path, last = dotted.split(".")
    for k in path:
        node = node[k]
    if last not in node:
        raise KeyError(f"unknown setting {dotted}")
    node[last] = yaml.safe_load(value) if isinstance(value, str) else value


def new_run(overrides=(), smoke=False):
    """Create runs/<id>/ with its own copy of the settings; return its path."""
    cfg = copy.deepcopy(load_config())
    for item in overrides:
        key, _, value = item.partition("=")
        _set(cfg, key.strip(), value.strip())
    if smoke:
        cfg["data"]["max_train_patients"] = SMOKE["max_train_patients"]
        cfg["data"]["val_patients"] = SMOKE["val_patients"]
        cfg["train"]["epochs"] = SMOKE["epochs"]
    run_id = datetime.now().strftime("%Y%m%d-%H%M%S")
    run_dir = os.path.join(RUNS_DIR, run_id)
    os.makedirs(run_dir)
    cfg["run"] = {"id": run_id, "created": datetime.now().isoformat(timespec="seconds"),
                  "smoke": smoke, "max_patients": SMOKE["max_patients"] if smoke else None}
    with open(os.path.join(run_dir, "run_config.yaml"), "w", encoding="utf-8") as f:
        yaml.safe_dump(cfg, f, sort_keys=False)
    Status(run_dir).update(state="queued", created=cfg["run"]["created"])
    return run_dir


def _pools(cfg):
    d = cfg["data"]
    train = list_patients(d["train_pool"])
    val = list_patients(d["val_pool"])
    if d.get("max_train_patients"):
        train = train[:d["max_train_patients"]]
    if d.get("val_patients"):
        val = val[:d["val_patients"]]
    return train, val


def _append_csv(path, row):
    new = not os.path.exists(path)
    with open(path, "a", encoding="utf-8", newline="") as f:
        if new:
            f.write(",".join(CSV_COLUMNS) + "\n")
        f.write(",".join("" if row.get(c) is None else f"{row[c]:g}" for c in CSV_COLUMNS) + "\n")


def _save(path, model, optimizer, cfg, epoch, val_dice3d):
    """Only `last.pt` carries the optimiser: it is the one a resume continues from."""
    state = {"model": model.sam2.state_dict(), "epoch": epoch, "val_dice3d": val_dice3d,
             "config": cfg}
    if os.path.basename(path) == "last.pt":
        state["optimizer"] = optimizer.state_dict()
    torch.save(state, path + ".tmp")
    os.replace(path + ".tmp", path)


def train(run_dir, cfg, status, resume=False):
    """The fine-tune itself: clips in, per-frame mask logits out, Dice + BCE on every frame."""
    t = cfg["train"]
    device = f"cuda:{t['device']}" if torch.cuda.is_available() else "cpu"
    torch.manual_seed(t["seed"])
    np.random.seed(t["seed"])

    train_patients, val_patients = _pools(cfg)
    sources = clip_sources(cfg, train_patients)
    per_round = sum(min(cfg["data"]["clips_per_patient"], s["n"]) for s in sources)
    status.update(train_patients=len(train_patients), val_patients=len(val_patients),
                  train_clips=per_round)
    print(f"[train] ~{per_round} clips of {cfg['video']['num_frames']} slices per round, drawn "
          f"fresh each round from {len(sources)} patients; checking on {len(val_patients)} val "
          f"patients", flush=True)

    model = build_train_model(cfg, device=device)
    optimizer = torch.optim.AdamW(param_groups(model, cfg), weight_decay=float(t["weight_decay"]))
    trainable = sum(p.numel() for p in model.parameters() if p.requires_grad)
    total = sum(p.numel() for p in model.parameters())
    print(f"[train] training {trainable/1e6:.1f}M of {total/1e6:.1f}M weights ({t['unfreeze']})", flush=True)
    # the checker is the real inference path, kept alongside and re-synced every round
    checker, _ = build_predictor(cfg, device=device)

    weights_dir = os.path.join(run_dir, "weights")
    os.makedirs(weights_dir, exist_ok=True)
    best_path = os.path.join(weights_dir, "best.pt")
    last_path = os.path.join(weights_dir, "last.pt")
    start_epoch, best, since_best = 0, -1.0, 0
    if resume and os.path.exists(last_path):
        state = torch.load(last_path, map_location="cpu", weights_only=False)
        model.sam2.load_state_dict(state["model"])
        optimizer.load_state_dict(state["optimizer"])
        start_epoch = int(state["epoch"])
        best = float(read_json(os.path.join(run_dir, "best.json"), {}).get("val_dice3d", -1.0))
        print(f"[train] resuming after round {start_epoch}", flush=True)

    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=max(1, t["epochs"]))
    for _ in range(start_epoch):
        scheduler.step()
    amp = torch.autocast("cuda", dtype=torch.bfloat16, enabled=bool(t["amp"]) and device != "cpu")

    for epoch in range(start_epoch, t["epochs"]):
        # New clips every round (see dataset.sample_clips): same patients, different windows.
        items = sample_clips(cfg, sources, seed=t["seed"] + epoch)
        loader = DataLoader(ClipDataset(cfg, items, train=True), batch_size=cfg["video"]["batch"],
                            sampler=PatientShuffle(items, seed=t["seed"] + epoch),
                            num_workers=t["workers"], collate_fn=collate, pin_memory=True,
                            drop_last=False)
        model.train()
        sums, n, seen, t0, last_report = {}, 0, 0, time.time(), 0.0
        optimizer.zero_grad(set_to_none=True)
        for step, batch in enumerate(loader):
            clips = Clips(batch["images"], batch["masks"], batch["obj_to_frame_idx"]).to(device)
            prompts = batch["prompts"].to(device, non_blocking=True)
            with amp:
                outputs = model(clips, prompts, anchors=[0])
            loss, parts = losses(outputs, clips, cfg)
            (loss / t["accum"]).backward()
            if (step + 1) % t["accum"] == 0:
                torch.nn.utils.clip_grad_norm_([p for p in model.parameters() if p.requires_grad], 1.0)
                optimizer.step()
                optimizer.zero_grad(set_to_none=True)
            for k, v in parts.items():
                sums[k] = sums.get(k, 0.0) + v
            n += 1
            seen += len(batch["patient"])
            if time.time() - last_report > 2:   # no need to hit the disk every step
                last_report = time.time()
                status.update(epoch=epoch + 1, epochs=t["epochs"], batch=step + 1,
                              batches=len(loader), loss=round(sums["loss"] / n, 4))
        optimizer.step()               # whatever is left in the accumulator
        optimizer.zero_grad(set_to_none=True)
        scheduler.step()

        model.eval()
        checker.load_state_dict(model.sam2.state_dict())
        val_dice3d = quick_dice(checker, cfg, val_patients, device=device)
        row = {"epoch": epoch + 1, "time": round(time.time() - t0, 1), "val_dice3d": round(val_dice3d, 4),
               "lr": optimizer.param_groups[0]["lr"], "clips": seen,
               **{k: round(v / max(n, 1), 4) for k, v in sums.items()}}
        _append_csv(os.path.join(run_dir, "results.csv"), row)
        _save(last_path, model, optimizer, cfg, epoch + 1, val_dice3d)
        if val_dice3d > best:
            best, since_best = val_dice3d, 0
            _save(best_path, model, optimizer, cfg, epoch + 1, val_dice3d)
            write_json(os.path.join(run_dir, "best.json"), {"epoch": epoch + 1, "val_dice3d": val_dice3d})
        else:
            since_best += 1
        status.update(epoch=epoch + 1, epochs=t["epochs"], epoch_time=row["time"],
                      val_dice3d=round(val_dice3d, 4), best_val_dice3d=round(best, 4),
                      since_best=since_best, batch=len(loader), batches=len(loader))
        print(f"[train] round {epoch + 1}/{t['epochs']}: loss {row['loss']:.4f} "
              f"(dice {row['dice']:.4f}, bce {row['bce']:.4f}) val 3D Dice {val_dice3d:.4f} "
              f"{'(best)' if since_best == 0 else f'({since_best} since best)'} "
              f"in {row['time']:.0f}s", flush=True)
        if t["patience"] and since_best >= t["patience"]:
            print(f"[train] no better round for {since_best} rounds — stopping early", flush=True)
            break
    del checker
    torch.cuda.empty_cache()
    return best_path


def run(run_dir, resume=False, evaluate_only=False):
    from .yolo_prompts import ensure_cache, prompt_key, yolo_run

    with open(os.path.join(run_dir, "run_config.yaml"), "r", encoding="utf-8") as f:
        cfg = yaml.safe_load(f)
    status = Status(run_dir)
    max_patients = cfg["run"].get("max_patients")
    status.update(state="running", pid=os.getpid(), started=datetime.now().isoformat(timespec="seconds"),
                  error=None, stage="prompts")
    try:
        chosen = yolo_run(cfg["prompt"]["yolo_run"])
        status.update(yolo_run=chosen["id"], prompt_key=prompt_key(cfg))
        print(f"[prompts] YOLO run {chosen['id']} ({chosen['which']}), variant {cfg['prompt']['variant']}; "
              f"anchors {cfg['anchors']['pick']} x{cfg['anchors']['count']}, "
              f"HITL up to {cfg['hitl']['rounds']} round(s)", flush=True)
        if not evaluate_only:
            train_patients, val_patients = _pools(cfg)
            ensure_cache(cfg, train_patients + val_patients,
                         progress=lambda d, t: status.update(prompt_done=d, prompt_total=t))
            status.update(stage="train")
            train(run_dir, cfg, status, resume=resume)

        # evaluate() builds whatever prompts val and test still need itself.
        status.update(stage="evaluate", eval_done=0, eval_total=None)
        evaluate(run_dir, cfg, os.path.join(run_dir, "eval"), max_patients=max_patients,
                 progress=lambda d, t, split: status.update(eval_done=d, eval_total=t, eval_split=split),
                 device=f"cuda:{cfg['train']['device']}" if torch.cuda.is_available() else "cpu")
        status.update(state="done", stage="done", finished=datetime.now().isoformat(timespec="seconds"))
    except Exception as e:
        traceback.print_exc()
        status.update(state="failed", error=f"{type(e).__name__}: {e}",
                      finished=datetime.now().isoformat(timespec="seconds"))
        raise


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--run", help="existing run folder (created by the page or a previous call)")
    ap.add_argument("--set", action="append", default=[], metavar="KEY=VALUE",
                    help="override a config.yaml setting for a new run, e.g. train.epochs=5")
    ap.add_argument("--smoke", action="store_true", help="2 patients, 1 round")
    ap.add_argument("--resume", action="store_true", help="continue from the run's last.pt")
    ap.add_argument("--evaluate-only", action="store_true", help="re-score best.pt")
    args = ap.parse_args()
    run_dir = os.path.abspath(args.run or new_run(args.set, args.smoke))
    print(f"[run] {run_dir}", flush=True)
    run(run_dir, resume=args.resume, evaluate_only=args.evaluate_only)


if __name__ == "__main__":
    main()
