"""Re-split the dataset pools BY PATIENT: every scan of a person lands in the same pool.

A scan folder is BraTS-GLI-XXXXX-YYY (XXXXX = patient, YYY = timepoint). Percentages are of
scans (patient + timepoint), but whole patients are assigned, so the shares come out close to,
not exactly, the targets. Moves the folders on disk and rewrites split_manifest.json (the old
one is kept as split_manifest.backup-<date>.json).

    python maunal_code/02_split_by_patient.py            # dry run: print the plan
    python maunal_code/02_split_by_patient.py --apply    # move folders + write manifest
"""
import argparse
import datetime
import json
import os
import random
import re
import shutil
from collections import defaultdict

import yaml

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
SEED = 42
TARGETS = {"test": 0.10, "yolo_val": 0.10, "yolo_train": 0.35, "medsam2_val": 0.10, "medsam2_train": 0.35}
SCAN_RE = re.compile(r"^BraTS-[A-Z]+-(\d{5})-(\d{3})$")


def plan(scans_by_patient):
    """Patients in random order, each to the pool furthest (relatively) below its scan target.

    Random order, not largest-first, so every pool gets the same mix of 1-scan and many-scan
    patients; largest-first piled the many-scan patients into the 35% pools.
    """
    total = sum(len(s) for s in scans_by_patient.values())
    want = {k: v * total for k, v in TARGETS.items()}
    have = {k: 0 for k in TARGETS}
    patients = sorted(scans_by_patient)
    random.Random(SEED).shuffle(patients)
    assign = {}
    for p in patients:
        pool = max(TARGETS, key=lambda k: (want[k] - have[k]) / want[k])
        assign[p] = pool
        have[pool] += len(scans_by_patient[p])
    return assign


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--apply", action="store_true", help="move folders and write the manifest")
    args = ap.parse_args()

    with open(os.path.join(ROOT, "config.yaml"), "r", encoding="utf-8") as f:
        paths = yaml.safe_load(f)["paths"]
    pool_dir = {k: os.path.join(ROOT, paths[k]) for k in TARGETS}
    dataset = os.path.join(ROOT, paths["dataset"])

    where = {}                                    # scan -> current pool
    scans_by_patient = defaultdict(list)
    for pool, d in pool_dir.items():
        os.makedirs(d, exist_ok=True)
        for name in os.listdir(d):
            m = SCAN_RE.match(name)
            if m and os.path.isdir(os.path.join(d, name)):
                where[name] = pool
                scans_by_patient[m.group(1)].append(name)

    assign = plan(scans_by_patient)
    total = len(where)
    moves = [(s, where[s], assign[p]) for p, ss in scans_by_patient.items() for s in ss
             if where[s] != assign[p]]

    print(f"{total} scans, {len(scans_by_patient)} patients, {len(moves)} scans to move")
    print(f"{'pool':<15}{'patients':>9}{'scans':>8}{'% scans':>9}{'target':>8}")
    for k in TARGETS:
        pats = [p for p, a in assign.items() if a == k]
        n = sum(len(scans_by_patient[p]) for p in pats)
        print(f"{k:<15}{len(pats):>9}{n:>8}{100 * n / total:>8.1f}%{100 * TARGETS[k]:>7.0f}%")
    if not args.apply:
        print("dry run - pass --apply to move the folders")
        return

    for scan, src, dst in moves:
        shutil.move(os.path.join(pool_dir[src], scan), os.path.join(pool_dir[dst], scan))

    manifest_path = os.path.join(dataset, "split_manifest.json")
    today = datetime.date.today().isoformat()
    if os.path.exists(manifest_path):
        shutil.copy2(manifest_path, os.path.join(dataset, f"split_manifest.backup-{today}.json"))
    assignment = {s: assign[p] for p, ss in scans_by_patient.items() for s in ss}
    manifest = {
        "seed": SEED,
        "note": (f"{today}: full re-split BY PATIENT (BraTS-GLI-XXXXX, all timepoints together), "
                 "so no person is in two pools, test/ included. Targets are % of scans: "
                 + ", ".join(f"{k} {int(v * 100)}%" for k, v in TARGETS.items())
                 + ". Made by maunal_code/02_split_by_patient.py. Earlier splits: see the backups."),
        "targets": TARGETS,
        "counts": {k: sum(1 for a in assignment.values() if a == k) for k in TARGETS},
        "subject_counts": {k: sum(1 for a in assign.values() if a == k) for k in TARGETS},
        "assignment": dict(sorted(assignment.items())),
        "moves": [{"date": today, "from": "previous split", "to": "per-patient re-split",
                   "seed": SEED, "scans": len(moves), "ids": sorted(s for s, _, _ in moves)}],
    }
    with open(manifest_path, "w", encoding="utf-8") as f:
        json.dump(manifest, f, indent=2)
    print(f"moved {len(moves)} scans, wrote {manifest_path}")


if __name__ == "__main__":
    main()
