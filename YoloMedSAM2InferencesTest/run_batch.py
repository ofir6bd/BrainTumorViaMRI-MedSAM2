"""Batch runner — sweep patients x variants offline, then read the results in the UI.

Run from the repo root (paths in `config.yaml` are relative to it):

    .venv/Scripts/python.exe -m YoloMedSAM2InferencesTest.run_batch --patients 10

Every (patient, variant) result is cached the moment it finishes, so the sweep is
resumable: re-running skips what is already on disk and picks up where it stopped.

Start with the pre-flight, which needs no GPU time worth speaking of:

    ... run_batch --probe-only --patients 30

If confidence does not track Dice there, B / Bg / C1 / C2 have nothing to work with.
"""
import argparse
import os
import sys
import time

_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if _ROOT not in sys.path:
    sys.path.insert(0, _ROOT)

from YoloMedSAM2InferencesTest.pipeline import (PatientRun, aggregate,  # noqa: E402
                                                is_cached, test_patients)
from YoloMedSAM2InferencesTest.variants import ORDER, VARIANTS  # noqa: E402
from YOLO.pipeline import list_weight_files  # noqa: E402


def _weights(name):
    if not name:
        return None
    for e in list_weight_files():
        if e["filename"] == name:
            return e["path"]
    raise SystemExit(f"unknown weights file: {name}")


def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--patients", type=int, default=None,
                    help="how many test-split patients (default: all)")
    ap.add_argument("--start", type=int, default=0, help="skip this many patients first")
    ap.add_argument("--variants", default=",".join(ORDER),
                    help=f"comma-separated subset of {','.join(ORDER)}")
    ap.add_argument("--weights", default=None,
                    help="YOLO checkpoint filename (default: the pipeline default)")
    ap.add_argument("--probe-only", action="store_true",
                    help="YOLO-only confidence-vs-Dice pre-flight, no MedSAM2")
    ap.add_argument("--probe-conf", type=float, default=0.25)
    ap.add_argument("--force", action="store_true", help="recompute cached results")
    ap.add_argument("--progress", action="store_true",
                    help="keep SAM2's per-frame tqdm bars (off by default: a sweep emits "
                         "one bar per propagation and they bury the results)")
    args = ap.parse_args(argv)

    if not args.progress:
        os.environ["TQDM_DISABLE"] = "1"

    variants = [v.strip() for v in args.variants.split(",") if v.strip()]
    unknown = [v for v in variants if v not in VARIANTS]
    if unknown:
        raise SystemExit(f"unknown variants: {unknown}; known: {ORDER}")

    weights_path = _weights(args.weights)
    patients = test_patients()[args.start:]
    if args.patients is not None:
        patients = patients[:args.patients]

    print(f"{len(patients)} patient(s) x {len(variants)} variant(s) "
          f"-> {os.path.relpath(os.path.dirname(os.path.abspath(__file__)))}")
    t0 = time.time()
    probes = []

    for i, p in enumerate(patients, 1):
        pid = p["patient_id"]
        print(f"[{i}/{len(patients)}] {pid}")
        run = PatientRun(p["dir"], weights_path=weights_path)
        try:
            if args.probe_only:
                probes.append(run.confidence_probe(conf=args.probe_conf,
                                                   force=args.force))
                continue
            for vid in variants:
                if is_cached(pid, vid, run.weights_path) and not args.force:
                    print(f"    [{vid}] cached")
                    continue
                result, _ = run.run_variant(vid, force=args.force)
                print(f"    [{vid}] dice={result['final_dice']:.4f} "
                      f"anchors={result['anchors_used']} "
                      f"({result['stop_reason']}, {result['seconds']:.1f}s)")
        except Exception as e:  # noqa: BLE001 - one bad patient must not kill the sweep
            print(f"    !! {type(e).__name__}: {e}")
        finally:
            run.release()

    print(f"\ndone in {(time.time() - t0) / 60:.1f} min")

    if args.probe_only:
        scored = [p for p in probes if p.get("pearson_conf_dice") is not None]
        if scored:
            import numpy as np
            pear = np.mean([p["pearson_conf_dice"] for p in scored])
            spear = np.mean([p["spearman_conf_dice"] for p in scored
                             if p.get("spearman_conf_dice") is not None])
            print(f"confidence vs Dice over {len(scored)} patients: "
                  f"mean pearson={pear:.3f}  mean spearman={spear:.3f}")
            print("A value near 0 means YOLO's confidence does not predict its own "
                  "quality, and the confidence-driven variants cannot help.")
        return

    agg = aggregate(weights_path=weights_path)
    print(f"\n{agg['n_patients_complete']} patient(s) complete across all variants")
    print(f"{'id':<4} {'variant':<28} {'n':>4} {'mean':>7} {'median':>7} "
          f"{'vs O':>7} {'W/L':>8} {'anch':>5}")
    for row in agg["variants"]:
        if not row["n"]:
            continue
        print(f"{row['id']:<4} {row['label'][:28]:<28} {row['n']:>4} "
              f"{row['mean_dice']:>7.4f} {row['median_dice']:>7.4f} "
              f"{row['mean_delta_vs_ref']:>+7.4f} "
              f"{str(row['wins_vs_ref']) + '/' + str(row['losses_vs_ref']):>8} "
              f"{row['mean_anchors']:>5.1f}")


if __name__ == "__main__":
    main()
