"""Per-patient tumour statistics for the Analytics tab.

Every number comes from the patient's own `-seg` and FLAIR volumes — nothing is
estimated or filled in. A patient whose files can't be read is reported as an error,
never given substitute values.

Results are cached in `<paths.outputs>/dashboard/dataset_stats.json`, one record per
patient id, keyed by the seg file's size + mtime. Moving a folder between pools keeps
both (os.rename), so re-splitting the dataset costs no recompute: the pool is not
cached, it's read from where the folder is *now* (see `app.py`).

Run directly to (re)build the cache from a terminal:
    .venv\\Scripts\\python.exe Frontend\\dataset_stats.py
"""
import json
import os
import threading
import time
from concurrent.futures import ThreadPoolExecutor, as_completed
from datetime import datetime

import nibabel as nib
import numpy as np
from scipy import ndimage

LABELS = {1: "NETC", 2: "SNFH", 3: "ET", 4: "RC"}
CACHE_VERSION = 1
WORKERS = 8

# Which world direction a voxel axis increases toward, per nibabel axis code, as
# (component, +1 if it increases toward Left / Anterior / Superior else -1).
_AXIS = {"L": (0, 1), "R": (0, -1), "A": (1, 1), "P": (1, -1), "S": (2, 1), "I": (2, -1)}


def _file_key(path):
    st = os.stat(path)
    return [st.st_size, int(st.st_mtime)]


def patient_stats(patient):
    """Statistics for one patient dict from `app.find_patients` (needs `seg_file` and
    `modalities["FLAIR"]`)."""
    seg_img = nib.load(patient["seg_file"])
    seg = np.rint(np.asarray(seg_img.dataobj)).astype(np.uint8)
    zooms = [float(z) for z in seg_img.header.get_zooms()[:3]]
    vox_ml = zooms[0] * zooms[1] * zooms[2] / 1000.0
    codes = nib.aff2axcodes(seg_img.affine)

    brain = np.asarray(nib.load(patient["modalities"]["FLAIR"]).dataobj) != 0
    if brain.shape != seg.shape:
        raise ValueError(f"FLAIR shape {brain.shape} != seg shape {seg.shape}")

    counts = np.bincount(seg.ravel(), minlength=5)
    wt = seg > 0
    wt_vox = int(wt.sum())
    ml = {name: round(int(counts[i]) * vox_ml, 3) for i, name in LABELS.items()}
    ml["WT"] = round(wt_vox * vox_ml, 3)
    ml["TC"] = round((int(counts[1]) + int(counts[3]) + int(counts[4])) * vox_ml, 3)
    brain_ml = round(int(brain.sum()) * vox_ml, 3)

    # Axial slices = array axis 2. Profile ordered inferior -> superior.
    per_slice = wt.sum(axis=(0, 1)).astype(int)
    if _AXIS[codes[2]][1] < 0:
        per_slice = per_slice[::-1]
    area_mm2 = zooms[0] * zooms[1]
    tumour_z = np.nonzero(wt.sum(axis=(0, 1)))[0]

    rec = {
        "id": patient["patient_id"],
        "subject": "-".join(patient["patient_id"].split("-")[:3]),
        "timepoint": patient["patient_id"].split("-")[-1],
        "shape": list(seg.shape),
        "spacing_mm": zooms,
        "axcodes": "".join(codes),
        "ml": ml,
        "brain_ml": brain_ml,
        "wt_pct_brain": round(100.0 * ml["WT"] / brain_ml, 3) if brain_ml else None,
        "present": [name for i, name in LABELS.items() if counts[i] > 0],
        "n_slices": int(tumour_z.size),
        "peak_z": int(np.argmax(wt.sum(axis=(0, 1)))) if wt_vox else None,
        "peak_area_mm2": round(float(per_slice.max()) * area_mm2, 1),
        "z_profile_mm2": [round(float(v) * area_mm2, 1) for v in per_slice],
        "extent_mm": None,
        "centroid_rel": None,
        "left_frac": None,
        "n_cc": 0,
        "largest_cc_frac": None,
    }
    if not wt_vox:
        return rec

    # Extent + position, both in (L-R, A-P, S-I) order regardless of voxel axis order.
    t_idx = np.nonzero(wt)
    b_idx = np.nonzero(brain)
    extent = [0.0, 0.0, 0.0]
    rel = [0.0, 0.0, 0.0]
    for ax in range(3):
        comp, sign = _AXIS[codes[ax]]
        extent[comp] = round(float(t_idx[ax].max() - t_idx[ax].min() + 1) * zooms[ax], 1)
        lo, hi = float(b_idx[ax].min()), float(b_idx[ax].max())
        r = (float(t_idx[ax].mean()) - lo) / max(hi - lo, 1.0)
        rel[comp] = round(r if sign > 0 else 1.0 - r, 4)
    rec["extent_mm"] = extent
    rec["centroid_rel"] = rel  # 0..1 toward Left, Anterior, Superior (brain bbox)

    # Share of tumour on the patient's left of the brain's centroid plane.
    lr_ax = next(ax for ax in range(3) if _AXIS[codes[ax]][0] == 0)
    mid = float(b_idx[lr_ax].mean())
    toward_left = t_idx[lr_ax] > mid if _AXIS[codes[lr_ax]][1] > 0 else t_idx[lr_ax] < mid
    rec["left_frac"] = round(float(toward_left.mean()), 4)

    lab, n_cc = ndimage.label(wt)
    rec["n_cc"] = int(n_cc)
    rec["largest_cc_frac"] = round(float(np.bincount(lab.ravel())[1:].max()) / wt_vox, 4)
    return rec


class StatsStore:
    """Cached statistics + a background job that fills in whatever is missing."""

    def __init__(self, cache_path):
        self.cache_path = cache_path
        self._lock = threading.Lock()
        self._thread = None
        self.records = {}
        self.errors = {}
        self.computed_at = None
        self.progress = {"done": 0, "total": 0}
        self._load()

    def _load(self):
        if not os.path.exists(self.cache_path):
            return
        try:
            with open(self.cache_path, "r", encoding="utf-8") as f:
                data = json.load(f)
        except (OSError, ValueError):
            return
        if data.get("version") == CACHE_VERSION:
            self.records = data.get("records", {})
            self.computed_at = data.get("computed_at")

    def _save(self):
        os.makedirs(os.path.dirname(self.cache_path), exist_ok=True)
        tmp = self.cache_path + ".tmp"
        with open(tmp, "w", encoding="utf-8") as f:
            json.dump({"version": CACHE_VERSION, "computed_at": self.computed_at,
                       "records": self.records}, f)
        os.replace(tmp, self.cache_path)

    def missing(self, patients):
        return [p for p in patients
                if (r := self.records.get(p["patient_id"])) is None
                or r.get("_key") != _file_key(p["seg_file"])]

    @property
    def running(self):
        return self._thread is not None and self._thread.is_alive()

    def start(self, patients, force=False, log=None):
        """Compute every patient not already cached (all of them if `force`)."""
        with self._lock:
            if self.running:
                return False
            todo = list(patients) if force else self.missing(patients)
            if not todo:
                return False
            if force:
                self.records = {}
            self.errors = {}
            self.progress = {"done": 0, "total": len(todo)}
            self._thread = threading.Thread(target=self._run, args=(todo, log), daemon=True)
            self._thread.start()
            return True

    def _run(self, todo, log):
        t0 = time.time()
        with ThreadPoolExecutor(max_workers=WORKERS) as pool:
            futures = {pool.submit(patient_stats, p): p for p in todo}
            for fut in as_completed(futures):
                p = futures[fut]
                try:
                    rec = fut.result()
                    rec["_key"] = _file_key(p["seg_file"])
                    self.records[p["patient_id"]] = rec
                except Exception as e:  # reported to the UI, never replaced with fake values
                    self.errors[p["patient_id"]] = f"{type(e).__name__}: {e}"
                self.progress["done"] += 1
                if log and self.progress["done"] % 100 == 0:
                    log(f"[stats] {self.progress['done']}/{self.progress['total']}")
        self.computed_at = datetime.now().isoformat(timespec="seconds")
        self._save()
        if log:
            log(f"[stats] done: {len(todo)} patients in {time.time() - t0:.0f}s, "
                f"{len(self.errors)} errors")

    def status(self, patients):
        n_missing = 0 if self.running else len(self.missing(patients))
        state = "running" if self.running else ("ready" if not n_missing else "missing")
        return {"state": state, "done": self.progress["done"], "total": self.progress["total"],
                "missing": n_missing, "errors": self.errors, "computed_at": self.computed_at}


if __name__ == "__main__":
    import sys

    sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
    from app import PATIENTS, stats_store

    if not stats_store.start(PATIENTS, force="--force" in sys.argv, log=print):
        print(f"[stats] cache already complete for {len(PATIENTS)} patients "
              f"({stats_store.cache_path}); pass --force to rebuild")
    elif stats_store._thread:
        stats_store._thread.join()
