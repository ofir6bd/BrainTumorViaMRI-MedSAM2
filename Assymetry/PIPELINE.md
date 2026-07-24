# Asymmetry Pipeline — Spec (IMPLEMENTED)

> Status: **implemented.** Code lives in `Assymetry/` (`pipeline.py`, `render.py`,
> `routes.py`); the UI is wired into `Frontend/` (sidebar "Asymmetry" button).
> Launch with `run_web.bat` and open the **Asymmetry** tab.
>
> As-built decisions are recorded in §6. Defaults live in `PARAMS` at the top of
> `pipeline.py` (`n_min_voxel=500`, `k_max=4`, ...).
>
> **No skull-stripping stage:** BraTS volumes are already skull-stripped, so the former
> Stage A (T1/T1C histogram matching → adaptive threshold → brain mask) was removed. The
> brain region is simply the **nonzero FLAIR voxels**, and the pipeline runs on FLAIR
> directly.

---

## 0. Scope & constraints

- **Pipeline code, outputs, and rendering logic live in `Assymetry/`.** The UI hooks into
  the **existing viewer in `Frontend/`**: a new **"Asymmetry"** sidebar entry calls into
  `Assymetry/`. Files touched outside `Assymetry/` are the web app's `index.html`,
  `app.js` (nav switch), `asymmetry.js` (new), `style.css`, and `app.py` (blueprint wiring).
- **Clustering = Gaussian Mixture Models** (`sklearn.mixture.GaussianMixture`) — used both
  for per-hemisphere clustering (Stage B) and for the Area Difference feature (Stage C).
- Data source is **`data/sample/`** only (read-only). Current patients:
  - `BraTS-GLI-00046-101`
  - `BraTS-GLI-00060-100`
- Only **FLAIR** (`-t2f`) and the **GT segmentation** (`-seg`) are used. T1/T1C/T2 exist in
  the dataset but are no longer read (they were only used by the removed skull-strip stage).
- Volumes are 3D NIfTI (`.nii.gz`). We process **axial slices** (index `z` along axis 2),
  matching the existing viewer's convention.

---

## 1. Goal

For each sample patient, detect the **whole tumor** by exploiting **left/right brain
asymmetry** in the skull-stripped FLAIR image, then score the result against the ground
truth (union of all 4 BraTS labels) using **Dice**. Every intermediate step is rendered
as an image and steppable in a small web UI.

---

## 2. Pipeline stages

> **Input:** BraTS FLAIR, already skull-stripped. `brain_mask = FLAIR > 0`;
> `flair_stripped` = FLAIR normalised to [0, 1]. (There is no Stage A — see the note at
> the top of this file.)

### Stage B — Per-slice asymmetry processing (on FLAIR)
Iterate over every axial slice `z`:
1. Count brain voxels in the slice. If `count < N_min_voxel` → **skip this slice** (no
   tumor decision, not shown unless "show skipped" is on — see Q6).
2. Otherwise:
   a. Place a **vertical mid-sagittal line** splitting the slice into left/right
      hemispheres (see Q5 for how the midline is located).
   b. **Cluster each hemisphere independently** with a **GMM** of `K` components (fit on
      the hemisphere's brain-voxel intensities).
   c. Replace every pixel with its **cluster's mean intensity** → a quantized image
      per hemisphere.
   d. **Flip** one hemisphere across the midline so it overlays the other.
   e. **Subtract** the two hemispheres → an **asymmetry/difference map**.
   f. Threshold the difference map (see Q7) → candidate **tumor mask** for slice `z`.

**Output per slice:** midline, per-hemisphere quantized images, flipped overlay,
difference map, candidate mask.

### Stage C — Symmetry feature extraction (paper features)
Assuming the vertical mid-sagittal plane splits FLAIR into two halves, compute per slice
and aggregated per volume:
- **AD — Area Difference**: difference in cluster areas between hemispheres, via GMM.
- **MD — Mean Difference**: difference of mean gray levels between hemispheres.
- **BC — Bhattacharyya Coefficient**: overlap of the two hemispheres' intensity histograms.

**Output:** a small feature table (`AD`, `MD`, `BC`) per processed slice.

### Stage D — Evaluation (Dice vs. whole tumor)
1. Build **ground-truth whole tumor** = `seg > 0` (union of labels 1–4).
2. Stack the per-slice candidate masks into a 3D predicted mask.
3. Compute **Dice** between predicted mask and GT whole tumor:
   - per slice, and
   - for the whole volume.

**Output:** Dice scores (per slice + volume), overlay of prediction vs. GT.

---

## 3. Web UI (in `Frontend/`, backed by `Assymetry/`)

The `Frontend/` viewer has a **left sidebar** with an **"Asymmetry"** entry that opens the
step-through viewer; it calls the `/asym` blueprint served from `Assymetry/routes.py`.

**Interaction model — "Next" walks the whole process:**
- A big **Next** button advances **one step at a time** through the stages, rendering the
  image for that step so the process is visible incrementally. As-built step order
  (defined in `render.py:STEP_DEFS`; ids adapt automatically if steps are inserted):
  1. Current slice: midline drawn
  2. Per-hemisphere GMM clustering (mean-color quantized)
  3. Flipped-hemisphere overlay + |difference|
  4. Difference map + candidate mask
  5. Feature values (AD / MD / BC) for the slice
  6. Prediction vs. GT overlay + slice Dice
  7. Volume summary (whole-tumour Dice + mean features)
  - Steps 1-6 repeat per qualifying slice; after the last per-slice step, **Next** moves to
    the next qualifying slice and resumes at the midline step; after the last slice it shows
    step 7, the **volume Dice** summary.
- **Skip slice** button → jump straight to the next qualifying slice.
- **Skip patient** button → jump to the next patient.
- Patient selector (dropdown) for the two sample patients.

Rendering follows the existing app's approach: **server-rendered matplotlib PNGs**, so it
matches the current look and needs no JS charting.

---

## 4. Parameters (`PARAMS` in `pipeline.py`)

| Name          | Meaning                                        | Default |
|---------------|------------------------------------------------|---------|
| `n_min_voxel` | min brain voxels for a slice to be processed   | `500`   |
| `k_max`       | max GMM components per hemisphere (auto-K, BIC)| `4`     |
| `feat_bins`   | histogram bins for the Bhattacharyya coeff.    | `64`    |

Difference-map threshold is **Otsu** on the `|difference|` values (not a fixed parameter).

---

## 5. Outputs / artifacts

- Rendered PNGs streamed to the browser on demand.
- Per-slice `z, brain_voxels, AD, MD, BC, dice` (+ a `volume` Dice row) saved to
  `Assymetry/outputs/<patient>/features.csv` on the summary step.

---

## 6. As-built decisions

_UI = added into the `Frontend/` viewer. Clusterer = GMM with auto-K by BIC (K =
2..`k_max`, default 4). **No skull-stripping stage** — see the note at the top._

1. **Brain region:** `FLAIR > 0` (BraTS is pre-skull-stripped); FLAIR normalised to [0,1].
2. **Midline:** **estimated per slice** = centroid column of the brain mask (fallback =
   geometric centre). Halves cropped to a symmetric width around it before flipping.
3. **Skipped slices:** **reported and rendered as a "skipped" frame** when stepping.
4. **Difference-map threshold:** **Otsu** on the `|difference|` values.
5. **Features:** computed **per slice and aggregated per volume** (mean).
6. **Persist outputs:** per-slice `AD,MD,BC,dice` **CSV saved to
   `Assymetry/outputs/<patient>/features.csv`** on the summary step.
7. **Volume Dice denominator:** **all slices** (skipped slices predict empty).
