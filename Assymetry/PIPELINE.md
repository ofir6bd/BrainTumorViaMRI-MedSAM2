# Asymmetry Pipeline — Spec (DRAFT, awaiting confirmation)

> Status: **draft for review.** No code has been written yet. Refine this file with me
> until you approve it, then I'll implement it. **All changes stay inside `Assymetry/`.**

---

## 0. Scope & constraints

- **Pipeline code, outputs, and rendering logic live in `Assymetry/`.** The UI hooks into
  the **existing `src/web` viewer** (decided): a new **"Asymmetry"** sidebar entry is added
  there, and it calls into `Assymetry/`. So the only files touched outside `Assymetry/` are
  the web app's `index.html`, `app.js`, `style.css`, and `app.py` (thin route wiring).
- **Clustering = Gaussian Mixture Models** (`sklearn.mixture.GaussianMixture`) — used both
  for per-hemisphere clustering (Stage B) and for the Area Difference feature (Stage C).
- Data source is **`data/sample/`** only (read-only). Current patients:
  - `BraTS-GLI-00046-101`
  - `BraTS-GLI-00060-100`
- Modality file suffixes in this dataset:
  | Role  | Suffix   |
  |-------|----------|
  | T1    | `-t1n`   |
  | T1C   | `-t1c`   |
  | FLAIR | `-t2f`   |
  | T2    | `-t2w`   |
  | GT seg| `-seg`   |
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

### Stage A — Preprocessing & Skull Stripping (uses T1 + T1C)
1. Load T1 (`-t1n`) and T1C (`-t1c`) volumes.
2. **Histogram matching**: match T1C's intensity histogram to T1 (or vice-versa —
   see Q3) so the two are on a comparable intensity scale.
3. **Adaptive threshold**: build the histogram of each, and find the intensity at which
   the **difference between the two histograms is maximal**; use that intensity as the
   skull-stripping threshold.
4. Apply the threshold to produce a **brain mask** (largest connected component, hole-fill
   — see Q4), and apply the same mask to **FLAIR** to get skull-stripped FLAIR.

**Output:** `brain_mask` (3D bool), `flair_stripped` (3D float).

### Stage B — Per-slice asymmetry processing (on skull-stripped FLAIR)
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
Assuming the vertical mid-sagittal plane splits skull-stripped FLAIR into two halves,
compute, per slice (and/or aggregated per volume — see Q8):
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

## 3. Web UI (self-contained, inside `Assymetry/`)

A standalone page served by `Assymetry/` with a **left sidebar** whose main entry is an
**"Asymmetry"** button. Selecting it opens the step-through viewer.

**Interaction model — "Next" walks the whole process:**
- A big **Next** button advances **one step at a time** through the stages, rendering the
  image for that step so the process is visible incrementally. Proposed step order:
  1. T1 & T1C raw
  2. Histogram matching result
  3. Adaptive-threshold histogram (with chosen threshold marked)
  4. Brain mask + skull-stripped FLAIR
  5. Current slice: midline drawn
  6. Per-hemisphere clustering (mean-color quantized)
  7. Flipped-hemisphere overlay
  8. Difference map + candidate mask
  9. Feature values (AD / MD / BC) for the slice
  10. Prediction vs. GT overlay + slice Dice
  - After the last per-slice step, **Next** moves to the next qualifying slice and
    resumes at step 5; after the last slice, it shows the **volume Dice** summary.
- **Skip slice** button → jump straight to the next qualifying slice.
- **Skip patient** button → jump to the next patient.
- Patient selector (dropdown) for the two sample patients.

Rendering follows the existing app's approach: **server-rendered matplotlib PNGs**, so it
matches the current look and needs no JS charting.

---

## 4. Parameters (defaults — adjust in review)

| Name           | Meaning                                   | Proposed default |
|----------------|-------------------------------------------|------------------|
| `N_min_voxel`  | min brain voxels for a slice to be processed | `500`         |
| `K`            | GMM components per hemisphere              | `3`             |
| `diff_thresh`  | threshold on difference map for candidate mask | see Q5      |

---

## 5. Outputs / artifacts

- Rendered PNGs streamed to the browser (no files needed), **or** also saved under
  `Assymetry/outputs/<patient>/` for later inspection (see Q9).
- Optional CSV of per-slice `AD, MD, BC, Dice` under `Assymetry/outputs/`.

---

## 6. Open questions (please answer / correct)

_Resolved: UI = added into existing `src/web` viewer. Clusterer = GMM._

1. **Histogram-matching direction:** match T1C→T1 or T1→T1C as reference?
2. **Brain mask cleanup:** after thresholding, keep largest connected component + fill
   holes? Or use the raw threshold mask as-is?
3. **Midline location:** fixed geometric center column, or estimated (e.g. best
   left/right symmetry / centroid of brain mask)?
4. **Skipped slices:** hide them entirely, or show a "skipped (below N_min_voxel)" frame
   when stepping?
5. **Difference-map threshold:** fixed value, percentile (e.g. top 2%), or Otsu on the
   difference map?
6. **Features per-slice or per-volume:** compute AD/MD/BC per processed slice, aggregate
   to a per-volume number, or both?
7. **Persist outputs:** stream PNGs only, or also save PNGs/CSV under `Assymetry/outputs/`?
8. **Which slices count for volume Dice:** only slices we actually processed (≥
   `N_min_voxel`), or all slices (skipped slices predict empty)?
