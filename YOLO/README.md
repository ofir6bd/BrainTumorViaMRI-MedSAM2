# YOLO Tumour Segmentation Pipeline — Spec (PLANNED)

> Status: **spec / not yet implemented.** This file describes how to build and fine-tune a
> YOLO segmentation model for the BraTS whole tumour, and how to wire it into the existing web viewer.
> Code will live in `YOLO/` (`pipeline.py`, `render.py`, `routes.py`, plus a `train.py` used
> once to fine-tune the weights); the UI hooks into `Frontend/` (a new sidebar
> **"YOLO Detection"** entry), exactly like the `FCMSegmentation/` and `Assymetry/` packages.
>
> Defaults live in a `PARAMS` dict at the top of `pipeline.py` (see §4).

---

## 0. Scope & constraints

- **All new code, weights, datasets, and rendering logic live in `YOLO/`.** The only files
  touched outside `YOLO/` are the web app's `Frontend/templates/index.html` (new sidebar
  button), `Frontend/static/app.js` (nav switch), `Frontend/static/yolo.js` (new),
  `Frontend/static/style.css`, and `Frontend/app.py` (blueprint wiring) — the same pattern
  the FCM tab uses.
- **Model = YOLO-seg via Ultralytics** (`ultralytics`, YOLO11-seg). Fine-tune a pretrained
  `yolo11n-seg.pt` (or `yolo11s-seg.pt`) on our data — no training from scratch.
- **Ground truth is the expert `-seg.nii.gz` mask** (labels 1–4 merged to WT). Training labels
  are segmentation polygons derived from this mask. No synthetic labels.
- Data source follows `config.yaml` (`paths.extract_to`, `data/dataset`):
  - **Train split source**: `data/dataset/training_data1_v2` +
    `data/dataset/training_data_additional` (1350 + 271 patients, **labeled** —
    each has an expert `-seg.nii.gz` mask).
  - **Valid split source**: patient holdout from the combined train source above
    (default `val_fraction=0.2`, deterministic by `seed`) — also **labeled**, so
    Dice/mAP can be computed on it.
  - **Test split source**: `data/dataset/validation_data` (188 patients). **This is
    the official BraTS challenge validation split and ships with no `-seg` mask** —
    it is **unlabeled / inference-only**. It is used to preview predictions
    qualitatively, never for Dice or loss (no ground truth exists to compare against).
  - **Smoke test / demo** reads the single-patient `paths.sample` (`data/sample/`) when
    `inference.use_sample: true`, same toggle the other stages use.
  - **No synthetic labels**: slices from `validation_data` never get a label file —
    fabricating an empty/negative label there would assert "no tumour" as if it were
    ground truth, which it is not.
- Volumes are 3D NIfTI (`.nii.gz`). We process **axial slices** along axis 2. Slices are
  **numbered from 1** — the first axial slice is `z = 1`, the last is `z = D` (depth). This
  1-based number is what appears in filenames and in the web UI; internally it maps to array
  index `z − 1`.

BraTS labels: `1 = NETC`, `2 = SNFH`, `3 = ET`, `4 = RC`. **Whole tumour (WT) = 1+2+3+4**.

---

## 1. Goal

tumour appears as several disconnected regions on a slice, **each connected component gets its
Fine-tune a YOLO segmentation model that, given a single axial slice rendered as an RGB image,
predicts **whole-tumour mask(s)**. There is **one class, `tumour`**. When the tumour appears as
several disconnected regions on a slice, **multiple masks for that same slice are valid and
expected** (one per connected component). The web UI presents the selected slice with predicted
vs expert masks and reports Dice for that slice, plus a summary table of Dice across slices.

---

## 2. RGB channel definition

Each axial slice becomes one RGB image. The channels are built **per slice** from four
co-registered modalities:

| Channel | Source                          | Meaning                                       |
|---------|---------------------------------|-----------------------------------------------|
| **R**   | `T1C − T1` (subtraction)        | contrast uptake — highlights the enhancing rim |
| **G**   | `T2` (`-t2w`)                   | oedema / tumour signal                        |
| **B**   | `FLAIR` (`-t2f`)                | oedema / whole-tumour extent                  |

File suffixes on disk: `-t1c` (T1C), `-t1n` (T1), `-t2w` (T2), `-t2f` (FLAIR), `-seg` (GT).

**R-channel (subtraction) details:**
1. Load raw T1C and T1 volumes.
2. Subtract per slice: `diff = T1C − T1` (float). Clip negatives to 0 — only *enhancement*
   (uptake) is kept, not signal drop.
3. Normalise the R channel with the **same per-slice percentile scheme** the project already
   uses (`_norm_slice_uint8`: clip to p0.5–p99.5 of the non-zero voxels, rescale to 0–255).
   Slices with fewer than `MIN_FOREGROUND_VOXELS_PER_SLICE` foreground voxels → black.

**G and B channels:** each is its own modality normalised with the same `_norm_slice_uint8`
scheme (T2 → G, FLAIR → B).

The three uint8 channels are stacked into an `H×W×3` image and saved as a JPEG/PNG frame,
identical in spirit to `prepare_rgb_frames_cached`, only with the R channel swapped from raw
T1C to the `T1C − T1` subtraction.

---

## 3. Pipeline stages

### Stage A — Build RGB frames (`pipeline.py`)
For every patient, for every axial slice `z` (numbered `1 … D`):
1. Build the RGB image per §2.
2. Keep a slice only if it has brain foreground (skip all-black slices).
3. Write `YOLO/dataset/images/<split>/<patient>_z<zzz>.png`, where `<zzz>` is the **1-based**
   slice number (e.g. the first slice is `z001`).

### Stage B — Derive YOLO-seg labels from the expert `-seg` mask
For every kept slice `z` (1-based; array index `z − 1`):
1. `wt = (seg[:, :, z - 1] > 0)` — whole-tumour mask (labels 1–4 merged).
2. Split `wt` into connected components (`scipy.ndimage.label`), suppressing components
  smaller than `min_mask_area`.
3. For each surviving component, extract a contour/polygon in image coordinates.
4. Convert each component to **YOLO-seg normalised polygon format** (class id `0` followed by
  polygon points):
  ```
  0  x1/W y1/H x2/W y2/H ... xN/W yN/H
  ```
  where `W`, `H` are slice width/height and all coordinates are in `[0, 1]`.
5. Write `YOLO/dataset/labels/<split>/<patient>_z<zzz>.txt`.
   - Slices that have brain but **no tumour** get an **empty label file** (a negative
     example) so the model learns to output no tumour mask there.

### Stage C — Dataset assembly & split
- Split **by patient** (all slices of a patient go to the same split) to avoid leakage.
- Use these sources:
  - `train`: patients from `training_data1_v2` + `training_data_additional`, after
    removing the holdout set used for `val`. Writes `images/train/` + `labels/train/`.
  - `val`: holdout patients sampled only from the combined train source above
    (default 20% via `val_fraction`, deterministic by `seed`). Writes `images/val/` +
    `labels/val/`.
  - `test`: all patients from `validation_data` (never used for training). **No**
    `-seg` mask exists for this split, so only `images/test/` is written — there is no
    `labels/test/`. It's for qualitative inference preview only; Ultralytics `model.val()`
    metrics are only meaningful on `train`/`val`.
- Emit `YOLO/dataset/data.yaml`:
  ```yaml
  path: YOLO/dataset
  train: images/train
  val: images/val
  test: images/test
  names:
    0: tumour
  ```

### Stage D — Fine-tune YOLO-seg (`train.py`, run once)
```powershell
.\.venv\Scripts\Activate.ps1
python YOLO\train.py            # wraps ultralytics; reads PARAMS
```
Internally:
```python
from ultralytics import YOLO
model = YOLO("yolo11n-seg.pt")             # pretrained COCO segmentation weights
model.train(data="YOLO/dataset/data.yaml",
            epochs=PARAMS["epochs"],
            imgsz=PARAMS["imgsz"],
            batch=PARAMS["batch"])
```
Best weights are copied to **`YOLO/weights/best.pt`** (the path the web UI loads).

### Stage E — Inference & evaluation
- **Segmentation inference:** load `YOLO/weights/best.pt`, run on a slice's RGB image →
  predicted instance masks with confidence scores; keep instances above `conf_threshold`.
  A slice may contain multiple predicted masks.
- **Per-slice Dice:** merge all predicted tumour instances on a slice into one binary WT mask
  (`pred_wt`), merge expert labels (`gt_wt = seg > 0`), then compute:
  $$
  Dice(z) = \frac{2\,|pred\_wt(z) \cap gt\_wt(z)|}{|pred\_wt(z)| + |gt\_wt(z)| + \epsilon}
  $$
  with a tiny `epsilon` for numerical stability.
- **Dataset metrics:** keep Ultralytics segmentation validation metrics (`mAP@50 / mAP@50-95`).
  Dice is additionally reported slice-by-slice for clinical interpretability.

### Stage F — Web UI integration (`routes.py`, `render.py`, `Frontend/`)
A Flask blueprint `yolo_bp` (prefix `/yolo`), registered in `Frontend/app.py`, mirrors the
FCM tab:
- `GET /yolo/api/patients` — patient list.
- `GET /yolo/api/patient/<idx>` — depth, best slice, slice indices.
- `GET /yolo/segment.png?id=<idx>&z=<z>` — server-rendered matplotlib PNG of the selected
  slice with **predicted masks** and **expert GT mask** overlaid (alpha colors), including
  Dice for that selected slice.
- `GET /yolo/api/dice?id=<idx>` — returns per-slice Dice values (for all valid slices) used by
  the frontend summary table.
A new **"YOLO Detection"** sidebar button (`index.html`), a nav switch in `app.js`, a new
`yolo.js` (patient dropdown + slice slider, same controls as Explore), and matching
`style.css` rules. In the YOLO tab, the user-selected slice is shown with mask overlays and its
Dice score, and a summary table lists Dice per slice.

---

## 4. Parameters (`PARAMS` in `pipeline.py`)

| Name              | Meaning                                              | Default |
|-------------------|------------------------------------------------------|---------|
| `min_mask_area`   | drop tumour components smaller than this (px)        | `50`    |
| `min_fg_voxels`   | min foreground voxels for a slice to be kept         | `100`   |
| `val_fraction`    | fraction of **patients** held out for validation     | `0.2`   |
| `seed`            | RNG seed for the deterministic patient split         | `42`    |
| `imgsz`           | YOLO training/inference image size                   | `512`   |
| `epochs`          | fine-tuning epochs                                   | `100`   |
| `batch`           | training batch size                                  | `16`    |
| `conf_threshold`  | min confidence to display a predicted box            | `0.25`  |
| `mask_threshold`  | binarisation threshold for predicted mask logits     | `0.5`   |

---

## 5. Folder structure (`YOLO/`)

```
YOLO/
├── README.md            # this spec
├── pipeline.py          # RGB frame + label generation, dataset assembly
├── train.py             # one-shot ultralytics fine-tuning wrapper
├── render.py            # matplotlib PNG rendering with mask overlays + Dice text
├── routes.py            # Flask blueprint (/yolo), incl. per-slice Dice API
├── weights/
│   └── best.pt          # fine-tuned weights (loaded by the web UI)
└── dataset/             # generated by pipeline.py
    ├── data.yaml
    ├── images/{train,val,test}/<patient>_z<zzz>.png
    └── labels/{train,val}/<patient>_z<zzz>.txt   # YOLO-seg polygon labels (test is unlabeled)
```

---

## 6. Outputs / artifacts

- `YOLO/dataset/` — the generated YOLO detection dataset (images + normalised label txts +
  `data.yaml`).
- `YOLO/weights/best.pt` — fine-tuned segmentation model.
- Ultralytics `runs/` training logs + segmentation metrics from `model.val()`.
- Per-slice Dice outputs (e.g., in API JSON and optional CSV cache) for each patient.
- Rendered segmentation PNGs streamed to the browser on demand (predicted vs. expert masks).

---

## 7. Dependency note

Add **`ultralytics`** to `requirements.txt` (it pulls YOLO; torch is already installed via the
CUDA build described in the root `README.md`). Nothing else new is required — `numpy`,
`scipy`, `nibabel`, `Pillow`, `matplotlib`, and `Flask` are already project dependencies.


command 
.\.venv\Scripts\Activate.ps1
python YOLO\train.py --rebuild-dataset --data-fraction 0.5 --epochs 100