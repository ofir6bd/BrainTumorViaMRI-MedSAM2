# YOLO Tumour Segmentation Pipeline — Spec (IMPLEMENTED)

> Status: **implemented.** This file describes how the YOLO segmentation model for the BraTS
> whole tumour is built/fine-tuned, and how it's wired into the existing web viewer.
> Code lives in `YOLO/` (`pipeline.py`, `render.py`, `routes.py`, `evaluate.py`, plus a
> `train.py` used to fine-tune the weights); the UI hooks into `Frontend/` (a sidebar
> **"YOLO Detection"** entry), the same pattern as the `FCMSegmentation/` and `Assymetry/`
> packages. See [PIPELINE.md](PIPELINE.md) for a step-by-step walkthrough of the actual code.
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
  - **Patient pool**: `data/dataset/training_data1_v2` + `data/dataset/training_data_additional`
    (1350 + 271 patients, all **labeled** — each has an expert `-seg.nii.gz` mask).
  - Split by patient into **train (60%) / val (20%) / test (20%)** (`val_fraction`/
    `test_fraction`, deterministic by `seed`). All three splits are **labeled**, so Dice
    can be computed on any of them. `test` is simply never trained on — it's held out
    for the post-training Dice quality summary (`evaluate_test_split` / `evaluate.py`)
    and is also the patient pool the web UI's Patient dropdown offers.
  - The old BraTS challenge `validation_data` folder (unlabeled, no `-seg`) is **not**
    read by the pipeline — labeled data alone is used for everything.
  - **Smoke test / demo** reads the single-patient `paths.sample` (`data/sample/`) when
    `inference.use_sample: true`, same toggle the other stages use (unrelated to the
    YOLO tab, which uses the labeled test split instead — see Stage F).
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
- All three splits come from the same labeled pool (`training_data1_v2` +
  `training_data_additional`), deterministic 60/20/20 by `seed`:
  - `train` (60%): writes `images/train/` + `labels/train/`.
  - `val` (20%): writes `images/val/` + `labels/val/`.
  - `test` (20%, never trained on): writes `images/test/` + `labels/test/` — it has
    ground truth too, just held out. Used for the post-training Dice quality summary
    (`evaluate_test_split` / `evaluate.py`) and as the web UI's patient list.
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
Best weights are evaluated on the held-out **test** split (`evaluate_test_split`, computing
the overall mean Dice) and saved to a uniquely named
**`YOLO/weights/best_<timestamp>_valloss<v>_testdice<v>.pt`** — never overwriting a previous
checkpoint — plus a sidecar `.json` with the same stem holding the full per-epoch
`results.csv` history, run params, and per-patient test Dice. The web UI's "Model" dropdown
picks among all saved checkpoints (best-first); its "Show training data" button reads that
sidecar. To re-run just the Dice summary on an existing checkpoint: `python YOLO\evaluate.py`.

### Stage E — Inference & evaluation
- **Segmentation inference:** load the selected `YOLO/weights/best_*.pt` checkpoint (default:
  best available, by `test_dice`/`val_loss`), run on a slice's RGB image → predicted instance
  masks with confidence scores; keep instances above `conf_threshold`. A slice may contain
  multiple predicted masks.
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
FCM tab, with patients drawn from the labeled, held-out **test** split (`list_test_patients`)
instead of `data/sample`:
- `GET /yolo/api/patients` — test-split patient list.
- `GET /yolo/api/patient/<idx>` — depth, best slice, slice indices.
- `GET /yolo/segment.png?id=<idx>&z=<z>&weights=<file>` — server-rendered matplotlib PNG of
  the selected slice with **predicted masks** and **expert GT mask** overlaid (alpha colors),
  including Dice for that selected slice.
- `GET /yolo/api/dice?id=<idx>&weights=<file>` — returns per-slice Dice values (for all valid
  slices) used by the frontend summary table.
- `GET /yolo/api/weights` — list of trained checkpoints (`val_loss`, `test_dice`), best-first,
  for the "Model" dropdown.
- `GET /yolo/api/weights/<file>/info` — that checkpoint's full training metadata (per-epoch
  history, run params, per-patient test Dice) for the "Show training data" button.
A new **"YOLO Detection"** sidebar button (`index.html`), a nav switch in `app.js`, a new
`yolo.js` (patient dropdown, model dropdown, slice slider, same controls as Explore), and
matching `style.css` rules. In the YOLO tab, the user-selected slice is shown with mask
overlays and its Dice score, a summary table lists Dice per slice, and the training-data panel
shows that checkpoint's metrics.

---

## 4. Parameters (`PARAMS` in `pipeline.py`)

| Name              | Meaning                                              | Default |
|-------------------|------------------------------------------------------|---------|
| `min_mask_area`   | drop tumour components smaller than this (px)        | `50`    |
| `min_fg_voxels`   | min foreground voxels for a slice to be kept         | `100`   |
| `val_fraction`    | fraction of **patients** held out for validation     | `0.2`   |
| `test_fraction`   | fraction of **patients** held out for the test split | `0.2`   |
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
├── pipeline.py          # RGB frame + label generation, dataset assembly, evaluate_test_split
├── train.py             # one-shot ultralytics fine-tuning wrapper + test-Dice tagging
├── evaluate.py          # standalone Dice quality summary on an existing checkpoint
├── render.py            # matplotlib PNG rendering with mask overlays + Dice text
├── routes.py            # Flask blueprint (/yolo), incl. per-slice Dice + weights API
├── weights/
│   ├── best_<timestamp>_valloss<v>_testdice<v>.pt   # one per training run
│   └── best_<timestamp>_....json                     # sidecar training metadata
└── dataset/             # generated by pipeline.py
    ├── data.yaml
    ├── images/{train,val,test}/<patient>_z<zzz>.png
    └── labels/{train,val,test}/<patient>_z<zzz>.txt   # YOLO-seg polygon labels (all 3 splits)
```

---

## 6. Outputs / artifacts

- `YOLO/dataset/` — the generated YOLO detection dataset (images + normalised label txts +
  `data.yaml`).
- `YOLO/weights/best_<timestamp>_valloss<v>_testdice<v>.pt` (+ sidecar `.json`) — one fine-tuned
  segmentation checkpoint per training run, with its held-out test Dice tagged in the filename.
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