# YOLO Pipeline — Detailed Walkthrough (Training → UI)

Here's the full pipeline end-to-end, stage by stage, based on the actual code in [pipeline.py](pipeline.py), [train.py](train.py), [render.py](render.py), and [routes.py](routes.py).

## 1. Patient discovery & split — `split_patients()`

[pipeline.py](pipeline.py#L107) reads `config.yaml` → `paths.extract_to` (`data/dataset`) and scans three folders:
- **Train pool**: `training_data1_v2` + `training_data_additional` — patients that have all 4 modalities (`-t1c`, `-t1n`, `-t2w`, `-t2f`) **and** `-seg` (ground truth).
- **Test**: `validation_data` — same 4 modalities but **no** `-seg` (official BraTS challenge holdout, unlabeled).

The train pool is sorted by patient id (deterministic), shuffled with `random.Random(seed=42)`, then split 80/20 into `train`/`val` by `val_fraction`. Optional `fraction`/`max_patients` knobs subsample this without breaking the ratio.

## 2. RGB frame construction — `build_rgb_slice()`

For each patient, each axial slice `z`, four raw NIfTI volumes (`T1C`, `T1`, `T2`, `FLAIR`) are loaded and one RGB image is built per slice:
- **R** = `clip(T1C − T1, 0, ∞)` — contrast-uptake subtraction, normalized (`_norm_slice_uint8`: clip to p0.5–p99.5 of nonzero voxels, rescale to 0–255).
- **G** = `T2`, normalized the same way.
- **B** = `FLAIR`, normalized the same way.

A slice is **skipped entirely** (no image, no label) if `FLAIR` has fewer than `min_fg_voxels` (100) nonzero voxels — i.e., essentially no brain tissue in view.

## 3. Label generation — `_polygons_from_mask()` / `_yolo_seg_lines()`

For slices that are kept and belong to a labeled split (`train`/`val`):
1. `wt = seg[:,:,z] > 0` merges BraTS labels 1–4 into one whole-tumour binary mask.
2. `scipy.ndimage.label` finds connected components; components smaller than `min_mask_area` (50 px) are dropped.
3. `skimage.measure.find_contours` extracts a polygon per surviving component.
4. Each polygon is written as a YOLO-seg line: `0 x1/W y1/H x2/W y2/H ...` (class `0` = tumour, normalized 0–1 coords).
5. Written to `dataset/labels/<split>/<patient>_z<zzz>.txt`. Slices with brain but no tumour get an **empty** file (valid negative example). `test` gets no label file at all (no ground truth exists — never fabricated).

Images go to `dataset/images/<split>/<patient>_z<zzz>.png` (`<zzz>` = 1-based slice number).

## 4. Dataset assembly — `build_dataset()`

Loops over all three splits, writes all images (+ labels where applicable), then emits `dataset/data.yaml`:
```yaml
path: YOLO/dataset
train: images/train
val: images/val
test: images/test
names:
  0: tumour
```

## 5. Training — `train.py`

```powershell
python YOLO\train.py --rebuild-dataset --data-fraction 0.01 --epochs 100
```
- Builds the dataset (if missing or `--rebuild-dataset`) via `build_dataset()`.
- Loads pretrained **`yolo11n-seg.pt`** (Ultralytics COCO segmentation checkpoint) and fine-tunes it: `model.train(data=data.yaml, epochs=..., imgsz=512, batch=16)` — standard Ultralytics training loop (box/seg/cls/dfl losses, AdamW auto-optimizer, augmentation, etc.).
- Ultralytics writes its own run artifacts to `runs/segment/train-N/weights/best.pt` (checkpoint with best validation performance across epochs).
- `train.py` then copies that file to **`YOLO/weights/best.pt`** — the single fixed path the rest of the app expects.

## 6. Per-patient inference — `YoloPipeline` (Stage E)

Used by both `render.py` and `routes.py`. For a given patient directory:
- Lazily loads the 5 NIfTI volumes (`t1c`, `t1`, `t2w`, `flair`, `seg` properties, cached).
- `slice_indices` — which `z` values pass the `min_fg_voxels` brain-foreground check.
- `best_slice_index()` — picks the slice with the largest tumour area (for a sensible default view).
- `_model_instance()` — lazily loads `YOLO/weights/best.pt` via Ultralytics; if the file doesn't exist yet, it records an error message instead of crashing (falls back to "GT only" display).
- `process_slice(z)`:
  1. Builds the same RGB frame as training (`rgb_slice`).
  2. Computes ground-truth `gt_wt = seg[:,:,z] > 0`.
  3. If the model loaded, runs `model.predict(rgb, conf=0.25)`, thresholds each predicted mask at `mask_threshold=0.5`, resizes if needed (`ndimage.zoom`) to match the GT mask's resolution, and OR-merges all instances into one `pred_wt` binary mask.
  4. Computes Dice: $Dice = \frac{2|pred \cap gt|}{|pred| + |gt| + \epsilon}$.
  5. Caches the result (`rgb`, `gt_wt`, `pred_wt`, `dice`, `model_error`).
- `dice_table()` — Dice for every valid slice of that patient (used by the summary table in the UI).

## 7. Rendering — `render.py` (Stage F, image generation)

`render(pipeline, z, slice_no)`:
- If the slice was skipped (too little brain), shows the raw FLAIR with a yellow "skipped" note.
- Otherwise, shows the RGB frame with a colored overlay: **green** = expert GT only, **red** = predicted only, **yellow** = overlap. Title shows the Dice score, and if no weights are loaded yet, appends "(no model — GT only)".
- Returns a PNG in-memory buffer via matplotlib's `Agg` backend.

## 8. Flask blueprint — `routes.py` (Stage F, API)

`yolo_bp` (prefix `/yolo`) reads patients from `data/sample` (config.yaml → `paths.sample`) — the small smoke-test set, not the full training data:
- `GET /yolo/api/patients` — list of `{id, label}` for the dropdown.
- `GET /yolo/api/patient/<idx>` — depth, valid slice indices, best slice index.
- `GET /yolo/segment.png?id=<idx>&z=<z>` — the rendered PNG (calls `render.render`).
- `GET /yolo/api/dice?id=<idx>` — per-slice Dice rows for the summary table.

Each patient gets one cached `YoloPipeline` instance (`_PIPELINES` dict) so volumes/model aren't reloaded on every request.

## 9. Frontend wiring

- `Frontend/app.py` registers `yolo_bp`.
- `index.html` has a "YOLO Detection" sidebar entry + `yoloPanel` (patient dropdown, slice slider, Best-slice button, Segmentation/Slice-summary tabs).
- `yolo.js` drives it: `initYolo()` fetches patients, `loadPatient(idx)` fetches depth/slice info, `renderSlice()` sets the `<img>` src to `/yolo/segment.png?...`, `loadDiceRows`/`renderSummaryTable` populate the Dice table.

## End-to-end flow summary

```
data/dataset (NIfTI)
  → split_patients (train/val/test by patient)
  → build_dataset (RGB PNGs + YOLO-seg polygon .txt labels + data.yaml)
  → train.py: yolo11n-seg.pt fine-tuned on it → weights/best.pt
  → YoloPipeline loads best.pt, runs inference per slice on data/sample patients
  → render.py draws GT vs predicted overlay + Dice
  → routes.py serves it as PNG/JSON
  → yolo.js/index.html displays it in the "YOLO Detection" tab
```
