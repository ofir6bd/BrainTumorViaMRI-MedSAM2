# YOLO Pipeline — Quick Summary

End-to-end flow implemented in [pipeline.py](pipeline.py), [train.py](train.py), [evaluate.py](evaluate.py), [render.py](render.py), [routes.py](routes.py).

The goal is to take raw multi-modal brain MRI (4 co-registered NIfTI volumes per patient) and produce a fine-tuned YOLO11 segmentation model that draws a **whole-tumour (WT)** mask on any single axial slice, then serve that model's predictions — side by side with the expert ground truth — through the project's Flask web UI.

1. **Patient split** — [pipeline.py](pipeline.py#L107) pools every patient with all 4 MRI modalities + expert `-seg` mask from `training_data1_v2` (1350) + `training_data_additional` (271) = **1621 patients**. Deterministic shuffle (`seed=42`), split by patient: **train 60% (≈973) / val 20% (≈324) / test 20% (≈324)**. All three splits are labeled; `test` is never trained on — only used for the post-training Dice summary and the web UI's patient list.
   - Splitting is done **per patient, not per slice** — every slice belonging to a given patient stays in exactly one split, so the model is never evaluated on a slice anatomically adjacent to one it trained on, which would otherwise leak information and inflate the reported Dice.
   - The seed makes the split fully reproducible: re-running the same code always produces the same three patient lists, so results from different training runs stay directly comparable.
   - `test` patients are the only ones ever shown in the web UI's Patient dropdown, so every Dice score a user sees in the app is always on data the model has genuinely never seen during training.

2. **RGB frame per slice** — R = `clip(T1C − T1, 0, ∞)` (contrast uptake), G = `T2`, B = `FLAIR`; each channel percentile-normalized to 0–255. Slices with < 100 foreground voxels are skipped.
   - The reasoning behind the channel choice: subtracting the pre-contrast T1 from the post-contrast T1C isolates the *enhancement* pattern (contrast agent leaking into tissue) — one of the strongest visual cues radiologists use to spot active tumour. T2 and FLAIR separately highlight oedema and the broader abnormal region, so stacking all three into one RGB image lets a single-channel-per-modality YOLO model "see" all three signals at once, the same way a normal colour photo lets a model see three colour channels at once.
   - Percentile normalisation (clipping to the 0.5th–99.5th percentile of non-zero voxels before rescaling to 0–255) stops a handful of extremely bright or dark outlier voxels from washing out the contrast in the rest of the slice.
   - Slices that are almost entirely background (skull stripped away, essentially no real brain tissue visible) are dropped from the dataset entirely — they'd otherwise just add noise with no useful learning signal.

3. **Labels from expert mask** — whole-tumour mask (`seg > 0`) → connected components (drop < 50 px) → polygon contours → YOLO-seg `.txt` lines (`0 x1/W y1/H ...`). Empty file = valid "no tumour" example.
   - BraTS ground-truth masks contain 4 distinct tumour sub-region labels (necrotic core, oedema, enhancing tumour, resection cavity). This pipeline merges all four into one binary "tumour vs. not tumour" mask, because the immediate goal is whole-tumour localisation, not sub-region classification.
   - Small stray components under 50 pixels are dropped as likely segmentation noise rather than real tumour tissue.
   - Tumour commonly appears as more than one disconnected blob on a slice (e.g. the main tumour body plus a separate satellite nodule) — each connected component becomes its own polygon/instance, so YOLO learns to output multiple masks per slice when appropriate.
   - Slices where the brain is visible but there's genuinely no tumour still get a label file — just an empty one — which teaches the model the equally important skill of correctly predicting "nothing here" instead of always guessing a mask exists.

4. **Dataset assembly** — writes `dataset/images|labels/{train,val,test}` + `data.yaml` (1 class: `tumour`).
   - This step just materialises everything from steps 1–3 onto disk in the folder layout Ultralytics' training code expects, and writes the small manifest file telling it where each split lives and what the single class is called.

5. **Training** (`train.py`) — fine-tunes pretrained `yolo11n-seg.pt` (Ultralytics) on train/val, `imgsz=512`, `batch=16`. Saves checkpoint as `weights/best_<timestamp>_valloss<v>_testdice<v>.pt` + a sidecar `.json` with per-epoch history, run params, and per-patient test Dice.
   - Starting from `yolo11n-seg.pt` (already pretrained on the general-purpose COCO dataset) rather than random weights means the model starts out already knowing generic shape/edge/texture features, so it needs far fewer epochs and far less data to specialise on tumour segmentation than training from scratch would.
   - Every run gets its own uniquely timestamped `.pt` file, so re-running training never silently overwrites a previously good checkpoint — older models stay available for comparison.
   - The sidecar JSON keeps a full audit trail per run (loss/metric per epoch, which hyperparameters were used, per-patient Dice on the test set), so the web UI's "Show training data" panel can display a run's history at any time, even long after the raw Ultralytics run folder has been cleaned up.

6. **Evaluation** (`evaluate.py` / `evaluate_test_split`) — runs the checkpoint over every held-out test patient, merges predicted instances per slice into one mask, computes $Dice = \frac{2|pred \cap gt|}{|pred|+|gt|+\epsilon}$, reports per-patient + overall mean Dice.
   - Dice measures how much the predicted mask and the expert ground-truth mask overlap, on a 0–1 scale (1 = perfect overlap, 0 = no overlap) — the standard metric for segmentation quality in medical imaging because, unlike plain pixel accuracy, it isn't dominated by the (usually huge) background region.
   - This evaluation can be re-run at any time against any saved checkpoint without retraining, which is useful for double-checking a number or comparing two checkpoints side by side.

7. **Inference & rendering** (`YoloPipeline` + `render.py`) — per patient/slice: builds the same RGB frame, runs the model, overlays GT (green) vs prediction (red) vs overlap (yellow), shows the Dice score.
   - Using the exact same RGB-construction code for live inference as was used to build the training images guarantees the model always sees data in the same format it was trained on.
   - The three-colour overlay makes it visually obvious, at a glance, where the model agrees with the expert (yellow), where it under-predicts (green only — a miss), and where it over-predicts (red only — a false positive) — far more informative than a single Dice number alone.
   - Volumes and the loaded model are cached per patient/checkpoint combination so scrubbing through slices in the UI feels responsive instead of reloading the whole MRI volume and the model from disk on every request.

8. **Web UI** (`routes.py` + `Frontend/`) — `/yolo` blueprint serves the test-split patient dropdown, model-checkpoint dropdown, segmentation PNG, per-slice Dice table, and a "Show training data" panel sourced from the checkpoint's sidecar JSON.
   - Lets a non-technical reviewer pick any held-out test patient, pick any trained checkpoint, scrub through slices, and immediately see the overlay + Dice, plus inspect that checkpoint's full training history — all without touching the command line.

## Parameters (defaults, see `PARAMS` in pipeline.py)

| Parameter | Meaning | Default |
|---|---|---|
| `min_mask_area` | A connected component of the expert whole-tumour mask must cover at least this many pixels to become a training label. Anything smaller is treated as segmentation noise/artifact rather than real tumour tissue and is silently dropped before polygon extraction — prevents the model from being trained to chase tiny, unreliable specks. | 50 |
| `min_fg_voxels` | The minimum number of non-zero (brain) voxels a slice's FLAIR image must contain to be kept at all. Slices below this threshold are almost entirely skull-stripped background/air (e.g. the very top/bottom of the volume) and are skipped entirely — no RGB image and no label file are produced for them. | 100 |
| `val_fraction` / `test_fraction` | The proportion of the total patient pool set aside for validation and for the held-out test set, respectively (the remainder — 1 minus both — becomes the training set). Splitting is done per patient so every slice of a given patient stays in exactly one split, avoiding data leakage between train/val/test. | 0.2 / 0.2 |
| `seed` | The seed for the random-number generator used to shuffle the patient list before splitting. Fixing this value makes the train/val/test assignment fully deterministic — re-running the pipeline always produces the exact same three patient lists, so results stay reproducible and comparable across runs. | 42 |
| `imgsz` | The square resolution (pixels) that every RGB slice is resized to before being fed into YOLO, both during training and at inference time. Larger values preserve more fine detail (useful for small tumour regions) at the cost of more GPU memory and slower training/inference. | 512 |
| `epochs` / `batch` | `epochs` is how many full passes over the training set are performed; `batch` is how many slice images are processed together in one forward/backward pass. More epochs generally improve fit (up to a point of diminishing returns/overfitting); larger batches make gradient estimates more stable but need more GPU memory. | 25 (current run) / 16 |
| `conf_threshold` | The minimum confidence score (0–1) a predicted tumour instance must have to be kept at inference time; lower-confidence detections are discarded as likely false positives before masks are merged and Dice is computed. | 0.25 |
| `mask_threshold` | YOLO's raw segmentation output is a per-pixel probability map for each predicted instance. This is the cutoff probability above which a pixel is counted as part of the predicted mask (i.e. binarised to 0/1) before merging instances and computing Dice against the ground truth. | 0.5 |

## Current best checkpoint

`best_20260821-013417_valloss2p9192_testdice0p8662.pt` — 25 epochs, imgsz 512, batch 16 → **val loss 2.919**, **mean test Dice 0.866** over 324 test patients.

```
data/dataset (NIfTI, 1621 patients)
  → split (train 973 / val 324 / test 324)
  → RGB slices + YOLO-seg polygon labels
  → train.py: yolo11n-seg.pt fine-tuned → evaluate_test_split on test
    → weights/best_*.pt + sidecar .json
  → YoloPipeline inference per slice → render.py GT vs prediction overlay
  → routes.py → yolo.js/index.html ("YOLO Detection" tab)
```
