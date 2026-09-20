# YOLO_finetune

Fine-tunes **YOLO11m-seg** (COCO-pretrained) to outline the **whole tumour** on single axial
MRI slices, and gives it a page in the main viewer to start runs, follow them live and analyse
the results. Everything this folder writes stays inside it; the patient pools are only read,
from the project `config.yaml -> paths`.

| Pool | Patients | Role |
|---|---|---|
| `yolo_train` | 250 | training |
| `yolo_val` | 81 | best-epoch choice (and threshold choice) |
| `test` | 324 | final score — never trained on, never used for choosing |

The `medsam2_*` pools are never touched, so a YOLO trained here can later give MedSAM2 honest
prompts for people it has not seen.

## Run it

Start the viewer with **`Frontend\run_web.bat`**, click **YOLO_finetune** in the sidebar
(http://localhost:5000/finetune/), set the options, press **Run fine-tune**. Or from the repo
root, without the page:

```powershell
.venv\Scripts\python.exe -m YOLO_finetune.train                        # full run with config.yaml
.venv\Scripts\python.exe -m YOLO_finetune.train --smoke                # 2 patients per pool, 2 epochs
.venv\Scripts\python.exe -m YOLO_finetune.train --set train.epochs=30  # override any setting
```

The first run of a model downloads its COCO weights (`yolo11m-seg.pt`, ~45 MB) into
`pretrained/`, and Ultralytics' mixed-precision check fetches a small model into the run folder.

## What a run does

1. **Dataset** (`build_dataset.py`) — one PNG per axial slice with brain in it
   (≥ `min_fg_voxels` non-zero FLAIR voxels): **R = T1C**, **G = T2**, **B = FLAIR**, each
   scaled to its own p0.5–p99.5. Label = outline of each
   whole-tumour piece (`seg > 0`) of ≥ `min_mask_area` px; slices without tumour keep an empty
   label. Built once into `dataset/<hash>/` and reused by every run with the same data settings.
2. **Training** (`train.py`) — Ultralytics fine-tune with the settings in `config.yaml`.
   Hue/saturation jitter is off: the three channels are MRI sequences, not colours.
3. **Evaluation** (`evaluate.py`) — `best.pt` on every brain slice of **val** and **test**,
   straight from the NIfTI files, masks at the slice's own size (`retina_masks`). One pass
   scores every confidence threshold in `evaluate.sweep`.

Each run is `runs/<YYYYMMDD-HHMMSS>/`: `run_config.yaml` (its exact settings), `status.json`,
`log.txt`, `train/` (Ultralytics: `results.csv`, `weights/best.pt`, `last.pt`, plots) and
`eval/` (`val.json`, `test.json`, `summary.json`). A run is its own process — closing the page
(or restarting the viewer) does not stop it; **Stop** does, and **Resume** continues from `last.pt`.

## Metrics

- **3D Dice** (headline) — per patient, over all its brain slices; then the mean over patients.
- **Tumour-slice Dice** — mean over slices that contain tumour.
- **Slice sensitivity / specificity** — tumour slices found / tumour-free slices left empty.
- **Legacy slice Dice** — the metric behind the old `YOLO/` 0.8662: mean over *all* brain
  slices, where a tumour-free slice with no prediction counts as a perfect 1.0. It is shown only
  for comparison — a model that predicts nothing already scores ~0.5 on it.
- **Threshold** — reported at `evaluate.conf`; the sweep shows where **val** peaks. Choose the
  threshold on val, never on test.

## The page

- **Start** — model, epochs, patience, image size, batch, smoke test; the full `config.yaml`
  shown alongside.
- **Live run** — stage stepper with progress (patients / epoch + batch / patients), elapsed,
  time per epoch, time left (estimate), best epoch and epochs since best (vs patience), live log
  with filter, and a live GPU readout.
- **Runs** — every run with its state and scores; tick up to 3 to overlay their curves.
- **Training curves** — losses (train vs val), mask and box metrics, learning rate; one synced
  crosshair across all charts, best-epoch marker, and a note on where val loss bottomed out.
- **Dataset** — slices per split with / without tumour, tumour-size distribution.
- **Evaluation** (val / test) — KPIs, threshold sweep, 3D Dice histogram (click a bar to filter
  the table), Dice vs tumour size, Dice by size bucket, slice-level confusion table, sortable
  patient table, per-patient CSV export.
- **See a slice** — pick **any finished model** (any run's `best.pt`), a pool (val / test) and a
  patient, and run it now: the RGB input next to FLAIR with found / missed / false tumour pixels,
  and a confidence slider. **Run this patient** sends every brain slice through that model
  (a few seconds) and draws one chart with two scales — Dice per slice on the left (0–1), expert
  and predicted tumour pixels on the right — plus that patient's 3D Dice. Click the chart to jump
  to a slice. Results are cached per model / patient / threshold.
- **Plots** — everything Ultralytics saved (PR curves, confusion matrix, batches).

The URL keeps the view (run, split, patient, slice, threshold, compared runs) — **Copy link**.
Keys: `/` search, `←`/`→` slice, `[`/`]` patient.

## Files

| File | |
|---|---|
| `config.yaml` | the fine-tune settings (commented) |
| `common.py` | paths, pools, RGB frame, label outlines |
| `build_dataset.py` | NIfTI → YOLO-seg dataset + statistics |
| `train.py` | one run: dataset → training → evaluation; CLI (`python -m YOLO_finetune.train`) |
| `evaluate.py` | val + test scoring and threshold sweep |
| `routes.py` | the page's API — a Flask blueprint the viewer mounts at `/finetune/` |
| `templates/finetune.html`, `static/` | the page |
