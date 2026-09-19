# BrainTumorViaMRI · BraTS 2024 · MedSAM2

## Rules

1. **Don't touch git.** No `git add` / `mv` / `commit` / `rm` — files are managed by hand.
2. **Keep it simple.** Add only what is asked for; no extra features, no cleverness.
3. **Don't use fallback or fake/random data.** Use real inputs only — no synthetic,
   placeholder, or randomly generated data, and no silent substitution of one input for another.
4. **Don't keep redundant code.** When changing the approach, remove the old parts —
   no dead code, no leftover alternatives, no duplicated logic.
5. **Don't touch/change MedSAM2 folder**
---

## What this project is

A BraTS 2024 brain-tumour segmentation project built on Meta's **MedSAM2** (medical Segment
Anything 2). MRI modalities are stacked into RGB "video" frames, ground-truth masks at a few
anchor slices are used as prompts, and SAM2 propagates the segmentation through the volume.
Pipeline stages: acquire data (**`01`**) → analyse and explore it in the web viewer (**`Frontend/`**, via `Frontend\run_web.bat`) → run inference (**`05`**).

BraTS 2024 labels: `1 = NETC`, `2 = SNFH`, `3 = ET`, `4 = RC`.
Composites: `WT` (whole tumour = 1+2+3+4), `TC` (tumour core = 1+3+4).

---

## Folder structure

```
BrainTumorViaMRI-MedSAM2/
├── README.md               # this file (the only project doc)
├── config.yaml             # pipeline parameters (paths, dataset pools, num_patients, ...)
├── secrets.json            # Synapse credentials (git-ignored)
├── .venv/                  # local virtual environment (not committed)
├── maunal_code/            # pipeline scripts
│   ├── 01_acquire_data.py
│   └── 05_infer_multibbox_hitl.py
├── Frontend/               # web viewer: Analytics dashboard + Explore
│   ├── run_web.bat         #   launches it (localhost, live-reload)
│   ├── app.py              #   Flask backend: patient list, rendered views, dashboard API
│   ├── dataset_stats.py    #   per-patient tumour statistics + cache (also runnable)
│   ├── serve.py            #   dev server with browser live-reload
│   ├── templates/index.html
│   └── static/             #   app.js (shell + Explore), dashboard.js, style.css
├── data/                   # all INPUTS (git-ignored)
│   ├── raw/                # downloaded BraTS archives
│   ├── dataset/            # patient folders in yolo_train/ yolo_val/ medsam2_train/ medsam2_val/ test/
│   └── sample/             # single-patient smoke test
├── outputs/                # all generated artifacts (git-ignored)
│   ├── dashboard/          # dataset_stats.json — cached Analytics statistics
│   ├── segs_nifti/  visualizations/  results_*.csv
│   └── tmp_frames/         # ephemeral RGB frames, auto-cleaned
└── MedSAM2/                # bundled MedSAM2 repo + checkpoints (installed editable)
```

All input/output locations are defined once in `config.yaml` (`paths:` section) and read by the
scripts — no paths are hard-coded.

---

## Environment setup

Only **Anaconda Python 3.12.7** is on this machine, so the local `.venv` is built from it.

```powershell
& "C:\ProgramData\anaconda3\python.exe" -m venv .venv     # create local env
.\.venv\Scripts\Activate.ps1                              # activate (PowerShell)
python -m pip install --upgrade pip
pip install -r requirements.txt

# GPU build of PyTorch (see GPU section) — RTX 5090 needs CUDA 12.8 wheels:
pip install torch torchvision --index-url https://download.pytorch.org/whl/cu128

# Bundled MedSAM2 (the "sam2" package), editable:
pip install -e ./MedSAM2 --no-build-isolation
```

## GPU (used every time)

This machine has an **NVIDIA GeForce RTX 5090** (32 GB, driver 595.95, CUDA 13.2). The inference
scripts use CUDA autocast (`torch.autocast(device_type="cuda")`), so the project runs on the GPU.
Installed: `torch 2.11.0+cu128`, `torchvision 0.26.0+cu128` — verified `torch.cuda.is_available()`
is `True`, device = *RTX 5090*, compute capability `sm_120`.

The RTX 5090 is Blackwell (`sm_120`) and needs the **CUDA 12.8+** wheels (`.../whl/cu128`). The
default PyPI `torch` is CPU-only and older CUDA wheels don't support `sm_120`. If you recreate the
environment, reinstall the GPU build or inference falls back to (very slow) CPU.

## How to run

Activate `.venv`, then run each script **from the repo root** (scripts read `config.yaml`,
`secrets.json`, `MedSAM2/`, and the `data/`/`outputs/` trees relative to the current directory):

```powershell
.\.venv\Scripts\Activate.ps1
```

For the interactive viewer, use `Frontend\run_web.bat` (see below) instead of a script.

## Web viewer

Double-click **`Frontend\run_web.bat`** (or run it from a terminal). It starts a local server and
opens `http://localhost:5000` in your browser. It reads patients from `config.yaml` →
`paths.dataset`, and knows each patient's pool from the folder it sits in.

- **Analytics** (landing page) — dashboard over every patient's real tumour statistics (WT/TC and
  per-label volumes, % of brain, tumour slices, connected parts, extent, location, side). They are
  computed from the `-seg` and FLAIR volumes on first open (a few minutes, with a progress bar)
  and cached in `outputs/dashboard/dataset_stats.json`; later only new or changed patients are
  recomputed, and moving a folder between pools needs no recompute
  (`python Frontend\dataset_stats.py` builds the cache from a terminal). It shows split integrity
  (subjects shared between pools — YOLO↔MedSAM2 must be 0 — and manifest vs disk), pool sizes,
  scans per subject, per-pool box plots with a KS test against `test`, a brushable histogram,
  label make-up and presence, a scatter explorer with box selection, a tumour-location heatmap,
  the mean tumour profile along the head, and a sortable patient table. Every chart filters the
  others; clicking a patient opens a drawer (slice viewer, measurements, other scans of the same
  subject, *Open in Explore*). The URL holds the whole view (*Copy link*), and *Export CSV* saves
  the filtered patients. Keys: `/` search, `Esc` close, `←`/`→` previous/next patient, `R` reset.
- **Explore** — pick a patient (searchable, grouped by pool), then switch between five views —
  **Panels**, **Modalities + Seg**, **BBox per label**, **3D scatter**, and **RGB** (the YOLO
  input frame `T1C−T1 / T2 / FLAIR` next to the raw-T1C version). The slice slider applies to
  Panels, Modalities and RGB. `#explore/<patient_id>/<z>` links straight to a patient and slice.

Label colours are the same everywhere: NETC red, SNFH green, ET violet, RC yellow.

The site is served with **live-reload**: while it's running, editing any file under `Frontend/`
(`templates/index.html`, `static/*.js`, `static/style.css`) makes the browser refresh
automatically. (Backend changes in `Frontend/app.py` or `dataset_stats.py` take effect after
re-running the bat.)

## Prerequisites the code expects

1. **`secrets.json`** at the repo root (needed by script 1):
   ```json
   { "synapse_auth_token": "<your Synapse token>", "synapse_dataset_id": "<Synapse dataset id>" }
   ```
2. **Extracted dataset** at `data/dataset/{yolo_train,yolo_val,medsam2_train,medsam2_val,test}/` — produced by script 01's extract
   step, read by the viewers and inference. Set `inference.use_sample: false` in `config.yaml` to
   use it; `true` reads the single-patient `data/sample/` folder instead.
3. **MedSAM2 checkpoint** `MedSAM2/checkpoints/MedSAM2_latest.pt` — already present.

## Configuration (`config.yaml`)

All tunable settings live in `config.yaml`:

- `paths:` — where inputs/outputs are read/written (`raw`, `extract_to`, `dataset`, `sample`,
  `outputs`, `tmp`, `medsam2`), plus the five physical dataset pools (`yolo_train`, `yolo_val`,
  `medsam2_train`, `medsam2_val`, `test`), split by subject so no person is in both a YOLO and a
  MedSAM2 pool. `data/dataset/split_manifest.json` records who is where.
- `inference.num_patients` — how many patients to process (`null` = all).
- `inference.use_sample` — `true` runs the one-patient sample, `false` the full dataset.

## Dataset split

The pools are split **by subject** (one person = `BraTS-GLI-XXXXX`; the `-100`, `-101`, … suffix
is a scan date), and all scans of a person sit in the same pool:

| Pool | Patients | People | Used for |
|---|---|---|---|
| `yolo_train` | 250 | 132 | YOLO training |
| `yolo_val` | 81 | 41 | YOLO best-epoch choice |
| `medsam2_train` | 721 | 370 | MedSAM2 training (on YOLO prompts for people YOLO never saw) |
| `medsam2_val` | 245 | 139 | MedSAM2 best-epoch choice |
| `test` | 324 | 262 | Final test for both, never trained on |

- **No person is in two pools**, except `test/` (below). This is checked live in the Analytics tab
  under **Split integrity**; every cell outside the `test` row/column must read ✓ 0.
- **2026-09-19 fix:** people who had scans in both train and val of the same model (45 in YOLO,
  138 in MedSAM2) were moved wholly to one side — 49 + 149 folders — keeping the val pools at
  their size. Otherwise each model's val score would be optimistic.
- **To change the split**, move a patient's folder — always all scans of that person together,
  never between a YOLO and a MedSAM2 pool — then update `split_manifest.json` (the dashboard's
  *Manifest vs disk* check flags anything out of date). No statistics recompute is needed.

**Why `test/` still shares people with other pools:** moving any of them would change the test
set, and every Dice number so far was measured on these exact 324 scans. The overlap was
measured and does not flatter the score (test scans whose person is also in a training pool:
Dice 0.863; the others: 0.871), so it is kept and noted here.
