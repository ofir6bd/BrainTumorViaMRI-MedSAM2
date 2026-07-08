# BrainTumorViaMRI · BraTS 2024 · MedSAM2

## Rules

1. **Don't touch git.** No `git add` / `mv` / `commit` / `rm` — files are managed by hand.
2. **Keep it simple.** Add only what is asked for; no extra features, no cleverness.
3. **Don't use fallback or fake/random data.** Use real inputs only — no synthetic,
   placeholder, or randomly generated data, and no silent substitution of one input for another.
4. **Don't keep redundant code.** When changing the approach, remove the old parts —
   no dead code, no leftover alternatives, no duplicated logic.

---

## What this project is

A BraTS 2024 brain-tumour segmentation project built on Meta's **MedSAM2** (medical Segment
Anything 2). MRI modalities are stacked into RGB "video" frames, ground-truth masks at a few
anchor slices are used as prompts, and SAM2 propagates the segmentation through the volume.
Pipeline stages follow the numeric prefixes of the scripts: **01 → 02 → 03 → 04 → 05**.

BraTS 2024 labels: `1 = NETC`, `2 = SNFH`, `3 = ET`, `4 = RC`.
Composites: `WT` (whole tumour = 1+2+3+4), `TC` (tumour core = 1+3+4).

---

## Folder structure

```
BrainTumorViaMRI-MedSAM2/
├── README.md               # this file (the only project doc)
├── requirements.txt        # Python dependencies
├── config.yaml             # pipeline parameters (paths, num_patients, ...)
├── secrets.json            # Synapse credentials (git-ignored)
├── .venv/                  # local virtual environment (not committed)
├── src/                    # all project scripts
│   ├── 01_acquire_data.py
│   ├── 02_view_slices.py
│   ├── 03_view_modalities.py
│   ├── 04_infer_baseline.py
│   └── 05_infer_multibbox_hitl.py
├── data/                   # all INPUTS (git-ignored)
│   ├── raw/                # downloaded BraTS archives
│   ├── dataset/            # extracted dataset (dataset/training_data1_v2)
│   └── sample/             # single-patient smoke test
├── outputs/                # all generated artifacts (git-ignored)
│   ├── segs_nifti/  visualizations/  results_*.csv
│   └── tmp_frames/         # ephemeral RGB frames, auto-cleaned
└── MedSAM2/                # bundled MedSAM2 repo + checkpoints (installed editable)
```

All input/output locations are defined once in `config.yaml` (`paths:` section) and read by the
scripts — no paths are hard-coded.

## What each script does

- **`src/01_acquire_data.py`** — Downloads the BraTS 2024 dataset from Synapse (token in `secrets.json`) into `data/raw/` and extracts the archives into `data/dataset/`.
- **`src/02_view_slices.py`** — Interactive viewer: 2D brain / segmentation / overlay / binary-mask panels and an optional 3D scatter of the tumour labels.
- **`src/03_view_modalities.py`** — Interactive viewer: all MRI modalities plus the segmentation, and a per-label bounding-box view on each label's peak slice.
- **`src/04_infer_baseline.py`** — Baseline MedSAM2 pipeline: 5 penta-anchor slices per label (and WT/TC), bidirectional shared-encoder propagation, mask merge, and Dice/HD95 metrics.
- **`src/05_infer_multibbox_hitl.py`** — Adds a human-in-the-loop iterative anchor-refinement loop and multi-bounding-box prompting (one box per disconnected component).

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
python src\01_acquire_data.py            # download + extract dataset
python src\02_view_slices.py             # interactive 2D/3D viewer
python src\03_view_modalities.py         # interactive modalities + bbox viewer
python src\04_infer_baseline.py          # baseline MedSAM2 inference (GPU)
python src\05_infer_multibbox_hitl.py    # HITL + multi-bbox inference (GPU)
```

## Prerequisites the code expects

1. **`secrets.json`** at the repo root (needed by script 1):
   ```json
   { "synapse_auth_token": "<your Synapse token>", "synapse_dataset_id": "<Synapse dataset id>" }
   ```
2. **Extracted dataset** at `data/dataset/training_data1_v2/` — produced by script 01's extract
   step, read by the viewers and inference. Set `inference.use_sample: false` in `config.yaml` to
   use it; `true` reads the single-patient `data/sample/` folder instead.
3. **MedSAM2 checkpoint** `MedSAM2/checkpoints/MedSAM2_latest.pt` — already present.

## Configuration (`config.yaml`)

All tunable settings live in `config.yaml`:

- `paths:` — where inputs/outputs are read/written (`raw`, `extract_to`, `dataset`, `sample`,
  `outputs`, `tmp`, `medsam2`).
- `inference.num_patients` — how many patients to process (`null` = all).
- `inference.use_sample` — `true` runs the one-patient sample, `false` the full dataset.
