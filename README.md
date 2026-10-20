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
Pipeline stages: acquire data (**`01`**) → analyse it in the web viewer (**`Frontend/`**, via `Frontend\run_web.bat`) → run inference (**`05`**).

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
├── Frontend/               # web viewer: Analytics + Results pages, and the shell of every page
│   ├── run_web.bat         #   launches it (localhost, live-reload)
│   ├── app.py              #   Flask backend: dashboard API + slice images
│   ├── results.py          #   the Results page's API (reads every run's saved scores)
│   ├── dataset_stats.py    #   per-patient tumour statistics + cache (also runnable)
│   ├── serve.py            #   dev server with browser live-reload (--port for a second copy)
│   ├── templates/          #   base.html (shared shell), index.html, results.html
│   └── static/             #   kit/ (kit.css, kit.js, charts.js — shared by all pages), pages/
├── YOLO_finetune/          # YOLO11m-seg fine-tuning + its page at /finetune/ (see its README.md)
├── MedSAM2_Finetune/       # MedSAM2 fine-tuned on YOLO prompts + its page at /medsam2/ (see its README.md)
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
opens `http://localhost:5000`. It reads patients from `config.yaml` → `paths.dataset` and knows
each patient's pool from the folder it sits in — if folders are moved while it runs, it notices
and re-lists them (no restart needed). All pages share one shell: the sidebar (pages, sections of
the current page, GPU readout), light / dark theme, a command palette (**Ctrl+K**: jump to a
section, open a patient or a run, run an action), keyboard help (**?**), saved views (★ Views)
and *Copy link* — every page keeps its whole view in the URL.

| Page | What it answers |
|---|---|
| **Analytics** (`/`) | What is in the dataset, and is the split clean? |
| **Results** (`/results/`) | How good is each YOLO and MedSAM2 run, and where does each one fail? |
| **YOLO_finetune** (`/finetune/`) | Start, follow and analyse YOLO11-seg fine-tunes — `YOLO_finetune/README.md` |
| **MedSAM2_Finetune** (`/medsam2/`) | Start, follow and analyse MedSAM2 fine-tunes on YOLO hints — `MedSAM2_Finetune/README.md` |

**Analytics** — every patient's real tumour statistics (WT/TC and per-label volumes, % of brain,
tumour slices, pieces, extent, location, side), computed from the `-seg` and FLAIR volumes on
first open and cached in `outputs/dashboard/dataset_stats.json` (`python Frontend\dataset_stats.py`
builds it from a terminal). Sections: split integrity (people shared between pools — a YOLO and a
MedSAM2 pool must share nobody — manifest vs folders, and every re-split so far), pool sizes and
scans per person, each pool against `test` (box + strip plots, Kolmogorov–Smirnov table), one
measure in detail (histogram or cumulative curve), **brushable mini-histograms of ten measures that
cross-filter each other**, tumour-part make-up / presence / combinations, a scatter explorer
(box-select, shift-drag zoom, colour by pool / side / pieces / cavity / scan number, trend line)
with a **Spearman correlation matrix** that sets its axes, tumour-centre heatmaps and the tumour
profile along the head, an **automatic list of unusual patients**, and the patient table (sort,
search, column picker, CSV). Every chart-made filter shows as a removable chip.
A patient opens in a **drawer**: T1C / T1 / T2 / FLAIR / T1C − T1 or all five, labels on/off,
a slice slider and **Play** (cine through the tumour slices), the tumour profile (click to jump),
parts, measurements with percentile ranks, and a timeline of the person's scans. **☆ Compare**
pins up to four patients for a side-by-side table, images and profiles.

**Results** — joins every run's saved per-patient scores (no GPU) with the dataset statistics.
Pick a pool (`test`, `medsam2_val`, `yolo_val`), a YOLO run and a MedSAM2 run (and their cut-offs).
It shows the headline with bootstrap 95% ranges and a Wilcoxon test of MedSAM2 against its own
YOLO hint, a leaderboard of every scored run, MedSAM2-vs-reference per patient, the change
histogram, cumulative score curves, cut-off sweeps, **scores split by tumour size, parts, cavity,
side, pieces or position** (with per-group tests), score against size, missed / false slices,
a **run-against-run** paired comparison, and a per-patient panel (slice-by-slice Dice of YOLO,
the hint and MedSAM2; the expert labels; and, on request, the models' own pictures of a slice).

Keys (on any page): `Ctrl+K` palette, `?` shortcuts, `T` theme, `G` next page; per page e.g.
`/` search, `Esc` close, `←`/`→` previous / next patient or slice, `[`/`]` slice or patient,
`P` play, `L` labels, `1`–`6` image, `C` compare, `R` reset, `N` new run. Label colours are the
same everywhere: NETC red, SNFH green, ET violet, RC yellow; model colours: YOLO blue, MedSAM2
orange, test green. Every chart saves as PNG / SVG / CSV from its corner.

The code: `Frontend/static/kit/` (the shared design `kit.css`, helpers `kit.js`, chart library
`charts.js`), `Frontend/templates/base.html` (the shell every page extends) and one script per page
(`Frontend/static/pages/`, `YOLO_finetune/static/finetune.js`, `MedSAM2_Finetune/static/medsam2.js`).
The site is served with **live-reload**: editing a template or script refreshes the browser.
Backend changes (`*.py`) take effect after re-running the bat. A second copy can run next to it
with `python Frontend\serve.py --port 5001`.

## Prerequisites the code expects

1. **`secrets.json`** at the repo root (needed by script 1):
   ```json
   { "synapse_auth_token": "<your Synapse token>", "synapse_dataset_id": "<Synapse dataset id>" }
   ```
2. **Extracted dataset** at `datasets/BraTS2024/{yolo_train,yolo_val,medsam2_train,medsam2_val,test}/` — produced by script 01's extract
   step, read by the viewers and inference. Set `inference.use_sample: false` in `config.yaml` to
   use it; `true` reads the single-patient `datasets/sample/` folder instead.
3. **MedSAM2 checkpoint** `MedSAM2/checkpoints/MedSAM2_latest.pt` — already present.

## Configuration (`config.yaml`)

All tunable settings live in `config.yaml`:

- `paths:` — where inputs/outputs are read/written (`raw`, `extract_to`, `dataset`, `sample`,
  `outputs`, `tmp`, `medsam2`), plus the five physical dataset pools (`yolo_train`, `yolo_val`,
  `medsam2_train`, `medsam2_val`, `test`), split by subject so no person is in both a YOLO and a
  MedSAM2 pool. `datasets/BraTS2024/split_manifest.json` records who is where.
- `inference.num_patients` — how many patients to process (`null` = all).
- `inference.use_sample` — `true` runs the one-patient sample, `false` the full dataset.

## Dataset split

The pools are split **by patient** (one person = `BraTS-GLI-XXXXX`; the `-100`, `-101`, … suffix
is a timepoint), and all scans of a person sit in the same pool — **no person is in two pools,
`test/` included**. Shares are % of scans (patient + timepoint), assigned in whole patients:

| Pool | Scans | People | % scans | Used for |
|---|---|---|---|---|
| `test` | 162 | 71 | 10% | Final test for both, never trained on |
| `yolo_val` | 165 | 72 | 10% | YOLO best-epoch choice |
| `yolo_train` | 567 | 258 | 35% | YOLO training |
| `medsam2_val` | 162 | 72 | 10% | MedSAM2 best-epoch choice |
| `medsam2_train` | 565 | 258 | 35% | MedSAM2 training (on YOLO prompts for people YOLO never saw) |

- Made by `maunal_code/02_split_by_patient.py` (seed 42; dry run by default, `--apply` moves the
  folders and rewrites `split_manifest.json`). Patients are taken in random order and each goes
  to the pool furthest below its scan target, so every pool gets the same mix of one-scan and
  many-scan patients.
- **2026-10-20 re-split:** the earlier split kept `test/` from the original per-scan split, so 213
  people had scans both in `test/` and in a train/val pool. Everything was re-split by patient
  (1223 of 1621 scans moved). **All models trained before 2026-10-20, and every Dice number from
  them, belong to the old split** — retrain YOLO, then MedSAM2 on prompts from that YOLO.
  Older manifests: `split_manifest.backup-2026-09-29.json`, `split_manifest.backup-2026-10-20.json`.
- **To change the split**, move a patient's folder — always all scans of that person together —
  then update `split_manifest.json` (the dashboard's *Manifest vs disk* check flags anything out
  of date). Moving people between a YOLO and a MedSAM2 pool means **both models must be
  retrained**, so MedSAM2 still only learns from people YOLO never saw.
