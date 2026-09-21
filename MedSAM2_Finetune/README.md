# MedSAM2_Finetune

Fine-tune **MedSAM2** to clean up a **YOLO** guess, on BraTS 2024 glioma MRI.

YOLO-seg produces two things per slice: a **confidence score per blob** and, underneath the
binary mask it normally shows, a **probability for every pixel**. This folder keeps the
second one and hands it to MedSAM2 as its prompt. The expert segmentation is only ever the
target of the loss — never a prompt — so the model learns to improve the hint it will
actually get at inference, not a perfect hint it will never have.

Page: <http://localhost:5000/medsam2/> (the same server as Analytics and YOLO_finetune).

---

## The chain

```
patient NIfTI ──► RGB slice (R=T1C, G=T2, B=FLAIR)  ─┐
                                                     ├─► YOLO best.pt ─► per-pixel probability
expert -seg.nii.gz ──────────────────────────────────┘                   + per-blob score
                                                                                 │
                   ┌─────────────────────────────────────────────────────────────┘
                   ▼
   prompt (logits) ──► MedSAM2 prompt encoder ──► mask decoder ──► mask logits
                                                                         │
   expert mask ──────────────────────────────► Dice + BCE loss ◄─────────┘
```

The frame builder is **imported from `YOLO_finetune.common`**, not copied. The prompting
model must see exactly the pictures it was trained on; a second copy of that code could
drift and silently feed it the wrong channels — the same trap that cost Dice before.

## Files

| file | what it does |
|---|---|
| `config.yaml` | every setting, with the reason for the non-obvious defaults |
| `common.py` | paths, the vendored `sam2` import, the checkpoint/YOLO-run pickers |
| `yolo_prompts.py` | YOLO → per-voxel probability; builds and reads `cache/<key>/<patient>.npz` |
| `dataset.py` | one slice per sample; which slices, and the flip augmentation |
| `model.py` | `SliceSAM2` (SAM2 on a single slice), what may be trained, the loss |
| `train.py` | one run: prompts → train → score. Also the CLI |
| `evaluate.py` | scores MedSAM2 **and** the YOLO prompt on the same patients |
| `routes.py` | the Flask blueprint behind `/medsam2/` |
| `templates/`, `static/` | the page. The look and the chart kit are the YOLO_finetune ones |

Generated (git-ignored): `cache/` (prompt maps, ~5 MB per patient) and `runs/`.

## Run it

From the repo root:

```bash
python -m MedSAM2_Finetune.train --smoke
```

```bash
python -m MedSAM2_Finetune.train --set train.epochs=30 --set prompt.variant=max_weighted
```

`--run <dir>` continues a folder the page made, `--resume` picks up from `last.pt`, and
`--evaluate-only` re-scores `best.pt`. Everything the page does goes through the same CLI.

## The prompt styles

| `prompt.variant` | the hint each voxel gets |
|---|---|
| `max` | the highest probability any YOLO blob gives it |
| `max_weighted` | the same, multiplied by that blob's confidence score |
| `conf_filtered` | blobs below `prompt.score_min` are dropped first |
| `binary` | the plain YOLO mask, 0 or 1 — no soft values |
| `box` | a filled rectangle per kept blob |
| `none` | nothing: MedSAM2 has to find the tumour unaided (the control) |

The cache stores the three probability maps and every blob's box and score, so switching
between these costs no rebuild. Only `prompt.yolo_run`, `prompt.yolo_conf`,
`prompt.score_min` and `data.min_fg_voxels` change the cache — the page says so before you
start.

### Two defaults that were measured, not guessed

- **`prompt.logit_scale: 8.0`** — YOLO's logits reach about ±5.5; SAM2's mask prompt was
  trained on a wider scale and barely reacts to them raw. On four val patients with the
  untrained checkpoint: scale 1 → 0.379 Dice, 2 → 0.627, 4 → 0.726, 8 → 0.757, 12 → 0.760
  (the prompt by itself scores 0.819). Above 8 it stops mattering.
- **`evaluate.use_obj_score: false`** — SAM2's "is there an object here" head, before any
  training on this data, says *no* on almost every brain slice (0 of 138 on the first val
  patient). Gating on it would report 0.0 and make the best-round choice meaningless.
  `train.obj_weight` teaches that head; turn the gate on once it has had rounds to learn.

## Scores

Everything is measured at the slice's own size, so it is comparable with the
YOLO_finetune page.

- **3D Dice per patient** — the headline, over all of that patient's brain slices.
- **The same number for the YOLO prompt alone**, on the same slices (blobs at
  `prompt.score_min`, probability cut at 0.5 — exactly Ultralytics' own mask at that
  confidence). The whole point of the folder is the difference between the two.
- Tumour-slice Dice, the old-style slice Dice (empty vs empty = 1.0, shown only so the
  older numbers can be compared), slice-level found/missed/false-alarm.
- A threshold sweep: one forward pass gives a logit map, so every cut-off in
  `evaluate.sweep` is free. **The cut-off may be chosen on val and is only ever reported on
  test.**

`runs/<id>/eval/{val,test}.json` keep the per-patient, per-slice record, so the summary can
be recomputed without touching the GPU again.

## Notes

- **Nothing in `MedSAM2/` is edited.** `model.py` subclasses and calls; it never patches.
  The checkpoints are read from `MedSAM2/checkpoints/`.
- **Nothing in Ultralytics is edited either.** `yolo_prompts.py` wraps
  `process_mask_native` for the length of its own call to keep the soft mask that function
  is about to throw away, and puts it back afterwards.
- Slices are handed out patient by patient so each cache file is read once. SAM2 normalises
  per sample (LayerNorm), so a batch from one patient is not a batch-statistics problem.
- The left-right flip is along array axis 0, which is the head's left-right axis — a flipped
  slice is still a plausible brain.
- A run is its own process. Closing the page, or restarting the viewer, does not stop it.
