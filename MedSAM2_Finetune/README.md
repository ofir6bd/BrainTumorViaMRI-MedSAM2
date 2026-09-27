# MedSAM2_Finetune

Fine-tune **MedSAM2** to clean up a **YOLO** guess, on BraTS 2024 glioma MRI — running SAM2
the way it was designed to run: **the stack of slices is a video**.

A few slices (the **anchors**) are prompted with YOLO's per-voxel probability; the rest of
the volume is reached by propagating through SAM2's memory, **forwards and backwards**. A
**HITL** loop then adds another anchor where the result disagrees most with the hint and
runs the whole thing again. The expert segmentation is only ever the target of the loss —
never a prompt.

Page: <http://localhost:5000/medsam2/> (the same server as Analytics and YOLO_finetune).

---

## The chain

```
patient NIfTI ──► RGB slice (R=T1C, G=T2, B=FLAIR) ──► YOLO best.pt ──► per-pixel probability
                                                                        + per-blob score
                                                                                │
   anchors.initial  picks the slice(s) to prompt  ◄─────────────────────────────┘
                                │
                                ▼
        anchor slice ──► MedSAM2 prompt encoder ──► mask ──► memory
                                                               │
             ┌── propagate FORWARD from the lowest anchor ──────┤
             └── propagate BACKWARD from the highest anchor ────┘
                                │
              anchors.next_anchor adds one more, and it runs again  (up to hitl.rounds)
                                │
   expert mask ──────────► Dice + BCE on every frame  (training only)
```

The frame builder is **imported from `YOLO_finetune.common`**, not copied. The prompting
model must see exactly the pictures it was trained on; a second copy of that code could
drift and silently feed it the wrong channels.

## Files

| file | what it does |
|---|---|
| `config.yaml` | every setting, with the reason for the non-obvious defaults |
| `common.py` | paths, the vendored `sam2` import, the checkpoint/YOLO-run pickers |
| `yolo_prompts.py` | YOLO → per-voxel probability; builds and reads `cache/<key>/<patient>.npz` |
| `anchors.py` | **which slices get prompted** — the first ones and the ones a HITL round adds |
| `dataset.py` | clips: the anchor plus the slices after it, half of them read backwards |
| `model.py` | `VideoSAM2` (the vendored `SAM2Train` with our prompts), what trains, the loss |
| `train.py` | one run: prompts → train → score. Also the CLI |
| `evaluate.py` | propagation + the HITL loop; scores MedSAM2 **and** the YOLO prompt |
| `routes.py` | the Flask blueprint behind `/medsam2/` |
| `templates/`, `static/` | the page. The look and the chart kit are the YOLO_finetune ones |

Generated (git-ignored): `cache/` (prompt maps, ~5 MB per patient) and `runs/`.

## Run it

From the repo root:

```bash
python -m MedSAM2_Finetune.train --smoke
```

```bash
python -m MedSAM2_Finetune.train --set train.epochs=30 --set hitl.rounds=5
```

`--run <dir>` continues a folder the page made, `--resume` picks up from `last.pt`, and
`--evaluate-only` re-scores `best.pt`. Everything the page does goes through the same CLI.

## Training vs inference

They are deliberately the same shape, at different scale.

|  | training | inference |
|---|---|---|
| unit | a **clip**: `video.num_frames` consecutive slices, **drawn again every round** | the **whole volume** |
| anchors | frame 0 of the clip | `anchors.count`, then one per HITL round |
| direction | forward; `video.reverse_fraction` of clips are fed in descending z, which is how backwards propagation is learned | forward from the lowest anchor, backward from the highest |
| prompts | YOLO's map on the anchor | the same, on every anchor |
| loss / score | Dice + BCE on **every** frame, not just the anchor | 3D Dice per patient |

The after-every-round check runs the real inference path (one HITL round), so the score
that picks the best round is the score that gets reported.

## The prompt styles

| `prompt.variant` | the hint each voxel gets |
|---|---|
| `max` | the highest probability any YOLO blob gives it |
| `max_weighted` | the same, multiplied by that blob's confidence score |
| `conf_filtered` | blobs below `prompt.score_min` are dropped first |

All three are soft maps. Three hard styles were dropped — `binary` (the same mask with the
shading thrown away), `box` (a filled rectangle per blob) and `none` (no hint at all).

Only `prompt.yolo_run`, `prompt.yolo_conf`, `prompt.score_min` and `data.min_fg_voxels`
change the cache; switching variant, anchors or HITL settings costs no rebuild.

## Anchors and HITL

`anchors.pick` chooses the first anchor: `yolo_peak` (biggest confident area — the automatic
stand-in for the slice a radiologist would start on), `yolo_score` or `spread`. Each HITL
round then adds the anchor where the prediction disagrees most with **YOLO's** mask.

**Nothing in the anchor or HITL logic ever reads the expert mask.** The old
`maunal_code/05_infer_multibbox_hitl.py` chose its anchors from the ground truth, which makes
its Dice an upper bound rather than a result; that mode is deliberately absent here, so every
score this folder produces is one the pipeline could reproduce on a new patient.

A new anchor is only ever placed on a slice where YOLO **found** something, so it is always a
positive prompt. Without that rule the loop picked slices where the prediction had drawn and
YOLO had not, prompted them with a confident "nothing", and the memory carried that erasure
into the neighbours — round 2 scored below round 1 on three of four patients.

### Three defaults that were measured, not guessed

- **`model.use_mask_input_as_output_without_sam: false`** — SAM2 short-circuits any slice that
  has a mask prompt and hands the prompt back as the answer, which is right for a mask a
  person drew and wrong for a YOLO guess. With it on the anchors were never segmented at all:
  on BraTS-GLI-00008-101 z=70 the output was the prompt pixel for pixel (99.8% identical,
  1721 px of which 1183 lay outside the head) and scored 0.33 Dice where YOLO's own mask
  scored 0.76. A checkpoint trained with it *on* cannot simply be re-scored with it off — its
  decoder has never had to segment an anchor frame (mask prompt, no memory behind it) and
  returns empty masks; measured 0.7424 -> 0.6970 on that patient. It needs a retrain.
- **`train.memory_lr: 1.0e-5`** — the memory carries the answer between slices, so it takes
  smaller steps than the decoder. Both published works that train it at all agree: SurgSAM-2
  uses 2e-4 / 2e-5 (10x), Medical SAM 2 uses 1e-4 / 1e-8 (10,000x).

- **`prompt.logit_scale: 8.0`** — YOLO's logits reach about ±5.5; SAM2's mask prompt was
  trained on a wider scale and barely reacts to them raw. On four val patients with the
  untrained checkpoint: scale 1 → 0.379 Dice, 2 → 0.627, 4 → 0.726, 8 → 0.757, 12 → 0.760
  (the prompt by itself scores 0.819). Above 8 it stops mattering.
- **`evaluate.use_obj_score: false`** — gating a slice on SAM2's object head moved 3D Dice
  by +0.0001 while costing sensitivity: the false alarms are confident, not leaks.

## Scores

Everything is measured at the slice's own size, so it is comparable with the
YOLO_finetune page.

- **3D Dice per patient** — the headline, over all of that patient's brain slices.
- **The same number for the YOLO prompt alone** (blobs at `prompt.score_min`, probability
  cut at 0.5 — exactly Ultralytics' own mask at that confidence). The whole point of the
  folder is the difference between the two.
- Tumour-slice Dice, the old-style slice Dice (empty vs empty = 1.0, for comparison with
  older numbers), slice-level found/missed/false-alarm, anchors and rounds per patient.
- A threshold sweep: one propagation gives a logit map, so every cut-off in
  `evaluate.sweep` is free. **The cut-off may be chosen on val and is only ever reported on
  test.**

`runs/<id>/eval/{val,test}.json` keep the per-patient, per-slice record including what each
HITL round did, so the summary can be recomputed without touching the GPU again.

## Notes

- **Nothing in `MedSAM2/` is edited.** `model.py` subclasses the vendored `SAM2Train` and
  replaces the prompts it would have taken from the ground truth; `evaluate.py` uses the
  vendored `SAM2VideoPredictorNPZ` as-is. Two module attributes are swapped at runtime and
  put back: Ultralytics' `process_mask_native` (to keep the soft mask it discards) and the
  predictor's `tqdm` (its per-pass progress bar would be hundreds of lines in the run log).
- A soft prompt survives `add_new_mask` only because it is handed in already at
  `image_size`: a mask that needs resizing is binarised at 0.5 on the way through.
- The video predictor writes **-1024** for "no object on this frame". That is below every
  threshold in the sweep, so such a slice reads as empty without any special case.
- The left-right flip is along array axis 0, which is the head's left-right axis (the
  volumes are `('L','A','S')`) — a flipped slice is still a plausible brain.
- A run is its own process. Closing the page, or restarting the viewer, does not stop it.
