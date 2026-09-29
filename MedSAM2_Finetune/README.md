# MedSAM2_Finetune

Fine-tune **MedSAM2** to fix a **YOLO** guess on BraTS 2024 glioma MRI. The stack of slices
is a video in which **every slice is prompted** with YOLO's own map for that slice, and
every slice also sees SAM2's **memory** of the slices just before it.

Page: <http://localhost:5000/medsam2/> (the same server as Analytics and YOLO_finetune).

---

## Why this design (and not the one before it)

Until 2026-09-27 this folder prompted only a few slices (the *anchors*, 1 to 8 per patient)
and reached everything else through memory, with a HITL loop adding anchors. Three real
runs of that design all scored **below YOLO alone** on test (0.849-0.863 vs 0.878; about
three patients hurt for every one helped). Slice by slice, it was below YOLO at every
distance from an anchor, including on the anchor itself. With 1 to 8 prompted slices out of ~140 it
had thrown away YOLO's answer on ~95% of the volume and could only copy it, with some loss, through memory.

Cheaper fixes were measured first on that old checkpoint (40 val patients, YOLO 0.8727):
using the cleaner `prob_f` hint (0.8581, worse), averaging the two passes (0.8604, worse),
cutting anything far from YOLO's 3D mask (0.8658, +0.002). None of them came close, so
the core was rewritten.

## The chain

```
patient NIfTI ──► RGB slice (R=T1C, G=T2, B=FLAIR) ──► YOLO best.pt ──► per-pixel probability
                                                                               │
          every slice's hint ◄────────────────────────────────────────────────┘
                 │
   start slice (most confident YOLO area): segmented from its hint alone
                 │
       ┌── pass UP:   each slice = its hint + memory of the slices below ──┐
       └── pass DOWN: each slice = its hint + memory of the slices above ──┘
                 │
     same again on the left-right mirrored patient, averaged  (evaluate.tta_flip)
                 │
     3D specks below evaluate.min_component voxels removed
                 │
   expert mask ─► Dice + BCE on every frame  (training only)
```

SAM2 supports a mask prompt on a *tracked* frame: on any frame that is not the first one, the
prompt goes to the decoder **and** the image features are memory-conditioned
(`sam2_base.py`, `_prepare_memory_conditioned_features`). So each slice is "YOLO's guess
here, corrected with what the neighbours looked like".

Training and inference call **the same function**, `model.track`, which runs the vendored
`SAM2Train.forward_tracking` with our prompts in place of the ground truth. Training feeds 8-slice clips,
and inference feeds the two half-volumes.

The frame builder is **imported from `YOLO_finetune.common`**, not copied: the prompting
model must see exactly the pictures it was trained on.

## Files

| file | what it does |
|---|---|
| `config.yaml` | every setting, with the reason for the non-obvious defaults |
| `common.py` | paths, the vendored `sam2` import, the checkpoint/YOLO-run pickers |
| `yolo_prompts.py` | YOLO → per-voxel probability; builds and reads `cache/<key>/<patient>.npz`; hint → logits |
| `anchors.py` | the start slice, and YOLO's own mask (what MedSAM2 is compared with) |
| `dataset.py` | clips with a hint on every frame, half read backwards; **hint augmentation** |
| `model.py` | `track` (shared by training and inference), what trains, the loss |
| `train.py` | one run: prompts → train → score. Also the CLI |
| `evaluate.py` | the two passes + TTA + clean-up; scores MedSAM2 **and** YOLO on the same patients |
| `routes.py` | the Flask blueprint behind `/medsam2/` |
| `templates/medsam2.html`, `static/medsam2.js` | the page, on the viewer's shared shell and kit (`Frontend/static/kit/`): live run, runs + settings diff, training charts, results, "did it help?", one patient slice by slice, a slice viewer with play, and a start dialog built from `routes.EDITABLE` |

Generated (git-ignored): `cache/` (prompt maps, ~5 MB per patient) and `runs/`.

## Run it

From the repo root:

```bash
python -m MedSAM2_Finetune.train --smoke
```

```bash
python -m MedSAM2_Finetune.train --set train.epochs=10 --set augment.p=0
```

`--run <dir>` continues a folder the page made, `--resume` picks up from `last.pt`,
`--evaluate-only` re-scores `best.pt`, and `--splits val` scores only the val pool at the
end (use it while comparing settings, so test is looked at once).

## Hint augmentation

If the hint in training is always YOLO's real answer, the cheapest thing to learn is to copy
it. So `augment.p` of the training frames get a **worse** hint on purpose: one blob dropped,
the slice blanked, the outline grown or shrunk (grey dilation/erosion), the map shifted a few
pixels, or the other confidence cut (`prob` ↔ `prob_f`). The target stays the expert mask.

## Results

All runs use YOLO `20260923-224333` for the hints. **Val = 245 patients, test = 324.**
Every setting was chosen on val; test was scored once, at the end.

| run | what changed | val 3D Dice | helped / hurt (val) |
|---|---|---|---|
| YOLO alone | — | 0.8644 | — |
| old anchor design, best (`20260925-082301`, deleted) | 8 HITL rounds | 0.8556 | 40 / 113 |
| S1 `20260927-221946` | new design, 10 rounds, hints damaged 30% | 0.8861 | 172 / 14 |
| S2 `20260928-000001` | same, no damage | 0.8859 | 162 / 9 |
| S3 `20260928-012232` | same as S1, image encoder trained too | 0.8864 | 183 / 10 |
| **final `20260928-025712`** | S3 settings, 15 rounds, 80-patient check | **0.8898** | 194 / 17 |
| final + clean-up | `min_component: 100` | 0.8903 | |

**Test (final + clean-up), scored once: MedSAM2 0.9008 vs YOLO 0.8776, +0.0232 ± 0.0016
(median +0.019). 254 patients helped, 12 hurt, 58 unchanged.** Tumour-slice Dice 0.852;
found 96.9% of tumour slices and left 96.6% of empty slices empty. The old anchor design's
best on the same test pool was 0.8605 (158 hurt).

What the numbers say:

- The redesign is the whole gain. Prompting every slice took val from 0.856 to 0.886 in
  10 rounds, and collapses are gone. The worst loss against YOLO is −0.06 on val and −0.18 on test
  (one patient; the next is −0.11). The anchor design lost up to −0.74
  (BraTS-GLI-02208-102) when the memory ran away.
- Hint damage (S1 vs S2) and training the image encoder (S3 vs S1) both **tie** on the mean
  (paired differences 0.0002 ± 0.002). Both are kept, because they cost little and S3 helped the most patients.
- Longer training (15 rounds vs 10) added about 0.003.
- The clean-up is nearly free and nearly useless: flip TTA +0.0004, removing pieces under 100
  voxels +0.0005. Cutting anything far from YOLO's 3D mask gained nothing.

## Settings that were measured, not guessed

- **`model.use_mask_input_as_output_without_sam: false`**: SAM2's own default hands a
  mask prompt straight back as the answer. With a hint on every slice, that would make the
  output *exactly* YOLO.
- **`prompt.logit_scale: 8.0`**: YOLO's logits reach about ±5.5, and SAM2's mask prompt barely
  reacts to them raw (untrained checkpoint: scale 1 → 0.379, 8 → 0.757, 12 → 0.760).
- **`train.memory_lr: 1.0e-5`**: the memory moves 10× slower than the decoder (the
  SurgSAM-2 ratio).
- **`prompt.yolo_run` is pinned** to `20260923-224333` (test 0.8776): its prompt cache
  exists for all 1,290 patients and every earlier run used it. The newest YOLO,
  `20260927-010028`, scores 0.8781, which is within noise.

## Scores

Everything is measured at the slice's own size, so it is comparable with the
YOLO_finetune page.

- **3D Dice per patient**: the headline, over all of that patient's brain slices.
- **The same number for YOLO alone** (blobs at `prompt.score_min`, probability cut at 0.5).
  The whole point of the folder is the difference between the two.
- Tumour-slice Dice, old-style slice Dice, slice-level found/missed/false-alarm, a threshold
  sweep. **Every setting is chosen on val; test is only reported.**

`runs/<id>/eval/{val,test}.json` keep the per-patient, per-slice record.

## Notes

- **Nothing in `MedSAM2/` is edited.** `model.py` builds the vendored `SAM2Train` through a
  Hydra `_target_` override and hands its `forward_tracking` our prompts.
- The untrained checkpoint scores 0 in this design: its object head says "no tumour" on
  almost every slice, and SAM2 then blanks the slice. One training round fixes that.
- A soft prompt keeps its shading only if it is handed in already at `image_size`. A mask
  SAM2 has to resize itself gets binarised.
- The left-right flip is along array axis 0, which is the head's left-right axis (the
  volumes are `('L','A','S')`), so a flipped slice is still a plausible brain.
- A run is its own process. Closing the page, or restarting the viewer, does not stop it.
