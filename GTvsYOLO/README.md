# GT vs YOLO — MedSAM2 prompt comparison

Runs MedSAM2 **twice per patient** over the same volume and asks one question: *how much
of MedSAM2's segmentation quality comes from having a perfect prompt?*

| | Arm GT | Arm YOLO |
|---|---|---|
| Prompt mask | expert mask, `seg > 0` (labels 1–4 merged) | YOLO-seg predicted mask (`YoloPipeline.process_slice`) |
| Model | `MedSAM2_latest.pt` | same |
| Config | `sam2/configs/sam2.1_hiera_t512.yaml` | same |
| Frames | axial slices as video frames, R:T1c / G:T2w / B:T2f | same |
| Prompt type | **mask** via `add_new_mask`, one object per anchor slice | same |
| Propagation | forward from first anchor, backward from last, logits merged per frame | same |
| Anchor slices | the same fixed list | the same fixed list |
| Scored against | GT whole tumour | GT whole tumour |

Everything except the prompt source is held constant, so a Dice difference is
attributable to prompt quality alone.

## Keeping the arms comparable

The hard part is anchor selection. `05_infer_multibbox_hitl.py` picks each next anchor
with `find_largest_error_slice` — the slice where the *current prediction* is worst
against GT. That is a feedback loop from one arm's output, so letting each arm run its
own loop would give them different slices and confound the comparison; letting the GT arm
drive would place anchors exactly where the GT arm struggles, biasing in its favour.

Instead:

1. **The anchor schedule is fixed before either arm runs**, from GT *geometry* only
   (`anchor_schedule`). Only slices carrying more than `MIN_ANCHOR_TUMOUR_PX` = **100**
   GT tumour pixels are eligible, so the few-pixel wisps at the ends of the tumour never
   become anchors. Anchor 0 is the largest-GT-area slice; each next anchor bisects the
   widest un-anchored stretch of the eligible z-range, taking the eligible slice nearest
   that midpoint. Deterministic, and independent of every model output. If no slice
   clears 100 px the schedule is empty and both arms report `no_anchors`.
2. **Both arms walk that one ordered list**, adding one anchor per round, each stopping
   on its own criterion (Dice ≥ 0.95, gain < 0.005, or list exhausted). An arm's anchor
   set is therefore always a *prefix* of the same list — never a different set of slices.
   "GT converged in 2 anchors, YOLO needed 3" is a result, not a confound.
3. **Where YOLO's mask is empty on an anchor**, that arm gets no prompt there for that
   round. No GT fallback (it would launder the miss), no substituting a different z (it
   would break the prefix property). The miss costs the arm a round — the measurement.
   If YOLO is empty on *every* anchor, the arm scores 0.0 and reports "no prompts"; the
   model is not called at all in that case.

Head-to-head at equal anchor count is only valid up to `min(anchors_used)`, reported as
`comparable_rounds` and marked in the UI.

HITL settings match `05_infer_multibbox_hitl.py`: 7 anchors max, min improvement 0.005,
Dice target 0.95.

## Mask prompts, not boxes

`05_infer_multibbox_hitl.py` prompts MedSAM2 with one bounding box per connected
component. This module prompts with **the mask itself** (`add_new_mask`), which is a
materially stronger prompt and lifts both arms substantially — on the first three
test-split patients the GT arm went 0.28 → 0.75, 0.77 → 0.89, 0.84 → 0.87 when switching
from boxes to masks.

It also changes what the comparison measures. With boxes, the arms differed only in where
the rectangle sat; with masks, the YOLO arm is handed YOLO's actual segmentation, so the
question becomes "how much does MedSAM2 add on top of YOLO's mask, versus on top of a
perfect one". The GT arm's per-slice Dice on an anchor is now near 1.0 by construction —
it was given the answer for that slice — so anchor slices are not informative on their
own; the volume-level Dice and the non-anchor slices are where the signal is.

## Layout

| File | Role |
|---|---|
| `medsam2_runner.py` | predictor singleton, JPEG frame writing, mask-prompt registration, propagation — adapted from `src/05_infer_multibbox_hitl.py`, which stays untouched (it is a script: at import it pip-installs, reads `config.yaml`, mutates `sys.path`) |
| `pipeline.py` | `anchor_schedule`, both arms, per-slice Dice rows, disk cache |
| `render.py` | the 3-panel slice figure |
| `routes.py` | Flask blueprint at `/gtvsyolo` |

UI: sidebar → **GT vs YOLO**. Patients come from the held-out test split only
(`data/dataset/training_data_additional/test/`). Selecting a patient runs the comparison;
results are cached to `outputs/gtvsyolo/<patient>__<weights>__mask.json` + `.npz`
(bit-packed masks), so a ✓ in the dropdown means it will load instantly. **Re-run**
discards the cache and recomputes.

The **Model** dropdown selects the YOLO checkpoint that supplies the YOLO arm's prompt
masks. MedSAM2 itself is always `MedSAM2_latest.pt`.

Cache files are keyed by prompt mode (`..._mask.json`), so box-era results can never be
served for a mask-era run.

## Two notes on the RGB, and on absolute numbers

**The two models see different RGB, deliberately.** MedSAM2's frames are R:T1c / G:T2w /
B:T2f (exactly what `05_infer_multibbox_hitl.py` feeds it). YOLO infers on its own
R:T1C−T1 / G:T2 / B:FLAIR composite, which is what it trained on. Each model gets the
input it expects; the 3-panel view shows the MedSAM2 frame, since that is what the
segmentation being displayed was produced from.

**Absolute MedSAM2 Dice is lower here than in `outputs/results_hitl_*.csv`** (July 2026)
*when compared like-for-like on box prompts*. On `BraTS-GLI-00046-101` with anchor z=118
the reference CSV records WT Dice 0.6487; this code, run with box prompts, got 0.5427 on
the same patient. That is not a porting difference — running 05's
`_infer_one_track` verbatim against this code's frames gives 0.5427 too (vs 0.5430 here;
bf16 run-to-run noise is ~0.0003). Data is unchanged (identical files, and GT voxel count
matches the CSV exactly at 104,331), `src/05_infer_multibbox_hitl.py` has not changed
since that run, and autocast precision does not explain it (bf16 0.5427 / fp16 0.5362 /
fp32 0.5357). The remaining suspect is torch/CUDA drift — SDPA now reports no available
flash-attention kernel and falls back. **It does not affect this comparison**: both arms
share one model, one set of frames and one code path, so the contrast between them is
unaffected even if the absolute level has shifted.

## Observed runtime

~15–55 s per patient on an RTX 5090 (both arms, early stopping active), dominated by
propagation. Measured with mask prompts: 53.4 s / 35.8 s / 15.2 s on the first three
test-split patients (the spread is anchor count — an arm that converges in 2 anchors does
a fraction of the work of one that runs to 6).

First results, mask prompts, `best_20260821-013417`:

| Patient | GT mask | YOLO mask | anchors (GT / YOLO) |
|---|---|---|---|
| BraTS-GLI-02742-100 | 0.7519 | 0.7360 | 6 / 6 |
| BraTS-GLI-02407-100 | 0.8888 | 0.7900 | 6 / 3 |
| BraTS-GLI-02484-100 | 0.8677 | 0.8334 | 2 / 2 |
