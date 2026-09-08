# YOLO confidence as a MedSAM2 prompt — eleven controlled variants

`GTvsYOLO/` asks *how much of MedSAM2's quality comes from having a perfect prompt*. This
module asks the follow-up: **YOLO knows how confident it is — does telling MedSAM2 help?**

Every variant runs the identical harness. Same patient, same anchor slices, same
checkpoint, same frames, same propagation, same scoring. The only thing that changes is
one function, `variants.build_prompt` — the mask handed to MedSAM2 on an anchor slice.
A Dice difference is therefore attributable to that one change and nothing else.

## The variants

| ID | Variant | The one thing that is different | What that tests |
|---|---|---|---|
| **O** | Reference (GTvsYOLO) | Nothing. This is the ordinary run that every other row is compared against. | *If this ever stops matching the GT vs YOLO tab, the test rig has drifted - not the model.* |
| **GT** | GT ceiling | MedSAM2 is handed the doctor's own mask instead of YOLO's. | *How good can the result get when the starting mask is perfect?* |
| **A** | Low threshold 0.10 | YOLO is allowed to report weaker findings - anything it is 10% sure of, instead of only 25% and above. | *Do the extra weak findings add real tumour, or just junk?* |
| **B** | Per-slice confidence | One confidence number for the whole slice. A slice YOLO is sure about is pushed on MedSAM2 strongly; an unsure one is offered more gently. | *Does telling MedSAM2 how sure YOLO is make it segment better?* |
| **Bg** | Per-slice gate 0.50 | If YOLO is less than 50% sure about an anchor slice, that slice is skipped and MedSAM2 gets no hint there at all. | *Is a doubtful hint worse than no hint?* |
| **C1** | Per-instance confidence | Like B, but each separate lump YOLO found carries its own confidence instead of one number covering the whole slice. | *Is confidence more useful lump by lump than slice by slice?* |
| **C2** | Per-pixel probability | Every single pixel carries its own confidence - strong in the middle of the tumour, fading towards the edge. | *The finest confidence YOLO can give. Does more detail help?* |
| **D** | Soft edge (control) | The edge is softened exactly like C2, but the softness comes from the shape alone. It carries no confidence from YOLO whatsoever. | *The control for C2. If this wins too, then softness helped - not confidence.* |
| **E** | Drop blobs under 0.50 | Lumps YOLO is less than 50% sure about are thrown away before the mask is built, so MedSAM2 never sees them. | *Are the weak lumps helping, or dragging the result down?* |
| **F** | Forced 7 anchors | The mask is unchanged. The run is simply not allowed to stop early, so all 7 anchor slices are used. | *How much Dice is the early-stop rule throwing away?* |
| **G** | Forced 15 anchors | Same as F, but with 15 anchor slices instead of 7. | *Do more anchor slices matter more than a better mask?* |

These two columns are what the UI table shows, written for someone who has not read the code — the same words appear next to every result.

**D is the one that makes C2 readable.** If a graded prompt beats a hard one, C2 alone
cannot say why: it could be that YOLO's confidence carries real information, or it could
be that MedSAM2 simply prefers a soft boundary. D has the softness without the
information, so `C2 − D` isolates the part confidence actually contributes. Without it,
a C2 win is uninterpretable.

## Two things that had to be true, and are

### MedSAM2 accepts graded masks — natively

The t512 config sets `use_mask_input_as_output_without_sam: true`, so a mask prompt skips
the SAM prompt encoder and goes through `_use_mask_as_output`
(`MedSAM2/sam2/modeling/sam2_base.py`):

```python
out_scale, out_bias = 20.0, -10.0
high_res_masks = mask_inputs_float * out_scale + out_bias
```

A prompt value `p` becomes the logit `20p − 10`, and the same float is written to the
memory bank, so the gradation propagates to the non-anchor slices. No change to MedSAM2
was needed.

Two things did have to be worked around, both on our side of the call
(`medsam2_soft.register_prompts`):

1. `add_new_mask` opens with `torch.tensor(mask, dtype=torch.bool)` for a numpy input, so
   float values die immediately → pass a `torch.Tensor`.
2. When the mask is not already at `image_size`, it resizes and then applies
   `(mask_inputs >= 0.5).float()`. BraTS is 240×240 and the model wants 512, so this
   always fires → pre-resize to 512 ourselves, bilinear + antialias, matching what SAM2
   would have done minus the threshold.

**0.5 is the decision boundary, not zero.** Multiplying a mask by a confidence of 0.3
does not weaken the prompt, it *inverts* it — logit −4, "this is background". So
confidence modulates within `[0.5, 1.0]`: 0.5 means "no opinion", 1.0 means "certain",
and only true background sits at 0.0. C2 is the exception — YOLO's per-pixel probability
already uses 0.5 as its own boundary, so it maps straight through.

### YOLO's per-pixel probability is recoverable

Per-instance scores (`boxes.conf`) were always available — `YOLO/pipeline.py` computes
them and drops them. Per-pixel probability is thrown away inside Ultralytics, whose
`ops.process_mask` ends:

```python
return crop_mask(masks.gt_(0.0).byte(), bboxes)
```

Those are logits, so `sigmoid(raw)` is the probability and `sigmoid(x) > 0.5` is exactly
`x > 0` — the cut Ultralytics makes anyway. `yolo_infer.soft_masks()` swaps in a
`process_mask` that applies the sigmoid instead of the threshold, for the duration of one
call. NMS, bbox cropping and letterboxing are untouched stock code, and thresholding the
result at 0.5 reproduces the stock mask exactly.

One consequence worth knowing: masks are cropped to their own detection box, so the
probability is 0 outside the boxes no matter what. The prompt panel in the UI shows this
as a faint rectangle around the tumour — that is real, and it is what MedSAM2 receives.

## Keeping the arms honest

**The anchor schedule** comes from `GTvsYOLO.pipeline.anchor_schedule` — GT geometry only,
never a model output. It is computed once at the largest budget any variant asks for (15).
Because it is built by appending one anchor at a time, `schedule[:7]` is exactly what a
7-anchor run would have produced, so F (7 anchors) and G (15) share their first seven
slices with everything else. Verified empirically, not just by reading the code.

**Hard content is held constant where it should be.** O, B, C1, C2, D, F and G all
binarise at the model's 0.5 to the *same* mask. B, C1 and D are built from the reference
union by construction; C2 is nudged onto it by `variants._align_to`, which only touches
the thin rim where nearest- and bilinear-resampling disagree. So the difference between
them is the gradation alone, never a different set of pixels being called tumour. A, Bg
and E deliberately *do* change the hard content — that is the thing they test.

**Empty prompts cost the round.** Where a variant has nothing to prompt with on an
anchor — YOLO found nothing, or Bg gated the slice off — it gets no prompt there. No GT
fallback (it would launder the miss) and no substituting a different z (it would break the
prefix property). Missing a slice costs that variant a round, which is the measurement.

**O is verified against the reference, not assumed equal to it.** On
`BraTS-GLI-02742-100`, O and GT reproduce the cached `GTvsYOLO` masks *bit for bit*
(`np.array_equal` on the unpacked volumes, 63,543 and 64,361 voxels), at Dice 0.7360 and
0.7519 — the numbers in `GTvsYOLO/README.md`. That is the harness's self-test: if it ever
stops matching, the harness has drifted, not the model.

## Run it

```bash
.venv/Scripts/python.exe -m YoloMedSAM2InferencesTest.run_batch --patients 10
```

Resumable — every (patient, variant) result is cached the moment it finishes, so
re-running skips what is done. `--variants O,GT,C2,D` for a subset, `--force` to
recompute, `--progress` to keep SAM2's per-frame bars.

**Do the pre-flight first.** It needs no meaningful GPU time:

```bash
.venv/Scripts/python.exe -m YoloMedSAM2InferencesTest.run_batch --probe-only --patients 30
```

It runs YOLO alone on every slice and correlates its confidence against its own Dice. If
that correlation is flat, B, Bg, C1 and C2 cannot work and the expensive sweep can be
skipped. On the first test patient it is Pearson 0.67 / Spearman 0.45 — confidence does
carry signal there.

### Cost

About **3.7 minutes per patient** for all eleven variants on an RTX 5090 (measured over
five patients: 14.6 min for four, plus one at 4.3). Most of it is G — 15 forced rounds is
15 propagations, against 2–6 for an early-stopping variant. The full 324-patient test
split is therefore roughly **20 hours**. Run it in slices with `--start` and `--patients`;
the cache makes that free.

## The UI

Sidebar → **Inference tests**, four views and eleven charts. Every chart stretches to the
panel width, names both of its axes, and — where the x axis is a slice index — spans the
whole volume (0..depth-1) rather than only the slices that happen to score, so two
variants are always read against the same range.

| View | Charts |
|---|---|
| **Variants** | the table (Dice, Δ vs O, anchors, stop reason, time), then: **Dice vs anchor count** — where each variant's curve stops, and what F/G's forced tails reveal that early stopping discarded; **Dice slice by slice** with GT tumour voxels on a second axis; **Δ vs the reference, slice by slice** — a near-zero volume Dice can hide large gains and losses that cancel, and this is where they show; **predicted tumour area vs the truth** — Dice cannot say *which way* a variant is wrong, this can |
| **Slice** | ground truth, the reference, the selected variant — plus, on an anchor, a heatmap of the prompt that variant was handed |
| **Confidence probe** | **confidence and Dice per slice**; **confidence vs Dice scatter**, one point per slice; **mean Dice per confidence band**, which makes the trend legible where the raw cloud does not |
| **All patients** | **mean Dice** per variant; **mean change vs O** as diverging bars, so a loss grows leftwards; **per-patient change vs O**, patients sorted hardest-first — this separates a safe change (every point above zero) from a gamble that averages out; **mean anchors used**, which exposes variants that changed two things at once |

Charts default to showing the reference, the ceiling and the selected variant — eleven
curves over 180 slices is unreadable. The rest are one legend click away, and hiding a
series rescales the axes rather than leaving the remainder squashed.

The sweep is driven from the browser one variant at a time, so the table fills in while
the GPU works and no single request is long enough to time out. **Stop** finishes the
variant in flight (it still gets cached) and starts nothing after it.

The aggregate view only counts a patient in the per-variant means once it has finished
*all eleven*, so the columns are always compared on the same patients rather than on
whichever subset happened to complete.

## Layout

| File | Role |
|---|---|
| `variants.py` | the registry and `build_prompt` — the one function that differs |
| `yolo_infer.py` | YOLO inference keeping per-instance and per-pixel confidence |
| `medsam2_soft.py` | graded-mask registration + propagation, on top of `GTvsYOLO/medsam2_runner.py` |
| `pipeline.py` | the harness, the disk cache, the confidence probe, the aggregate |
| `render.py` | the slice figure and the prompt heatmap |
| `routes.py` | Flask blueprint at `/inftest` |
| `run_batch.py` | offline sweep, resumable |

Frames are reused from `outputs/tmp_frames/gtvsyolo/` — the same bytes the reference run
saw. Results cache to `outputs/yolomedsam2test/<patient>__<weights>/<variant>.json|.npz`.

## First results (n=10 — early, but the shape is clear)

Ten test-split patients, weights `best_20260821-013417`, every variant complete on all ten.
`W/L` counts patients where the variant beat / lost to O.

| ID | Variant | Mean Dice | Median | SD | Δ vs O | W/L | Anchors |
|---|---|---|---|---|---|---|---|
| O | Reference | 0.7822 | 0.7782 | 0.0880 | — | — | 3.6 |
| GT | GT ceiling | 0.8037 | 0.8666 | 0.1235 | +0.0215 | 9/1 | 4.0 |
| A | Low threshold 0.10 | 0.7871 | 0.7960 | 0.0855 | +0.0049 | 3/1 | 3.5 |
| B | Per-slice confidence | 0.7771 | 0.7800 | 0.1038 | −0.0051 | 5/5 | 3.7 |
| Bg | Per-slice gate 0.50 | 0.7486 | 0.7512 | 0.1185 | −0.0336 | 0/3 | 3.3 |
| C1 | Per-instance confidence | 0.7777 | 0.7800 | 0.1030 | −0.0045 | 5/5 | 3.7 |
| C2 | Per-pixel probability | 0.7866 | 0.8035 | 0.0888 | +0.0044 | 5/5 | 3.8 |
| D | Soft edge (control) | 0.7832 | 0.7822 | 0.0874 | +0.0010 | 6/4 | 3.5 |
| E | Drop blobs under 0.50 | 0.7576 | 0.7512 | 0.1121 | −0.0246 | 1/3 | 3.5 |
| F | Forced 7 anchors | 0.7988 | 0.8073 | 0.0945 | +0.0166 | **6/0** | 7.0 |
| **G** | **Forced 15 anchors** | **0.8134** | 0.8178 | 0.0841 | **+0.0312** | **7/0** | 15.0 |

**Anchor count beats prompt quality, and it is not close.** G gains +0.0312 and loses to
the reference on *no* patient; it also outscores the GT ceiling on average (0.8134 vs
0.8037) and beats it outright on 5 of 10 patients. F changes nothing except switching
early stopping off and gains +0.0166, again with zero losses. The HITL rule
(`delta < 0.005`) fires on a flat round that later rounds recover from — on
`02504-101` the curve runs 0.8589 at 2 anchors to 0.9096 at 7, all of which the stop rule
discards. **F is a free win available today: same prompts, same code, one flag.**

**Confidence as prompt amplitude does not work.** B and C1 are exactly coin flips — 5
wins, 5 losses each, and both slightly negative on the mean. This is the idea the whole
module was built to test, and at n=10 it is not paying.

**C2 is the only confidence variant still positive, and the control eats most of it.**
C2 gains +0.0044; D — identical soft boundary, zero confidence in it — gains +0.0010. So
about +0.0034 is attributable to YOLO's confidence rather than to MedSAM2 preferring a
soft edge, down from +0.0075 at n=5 and shrinking as patients are added. This is exactly
what D exists to reveal: without it, the whole +0.0044 would look like a confidence win.

**Refusing a doubtful prompt is worse than using it.** Bg is the worst variant in the set
(0/3, −0.0336) and E is second worst. On `02407-100` the second anchor is gated off at
conf 0.328 and the patient falls 0.79 → 0.60. A weak prompt still beats no prompt.

**Bg and E diverged, as predicted.** At n=5 they produced byte-identical masks on every
patient; at n=10 they differ on 3 (`02103-106`, `02184-100`, `02345-100`). They can only
separate on an anchor carrying a *mixture* of strong and weak detections — where gating
the slice and dropping the weak blob are different acts — and it took ten patients for
early stopping to reach one. On single-detection anchors the slice confidence just is that
blob's confidence, so the two are the same operation.

**The GT ceiling is not always a ceiling.** On `00518-101` the expert-mask prompt scores
0.5488 against YOLO's 0.7663 — a perfect prompt doing markedly worse. This is not a
harness artefact: the cached `GTvsYOLO` run records the same two numbers. Worth a look on
its own, since it undercuts the framing that GT bounds what any prompt can achieve.

### Suggested next step

The confidence axis (B, Bg, C1, C2, E) is not delivering, while the anchor axis (F, G) is.
But the two are confounded in the current design: a graded prompt changes *when* the stop
rule fires, so B/C1/C2 also ran more anchors on average (3.7–3.8 vs 3.6) than the
reference. Forced-anchor versions of B, C1 and C2 would separate them and settle whether
graded prompts help at all once anchor count is held fixed.
