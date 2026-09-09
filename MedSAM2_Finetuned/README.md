# MedSAM2_Finetuned

Finetuning MedSAM2 on BraTS, on top of the stock `MedSAM2_latest.pt` checkpoint, so that
the model is trained on **the prompts it will actually receive in production**: YOLO's
per-pixel probability masks.

This folder is **fully self-contained, in both directions**. Nothing outside it is
modified or depended on for runtime behaviour — not `MedSAM2/`, not `src/`, not
`GTvsYOLO/`, not `Frontend/` — and nothing it produces is written outside it: no
`outputs/`, no frames cache in the project tree, no checkpoints anywhere but here. The
vendored MedSAM2 repo is used strictly read-only (we import its `training/` package and
load its checkpoint). Everything else — data prep, dataset loader, collate, model
subclass, config, launcher, **inference runner**, evaluation, **and every file any of them
writes** — lives here. §1.5 is the full list of what is read and what is written.

---

## 0. What is different from the current pipeline

Three deliberate changes, all of them driven by one idea: **train the model on the same
thing it will be given at inference time.**

| | Today | Here |
|---|---|---|
| Red channel | raw T1c | **T1c − T1n** (matches YOLO) |
| Training prompt | ground-truth mask, plus 50 % boxes/points | **YOLO per-pixel probability** |
| Colour augmentation | grayscale + saturation/hue jitter | **removed** |

Plus one consequence of them, not a change in its own right: inference moves off
`GTvsYOLO/medsam2_runner.py` onto this folder's own runner, because the frames and the
prompt registration both differ now.

The last two are training changes by definition — a prompt the model is never trained on
and an augmentation that is never applied are not things you can "apply" to a finished
checkpoint. The red channel is the odd one out: it is the only change you *could* make
without training at all, simply by feeding the base checkpoint different frames. Doing so
is not expected to help, and that is the point — it moves the input distribution the
encoder was pretrained on, so it is the change that makes finetuning **necessary** rather
than optional. §2.1 has the argument, and phase 8 row 2 measures it.

### 0.1 What is fixed, and what moves

Everything not under test is held still, so that a difference in the final number has as
few possible causes as it can.

| | Setting |
|---|---|
| Objects | one, `obj_id=1`, whole tumour (`seg > 0`, BraTS labels merged) |
| Prompt | YOLO per-pixel probability, `conf 0.25`, `max` over instances, `_align_to` |
| Architecture / t512 flags | `sam2.1_hiera_t512.yaml`, unmodified |
| Anchors / scoring | GT-derived schedule, ≤ 7 anchors, HITL stop rules, volume Dice |
| MedSAM2 frames | **R : T1c − T1n** instead of raw T1c (§2.1) |
| Weights | **finetuned** instead of the stock checkpoint |

**Two variables move, not one**, and that is the thing to keep straight when reading a
result: the frames changed *and* the weights changed. A single Dice number from the final
run cannot separate them. Phase 8 splits them apart with a row for each, and row 2 — base
checkpoint, new frames, no training — is the one that says whether the channel change was
worth making at all.

Every number this folder reports is produced **inside this folder**, by its own runner, on
its own frames. Nothing is compared against a figure computed elsewhere in the project:
another module's cached results were produced by a different runner on a different frame
build, and a difference between them would never be attributable to one cause.

One thing this costs, accepted knowingly: **the sub-label structure is thrown away.**
BraTS labels 1/2/3 are merged into one binary whole-tumour object, so the model cannot
learn to separate necrotic core from edema from enhancing tumour. Whole-tumour Dice is
what the result is measured in, so whole tumour is what we train.

---

## 1. The configuration we want to use

### 1.1 Starting point

| Item | Value |
|---|---|
| Base checkpoint | `MedSAM2/checkpoints/MedSAM2_latest.pt` (read-only) |
| Architecture | SAM 2.1 Hiera **tiny** — `embed_dim 96`, `num_heads 1`, `stages [1,2,7,2]`, `global_att_blocks [5,7,9]` |
| Neck | FPN, `d_model 256`, `backbone_channel_list [768,384,192,96]`, `fpn_top_down_levels [2,3]` |
| Input resolution | **512 × 512** (slices are 182 × 218 and get upscaled) |
| Memory bank | `num_maskmem: 7` |
| Objects | **one** — `obj_id=1`, whole tumour (`seg > 0`, BraTS labels 1–4 merged) |
| Precision | AMP **bfloat16** |
| Trainer model class | `MedSAM2_Finetuned.sam2_yolo_prompt.SAM2TrainYOLOPrompt` (subclass of `training.model.sam2.SAM2Train`, see §4.4) |

The architecture block is copied verbatim from
`MedSAM2/sam2/configs/sam2.1_hiera_t512.yaml`. **This must not drift.** If it differs by
even one flag, the resulting checkpoint will not load through
`build_sam2_video_predictor(config_file="configs/sam2.1_hiera_t512.yaml", ...)`, and the
finetuned model becomes unusable. Behaviour flags to keep exactly as the t512 config has
them:

```yaml
sigmoid_scale_for_mem_enc: 20.0
sigmoid_bias_for_mem_enc: -10.0
use_mask_input_as_output_without_sam: true   # see §6 — this one has consequences
directly_add_no_mem_embed: true
no_obj_embed_spatial: true
use_high_res_features_in_sam: true
multimask_output_in_sam: true
iou_prediction_use_sigmoid: true
use_obj_ptrs_in_encoder: true
add_tpos_enc_to_obj_ptrs: true
proj_tpos_enc_in_obj_ptrs: true
use_signed_tpos_enc_to_obj_ptrs: true
only_obj_ptrs_in_the_past_for_eval: true
pred_obj_scores: true
pred_obj_scores_mlp: true
fixed_no_obj_ptr: true
multimask_output_for_tracking: true
use_multimask_token_for_obj_ptr: true
multimask_min_pt_num: 0
multimask_max_pt_num: 1
use_mlp_for_obj_ptr_proj: true
```

### 1.2 Training hyperparameters

Adapted from `sam2.1_hiera_tiny512_FLARE_RECIST.yaml` (MedSAM2's own finetune recipe),
retuned for one RTX 5090 (32 GB) on Windows.

```yaml
scratch:
  resolution: 512
  train_video_batch_size: 2      # raise to 4 if VRAM allows; watch the 32 GB ceiling
  num_train_workers: 0           # MUST be 0 on Windows (see §4.5)
  num_frames: 8                  # slices per training clip
  max_num_objects: 1             # one object: the whole tumour, obj_id=1
  base_lr: 5.0e-5
  vision_lr: 3.0e-5              # lower LR for the image encoder
  phases_per_epoch: 1
  num_epochs: 40
```

Two dataset settings that are not stock defaults and are easy to leave behind:

```yaml
dataset:
  multiplier: 4                  # §3.1 — ~50 % tumour coverage per epoch instead of ~13 %
  sampler:
    reverse_time_prob: 0.5       # phase 1 — trains the backward pass, and makes the
                                 # leading tumour-free margin reachable at all
```

| Item | Value | Why |
|---|---|---|
| Optimizer | AdamW | stock |
| LR schedule | Cosine, `base_lr → base_lr/10` | stock |
| Image-encoder LR | separate cosine on `image_encoder.*`, `3e-5 → 3e-6` | encoder needs gentler updates than the heads |
| Layer decay | `0.9` on `image_encoder.trunk`, `*pos_embed*` overridden to `1.0` | stock |
| Weight decay | `0.1`, `0.0` for `*bias*` and LayerNorm | stock |
| Gradient clip | `max_norm 0.1`, L2 | stock |
| Loss | `MultiStepMultiMasksAndIous`, `loss_mask 20 / loss_dice 1 / loss_iou 1 / loss_class 1` | stock |
| AMP | bfloat16 | matches inference-time autocast |
| Distributed backend | **`gloo`** | NCCL is Linux-only; stock says `nccl` and crashes on Windows |
| `find_unused_parameters` | `true` | stock |

### 1.3 Prompt sampling — changed

This is the substantive change. The stock recipe spends half its training on boxes and
points sampled from ground truth. We do not want that: production prompts MedSAM2 with
**YOLO masks**, so training should too.

```yaml
prob_to_use_pt_input_for_train: 0.0    # was 0.5 — never points/boxes
prob_to_use_box_input_for_train: 0.0   # was 1.0
prob_to_use_pt_input_for_eval: 0.0
prob_to_use_box_input_for_eval: 0.0
num_init_cond_frames_for_train: 2      # 1-2 prompted anchor slices per clip
rand_init_cond_frames_for_train: true
num_init_cond_frames_for_eval: 1
forward_backbone_per_frame_for_eval: true
```

With `prob_to_use_pt_input_for_train: 0.0`, SAM2Train takes the mask-input branch every
time — and our subclass makes that mask the **YOLO probability map** instead of the ground
truth (§4.4). Correction clicks are automatically disabled in this branch
(`if not use_pt_input: frames_to_add_correction_pt = []`), which is correct: production
has no human clicking corrections either.

### 1.4 Augmentation — colour augmentation removed

```yaml
- RandomHorizontalFlip           (consistent)
- RandomAffine                   degrees 25, shear 20, consistent
                                 image_interpolation: bilinear  <- the IMAGE only;
                                 masks are always NEAREST (see phase 3)
- RandomResizeAPI                512, square, consistent
- ToTensorAPI
- NormalizeAPI                   ImageNet mean/std
```

**Dropped from the stock stack: `RandomGrayscale` and both `ColorJitter` passes.**

In our data colour *is* modality — R is one MRI sequence, G another, B a third. Turning a
clip grey destroys the modality separation entirely; jittering saturation and hue
re-weights the sequences against each other, which is not a nuisance variation the model
should learn to ignore, it is the signal. Brightness/contrast jitter is dropped along with
them because it is applied per-channel and so does the same damage in milder form; the
per-slice percentile windowing in §2 already normalises intensity, which is what that
jitter was there to cover.

Geometric augmentation (flip, affine) is kept — it does not touch channel relationships.

### 1.5 Where everything is written

**Every byte this folder produces stays inside this folder.** Nothing is written to
`outputs/`, to `MedSAM2/`, to `GTvsYOLO/`, or anywhere else in the repo. The rest of the
project must be able to carry on as if this folder did not exist, and deleting this folder
must leave nothing behind.

```
MedSAM2_Finetuned/
  configs/braTS_finetune512.yaml
  *.py                             # the modules in section 5
  data/npz/{train,val}/            # phase 1: one npz per patient   (~20-40 GB, ignored)
  data/npz/skipped.json            # phase 1: patients dropped, with the reason
  work/frames/<build>/<patient>/   # phase 8: JPEGs, build = diff|raw (regenerable, ignored)
  work/prompts/<patient>.npz       # phase 8: YOLO prompts for the test split (ignored)
  exp_log/braTS_finetune/<run>/    # one dir per run: run0_gt / run1_yolo / ...
    checkpoints/checkpoint.pt      # rewritten each epoch -> Ctrl-C is safe
    checkpoints/checkpoint_<N>.pt  # every 10 epochs
    checkpoints/checkpoint_best_<meter>.pt
    tensorboard/
    logs/                          # incl. best_stats.json
  eval/<run>/<patient>.npz         # phase 8: packed predicted mask per patient (ignored)
  eval/<run>/summary.json          # phase 8: per-patient Dice + the run's row of the table
  eval/table.md                    # phase 8: the four-row comparison
```

What is read from outside, all of it **read-only**, none of it written to:

| Path | Used for |
|---|---|
| `MedSAM2/checkpoints/MedSAM2_latest.pt` | starting weights |
| `MedSAM2/sam2/configs/sam2.1_hiera_t512.yaml` | the architecture block is copied from it (§1.1) |
| `MedSAM2/training/`, `MedSAM2/sam2/` | imported as packages |
| `data/dataset/training_data_additional/` + `split_manifest.json` | the patients |
| `YOLO/` weights (`.pt`) | prompt generation (§ phase 2) |

That is the whole list. **No other module's outputs are read** — not `outputs/`, not
another folder's cached results. Everything this folder reports it computes itself.

⚠ **Relative paths are not safe here.** `train_entry.py` makes `MedSAM2/` the working
directory (phase 5), so a bare `data/npz/` in the config resolves to
`MedSAM2/data/npz/` — outside this folder, inside the vendored repo, exactly what the
self-containment rule forbids, and it fails silently by creating an empty dir rather than
erroring. **Every path in the config and in every module is absolute, resolved from
`Path(__file__).parent`**, never from `os.getcwd()`. One helper, in `paths.py`:

```python
ROOT = Path(__file__).resolve().parent          # .../MedSAM2_Finetuned
NPZ, WORK, EXP_LOG, EVAL = ROOT/"data"/"npz", ROOT/"work", ROOT/"exp_log", ROOT/"eval"
MEDSAM2 = ROOT.parent/"MedSAM2"                 # read-only
```

and the config's paths are written out absolute by `train_entry.py` before `compose()`,
not hardcoded — so the folder still works if the repo is cloned elsewhere.

The `<run>` level is not decoration: `save_dir` hangs off `experiment_log_dir`, so two runs
pointed at the same directory overwrite each other's `checkpoint.pt` and interleave their
TensorBoard scalars. `train_entry.py` takes the run name as an argument and refuses to
start if the directory already holds checkpoints, unless explicitly resuming.

```yaml
launcher:
  experiment_log_dir: <abs path to MedSAM2_Finetuned/exp_log/braTS_finetune/<run>>
checkpoint:
  save_dir: ${launcher.experiment_log_dir}/checkpoints
  save_freq: 10
  # Without this, only epochs 10/20/30/40 and the rolling checkpoint.pt survive — a best
  # epoch at 27 would be unrecoverable, and phase 7 says to take the best one. The trainer
  # already supports it (`trainer.py:115`, used at `:918`): whenever a tracked meter
  # improves it writes checkpoint_best_<meter>.pt alongside the numbered ones.
  save_best_meters:
    - val_loss
  model_weight_initializer:
    _partial_: true
    _target_: training.utils.checkpoint_utils.load_state_dict_into_model
    strict: true
    model_weight_path: <abs path to MedSAM2/checkpoints/MedSAM2_latest.pt>
```

Confirm the meter key against what the val phase actually logs before trusting it — a name
that matches nothing fails silently, saving no best checkpoint at all, and you find out at
epoch 40. `exp_log/.../logs/best_stats.json` is written by the same mechanism and is the
quickest way to see which keys are live.

`.gitignore` inside this folder covers `data/`, `work/`, `exp_log/` and `eval/*/` —
everything except `eval/table.md`, which is the result and is worth committing.

---

## 2. Input representation

MedSAM2 is a video model: one patient = one video, each axial slice = one frame. The MRI
sequences are packed into RGB.

| Channel | Content |
|---|---|
| **R** | **T1c − T1n** (post-contrast minus pre-contrast, negatives clipped) |
| **G** | T2w |
| **B** | T2f (FLAIR) |

Each channel is windowed **per slice** to its own 0.5–99.5 percentile range, then scaled
to uint8:

```python
def norm_slice_uint8(slice_2d, min_fg_voxels=100):
    s = slice_2d.astype(np.float32)
    nonzero = s[s != 0]
    if nonzero.size < min_fg_voxels:
        return np.zeros(s.shape, np.uint8)          # near-empty slice -> black
    lo = float(np.percentile(nonzero, 0.5))
    hi = float(np.percentile(nonzero, 99.5))
    if hi - lo < 1e-8:
        return np.zeros(s.shape, np.uint8)
    return np.clip((s - lo) / (hi - lo) * 255, 0, 255).astype(np.uint8)

# build="diff" — what we train and infer on
def build_rgb_slice(t1c, t1n, t2w, t2f):
    diff = np.clip(t1c.astype(np.float32) - t1n.astype(np.float32), 0, None)
    return np.stack([norm_slice_uint8(diff),
                     norm_slice_uint8(t2w),
                     norm_slice_uint8(t2f)], axis=-1)

# build="raw" — today's frames, kept only for row 1 of the phase-8 table
def build_rgb_slice_raw(t1c, t2w, t2f):
    return np.stack([norm_slice_uint8(t1c),
                     norm_slice_uint8(t2w),
                     norm_slice_uint8(t2f)], axis=-1)

JPEG_QUALITY = 95        # the single constant both paths use

def jpeg_roundtrip(rgb):
    """Encode and decode exactly as inference does, so training sees the same pixels."""
    buf = io.BytesIO()
    Image.fromarray(rgb, mode="RGB").save(buf, format="JPEG", quality=JPEG_QUALITY)
    buf.seek(0)
    return np.asarray(Image.open(buf).convert("RGB"), dtype=np.uint8)
```

**The negative clip.** `T1c − T1n` goes negative where the pre-contrast scan is brighter.
Those voxels are clipped to 0 before normalisation, matching `build_rgb_slice` in
[YOLO/pipeline.py:328](../YOLO/pipeline.py#L328). Enhancement is a *positive* difference;
negatives are noise and misregistration. A signed difference is a different experiment.

**Each channel is normalised on its own scale, after the subtraction.** That is why the
red channel is not dark despite being a difference of two similar images: if the diff
spans 0–40 while T2w spans 0–900, red's own 0.5–99.5 window is ~0–40 and gets stretched
across the full 0–255. The percentiles are taken over **non-zero** voxels only, so once
negatives are clipped away the window is measured on real enhancement rather than on
background. The cost is that red no longer encodes *how much* uptake there is, only where
it is relative to the rest of that slice — a slice with no enhancement gets its noise
stretched to full range too.

**Two builders, one of them frozen.** `build_rgb_slice` is what this folder trains and
infers on. `build_rgb_slice_raw` reproduces today's composite (raw T1c) and exists only for
row 1 of the phase-8 table, the baseline the finetune has to beat; nothing trains on it and
it must never change — the moment it does, row 1 stops being today's number. `frames.py`
holds both, and
*both* the training data prep (§4.1) and this folder's inference runner (§4.7) import
them — train/inference frame mismatch is the same class of bug as the BGR one (silent
Dice loss, no error message), so there is exactly one implementation of each.

⚠ **The frame cache is keyed by build.** Frames live at
`work/frames/<build>/<patient>/`, `build ∈ {diff, raw}`. The plumbing this is
copied from reuses an existing directory when the **file count** matches — it never looks
at the pixels — so a single shared dir would silently serve `raw` frames to a `diff` run
and produce a plausible, wrong number. Put the build in the path, and delete the dir when
in doubt; regenerating is cheap.

`GTvsYOLO/medsam2_runner.py` is left alone and keeps building `R = raw T1c`. That is
correct for the base checkpoint it serves. **Do not run the finetuned checkpoint through
it** — it would load fine and hand the model the wrong red channel, with no error.

### 2.1 Why the red channel changes — and the check that says whether it helped

YOLO's own input is `R = clip(T1c − T1n, 0, None)`
([YOLO/pipeline.py:328](../YOLO/pipeline.py#L328)); MedSAM2's, today, is raw T1c. The two
models look at different pictures of the same slice, which is a real confound in the
YOLO→MedSAM2 pipeline. Matching them removes it, and it is the reason this folder
finetunes rather than just swapping the input:

**The base checkpoint has no idea what the red channel means.** It was never trained on a
composite where R, G and B carry three different sequences — the loader MedSAM2's own
recipe ships (`NPZRawDataset`) takes grayscale `(N,H,W)` and replicates one channel three
times. So today's R:T1c / G:T2w / B:T2f is *already* off-distribution for it, and moving
red somewhere else off-distribution buys nothing on its own. Training is what turns the
new channel into signal — the same way YOLO learned it, by being trained on it.

**Two reasons it might not pay off, worth knowing before spending the GPU hours.** The
subtraction earns its keep for a *detector*: raw T1c confuses enhancement with anything
natively bright on T1 (fat, blood products, calcification), and differencing isolates
genuine uptake. MedSAM2 is not detecting here — it takes a mask prompt and propagates it
across slices, so what it needs is clean boundaries and slice-to-slice continuity. A
difference image has lower SNR than either input, and T1c/T1n misregistration shows up as
a bright rim exactly at tissue edges, which is where the propagator is looking. On top of
that, `use_mask_input_as_output_without_sam: true` means the prompt bypasses the encoder
on anchor slices entirely (§6) — the frames only matter on the *propagated* slices, which
is precisely where the noise cost lands and the specificity benefit does not.

**The pre-flight that settles it, before any training.** Run the base checkpoint with the
YOLO prompts on the same patients, twice: once on `raw` frames, once on `diff` frames. Two
inference passes, no training. That is row 1 vs row 2 of the phase-8 table.

- `diff` already wins frozen → the channel change is doing real work and finetuning will
  very likely widen the gap.
- `diff` is roughly level → expected; the model has not learned the channel yet. Proceed,
  and row 3 tells you whether training closed it.
- `diff` is clearly *worse* frozen → not fatal, but read it as a warning that the encoder
  has to travel further, and check run #1's val loss early rather than at epoch 40.

### 2.2 The JPEG round-trip — training must see what inference sees

Inference cannot avoid JPEG. `init_state` reads a directory of image files, and the loader
opens them with PIL (`sam2/utils/misc.py:93`), so the runner writes `.jpg` at quality 95 —
the same thing `GTvsYOLO/medsam2_runner.write_frames` does.

JPEG is lossy: every pixel comes back slightly changed. So if `prep_npz` stored the exact
arrays, the model would learn on clean pixels and be judged on smudged ones. The difference
is small at q95 — but "train on what you will actually be given" is the entire premise of
this folder, and this is the same class of gap as the red channel, only quieter.

**So `prep_npz` stores `imgs` already round-tripped**, via `frames.jpeg_roundtrip`. Encoding
to a `BytesIO` and decoding with PIL is byte-for-byte what writing a `.jpg` and reopening it
produces, so no files are needed at prep time.

Three things to keep straight:

- **`JPEG_QUALITY` is one constant, used by both paths.** Two literals will drift, and the
  drift is invisible — no error, just a slightly worse number.
- **Only `imgs` is round-tripped.** `gts` and `yolo_prob` are never JPEGs at inference
  either: the prompt is handed to `add_new_mask` as an array, and the GT never leaves
  Python. Compressing them would quantise the probability field for no reason.
- **This costs nothing at eval time** and changes phase 8 not at all — it only makes the
  training input match the inference input.

---

## 3. The data

`data/dataset/training_data_additional/`, split by `split_manifest.json` (seed 42,
60/20/20):

| Split | Patients | Use here |
|---|---|---|
| `train/` | **973** | training |
| `val/` | **324** | validation each epoch |
| `test/` | 324 | **untouched** — final eval only |

Per patient: `-t1c`, `-t1n`, `-t2w`, `-t2f`, `-seg` NIfTI volumes, all
**182 × 218 × 182** (axial slices are 182 × 218; z is the third axis). The `-seg` labels
(1 = necrotic core, 2 = peritumoral edema, 3 = enhancing tumour) are **merged into one
whole-tumour mask**, `seg > 0`, carried as the single object `obj_id=1`. Hence
`max_num_objects: 1`.

The sub-labels are discarded, not modelled. The YOLO prompt is a whole-tumour probability
map with no class structure in it, and the Dice that decides whether this finetune worked
is whole-tumour Dice; training three objects the prompt cannot distinguish would optimise
something nothing here measures. Splitting the labels back out is a separate experiment,
and it needs a per-class prompt to be worth anything.

### 3.1 How many slices per patient actually reach the model

Measured over 40 training patients:

| Quantity | Value |
|---|---|
| Tumour z-extent per patient | median **62** slices (min 31, p25 51, p75 74, max 113) |
| Patients with fewer than `num_frames` (8) tumour slices | **0** — nothing gets skipped |
| Stored in the npz | tumour z-range + 12 tumour-free slices each side, so ~86 slices/patient |
| **Fed to the model per epoch** | **8 contiguous slices per patient** |
| Of those 8, prompted (conditioning) | 1–2 |
| Of those 8, actually learned from | 6–7 |

`RandomUniformSampler` picks `start = random.randrange(0, N - 8 + 1)` and takes 8
**contiguous** frames. With `multiplier: 1`, `VOSDataset` yields one clip per patient per
epoch. So per epoch the model sees 973 × 8 = **7,784 slices**, and each patient contributes
one random 8-slice window out of its ~62 — about **13 % of that patient's tumour per
epoch**. Over 40 epochs a patient gets 40 windows ≈ 320 slice-samples, so each of its
slices is seen roughly 5 times on average.

Clip *starts* are confined to tumour slices — the sampler rejects a window whose first
frame is empty (phase 1) — so the ~62 tumour slices are what the coverage figure is
measured against; the 24 margin slices are reached by propagating off either end.

**Raise `multiplier` to 4** (set in §1.2). Four windows per patient per epoch is ~50 %
coverage instead of 13 %, and clip starts are independent so they spread across the whole
tumour rather than clustering. Cost is linear — roughly 8 min/epoch instead of 2, so
40 epochs goes from ~1.5 h to ~5 h. On a 32 GB card with a tiny model that is the cheapest
accuracy available. Set it in `dataset.multiplier`, not by lengthening `num_frames`:
longer clips raise VRAM per step, more clips do not.

Do **not** raise `num_frames` past 8 without checking VRAM. It multiplies activation memory
directly, and the propagation length it buys matters less than coverage does.

### 3.2 ⚠ YOLO trained on these same patients

`YOLO/dataset/data.yaml` points at `YOLO/dataset/images/{train,val,test}`, built from
**this same `training_data_additional` split**. So on our 973 training patients, YOLO is
predicting on data it memorised: its masks there will be noticeably better than the ones
it produces on unseen patients.

Left unaddressed, MedSAM2 learns *"the prompt is nearly always right, just follow it"* —
and then meets weaker prompts at test time. The model would be tuned for a quality of
input it will never actually get.

Handle it in this order:

1. **Measure the gap first (cheap, do this before anything else).** Run YOLO over a sample
   of train patients and a sample of val patients and compare mean Dice. If the gap is
   small, the leak is not doing much harm and you can proceed as planned while noting it.
   If train Dice is far higher, the remaining options matter.
2. **Prompt degradation on train patients** (recommended if the gap is large). Randomly
   perturb YOLO prompts during training only — dilate/erode by 1–3 px, drop a whole
   instance with small probability, scale the probability field toward 0.5. Cheap, and it
   directly teaches the model to survive an imperfect prompt.
3. **Train MedSAM2 on the val split's YOLO output** (324 patients YOLO never saw), holding
   test back. Clean, but a third of the data.
4. **K-fold YOLO** so every train patient gets an out-of-fold prediction. Correct, and the
   most expensive.

Whatever you pick, write it in the run notes. A Dice number from this pipeline is not
interpretable without knowing which of these was used.

---

## 4. Finetuning plan

Eight phases. Each is verifiable on its own — do not start a 40-epoch run before phase 5
has produced one clean epoch.

### Phase 1 — Convert patients to NPZ clips

`MedSAM2_Finetuned/prep_npz.py`.

Read each patient, build the RGB volume per §2, pass each slice through
`frames.jpeg_roundtrip` (§2.2), write one `.npz` per patient into `data/npz/{train,val}/`:

| Key | Shape / dtype | Meaning |
|---|---|---|
| `imgs` | uint8 `(N,H,W,3)` | RGB frames, **JPEG-round-tripped** (§2.2) |
| `gts` | uint8 `(N,H,W)` | whole-tumour mask, `seg > 0` as 0/1 — the single object `obj_id=1`, **loss target** |
| `yolo_prob` | uint8 `(N,H,W)` | YOLO per-pixel probability × 255 — **prompt** (phase 2) |

Keep the contiguous tumour z-range **plus a margin of `MARGIN = 12` tumour-free slices at
each end**, clamped to the volume. `RandomUniformSampler` requires at least `num_frames`
contiguous frames *and* a tumour on the first sampled frame — patients that cannot satisfy
that are skipped and **logged to `data/npz/skipped.json`** with the reason, never silently
dropped.

#### Why the margin — the model has to learn where the tumour *stops*

Without it, every stored frame contains tumour (median z-extent is 62 slices, so the
`N >= num_frames` expansion essentially never fires), and every 8-frame clip is drawn from
inside the tumour. The model would never once be shown a slice whose correct answer is
"nothing here" — no gradient through `pred_obj_scores`, `no_obj_embed_spatial` or
`fixed_no_obj_ptr`, ever.

Phase 8 then propagates the **whole volume**: forward from the first anchor and backward
from the last, across ~180 slices of which only ~62 have tumour. Most frames the runner
touches would be a kind it had never seen in training, and the failure mode is the obvious
one — the mask keeps going past the tumour edge, because nothing ever taught it to stop.

The margin is only reachable because of how the sampler works, and this is worth being
precise about (`vos_sampler.py:42-70`):

- `start = random.randrange(0, N - num_frames + 1)`, then the clip is **rejected and
  resampled** (up to `MAX_RETRIES = 1000`) unless `frames[0]` contains a visible object.
- So clip *starts* stay on tumour slices, but a clip starting near the tail runs forward
  into the trailing margin. Those tumour-free frames arrive as **propagated** frames,
  never as conditioning frames — exactly their role at inference.
- ⚠ The reversal happens **before** that check, so with `reverse_time_prob: 0.0` the
  *leading* margin is unreachable: a window that starts in it is always rejected. Set
  **`reverse_time_prob: 0.5`** on the sampler. It makes the leading margin reachable, and
  independently it is the right setting anyway — the runner propagates backward from the
  last anchor, and at 0.0 the model is never trained on a backward pass at all.

`MARGIN = 12` is chosen to exceed `num_frames` (8), so a clip can sit entirely in the
transition rather than only clipping its edge. Cost is ~40 % more stored slices
(~62 → ~86 per patient); the budget below already accounts for it.

Store `yolo_prob` quantised to uint8: float32 would roughly triple the folder size, and
1/255 resolution is far finer than the probability field is meaningful to.

Merge the labels at prep time (`gts = (seg > 0).astype(np.uint8)`) rather than in the
dataset, so there is one place where whole-tumour is defined and the stored file is
already what the model trains on.

Checks: file count matches 973/324 minus the skip log; one npz opened by hand shows
`imgs.shape == (N,H,W,3)` with three visibly *different* channels (not a grey image);
`set(np.unique(gts)) ⊆ {0, 1}` — if a 2 or a 3 survives, the merge did not happen and the
dataset will hand SAM2 three objects; and **`gts[:MARGIN].sum() == 0` and
`gts[-MARGIN:].sum() == 0`** except where the margin was clamped at the volume edge — if
either end has tumour, the margin was not applied and the model is back to never seeing an
empty slice. Budget ~20–40 GB and 1–2 h. Use `np.savez_compressed`.

One more, for §2.2: write one slice to a real `.jpg` with the runner's own call, read it
back, and assert it equals the stored `imgs[z]` exactly. That is the only check that
catches a quality or encoder mismatch between prep and inference — the images look the
same either way.

### Phase 2 — Generate the YOLO prompts

`MedSAM2_Finetuned/yolo_prompts.py`, run as part of phase 1.

Reuse the technique from
[YoloMedSAM2InferencesTest/yolo_infer.py](../YoloMedSAM2InferencesTest/yolo_infer.py) —
copied into this folder, not imported, so the folder stays self-contained. Ultralytics
normally thresholds mask logits at 0.5 and throws the probabilities away; `soft_masks()`
patches `ops.process_mask` to apply a sigmoid instead and keep the crop, so pixels outside
a detection's own box stay 0.

Per slice:

```python
out = np.zeros(shape, np.float32)
for inst in instances:                 # conf >= 0.25
    np.maximum(out, inst["prob"], out=out)   # overlaps take the higher probability
prompt = _align_to(out, union)         # force the 0.5-crossing onto the binary union
```

`_align_to` nudges only the pixels where nearest- and bilinear-resampling disagree, so the
probability field binarises to exactly the same pixel set as the ordinary run. Without it
the graded prompt would be a *different mask*, not a graded version of the same one.

**Do not rescale the probability into [0.5, 1.0].** A *confidence* score would have to be —
a raw 0.3 would read to the model as "background" (§6). A per-pixel probability does not:
it already uses 0.5 as its own decision boundary, so it maps straight through unchanged.

⚠ **The BGR trap.** Ultralytics reads numpy arrays as **BGR**. Handing it our RGB frames
directly silently destroys accuracy — this bit the web UI once already. Convert before
every call.

Cost: ~1300 patients × ~80 slices ≈ 100k YOLO inferences. Budget ~30–60 min. Cache to disk
— you do not want to regenerate these per epoch.

### Phase 3 — An RGB-capable raw dataset

`MedSAM2_Finetuned/npz_rgb_raw_dataset.py`.

Stock `training.dataset.vos_raw_dataset.NPZRawDataset` assumes **grayscale** `imgs`
`(N,H,W)` and replicates one channel three times — it physically cannot carry three
modalities. Subclass `VOSRawDataset`, copy the stock file-walk and `file_list_txt`
handling, read `imgs` as real `(N,H,W,3)`, and additionally load `yolo_prob`.

**One object, plus its prompt.** `gts` is already the merged whole tumour (phase 1), so
the object list is exactly two entries per frame: object `0` is the GT whole-tumour mask —
the one that becomes SAM2's `obj_id=1` and the loss target — and object `1` carries the
YOLO probability for it. `max_num_objects: 1` counts *trained* objects, so the prompt
entry must be split out by the collate (phase 4a) before the sampler counts objects, not
left in the list where it would read as a second tumour.

Appending the prompt to the object list is what guarantees it receives byte-identical
geometric transforms to its GT — same flip, same affine, same resize — so the two can
never drift out of alignment.

**What each transform does to a mask, precisely.** The two steps do *not* behave the same
way, and it matters:

| Step | Interpolation on masks | Effect on a probability field |
|---|---|---|
| `RandomAffine` | `InterpolationMode.NEAREST` (`transforms.py:420`) | values preserved exactly, no blending |
| `RandomResizeAPI` → `resize()` | **bilinear** — `F.resize(obj.segment[None, None], size)` with the default (`transforms.py:97`) | values *are* blended at edges |

So the prompt's gradation survives the rotate/shear untouched but is smoothed by the
182×218 → 512 resize. That is fine — bilinear is the right thing for a probability field,
and it is what `medsam2_soft._prompt_tensor` does at inference too, so the two paths agree.
What is *not* fine is believing the masks are never blended: they are, on every sample.

Referenced from the config as
`_target_: MedSAM2_Finetuned.npz_rgb_raw_dataset.NPZRGBRawDataset` — an import path, so
nothing is dropped into `MedSAM2/training/dataset/`.

Check: instantiate standalone, pull video 0, assert three-channel frames, assert channel
0 ≠ channel 1, assert exactly one GT object and one prompt object, and assert the prompt
object holds values strictly between 0 and 1.

Do **not** assert the GT is still `{0, 1}` here — after the resize it legitimately carries
a blended rim (phase 3 table above), and it only becomes binary in the collate. `{0, 1}`
at this point would mean the resize did not run.

### Phase 4 — Keep the probabilities alive

Three places binarise a soft mask on the way in. All three must be handled, and none of
them raises an error — they just quietly turn your graded prompt into a plain 0/1 mask, and
you spend a week wondering why the per-pixel probability made no difference at all.

**4a. The collate function — the one that actually kills it.**
`MedSAM2/training/utils/data_utils.py:149` does `obj.segment.to(torch.bool)`, and
`BatchedVideoDatapoint.masks` is annotated `torch.BoolTensor`. Every probability dies right
there.

Fix: `MedSAM2_Finetuned/collate.py` — our own collate, pointed at from the config
(`collate_fn._target_` is already a config field, so this needs no MedSAM2 edit). It keeps
GT objects `bool` and prompt objects `float32`, and returns a small dataclass subclass of
`BatchedVideoDatapoint` with an extra `prompt_masks` field. Splitting them into separate
fields keeps the loss target unambiguously binary.

⚠ **`to(torch.bool)` on a bilinearly-resized GT dilates the loss target.** The two halves
of this are individually harmless and only bite together: `resize()` blends the GT edge
(phase 3), producing a rim of small non-zero values where the mask used to end — and
`to(torch.bool)` maps *any* non-zero to `True`. The ground truth the loss is computed
against therefore grows by roughly a pixel all the way round, on every sample.

This is stock MedSAM2 behaviour, but it is not harmless here: phase 8 scores against the
**un-dilated** GT at native resolution, so the model would be trained to a target
systematically fatter than the one it is graded on, and the bias is in the direction the
whole pipeline is already prone to — over-segmentation past the tumour edge.

Since the collate is ours anyway, **threshold instead of casting**: `(segment >= 0.5)`
rather than `segment.to(torch.bool)`. Bilinear upsampling puts the true boundary exactly at
0.5, so this recovers the real edge at essentially no cost, and it applies only to the GT
objects — the prompt objects stay `float32` and untouched.

It is a deliberate deviation from stock. Write it in the run notes: if a later result is
compared against a number produced with the stock cast, the two are not measuring quite the
same target.

**4b. `prepare_prompt_inputs` hardcodes GT.**
`MedSAM2/training/model/sam2.py:226` is literally
`backbone_out["mask_inputs_per_frame"][t] = gt_masks_per_frame[t]`. There is no config
switch for this. Handled by the subclass in phase 4.4.

**4c. The resize threshold.** When a mask prompt is not already at `image_size`, SAM2
resizes it and applies `(mask_inputs >= 0.5).float()`. In training this does **not** fire,
because `RandomResizeAPI` has already brought everything to 512 — but only as long as
`scratch.resolution` matches the model's `image_size`. If you ever change one, change both.
(At inference it *does* fire, which is why
[YoloMedSAM2InferencesTest/medsam2_soft.py](../YoloMedSAM2InferencesTest/medsam2_soft.py)
pre-resizes to 512 itself.)

**Gate before proceeding:** a unit check that pulls one batch and asserts
`prompt_masks` contains values strictly between 0 and 1. If it is all 0/1, one of the three
is still firing.

Second assertion in the same check, for the dilation above: take one GT object before and
after the collate and compare areas, scaled for the resize. `>= 0.5` should land within a
percent or so of the native area; `to(torch.bool)` will come out visibly larger. That is
the only way to see the difference — both produce a perfectly plausible-looking mask.

### Phase 4.4 — The model subclass

`MedSAM2_Finetuned/sam2_yolo_prompt.py`:

```python
class SAM2TrainYOLOPrompt(SAM2Train):
    def prepare_prompt_inputs(self, backbone_out, input, start_frame_idx=0):
        out = super().prepare_prompt_inputs(backbone_out, input, start_frame_idx)
        if self.use_yolo_prompt and not out["use_pt_input"]:
            # Redraw conditioning frames onto slices YOLO actually detected on;
            # never substitute GT (see below).
            out["init_cond_frames"] = self._redraw_cond_frames(out, input)
            for t in out["init_cond_frames"]:
                out["mask_inputs_per_frame"][t] = input.prompt_masks[t]
        return out
```

Everything else is inherited. `gt_masks_per_frame` is left untouched, so the **loss is
still computed against ground truth** — the model is prompted with YOLO and graded against
the truth. That asymmetry is the whole point: it teaches the model to *correct* YOLO rather
than to reproduce it.

`use_yolo_prompt` is a config flag so the GT-prompt baseline (§5, run #0) is the same code
path with one value flipped.

#### When YOLO found nothing on a conditioning frame

**Do not fall back to GT.** At inference there is no GT to fall back to: the runner leaves
a missed anchor unprompted and lets the memory bank fill that slice from its neighbours.
Substituting GT during training would hand the model a perfect prompt in exactly the place
YOLO is weakest, teaching it a situation that cannot occur in production — the same class
of train/inference gap as the red channel and the JPEG round-trip, and the one that would
flatter the result most.

Do this instead, in order:

1. **Prefer a different conditioning frame.** `init_cond_frames` is chosen at random from
   the clip; if the drawn frame has an empty prompt, redraw among the frames that have one.
   The clip is still trained, and on a real prompt.
2. **If no frame in the clip has a prompt, prompt with nothing** and let the loss fall on
   the propagated frames, exactly as inference would.

#### Counting it — and what not to count

An empty prompt means two completely different things now that the margin exists
(phase 1):

| GT on that frame | Prompt | Meaning | Count it? |
|---|---|---|---|
| non-empty | empty | **YOLO missed a real tumour** | **yes** |
| empty | empty | a margin slice, correctly nothing to find | no |

Count only the first. Lumping them together would tally the ~24 margin slices per patient
as YOLO failures and drown the real signal in noise that grows with `MARGIN`.

With the GT fallback gone, the counter no longer warns about reverting to GT-prompt
training — that cannot happen any more. It now measures something else, still worth
watching: **how often YOLO misses a tumour slice outright.** Log it per epoch alongside
the loss.

- A small, steady figure is normal and is exactly the imperfection the finetune is meant
  to learn to survive.
- A large one means most clips are falling to step 2 — trained with no prompt at all —
  and the run is closer to plain segmentation than to prompt correction. Look at the
  prompts before reading any Dice.
- A figure that differs sharply between train and val is the YOLO train-split leak (§3.2)
  showing up in a number, which is a useful early read on how big that problem is.

### Phase 5 — A launcher that works on Windows

`MedSAM2_Finetuned/train_entry.py`. Three jobs:

1. **Config discovery without touching MedSAM2.** `training/train.py` does
   `initialize_config_module("sam2")`, so it only sees configs inside the `sam2` package.
   Rather than copying our config in there, the launcher runs its own
   `hydra.initialize_config_dir(<this folder>/configs)` + `compose(...)`, then calls the
   *unmodified* `training.train.main(args, cfg)` — a module-level, importable function.
   `MedSAM2/` goes on `sys.path` and becomes the working directory so `training.*` /
   `sam2.*` targets resolve; this folder's parent goes on `sys.path` so
   `MedSAM2_Finetuned.*` targets resolve.
2. **Neutralise distributed collectives.** On Windows + RTX 5090 the `gloo` backend
   segfaults (access violation) on any collective. At `world_size=1` collectives are no-ops
   anyway, so monkeypatch `torch.distributed.all_reduce` / `barrier` to no-ops and swap
   `DistributedDataParallel` for a pass-through wrapper that still exposes `.module` — so
   saved checkpoints keep unprefixed keys and stay loadable by
   `build_sam2_video_predictor`.
3. Run with `--num-gpus 1`.

Gate: **one** epoch on a ~20-patient subset. Loss must decrease; a checkpoint must be
written; that checkpoint must load through `build_sam2_video_predictor` with the stock
t512 config. If it does not load here, it never will.

### Phase 6 — Full finetune or LoRA

`peft` is not yet in `.venv`; install it as part of this phase (`pip install peft`). It is
a training-only dependency and, like the other training-only deps, does not belong in the
project `requirements.txt`. **Recommendation: run #0 and #1 first, and treat run #2 (LoRA)
as optional** — the reasoning is under run #2 below.

**Run #0 — GT-prompt baseline.** Same everything, `use_yolo_prompt: false`. Without it you
cannot claim the YOLO-prompt training helped; you can only claim the finetune helped.

**Run #1 — full finetune, YOLO prompts.** Proven on this hardware, ~18 GB of 32 GB. At the
configured `multiplier: 4` that is ~8 min/epoch, so **40 epochs ≈ 5 h**; the ~2 min/epoch
and ~1.5 h figures are the stock `multiplier: 1`, which §3.1 argues against. No new
dependency, and the checkpoint drops straight into the existing inference call. With the
input channels moved (§2.1), the image encoder genuinely needs to travel — which full
finetuning allows and freezing prevents.

**Run #2 — LoRA. Recommended *against* for this project.** Install `peft` and keep the
option, but do not spend the first week on it. Four reasons, in order of weight:

1. **The input distribution changed, and that is the wrong shape of problem for LoRA.**
   Swapping the red channel from raw T1c to `T1c − T1n` (§2.1) changes the statistics the
   patch embedding and the earliest trunk layers see. LoRA adds low-rank updates to
   attention `qkv`/`proj`; it barely touches the patch embedding and MLPs, which is exactly
   where an input-domain shift has to be absorbed. A method designed to nudge *behaviour*
   while holding *representation* fixed is a poor fit when the representation is what has
   to move.
2. **Forgetting is not a cost we pay.** LoRA's headline benefit is preserving the base
   model's general ability. This checkpoint has one job — brain tumours on BraTS. There is
   no second task to protect, so that benefit buys nothing here.
3. **This is not a low-data regime.** 973 patients against a ~38 M-parameter tiny model,
   with 40 epochs at `multiplier: 4` giving ~1.2 M slice-samples. LoRA's implicit
   regularisation earns its keep on hundreds of examples, not on this.
4. **VRAM is not the constraint.** ~18 GB of 32 GB used. LoRA's efficiency argument is
   moot on this card.

Against that, full finetuning costs only a modest amount of extra compute per epoch — the
run length is set by `multiplier`, not by which parameters are trainable — and produces a
checkpoint that loads with zero extra steps, while LoRA adds `merge_and_unload()` and a
strict-loading trap that can silently produce an unusable file.

**When to revisit.** Two concrete triggers: (a) run #1's val loss starts rising while train
loss keeps falling — genuine overfitting, and LoRA's constraint becomes a cheap fix; or
(b) you later want several task-specific variants sharing one base, where swappable adapters
beat storing several full checkpoints. Neither is true today.

If you do run it: rank 8–16, alpha 16–32, dropout 0.05, targeting `qkv` and `proj` of the
Hiera trunk attention plus the mask-decoder attention; leave memory attention full-rank.
**The trap:** `model_weight_initializer` runs with `strict: True`, and injecting LoRA
renames parameters — so base weights must be loaded *before* adapters are injected, and
`merge_and_unload()` must produce a plain state dict at the end. An unmerged LoRA
checkpoint will not load in `build_sam2_video_predictor`.

### Phase 7 — Train

40 epochs, `checkpoint.pt` rewritten every epoch (Ctrl-C safe), numbered checkpoints every
10, plus `checkpoint_best_<meter>.pt` whenever the tracked val meter improves (§1.5).
`trainer.mode: train` with a `data.val` block over the 324 val patients, eval transforms
only (resize + tensor + normalize, no augmentation), `shuffle: False`, `val_epoch_freq: 1`.
Watch train vs val loss in TensorBoard; if val plateaus well before epoch 40, stop early.

#### Do not pick the final checkpoint on val loss alone

Val loss is measured under the **training** regime — 8-slice clips,
`num_init_cond_frames_for_eval: 1`, one prompt, seven propagated frames. Phase 8 measures
something materially different: up to 7 anchors, forward *and* backward propagation across
the whole ~180-slice volume, the HITL early-stop ladder, and volume Dice against `seg > 0`.
A checkpoint can win on the first and lose on the second — the two differ in propagation
length by an order of magnitude, and that is exactly the axis a finetune moves.

So treat val loss as the **cheap monitor** (is it still learning? has it started
overfitting?) and select on the real thing:

1. Hold out ~20 val patients as a selection set, fixed once and never changed.
2. Run the **phase-8 protocol** on them — this folder's `runner.py`, real anchors, real
   propagation, volume Dice — at the numbered checkpoints (10/20/30/40) and at
   `checkpoint_best_val_loss.pt`.
3. Ship whichever wins there. About 20 patients × 5 checkpoints of inference, minutes not
   hours, and it is the only comparison measured in the same units as the result.

Keep the selection set inside **val**, never test. Test stays untouched until phase 8, or
the four-row table stops meaning what it says.

### Phase 8 — Evaluate

`MedSAM2_Finetuned/runner.py` (inference) + `MedSAM2_Finetuned/evaluate.py` (scoring).

The runner is this folder's own copy of the video-predictor plumbing: builds frames via
`frames.py` (§2) at the **build** the row asks for (`diff` or `raw`), writes the JPEG
sequence to `work/frames/<build>/` in this folder at `frames.JPEG_QUALITY` (§2.2),
`init_state(async_loading_frames=False)`, registers **one object, `obj_id=1`**, with soft
mask prompts the way `medsam2_soft.py` does (float tensor pre-resized to 512, to dodge
trap 4c), propagates forward from the first anchor and backward from the last, merges
per-frame logits with `max`, thresholds at logit 0.

⚠ **Do not write into `outputs/tmp_frames/gtvsyolo/`.** That is where the plumbing this is
copied from puts its frames. Writing there would scatter this folder's output through the
project's `outputs/` tree and interfere with a cache another module depends on. Own dir,
`work/frames/<build>/`, regenerable, gitignored.

Prompts are generated with this folder's `yolo_prompts.py` and cached under
`work/prompts/`. Nothing is read from another module's outputs.

The harness around the model is fixed and identical for every row, or the rows are not
comparable with each other: the GT-derived `anchor_schedule` (`MIN_ANCHOR_TUMOUR_PX = 100`,
≤ 7 anchors), the HITL ladder (one anchor per round, stop at Dice ≥ 0.95 or Δ < 0.005, keep
the best round), prompts at `conf 0.25`, and volume Dice against `seg > 0`.

Score on the **test** split — all 324 patients, which neither model has seen. Same prompts,
same anchors, same scoring throughout; **one variable moves per row**:

| # | Checkpoint | Frames | Answers |
|---|---|---|---|
| 1 | base `MedSAM2_latest.pt` | `raw` (`R = T1c`) | the baseline — today's pipeline, nothing changed |
| 2 | base `MedSAM2_latest.pt` | `diff` (`R = T1c−T1n`) | how much is the channel change alone, with no training? |
| 3 | finetuned, GT-prompt (run #0) | `diff` | how much did finetuning help? |
| 4 | finetuned, YOLO-prompt (run #1/2) | `diff` | how much did training on the YOLO prompt specifically help? |

All four rows are computed here, by the same runner, in the same pass. That is the point:
every row shares its code path, so a difference between two rows is caused by the one thing
that differs between them and nothing else.

**Row 1 is the number to beat.** If rows 3 and 4 do not clear it, the finetune did not earn
its place, whatever the val loss said.

**Row 2 is the one people forget**, and it is also the §2.1 pre-flight — two inference
passes, no training, so it can be run on day one. Without it, an improvement in row 4
cannot be attributed to the finetune rather than to the channel change; they moved
together, and only this row tells them apart.

Rows 3 and 4 differ by `use_yolo_prompt` alone, so 4 − 3 is the value of training on the
YOLO prompt and 3 − 2 is the value of finetuning at all.

`evaluate.py` writes one packed mask per patient plus a `summary.json` under
`eval/<run>/` and renders the table to `eval/table.md` — again, in this folder, not in the
project's `outputs/`.

### Ordering note

Phases 1–5 carry all the risk — every one has a failure mode that only appears at runtime,
and phase 4 has three that fail *silently*. Phases 6–8 are mostly waiting. Write the
`.gitignore` **inside this folder** (`data/`, `work/`, `exp_log/`, `eval/*/`) before the
first run, or 20–40 GB of NPZ ends up staged for commit.

**Check this once, after phase 1 and again after phase 8:** `git status` from the repo
root should show nothing outside `MedSAM2_Finetuned/`, and `outputs/`, `MedSAM2/` and
`GTvsYOLO/` should have no new or modified files. If they do, a path escaped — find it
before the 40-epoch run, not after.

---

## 5. Files this folder will contain

| File | Role |
|---|---|
| `paths.py` | every absolute path, resolved from `__file__` (§1.5) — nothing outside this folder is written |
| `frames.py` | RGB construction (§2) + the JPEG round-trip (§2.2) — one source of truth, used by prep *and* inference |
| `yolo_prompts.py` | soft-mask YOLO inference, per-pixel probability prompt construction |
| `prep_npz.py` | patients → npz clips (`imgs`, `gts`, `yolo_prob`) |
| `npz_rgb_raw_dataset.py` | RGB-aware raw dataset + prompt objects |
| `collate.py` | float-preserving collate + `prompt_masks` field; thresholds GT at 0.5 (§ phase 4a) |
| `sam2_yolo_prompt.py` | `SAM2Train` subclass — YOLO prompt in, GT loss out |
| `configs/braTS_finetune512.yaml` | the config from §1 |
| `train_entry.py` | Windows-safe launcher |
| `runner.py` | inference — any checkpoint (base or finetuned) on either frame build, for all four phase-8 rows |
| `evaluate.py` | test-split scoring, the four-row table (§ phase 8) |
| `.gitignore` | `data/`, `work/`, `exp_log/`, `eval/*/` |

---

## 6. `use_mask_input_as_output_without_sam: true` — what it actually does

### The normal path

Normally a prompt goes through the **prompt encoder**: a small network that turns your box,
click or mask into an embedding. That embedding meets the image features in the mask
decoder, and the decoder *decides* what the mask should be. Your prompt is a suggestion;
the model interprets it.

### What the flag changes

With `use_mask_input_as_output_without_sam: true`, a **mask** prompt skips all of that. The
prompt encoder and mask decoder are bypassed entirely, in `sam2_base.py::_use_mask_as_output`:

```python
out_scale, out_bias = 20.0, -10.0          # sigmoid_scale/bias_for_mem_enc
high_res_masks = mask_inputs_float * out_scale + out_bias
```

That is the whole computation. Your mask **becomes** the model's output for that frame,
turned into logits by a straight-line formula. The model does not interpret it — it accepts
it, and writes it into the memory bank so it propagates to neighbouring slices.

### Why 0.5 is the neutral point

A pixel value `p` becomes the logit `20p − 10`:

| Your value `p` | Logit | What the model hears |
|---|---|---|
| 0.0 | **−10** | "definitely background" (a shout) |
| 0.25 | −5 | "background" |
| **0.5** | **0** | "no opinion" |
| 0.75 | +5 | "tumour" |
| 1.0 | **+10** | "definitely tumour" (a shout) |

A logit of 0 is a probability of 0.5 — a coin flip. So **zero confidence is written as 0.5,
not as 0.**

### The trap this creates

The natural move is to multiply the mask by the confidence: YOLO is 30 % sure, so write 0.3
and let the model treat it as a weak hint.

**That does the opposite of what you intend.** 0.3 becomes the logit −4 — a fairly
confident statement that the region is *background*. You have not whispered "maybe
tumour"; you have said "not tumour". A weak detection, expressed naively, actively
suppresses the very region it was pointing at. Nothing errors. You just get worse Dice.

### Hence the [0.5, 1.0] mapping

[YoloMedSAM2InferencesTest/variants.py](../YoloMedSAM2InferencesTest/variants.py) maps
confidence into `[0.5, 1.0]` via `NEUTRAL = 0.5`:

```
prompt_value = 0.5 + 0.5 * confidence
```

so `confidence = 0` → 0.5 ("no opinion", logit 0) and `confidence = 1` → 1.0 ("certain",
logit +10). Background stays a hard 0.0. Confidence now modulates **how hard the model is
pushed toward tumour**, and can never flip the sign of the claim.

**The per-pixel probability — the prompt we are training on — is the exception.** It
already uses 0.5 as its own decision boundary, so it maps straight through with no
rescaling. That is exactly why it is the right prompt to train on: its native scale already
matches what the model expects.

### Three consequences for this finetune

1. **Keep the flag on, and keep 20.0 / −10.0.** They are what `MedSAM2_latest.pt` was
   trained under, and the whole `[0.5, 1.0]` reasoning above is calibrated to them.
   Changing either constant changes what a prompt value means, and every number produced
   before the change stops being comparable with every number after it.
2. **On prompted frames the loss is nearly free.** Where a mask prompt is supplied, the
   "prediction" is an arithmetic transform of the prompt itself. Learning happens on the
   **propagated** frames — the ones with no prompt. That is why `num_frames: 8` with only
   1–2 conditioning frames matters: shrink the clip, and you shrink the part of the batch
   that actually teaches anything.
3. **This is what makes YOLO-prompt training work at all.** Because the prompt passes
   through untouched, prompting with YOLO and grading against GT puts the error exactly
   where it belongs — on the propagated slices, where the model must clean up YOLO's
   mistakes. That gradient is the thing we are trying to buy.
