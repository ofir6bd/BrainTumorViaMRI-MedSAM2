"""The variants under test — and the one function that differs between them.

Every variant runs the identical harness: same patients, same GT-derived anchor
schedule, same MedSAM2 checkpoint and config, same frames, same forward+backward
propagation, same scoring. The **only** thing that changes is `build_prompt` — the mask
handed to MedSAM2 on an anchor slice. Any difference in Dice is therefore attributable
to that one change.

## Why confidence maps to [0.5, 1.0] and not [0, 1]

The t512 config sets `use_mask_input_as_output_without_sam: true`, so a mask prompt
bypasses the SAM prompt encoder and goes straight through `_use_mask_as_output`
(`sam2_base.py`):

    out_scale, out_bias = 20.0, -10.0
    high_res_masks = mask_inputs_float * out_scale + out_bias

A prompt value `p` becomes the logit `20p - 10`. **0.5 is the decision boundary**: above
it the model is told "tumour", below it "not tumour", and 0 means "definitely not".

So multiplying a mask by a confidence of 0.3 does not weaken the prompt — it inverts it,
telling MedSAM2 the tumour is background. Confidence therefore modulates *within*
`[0.5, 1.0]` (`_weighted`), where 0.5 is "no opinion" and 1.0 is "certain", and
background stays at 0.0. Variant C2 is the exception: YOLO's per-pixel probability
already uses 0.5 as its own decision boundary, so it maps straight through.

## Keeping the hard content identical

O, B, C1, C2 and F/G all binarise (at the model's 0.5) to the *same* mask — the stock
YOLO union at conf 0.25. B and C1 are built from that union by construction; C2 is
nudged onto it by `_align_to`, which only touches pixels where nearest- and
bilinear-resampling disagree. So a Dice difference between them is caused by the
gradation alone, never by a different set of pixels being called tumour.

A, Bg and E deliberately *do* change the hard content — that is what they test. H and I
change the harness itself rather than the prompt: H prompts every brain slice in one
pass, and I skips MedSAM2 altogether.
"""
from dataclasses import dataclass
from typing import Optional

import numpy as np

from .yolo_infer import slice_confidence

# MedSAM2's decision boundary. A confidence c becomes 0.5 + 0.5*c, so c=0 is "no opinion"
# (logit 0) rather than "definitely background" (logit -10).
NEUTRAL = 0.5


@dataclass(frozen=True)
class Variant:
    id: str
    label: str
    # `changes` and `why` are read by the UI table, so they are written for a reader who
    # does not know the codebase: what is different, and what that difference is testing.
    # Plain words, no "amplitude" / "union" / "area-weighted".
    changes: str            # the one thing that differs from O
    why: str                # the question this variant answers
    source: str = "yolo"    # "yolo" | "gt"
    conf: float = 0.25      # YOLO detection threshold
    mode: str = "binary"
    tau: Optional[float] = None
    max_anchors: int = 7
    early_stop: bool = True

    @property
    def wants_prob(self):
        return self.mode == "pixel_prob"

    @property
    def is_soft(self):
        """Does this variant ever emit a value strictly between 0 and 1?"""
        return self.mode in ("slice_conf", "inst_conf", "pixel_prob")


VARIANTS = {v.id: v for v in [
    Variant("O", "Reference (GTvsYOLO)",
            "Nothing. This is the ordinary run that every other row is compared against.",
            "If this ever stops matching the GT vs YOLO tab, the test rig has drifted - "
            "not the model."),
    Variant("GT", "GT ceiling",
            "MedSAM2 is handed the doctor's own mask instead of YOLO's.",
            "How good can the result get when the starting mask is perfect?",
            source="gt"),
    Variant("A", "Low threshold 0.10",
            "YOLO is allowed to report weaker findings - anything it is 10% sure of, "
            "instead of only 25% and above.",
            "Do the extra weak findings add real tumour, or just junk?",
            conf=0.10),
    Variant("B", "Per-slice confidence",
            "One confidence number for the whole slice. A slice YOLO is sure about is "
            "pushed on MedSAM2 strongly; an unsure one is offered more gently.",
            "Does telling MedSAM2 how sure YOLO is make it segment better?",
            mode="slice_conf"),
    Variant("Bg", "Per-slice gate 0.50",
            "If YOLO is less than 50% sure about an anchor slice, that slice is skipped "
            "and MedSAM2 gets no hint there at all.",
            "Is a doubtful hint worse than no hint?",
            mode="slice_gate", tau=0.50),
    Variant("C1", "Per-instance confidence",
            "Like B, but each separate lump YOLO found carries its own confidence "
            "instead of one number covering the whole slice.",
            "Is confidence more useful lump by lump than slice by slice?",
            mode="inst_conf"),
    Variant("C2", "Per-pixel probability",
            "Every single pixel carries its own confidence - strong in the middle of the "
            "tumour, fading towards the edge.",
            "The finest confidence YOLO can give. Does more detail help?",
            mode="pixel_prob"),
    Variant("E", "Drop blobs under 0.50",
            "Lumps YOLO is less than 50% sure about are thrown away before the mask is "
            "built, so MedSAM2 never sees them.",
            "Are the weak lumps helping, or dragging the result down?",
            mode="inst_filter", tau=0.50),
    Variant("F", "Forced 7 anchors",
            "The mask is unchanged. The run is simply not allowed to stop early, so all "
            "7 anchor slices are used.",
            "How much Dice is the early-stop rule throwing away?",
            early_stop=False),
    Variant("G", "Forced 15 anchors",
            "Same as F, but with 15 anchor slices instead of 7.",
            "Do more anchor slices matter more than a better mask?",
            max_anchors=15, early_stop=False),
    Variant("H", "Every brain slice prompted",
            "Like G, but there is no anchor budget at all: every slice of the brain is "
            "handed YOLO's mask at once, not just a chosen few. Nothing is held back for "
            "MedSAM2 to guess, and the run is done in one pass instead of round by round.",
            "With nothing left to guess, how good can the MedSAM2 arm possibly get?",
            mode="all_slices", max_anchors=0, early_stop=False),
    Variant("I", "YOLO alone, no MedSAM2",
            "MedSAM2 is not used at all. YOLO is run on every slice of the volume and "
            "its own masks are stacked into the result, so there are no anchor slices "
            "and nothing is propagated.",
            "Is MedSAM2 earning its place? If YOLO on its own already scores this well, "
            "the propagation is not adding anything.",
            mode="yolo_only", max_anchors=0),
]}

ORDER = list(VARIANTS)
REFERENCE = "O"
CEILING = "GT"

# Stable per-variant colours for the charts (kept in sync with inferencetest.js).
COLORS = {
    "O": "#ff5a26", "GT": "#40a6ff", "A": "#ffd23f", "B": "#4ade80", "Bg": "#2dd4bf",
    "C1": "#c084fc", "C2": "#f472b6", "E": "#fb923c", "F": "#38bdf8",
    "G": "#a3e635", "H": "#e879f9", "I": "#f87171",
}


# ---------------------------------------------------------------------------
# Prompt construction - the one thing that differs between variants
# ---------------------------------------------------------------------------
def _weighted(mask, c):
    """Binary mask carrying confidence `c` as its amplitude, in [NEUTRAL, 1]."""
    amp = NEUTRAL + (1.0 - NEUTRAL) * float(np.clip(c, 0.0, 1.0))
    return np.where(mask, amp, 0.0).astype(np.float32)


def _align_to(prob, hard):
    """Force `prob`'s 0.5-crossing onto `hard` without disturbing its gradation.

    `hard` is nearest-resampled and `prob` bilinear, so they disagree on a thin rim of
    edge pixels. Left alone that would make C2 a different *set* of pixels from O, not
    just a graded version of it, and the comparison would confound the two.
    """
    p = np.array(prob, dtype=np.float32, copy=True)
    eps = np.float32(1e-3)
    p[hard & (p < 0.5)] = np.float32(0.5) + eps
    p[(~hard) & (p >= 0.5)] = np.float32(0.5) - eps
    return p


def build_prompt(variant, info):
    """The prompt mask for one anchor slice, as float in [0, 1]. `None` = no prompt here.

    `info` is a `yolo_infer.predict_slice` result (or `{"union": gt_slice}` for the GT
    arm). An all-zero return is treated as "nothing to prompt with", exactly as the
    GTvsYOLO reference treats an empty YOLO mask.
    """
    if variant.mode == "yolo_only":
        raise ValueError("I never prompts MedSAM2 — it is scored from YOLO's own masks")

    union = info["union"]
    if variant.source == "gt" or variant.mode in ("binary", "all_slices"):
        return union.astype(np.float32)

    instances = info.get("instances") or []

    if variant.mode == "slice_gate":
        c = slice_confidence(instances)
        if c is None or c < variant.tau:
            return None
        return union.astype(np.float32)

    if variant.mode == "slice_conf":
        c = slice_confidence(instances)
        if c is None:
            return None
        return _weighted(union, c)

    if variant.mode == "inst_conf":
        # Overlapping blobs take the higher confidence — the union is an OR, so a pixel
        # claimed by a confident detection should not be weakened by a doubtful one.
        out = np.zeros(union.shape, dtype=np.float32)
        for inst in instances:
            np.maximum(out, _weighted(inst["mask"], inst["conf"]), out=out)
        return out

    if variant.mode == "pixel_prob":
        out = np.zeros(union.shape, dtype=np.float32)
        for inst in instances:
            p = inst["prob"]
            if p is None:
                p = inst["mask"].astype(np.float32)
            np.maximum(out, p, out=out)
        return _align_to(out, union)


    if variant.mode == "inst_filter":
        out = np.zeros(union.shape, dtype=bool)
        for inst in instances:
            if inst["conf"] >= variant.tau:
                out |= inst["mask"]
        return out.astype(np.float32)

    raise ValueError(f"unknown prompt mode: {variant.mode}")
