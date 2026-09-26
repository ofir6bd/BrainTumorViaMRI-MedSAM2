"""Which slices get a prompt — the first ones, and the ones a HITL round adds.

In video mode only a few slices are prompted; the rest of the volume is reached through
SAM2's memory. Choosing those few is most of the work, and it is the same question the old
`05_infer_multibbox_hitl.py` answered with "the slice with the most tumour, then wherever
the prediction is worst" — except that script read the answer sheet to decide.

Here the reference is always **YOLO**. Nothing in this file ever looks at the expert mask,
so no score this pipeline produces can be inflated by a hint it would not have in practice.
"""
import numpy as np

from .yolo_prompts import prompt_logits


def yolo_mask(entry, cfg):
    """What YOLO itself would publish: blobs at `score_min`, probability cut at 0.5."""
    return entry["prob_f"] > 127


def strength(entry, cfg):
    """How good a starting slice each slice would make: big and confident.

    Two other rules used to live here — the single highest-scoring blob, and evenly spaced
    over everything YOLO marked. Measured on three test patients at 5 rounds, all three
    finished within 0.0018 of each other (0.8194 / 0.8198 / 0.8180): the correction rounds
    wash out where the first anchor sat, so the choice was not worth keeping as a setting.
    """
    area = (entry["prob"] > 127).reshape(len(entry["z"]), -1).sum(axis=1).astype(np.float32)
    return area * entry["scores"]


def _spaced(order, count, min_gap, taken=()):
    """Take the best `count` candidates that stay `min_gap` slices apart."""
    chosen = list(taken)
    out = []
    for k in order:
        if len(out) >= count:
            break
        if all(abs(int(k) - int(c)) >= min_gap for c in chosen):
            out.append(int(k))
            chosen.append(int(k))
    return out


def initial(entry, cfg):
    """The anchors a run starts from, best first."""
    a = cfg["anchors"]
    s = strength(entry, cfg)
    if not s.any():
        # YOLO found nothing anywhere: start in the middle of the brain so the pass still
        # happens and the model gets its chance to say "nothing here".
        return [len(entry["z"]) // 2]
    order = np.argsort(-s)
    order = order[s[order] > 0]
    return sorted(_spaced(order, a["count"], a["min_gap"]))


def next_anchor(entry, cfg, pred, taken):
    """The slice a HITL round adds: where the current prediction disagrees most with YOLO.

    The disagreement is a plain voxel count — how many voxels one of them calls tumour and
    the other does not — so a big argument outranks a small one. Any slice may win, including
    one YOLO left empty: that anchor is prompted with `prompt.empty_logit`, a confident
    "nothing here", which is the only way to answer a run-away propagation where memory keeps
    drawing tumour far past the end of it.

    Slices YOLO left empty used to be barred, so that every anchor said "the tumour is here"
    and no confident "nothing" could be carried into the neighbours by the memory. Measured on
    12 test patients with an 8-round schedule, that guard bought nothing and cost a patient:
    0.8263 against 0.8679 without it. The two rules pick the same anchors on 7 of the 12 and
    average 0.0009 apart on the 11 that behave; the whole gap is BraTS-GLI-02254-100, where
    YOLO's tumour ends at z=97 but memory drew up to 1100 voxels a slice out to z=150 — the
    guard could not point at any of those slices (0.2450), and without it z=130 and z=135
    became anchors (0.7342). It also used to stop the loop early on that patient, at 4 of 8
    rounds, because every YOLO-positive slice was already within `anchors.min_gap` of one.
    """
    reference = yolo_mask(entry, cfg)
    n = len(entry["z"])
    wrong = np.logical_xor(pred, reference).reshape(n, -1).sum(axis=1)
    order = np.argsort(-wrong)
    order = order[wrong[order] > 0]
    picks = _spaced(order, 1, cfg["anchors"]["min_gap"], taken)
    return picks[0] if picks else None


def prompt_for(entry, cfg, index):
    """The prompt tensor for one anchor slice, as logits at the slice's own size."""
    return prompt_logits(entry, cfg, index)[0]
