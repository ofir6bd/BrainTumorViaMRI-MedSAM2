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


def strength(entry, cfg, rule=None):
    """How good a starting slice each slice would make, one number per slice."""
    rule = rule or cfg["anchors"]["pick"]
    if rule == "yolo_score":
        return entry["scores"].astype(np.float32)
    area = (entry["prob"] > 127).reshape(len(entry["z"]), -1).sum(axis=1).astype(np.float32)
    if rule == "spread":
        return (area > 0).astype(np.float32)      # every slice YOLO marked is equally eligible
    return area * entry["scores"]                  # yolo_peak: big and confident


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
    if a["pick"] == "spread" and a["count"] > 1:
        marked = np.nonzero(s > 0)[0]
        picks = np.linspace(0, len(marked) - 1, a["count"]).round().astype(int)
        return sorted({int(marked[p]) for p in picks})
    order = np.argsort(-s)
    order = order[s[order] > 0]
    return sorted(_spaced(order, a["count"], a["min_gap"]))


def next_anchor(entry, cfg, pred, taken):
    """The slice a HITL round adds: where the current prediction disagrees most with YOLO.

    Only slices where YOLO *found* something are eligible, so the new anchor is always a
    positive prompt ("the tumour is here"). Measured the hard way: without this, the rule
    happily picked a slice where the prediction had drawn and YOLO had not, prompted it with
    a confident "nothing", and the memory carried that erasure into its neighbours — round 2
    scored *below* round 1 on three of four patients. It is also what the old hand-prompted
    pipeline did: its anchors were boxes, which can only ever say "here".
    """
    reference = yolo_mask(entry, cfg)
    n = len(entry["z"])
    has_reference = reference.reshape(n, -1).any(axis=1)
    wrong = np.logical_xor(pred, reference).reshape(n, -1).sum(axis=1) * has_reference
    order = np.argsort(-wrong)
    order = order[wrong[order] > 0]
    picks = _spaced(order, 1, cfg["anchors"]["min_gap"], taken)
    return picks[0] if picks else None


def prompt_for(entry, cfg, index):
    """The prompt tensor for one anchor slice, as logits at the slice's own size."""
    return prompt_logits(entry, cfg, index)[0]
