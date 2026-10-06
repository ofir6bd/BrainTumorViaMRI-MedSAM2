"""MedSAM2 on a stack of slices, as a video in which **every slice is prompted**.

Each slice gets YOLO's own per-voxel map for that slice as a mask prompt, *and* the memory
of the slices before it. Frame 0 (the anchor) is the one frame with no memory behind it;
every later frame is a SAM2 "tracked" frame that also carries a mask prompt. The vendored
`SAM2Base._track_step` supports exactly that — on a frame that is not the initial
conditioning frame, the mask prompt goes to the decoder *and* the features are memory-
conditioned (`sam2_base.py`, `_prepare_memory_conditioned_features`, `is_init_cond_frame`).

So the model's job is "fix YOLO's guess on this slice, knowing what the neighbours looked
like" — not "redraw the tumour from memory alone", which is what the anchor-only version
asked and why it never beat YOLO: with 1-8 anchors it threw away YOLO's answer on ~95% of
the slices, and scored below YOLO at every distance from an anchor, the anchor included.

Training and inference run **the same function** (`track`): `SAM2Train.forward_tracking`
with our prompts in place of the ground truth it would have used. Inference simply feeds a
whole half-volume (anchor -> top, anchor -> bottom) where training feeds a short clip.
Nothing in `MedSAM2/` is edited; the batch object is a small local stand-in for the
vendor's `BatchedVideoDatapoint` (the tracking code reads exactly the things in `Clips`).
"""
from dataclasses import dataclass

import torch
import torch.nn as nn
import torch.nn.functional as F

from .common import import_sam2, mask_prompt_override

LOGIT_CLAMP = 30.0  # keeps BCE finite if the decoder ever produces an extreme logit


@dataclass
class Clips:
    """A training batch. img_batch: [T, B, 3, S, S]   masks: [T, B, S, S] (one object per clip)"""

    img_batch: torch.Tensor
    masks: torch.Tensor

    @property
    def num_frames(self):
        return self.img_batch.shape[0]

    @property
    def flat_img_batch(self):
        return self.img_batch.transpose(0, 1).flatten(0, 1)      # [(B*T), 3, S, S], clip-major

    def to(self, device):
        return Clips(self.img_batch.to(device, non_blocking=True),
                     self.masks.to(device, non_blocking=True))


def build_train_model(cfg, device="cuda", weights=None):
    """The vendored SAM2 built as `SAM2Train`, from one of the MedSAM2 checkpoints, and
    optionally carrying a run's fine-tuned weights. Used for training *and* inference."""
    import os

    from .common import CHECKPOINTS_DIR

    build_sam2 = import_sam2()
    ckpt = os.path.join(CHECKPOINTS_DIR, cfg["model"]["checkpoint"])
    if not os.path.exists(ckpt):
        raise FileNotFoundError(f"no checkpoint at {ckpt}")
    model = build_sam2(
        cfg["model"]["config"], ckpt, device=device, mode="train",
        hydra_overrides_extra=[
            "++model._target_=training.model.sam2.SAM2Train",
            # always a mask prompt, never sampled clicks or boxes: the prompt is YOLO's map
            "++model.prob_to_use_pt_input_for_train=0.0",
            "++model.prob_to_use_box_input_for_train=0.0",
            "++model.num_init_cond_frames_for_train=1",
            "++model.rand_init_cond_frames_for_train=false",
            *mask_prompt_override(cfg),
        ])
    state = None
    if weights:
        state = torch.load(weights, map_location="cpu", weights_only=False)
        model.load_state_dict(state["model"])
    return VideoSAM2(model), state


@dataclass
class _Index:
    """The two things `forward_tracking` reads off its input when the image features are
    already computed: how many frames, and which feature row belongs to (frame, clip)."""

    num_frames: int
    flat_obj_to_img_idx: torch.Tensor


def features(sam2, images, chunk=16):
    """Image-encoder output for [N, 3, S, S] frames, computed `chunk` at a time."""
    parts = [sam2.forward_image(images[i:i + chunk]) for i in range(0, len(images), chunk)]
    return {"backbone_fpn": [torch.cat([p["backbone_fpn"][l] for p in parts])
                             for l in range(len(parts[0]["backbone_fpn"]))],
            "vision_pos_enc": [torch.cat([p["vision_pos_enc"][l] for p in parts])
                               for l in range(len(parts[0]["vision_pos_enc"]))]}


def select(feats, rows):
    """The feature rows of `rows` (a LongTensor), in that order."""
    return {k: [x[rows] for x in v] for k, v in feats.items()}


def track(sam2, feats, prompts, num_frames, num_clips):
    """Frame 0 of each clip is the anchor; every frame gets its mask prompt.

    `feats` rows are ordered clip-major (row = clip * T + frame, as `Clips.flat_img_batch`),
    `prompts` is [T, B, 1, S, S]. Returns the vendor's per-frame output dicts.
    """
    t, b = num_frames, num_clips
    backbone_out = dict(feats)
    backbone_out.update(
        num_frames=t, init_cond_frames=[0], frames_not_in_init_cond=list(range(1, t)),
        mask_inputs_per_frame={k: prompts[k] for k in range(t)},
        point_inputs_per_frame={}, gt_masks_per_frame={}, frames_to_add_correction_pt=[])
    frame = torch.arange(t, device=prompts.device)[:, None]
    clip = torch.arange(b, device=prompts.device)[None, :]
    return sam2.forward_tracking(backbone_out, _Index(t, clip * t + frame))


class VideoSAM2(nn.Module):
    """Runs one batch of clips: (frames, per-frame prompts) -> per-frame mask logits."""

    def __init__(self, sam2):
        super().__init__()
        self.sam2 = sam2

    def forward(self, clips, prompts):
        """`prompts` is [T, B, 1, S, S] logits, one YOLO hint per frame of every clip."""
        feats = self.sam2.forward_image(clips.flat_img_batch)
        feats = {"backbone_fpn": feats["backbone_fpn"], "vision_pos_enc": feats["vision_pos_enc"]}
        return track(self.sam2, feats, prompts, clips.num_frames, clips.img_batch.shape[1])


def set_trainable(model, unfreeze):
    """Freeze everything, then switch back on what this run is allowed to change.

    The prompt encoder reads YOLO's hint on every slice and the memory brings in the
    neighbours, so both are worth training — `decoder+prompt+memory` is the default. `all`
    also moves the image encoder, at `train.vision_lr`.
    """
    sam2 = model.sam2
    for p in sam2.parameters():
        p.requires_grad_(False)
    groups = {
        "decoder": [sam2.sam_mask_decoder],
        "decoder+prompt": [sam2.sam_mask_decoder, sam2.sam_prompt_encoder],
        "decoder+prompt+memory": [sam2.sam_mask_decoder, sam2.sam_prompt_encoder,
                                  sam2.memory_attention, sam2.memory_encoder],
        "all": [sam2.sam_mask_decoder, sam2.sam_prompt_encoder, sam2.memory_attention,
                sam2.memory_encoder, sam2.image_encoder],
    }
    if unfreeze not in groups:
        raise ValueError(f"unfreeze must be one of {list(groups)}")
    for module in groups[unfreeze]:
        for p in module.parameters():
            p.requires_grad_(True)
    # the object-pointer projection rides along with the decoder; it feeds the memory
    for p in sam2.obj_ptr_proj.parameters():
        p.requires_grad_(unfreeze != "decoder")
    return groups[unfreeze]


def param_groups(model, cfg):
    """Three learning rates, because the three parts tolerate very different step sizes.

    The memory is the delicate one: it carries the answer between slices, so a large update
    there disturbs every slice at once. Both published works that train it at all
    move it far more slowly than the decoder — SurgSAM-2 by 10x (2e-4 vs 2e-5), Medical SAM 2
    by 10,000x (1e-4 vs 1e-8). `memory_lr` defaults to `lr` if a run's config predates it.
    """
    t = cfg["train"]
    sam2 = model.sam2
    set_trainable(model, t["unfreeze"])
    encoder = {id(p) for p in sam2.image_encoder.parameters()}
    memory = {id(p) for m in (sam2.memory_attention, sam2.memory_encoder) for p in m.parameters()}
    head, mem, vision = [], [], []
    for p in model.parameters():
        if not p.requires_grad:
            continue
        (vision if id(p) in encoder else mem if id(p) in memory else head).append(p)
    out = [{"params": head, "lr": float(t["lr"])}]
    if mem:
        out.append({"params": mem, "lr": float(t.get("memory_lr", t["lr"]))})
    if vision:
        out.append({"params": vision, "lr": float(t["vision_lr"])})
    return out


def dice_loss(logits, target, eps=1.0):
    p = torch.sigmoid(logits).flatten(1)
    g = target.flatten(1)
    inter = (p * g).sum(1)
    return (1 - (2 * inter + eps) / (p.sum(1) + g.sum(1) + eps)).mean()


def losses(outputs, clips, cfg):
    """Dice + BCE on every frame of the clip, plus two small honesty terms.

    Every frame counts, not just the anchors: the whole point of propagation is what happens
    on the frames nobody prompted.
    """
    t = cfg["train"]
    logits = torch.stack([o["pred_masks_high_res"] for o in outputs])       # [T, O, 1, S, S]
    # The anchor runs with one mask, the tracked frames with three (the config's
    # `multimask_output_for_tracking`), and the one kept is the highest-scoring of them —
    # so the IoU that belongs to the returned mask is the largest of the row.
    ious = torch.stack([o["multistep_pred_ious"][-1].amax(dim=-1).reshape(-1) for o in outputs])
    objs = torch.stack([o["multistep_object_score_logits"][-1].reshape(-1) for o in outputs])
    logits = logits.flatten(0, 1).clamp(-LOGIT_CLAMP, LOGIT_CLAMP).float()  # [T*O, 1, S, S]
    target = clips.masks.flatten(0, 1)[:, None].float()                     # [T*O, 1, S, S]

    bce = F.binary_cross_entropy_with_logits(logits, target)
    dsc = dice_loss(logits, target)
    with torch.no_grad():
        hard = (logits > 0).float().flatten(1)
        g = target.flatten(1)
        inter = (hard * g).sum(1)
        union = hard.sum(1) + g.sum(1) - inter
        real_iou = torch.where(union > 0, inter / union.clamp(min=1.0), torch.ones_like(union))
        has_obj = (g.sum(1) > 0).float()
    iou = F.mse_loss(ious.flatten().float(), real_iou)
    obj = F.binary_cross_entropy_with_logits(objs.flatten().float(), has_obj)
    total = (float(t["dice_weight"]) * dsc + float(t["bce_weight"]) * bce
             + float(t["iou_weight"]) * iou + float(t["obj_weight"]) * obj)
    return total, {k: float(v.detach()) for k, v in
                   {"loss": total, "dice": dsc, "bce": bce, "iou": iou, "obj": obj}.items()}
