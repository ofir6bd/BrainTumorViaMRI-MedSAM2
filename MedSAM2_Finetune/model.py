"""MedSAM2 on a stack of slices, the way SAM2 is meant to run: as a video.

A few slices (the anchors) are prompted with YOLO's per-voxel map; the rest of the clip is
reached through SAM2's memory. The vendored `training.model.sam2.SAM2Train` already knows
how to run that — prompt the conditioning frames, then track the others with memory
attention — so it is *used*, not reimplemented. Two things are changed from the outside:

1. `SAM2Train.prepare_prompt_inputs` normally puts the **ground-truth mask** on the
   conditioning frames (`mask_inputs_per_frame[t] = gt_masks_per_frame[t]`). Here the
   anchors and their prompts are supplied by the caller instead, so the expert mask never
   reaches the model — it is only the target of the loss.
2. Nothing in `MedSAM2/` is edited. The class is subclassed and the batch object is a small
   local stand-in for the vendor's `BatchedVideoDatapoint` (the tracking code reads exactly
   four things off it, listed in `Clips` below).

Backwards propagation is learned by feeding half the clips in descending z
(`video.reverse_fraction`) — to the model that is still "forward", and at inference the
video predictor's own `reverse=True` pass then behaves the same way.
"""
from dataclasses import dataclass

import torch
import torch.nn as nn
import torch.nn.functional as F

from .common import import_sam2, mask_prompt_override

LOGIT_CLAMP = 30.0  # keeps BCE finite if the decoder ever produces an extreme logit


@dataclass
class Clips:
    """What `SAM2Train.forward_tracking` reads off its input, and nothing more.

    img_batch: [T, B, 3, S, S]   masks: [T, O, S, S] (O = B, one object per clip)
    obj_to_frame_idx: [T, O, 2] as (frame index, clip index)
    """

    img_batch: torch.Tensor
    masks: torch.Tensor
    obj_to_frame_idx: torch.Tensor

    @property
    def num_frames(self):
        return self.img_batch.shape[0]

    @property
    def flat_img_batch(self):
        return self.img_batch.transpose(0, 1).flatten(0, 1)      # [(B*T), 3, S, S]

    @property
    def flat_obj_to_img_idx(self):
        frame_idx, video_idx = self.obj_to_frame_idx.unbind(dim=-1)
        return video_idx * self.num_frames + frame_idx

    def to(self, device):
        return Clips(self.img_batch.to(device, non_blocking=True),
                     self.masks.to(device, non_blocking=True),
                     self.obj_to_frame_idx.to(device, non_blocking=True))


def build_train_model(cfg, device="cuda"):
    """The vendored SAM2 built as `SAM2Train`, from one of the MedSAM2 checkpoints."""
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
    return VideoSAM2(model)


class VideoSAM2(nn.Module):
    """Runs one batch of clips: (frames, prompts, anchors) -> per-frame mask logits."""

    def __init__(self, sam2):
        super().__init__()
        self.sam2 = sam2

    def forward(self, clips, prompts, anchors):
        """`prompts` is [A, B, 1, S, S] logits for the A anchor frames, `anchors` their
        frame indices inside the clip (the same for every clip in the batch)."""
        sam2 = self.sam2
        backbone_out = sam2.forward_image(clips.flat_img_batch)
        backbone_out = sam2.prepare_prompt_inputs(backbone_out, clips)
        # Replace the vendor's ground-truth prompts with ours. Everything after this point
        # is the vendored tracking code, unchanged.
        n = clips.num_frames
        backbone_out["init_cond_frames"] = list(anchors)
        backbone_out["frames_not_in_init_cond"] = [t for t in range(n) if t not in anchors]
        backbone_out["mask_inputs_per_frame"] = {int(t): prompts[i] for i, t in enumerate(anchors)}
        backbone_out["point_inputs_per_frame"] = {}
        backbone_out["frames_to_add_correction_pt"] = []
        return sam2.forward_tracking(backbone_out, clips)


def set_trainable(model, unfreeze):
    """Freeze everything, then switch back on what this run is allowed to change.

    Unlike the slice-by-slice version, the memory encoder and memory attention now carry
    the prediction from the anchors to the rest of the volume, so they are worth training —
    `decoder+prompt+memory` is the default.
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
    there disturbs every slice nobody prompted. Both published works that train it at all
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
