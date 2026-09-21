"""MedSAM2 on a single slice, prompted by YOLO's per-voxel map.

SAM2's own `_forward_sam_heads` is not used directly for two reasons:

1. With `pred_obj_scores: true` it replaces the mask with the constant -1024 whenever its
   object head says "nothing here" (`torch.where`). That is the right thing at inference,
   but during training it cuts the gradient off for exactly the slices we most want to
   teach — the empty ones. Calling the prompt encoder and the mask decoder ourselves keeps
   the raw logits, and lets the object head be trained explicitly as "is there tumour on
   this slice at all".
2. It gives us the object score as its own number, which the dashboard charts.

Everything here is a *use* of the vendored `sam2` package. Nothing in `MedSAM2/` is edited.
"""
import torch
import torch.nn as nn
import torch.nn.functional as F

LOGIT_CLAMP = 30.0  # keeps BCE finite if the decoder ever produces an extreme logit


class SliceSAM2(nn.Module):
    """Wraps a SAM2Base so it behaves like an ordinary image segmentation model:
    (frame, prompt) -> (mask logits at frame size, predicted IoU, object score)."""

    def __init__(self, sam2):
        super().__init__()
        self.sam2 = sam2

    def image_features(self, image):
        out = self.sam2.forward_image(image)
        _, vision_feats, _, feat_sizes = self.sam2._prepare_backbone_features(out)
        # Single slices have no memory to attend to, so the "no memory" embedding is added
        # directly — the same thing SAM2's own image predictor does.
        vision_feats[-1] = vision_feats[-1] + self.sam2.no_mem_embed
        b = image.shape[0]
        feats = [f.permute(1, 2, 0).view(b, -1, *size)
                 for f, size in zip(vision_feats[::-1], feat_sizes[::-1])][::-1]
        return feats[-1], feats[:-1]

    def forward(self, image, prompt):
        embed, high_res = self.image_features(image)
        b, _, h, w = image.shape
        size = self.sam2.sam_prompt_encoder.mask_input_size
        mask_prompt = prompt if prompt.shape[-2:] == tuple(size) else F.interpolate(
            prompt.float(), size=size, mode="bilinear", align_corners=False, antialias=True)
        coords = torch.zeros(b, 1, 2, device=image.device)
        labels = -torch.ones(b, 1, dtype=torch.int32, device=image.device)  # -1 = padding point
        sparse, dense = self.sam2.sam_prompt_encoder(points=(coords, labels), boxes=None,
                                                     masks=mask_prompt)
        low_res, ious, _tokens, obj_score = self.sam2.sam_mask_decoder(
            image_embeddings=embed, image_pe=self.sam2.sam_prompt_encoder.get_dense_pe(),
            sparse_prompt_embeddings=sparse, dense_prompt_embeddings=dense,
            multimask_output=False, repeat_image=False, high_res_features=high_res)
        logits = F.interpolate(low_res.float(), size=(h, w), mode="bilinear", align_corners=False)
        return logits.clamp(-LOGIT_CLAMP, LOGIT_CLAMP), ious.float(), obj_score.float()


def set_trainable(model, unfreeze):
    """Freeze everything, then switch back on what this run is allowed to change.

    The memory encoder / memory attention are never trained here: a single slice has no
    memory, so those weights see no gradient and would only waste optimiser state.
    """
    sam2 = model.sam2
    for p in sam2.parameters():
        p.requires_grad_(False)
    groups = {"decoder": [sam2.sam_mask_decoder],
              "decoder+prompt": [sam2.sam_mask_decoder, sam2.sam_prompt_encoder],
              "all": [sam2.sam_mask_decoder, sam2.sam_prompt_encoder, sam2.image_encoder]}
    if unfreeze not in groups:
        raise ValueError(f"unfreeze must be one of {list(groups)}")
    for module in groups[unfreeze]:
        for p in module.parameters():
            p.requires_grad_(True)
    return groups[unfreeze]


def param_groups(model, cfg):
    """Two learning rates: the image encoder is big and pre-trained, so it moves slower."""
    t = cfg["train"]
    set_trainable(model, t["unfreeze"])
    encoder = set(id(p) for p in model.sam2.image_encoder.parameters())
    head, vision = [], []
    for p in model.parameters():
        if p.requires_grad:
            (vision if id(p) in encoder else head).append(p)
    out = [{"params": head, "lr": float(t["lr"])}]
    if vision:
        out.append({"params": vision, "lr": float(t["vision_lr"])})
    return out


def dice_loss(logits, target, eps=1.0):
    p = torch.sigmoid(logits).flatten(1)
    g = target.flatten(1)
    inter = (p * g).sum(1)
    return (1 - (2 * inter + eps) / (p.sum(1) + g.sum(1) + eps)).mean()


def losses(logits, ious, obj_score, target, cfg):
    """Dice + BCE on the mask, plus two small honesty terms.

    `iou` teaches SAM2's quality head to predict the overlap it actually achieved, and
    `obj` teaches its object head whether this slice has any tumour — the model's own
    version of the empty-slice decision.
    """
    t = cfg["train"]
    bce = F.binary_cross_entropy_with_logits(logits, target)
    dsc = dice_loss(logits, target)
    with torch.no_grad():
        hard = (logits > 0).float().flatten(1)
        g = target.flatten(1)
        inter = (hard * g).sum(1)
        union = hard.sum(1) + g.sum(1) - inter
        real_iou = torch.where(union > 0, inter / union.clamp(min=1.0), torch.ones_like(union))
        has_obj = (g.sum(1) > 0).float()
    iou = F.mse_loss(ious.squeeze(1), real_iou)
    obj = F.binary_cross_entropy_with_logits(obj_score.squeeze(1), has_obj)
    total = (float(t["dice_weight"]) * dsc + float(t["bce_weight"]) * bce
             + float(t["iou_weight"]) * iou + float(t.get("obj_weight", 0.0)) * obj)
    return total, {k: float(v.detach()) for k, v in
                   {"loss": total, "dice": dsc, "bce": bce, "iou": iou, "obj": obj}.items()}
