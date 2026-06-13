# ------------------------------------------------------------------------
# Modification: EDA
# Created: 05/21/2022
# Author: Yanmin Wu
# E-mail: wuyanminmax@gmail.com
# https://github.com/yanmin-wu/EDA
# ------------------------------------------------------------------------
# BEAUTY DETR
# Copyright (c) 2022 Ayush Jain & Nikolaos Gkanatsios
# Licensed under CC-BY-NC [see LICENSE for details]
# All Rights Reserved
# ------------------------------------------------------------------------
# Parts adapted from Group-Free
# Copyright (c) 2021 Ze Liu. All Rights Reserved.
# Licensed under the MIT License.
# ------------------------------------------------------------------------

from scipy.optimize import linear_sum_assignment
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.distributed as dist
from utils.scatter_util import scatter_mean
import math
import numpy as np
from typing import Dict, List, Optional, Tuple # <-- Ensure Dict is imported here



def is_dist_avail_and_initialized():
    if not dist.is_available():
        return False
    if not dist.is_initialized():
        return False
    return True


def box_cxcyczwhd_to_xyzxyz(x):
    x_c, y_c, z_c, w, h, d = x.unbind(-1)
    w = torch.clamp(w, min=1e-6)
    h = torch.clamp(h, min=1e-6)
    d = torch.clamp(d, min=1e-6)
    assert (w < 0).sum() == 0
    assert (h < 0).sum() == 0
    assert (d < 0).sum() == 0
    b = [(x_c - 0.5 * w), (y_c - 0.5 * h), (z_c - 0.5 * d),
         (x_c + 0.5 * w), (y_c + 0.5 * h), (z_c + 0.5 * d)]
    return torch.stack(b, dim=-1)


def _volume_par(box):
    return (
            (box[:, 3] - box[:, 0])
            * (box[:, 4] - box[:, 1])
            * (box[:, 5] - box[:, 2])
    )


def _intersect_par(box_a, box_b):
    xA = torch.max(box_a[:, 0][:, None], box_b[:, 0][None, :])
    yA = torch.max(box_a[:, 1][:, None], box_b[:, 1][None, :])
    zA = torch.max(box_a[:, 2][:, None], box_b[:, 2][None, :])
    xB = torch.min(box_a[:, 3][:, None], box_b[:, 3][None, :])
    yB = torch.min(box_a[:, 4][:, None], box_b[:, 4][None, :])
    zB = torch.min(box_a[:, 5][:, None], box_b[:, 5][None, :])
    return (
            torch.clamp(xB - xA, 0)
            * torch.clamp(yB - yA, 0)
            * torch.clamp(zB - zA, 0)
    )


def _iou3d_par(box_a, box_b):
    intersection = _intersect_par(box_a, box_b)
    vol_a = _volume_par(box_a)
    vol_b = _volume_par(box_b)
    union = vol_a[:, None] + vol_b[None, :] - intersection
    return intersection / union, union


# BRIEF 3DIoU loss
def generalized_box_iou3d(boxes1, boxes2):
    """
    Generalized IoU from https://giou.stanford.edu/

    The boxes should be in [x0, y0, x1, y1] format
    Returns a [N, M] pairwise matrix, where N = len(boxes1)
    and M = len(boxes2)
    """
    # degenerate boxes gives inf / nan results
    # so do an early check

    assert (boxes1[:, 3:] >= boxes1[:, :3]).all()
    assert (boxes2[:, 3:] >= boxes2[:, :3]).all()
    iou, union = _iou3d_par(boxes1, boxes2)

    lt = torch.min(boxes1[:, None, :3], boxes2[:, :3])
    rb = torch.max(boxes1[:, None, 3:], boxes2[:, 3:])

    wh = (rb - lt).clamp(min=0)  # [N,M,3]
    volume = wh[:, :, 0] * wh[:, :, 1] * wh[:, :, 2]

    return iou - (volume - union) / volume


class SigmoidFocalClassificationLoss(nn.Module):
    """
    Sigmoid focal cross entropy loss.

    This class is taken from Group-Free code.
    """

    def __init__(self, gamma=2.0, alpha=0.25):
        """
        Args:
            gamma: Weighting parameter for hard and easy examples.
            alpha: Weighting parameter for positive and negative examples.
        """
        super().__init__()
        self.alpha = alpha
        self.gamma = gamma

    @staticmethod
    def sigmoid_cross_entropy_with_logits(input, target):
        """
        PyTorch Implementation for tf.nn.sigmoid_cross_entropy_with_logits:
        max(x, 0) - x * z + log(1 + exp(-abs(x))) in

        Args:
            input: (B, #proposals, #classes) float tensor.
                Predicted logits for each class
            target: (B, #proposals, #classes) float tensor.
                One-hot encoded classification targets

        Returns:
            loss: (B, #proposals, #classes) float tensor.
                Sigmoid cross entropy loss without reduction
        """
        loss = (
                torch.clamp(input, min=0) - input * target
                + torch.log1p(torch.exp(-torch.abs(input)))
        )
        return loss

    def forward(self, input, target, weights):
        """
        Args:
            input: (B, #proposals, #classes) float tensor.
                Predicted logits for each class
            target: (B, #proposals, #classes) float tensor.
                One-hot encoded classification targets
            weights: (B, #proposals) float tensor.
                Anchor-wise weights.

        Returns:
            weighted_loss: (B, #proposals, #classes) float tensor
        """
        pred_sigmoid = torch.sigmoid(input)
        alpha_weight = target * self.alpha + (1 - target) * (1 - self.alpha)
        pt = target * (1.0 - pred_sigmoid) + (1.0 - target) * pred_sigmoid
        focal_weight = alpha_weight * torch.pow(pt, self.gamma)

        bce_loss = self.sigmoid_cross_entropy_with_logits(input, target)

        loss = focal_weight * bce_loss
        loss = loss.squeeze(-1)

        assert weights.shape.__len__() == loss.shape.__len__()

        return loss * weights


def compute_points_obj_cls_loss_hard_topk(end_points, topk):
    box_label_mask = end_points['box_label_mask']
    seed_inds = end_points['seed_inds'].long()  # B, K
    seed_xyz = end_points['seed_xyz']  # B, K, 3
    seeds_obj_cls_logits = end_points['seeds_obj_cls_logits']  # B, 1, K
    gt_center = end_points['center_label'][:, :, :3]  # B, G=132, 3
    gt_size = end_points['size_gts'][:, :, :3]  # B, G, 3
    B = gt_center.shape[0]  # batch size
    K = seed_xyz.shape[1]  # number if points from p++ output  1024
    G = gt_center.shape[1]  # number of gt boxes (with padding) 132

    # Assign each point to a GT object
    point_instance_label = end_points['point_instance_label']  # B, num_points=5000
    obj_assignment = torch.gather(point_instance_label, 1, seed_inds)  # B, K=1024
    obj_assignment[obj_assignment < 0] = G - 1  # bg points to last gt
    obj_assignment_one_hot = torch.zeros((B, K, G)).to(seed_xyz.device)
    obj_assignment_one_hot.scatter_(2, obj_assignment.unsqueeze(-1), 1)

    # Normalized distances of points and gt centroids
    delta_xyz = seed_xyz.unsqueeze(2) - gt_center.unsqueeze(1)  # (B, K, G, 3)
    delta_xyz = delta_xyz / (gt_size.unsqueeze(1) + 1e-6)  # (B, K, G, 3)
    new_dist = torch.sum(delta_xyz ** 2, dim=-1)
    euclidean_dist1 = torch.sqrt(new_dist + 1e-6)  # BxKxG
    euclidean_dist1 = (
            euclidean_dist1 * obj_assignment_one_hot
            + 100 * (1 - obj_assignment_one_hot)
    )  # BxKxG
    euclidean_dist1 = euclidean_dist1.transpose(1, 2).contiguous()

    # Find the points that lie closest to each gt centroid
    topk_inds = (
            torch.topk(euclidean_dist1, topk, largest=False)[1]
            * box_label_mask[:, :, None]
            + (box_label_mask[:, :, None] - 1)
    )  # BxGxtopk
    topk_inds = topk_inds.long()  # BxGxtopk
    topk_inds = topk_inds.view(B, -1).contiguous()  # B, Gxtopk
    batch_inds = torch.arange(B)[:, None].repeat(1, G * topk).to(seed_xyz.device)
    batch_topk_inds = torch.stack([
        batch_inds,
        topk_inds
    ], -1).view(-1, 2).contiguous()

    # Topk points closest to each centroid are marked as true objects
    objectness_label = torch.zeros((B, K + 1)).long().to(seed_xyz.device)
    objectness_label[batch_topk_inds[:, 0], batch_topk_inds[:, 1]] = 1
    objectness_label = objectness_label[:, :K]
    objectness_label_mask = torch.gather(point_instance_label, 1, seed_inds)
    objectness_label[objectness_label_mask < 0] = 0

    # Compute objectness loss
    criterion = SigmoidFocalClassificationLoss()
    cls_weights = (objectness_label >= 0).float()
    cls_normalizer = cls_weights.sum(dim=1, keepdim=True).float()
    cls_weights /= torch.clamp(cls_normalizer, min=1.0)
    cls_loss_src = criterion(
        seeds_obj_cls_logits.view(B, K, 1),
        objectness_label.unsqueeze(-1),
        weights=cls_weights
    )
    objectness_loss = cls_loss_src.sum() / B

    return objectness_loss


class HungarianMatcher(nn.Module):
    """
    Assign targets to predictions.

    This class is taken from MDETR and is modified for our purposes.

    For efficiency reasons, the [targets don't include the no_object].
    Because of this, in general, there are [more predictions than targets].
    In this case, we do a 1-to-1 matching of the best predictions,
    while the others are un-matched (and thus treated as non-objects).
    """

    def __init__(self, cost_class=1, cost_bbox=5, cost_giou=2,
                 soft_token=False):
        """
        Initialize matcher.

        Args:
            cost_class: relative weight of the classification error
            cost_bbox: relative weight of the L1 bounding box regression error
            cost_giou: relative weight of the giou loss of the bounding box
            soft_token: whether to use soft-token prediction
        """
        super().__init__()
        self.cost_class = cost_class
        self.cost_bbox = cost_bbox
        self.cost_giou = cost_giou
        self.cost_masks = 0.0002  # mask weight
        assert cost_class != 0 or cost_bbox != 0 or cost_giou != 0
        self.soft_token = soft_token

    @torch.no_grad()
    def forward(self, outputs, targets):
        """
        Perform the matching.

        Args:
            outputs: This is a dict that contains at least these entries:
                "pred_logits" (tensor): [batch_size, num_queries, num_classes]
                "pred_boxes" (tensor): [batch_size, num_queries, 6], cxcyczwhd
            targets: list (len(targets) = batch_size) of dict:
                "labels" (tensor): [num_target_boxes]
                    (where num_target_boxes is the no. of ground-truth objects)
                "boxes" (tensor): [num_target_boxes, 6], cxcyczwhd
                "positive_map" (tensor): [num_target_boxes, 256]

        Returns:
            A list of size batch_size, containing tuples of (index_i, index_j):
                - index_i is the indices of the selected predictions
                - index_j is the indices of the corresponding selected targets
            For each batch element, it holds:
            len(index_i) = len(index_j) = min(num_queries, num_target_boxes)
        """
        # Notation: {B: batch_size, Q: num_queries, C: num_classes}
        bs, num_queries = outputs["pred_logits"].shape[:2]  # Q: num_queries = 256

        # We flatten to compute the cost matrices in a batch
        out_prob = outputs["pred_logits"].flatten(0, 1).softmax(-1)  # [B*Q, C=256]
        out_bbox = outputs["pred_boxes"].flatten(0, 1)  # [B*Q, 6]

        cost_masks = 0.0
        if "pred_masks" in outputs:
            cost_masks = []
            out_masks = None
            tgt_masks = torch.cat([v["masks"] for v in targets])  # (B, 50000)
            for idx in range(len(outputs["pred_masks"])):
                out_mask = outputs["pred_masks"][idx].squeeze(0)  # [Q, super_num]
                out_mask = (out_mask > 0).float()  # [Q, super_num]
                superpoint = outputs["superpoints"][idx].unsqueeze(0).expand(out_mask.shape[0], -1)  # (Q, 50000)
                out_mask = torch.gather(out_mask, 1, superpoint)  # (Q, 50000)
                if out_masks == None:
                    out_masks = out_mask
                else:
                    out_masks = torch.cat([out_masks, out_mask], dim=0)  # (B*Q, 50000)

            cost_masks = torch.cdist(out_masks, tgt_masks.float(), p=1)  # ([B*Q, 2]) 110 - 2092 * 0.0002

        # Also concat the target labels and boxes
        positive_map = torch.cat([t["positive_map"] for t in targets])  # (B, 256)
        tgt_ids = torch.cat([v["labels"] for v in targets])  # (B)
        tgt_bbox = torch.cat([v["boxes"] for v in targets])  # (B, 6)

        if self.soft_token:
            # pad if necessary
            if out_prob.shape[-1] != positive_map.shape[-1]:
                positive_map = positive_map[..., :out_prob.shape[-1]]
            cost_class = -torch.matmul(out_prob, positive_map.transpose(0, 1))  # (256, 1)
        else:
            # Compute the classification cost.
            # Contrary to the loss, we don't use the NLL,
            # but approximate it in 1 - proba[target class].
            # The 1 is a constant that doesn't change the matching,
            # it can be ommitted. DETR
            # out_prob = out_prob * out_objectness.view(-1, 1)
            cost_class = -out_prob[:, tgt_ids]

        # Compute the L1 cost between boxes
        cost_bbox = torch.cdist(out_bbox, tgt_bbox, p=1)  # ([B*Q, 2])  0.08 - 15.3 * 1

        # Compute the giou cost betwen boxes
        cost_giou = -generalized_box_iou3d(  # ([B*Q, 2])  -0.8 - 0.98 * 2
            box_cxcyczwhd_to_xyzxyz(out_bbox),
            box_cxcyczwhd_to_xyzxyz(tgt_bbox)
        )

        # Final cost matrix
        C = (
                self.cost_bbox * cost_bbox  # 0 *
                + self.cost_class * cost_class  # 1 * ([B*Q, 2])
                + self.cost_giou * cost_giou  # 2 * ([B*Q, 2])
                + self.cost_masks * cost_masks
        ).view(bs, num_queries, -1).cpu()

        sizes = [len(v["boxes"]) for v in targets]
        indices = [
            linear_sum_assignment(c[i])
            for i, c in enumerate(C.split(sizes, -1))
        ]
        return [
            (
                torch.as_tensor(i, dtype=torch.int64),  # matched pred boxes
                torch.as_tensor(j, dtype=torch.int64)  # corresponding gt boxes
            )
            for i, j in indices
        ]


def dice_loss(inputs, targets, num_boxes):
    """
    Compute the DICE loss, similar to generalized IOU for masks
    Args:
        inputs: A float tensor of arbitrary shape.
                The predictions for each example.
        targets: A float tensor with the same shape as inputs. Stores the binary
                 classification label for each element in inputs
                (0 for the negative class and 1 for the positive class).
    """
    inputs = inputs.sigmoid()
    inputs = inputs.flatten(1)
    numerator = 2 * (inputs * targets).sum(1)
    denominator = inputs.sum(-1) + targets.sum(-1)
    loss = 1 - (numerator + 1) / (denominator + 1)
    return loss.sum() / num_boxes


def sigmoid_focal_loss(inputs, targets, num_boxes, weight=1, alpha: float = 0.25, gamma: float = 2):
    """
    Loss used in RetinaNet for dense detection: https://arxiv.org/abs/1708.02002.
    Args:
        inputs: A float tensor of arbitrary shape.
                The predictions for each example.
        targets: A float tensor with the same shape as inputs. Stores the binary
                 classification label for each element in inputs
                (0 for the negative class and 1 for the positive class).
        alpha: (optional) Weighting factor in range (0,1) to balance
                positive vs negative examples. Default = -1 (no weighting).
        gamma: Exponent of the modulating factor (1 - p_t) to
               balance easy vs hard examples.
    Returns:
        Loss tensor
    """
    prob = inputs.sigmoid()
    ce_loss = F.binary_cross_entropy_with_logits(inputs, targets, reduction="none")
    ce_loss = ce_loss * weight
    p_t = prob * targets + (1 - prob) * (1 - targets)
    loss = ce_loss * ((1 - p_t) ** gamma)

    if alpha >= 0:
        alpha_t = alpha * targets + (1 - alpha) * (1 - targets)
        loss = alpha_t * loss

    return loss.mean(1).sum() / num_boxes


# BRIEF Compute loss
class SetCriterion(nn.Module):
    def __init__(self, matcher, losses={}, eos_coef=0.1, temperature=0.07):
        """
        Parameters:
            matcher: module that matches targets and proposals
            losses: list of all the losses to be applied
            eos_coef: weight of the no-object category
            temperature: used to sharpen the contrastive logits
        """
        super().__init__()
        self.matcher = matcher
        self.eos_coef = eos_coef  # 0.1
        self.losses = losses
        self.temperature = temperature

    def get_loss_by_prefix(self, prefix):
        """Get the specific loss function based on the prefix."""
        if prefix == '0head_':
            return self.loss_pos_align_first
        elif prefix == '1head_':
            return self.loss_pos_align_second
        elif prefix == '2head_':
            return self.loss_pos_align_third
        elif prefix == '3head_':
            return self.loss_pos_align_fourth
        else:
            return self.loss_pos_align

    #####################################
    # BRIEF dense position-aligned loss #
    #####################################
    def loss_pos_align(self, outputs, targets, indices, num_boxes, auxi_indices):
        logits = outputs["pred_logits"].log_softmax(-1)

        # text position label
        positive_map = torch.cat([t["positive_map"] for t in targets])  # main object
        modify_positive_map = torch.cat([t["modify_positive_map"] for t in targets])  # attribute(modify)
        pron_positive_map = torch.cat([t["pron_positive_map"] for t in targets])  # pron
        other_entity_map = torch.cat([t["other_entity_map"] for t in targets])  # other(auxi)
        rel_positive_map = torch.cat([t["rel_positive_map"] for t in targets])  # relation

        # Trick to get target indices across batches
        src_idx = self._get_src_permutation_idx(indices)
        tgt_idx = []
        offset = 0
        for i, (_, tgt) in enumerate(indices):
            tgt_idx.append(tgt + offset)
            offset += len(targets[i]["boxes"])
        tgt_idx = torch.cat(tgt_idx)

        # NOTE constract the position label of the target object
        tgt_pos = positive_map[tgt_idx]
        mod_pos = modify_positive_map[tgt_idx]
        pron_pos = pron_positive_map[tgt_idx]
        other_pos = other_entity_map[tgt_idx]
        rel_pos = rel_positive_map[tgt_idx]
        # TODO ScanRefer & NR3D
        tgt_weight_pos = tgt_pos * 0.6 + mod_pos * 0.2 + pron_pos * 0.2 + rel_pos * 0.1
        # TODO SR3D (5:1:1:1)/8 = 0.625: 0.125: 0.125: 0.125
        if outputs["language_dataset"][0] == "sr3d":
            tgt_weight_pos = tgt_pos * 0.625 + mod_pos * 0.125 + pron_pos * 0.125 + rel_pos * 0.125

        # mask, keep the positive term
        pos_mask = tgt_pos + mod_pos + pron_pos + rel_pos + other_pos
        target_mask = torch.zeros_like(logits)
        target_mask[:, :, -1] = 1
        target_mask[src_idx] = pos_mask

        target_sim = torch.zeros_like(logits)
        target_sim[:, :, -1] = 1
        target_sim[src_idx] = tgt_weight_pos

        # STEP Compute entropy
        entropy = torch.log(target_sim + 1e-6) * target_sim
        loss_ce = (entropy - logits * target_sim).sum(-1)

        # Weight less 'no_object'
        eos_coef = torch.full(
            loss_ce.shape, self.eos_coef,
            device=target_sim.device
        )
        eos_coef[src_idx] = 1
        loss_ce = loss_ce * eos_coef

        loss_ce = loss_ce.sum() / num_boxes

        losses = {"loss_ce": loss_ce}

        return losses

    def loss_pos_align_first(self, outputs, targets, indices, num_boxes, auxi_indices):
        logits = outputs["pred_logits"].log_softmax(-1)  # [B,K,L] P_obj

        # text position label
        positive_map = torch.cat([t["positive_map"] for t in targets])  # main object
        other_entity_map = torch.cat([t["other_entity_map"] for t in targets])  # other(auxi)

        # Trick to get target indices across batches
        src_idx = self._get_src_permutation_idx(indices)
        tgt_idx = []
        offset = 0
        for i, (_, tgt) in enumerate(indices):
            tgt_idx.append(tgt + offset)
            offset += len(targets[i]["boxes"])
        tgt_idx = torch.cat(tgt_idx)

        # NOTE constract the position label of the target object
        tgt_pos = positive_map[tgt_idx]
        other_pos = other_entity_map[tgt_idx]
        # TODO ScanRefer & NR3D
        tgt_weight_pos = tgt_pos
        # TODO SR3D
        if outputs["language_dataset"][0] == "sr3d":
            tgt_weight_pos = tgt_pos

        # mask, keep the positive term
        pos_mask = tgt_pos + other_pos
        target_mask = torch.zeros_like(logits)
        target_mask[:, :, -1] = 1
        target_mask[src_idx] = pos_mask

        target_sim = torch.zeros_like(logits)
        target_sim[:, :, -1] = 1
        target_sim[src_idx] = tgt_weight_pos

        # STEP Compute entropy
        entropy = torch.log(target_sim + 1e-6) * target_sim
        loss_ce = (entropy - logits * target_sim).sum(-1)

        # Weight less 'no_object'
        eos_coef = torch.full(
            loss_ce.shape, self.eos_coef,
            device=target_sim.device
        )
        eos_coef[src_idx] = 1
        loss_ce = loss_ce * eos_coef

        loss_ce = loss_ce.sum() / num_boxes

        losses = {"loss_ce": loss_ce}

        return losses

    def loss_pos_align_second(self, outputs, targets, indices, num_boxes, auxi_indices):
        logits = outputs["pred_logits"].log_softmax(-1)  # [B,K,L] predicted labels

        # text position label
        positive_map = torch.cat([t["positive_map"] for t in targets])  # main object [43,256]
        modify_positive_map = torch.cat([t["modify_positive_map"] for t in targets])  # attribute(modify)
        other_entity_map = torch.cat([t["other_entity_map"] for t in targets])  # other(auxi)

        # Trick to get target indices across batches
        src_idx = self._get_src_permutation_idx(indices)

        tgt_idx = []
        offset = 0
        for i, (_, tgt) in enumerate(indices):
            tgt_idx.append(tgt + offset)
            offset += len(targets[i]["boxes"])
        tgt_idx = torch.cat(tgt_idx)  # [43]
        # NOTE constract the position label of the target object
        tgt_pos = positive_map[tgt_idx]  # [43,256]
        mod_pos = modify_positive_map[tgt_idx]
        other_pos = other_entity_map[tgt_idx]

        # TODO ScanRefer & NR3D
        tgt_weight_pos = tgt_pos * 0.6 + mod_pos * 0.2
        # TODO SR3D (5:1:1:1)/8 = 0.625: 0.125: 0.125: 0.125
        if outputs["language_dataset"][0] == "sr3d":
            tgt_weight_pos = tgt_pos * 0.625 + mod_pos * 0.125

        # mask, keep the positive term
        pos_mask = tgt_pos + mod_pos + other_pos
        target_mask = torch.zeros_like(logits)
        target_mask[:, :, -1] = 1
        target_mask[src_idx] = pos_mask

        target_sim = torch.zeros_like(logits)  # [B,K,L]
        target_sim[:, :, -1] = 1  # set the last position to 1
        target_sim[src_idx] = tgt_weight_pos

        # STEP Compute entropy
        entropy = torch.log(target_sim + 1e-6) * target_sim
        loss_ce = (entropy - logits * target_sim).sum(-1)

        # Weight less 'no_object'
        eos_coef = torch.full(
            loss_ce.shape, self.eos_coef,
            device=target_sim.device
        )
        eos_coef[src_idx] = 1
        loss_ce = loss_ce * eos_coef

        loss_ce = loss_ce.sum() / num_boxes

        losses = {"loss_ce": loss_ce}

        return losses

    def loss_pos_align_third(self, outputs, targets, indices, num_boxes, auxi_indices):
        logits = outputs["pred_logits"].log_softmax(-1)  # [B,K,L] predicted labels

        # text position label
        positive_map = torch.cat([t["positive_map"] for t in targets])  # main object [43,256]
        modify_positive_map = torch.cat([t["modify_positive_map"] for t in targets])  # attribute(modify)
        pron_positive_map = torch.cat([t["pron_positive_map"] for t in targets])  # pron
        other_entity_map = torch.cat([t["other_entity_map"] for t in targets])  # other(auxi)

        # Trick to get target indices across batches
        src_idx = self._get_src_permutation_idx(indices)

        tgt_idx = []
        offset = 0
        for i, (_, tgt) in enumerate(indices):
            tgt_idx.append(tgt + offset)
            offset += len(targets[i]["boxes"])
        tgt_idx = torch.cat(tgt_idx)  # [43]
        # NOTE constract the position label of the target object
        tgt_pos = positive_map[tgt_idx]  # [43,256]
        mod_pos = modify_positive_map[tgt_idx]
        pron_pos = pron_positive_map[tgt_idx]
        other_pos = other_entity_map[tgt_idx]

        # TODO ScanRefer & NR3D
        tgt_weight_pos = tgt_pos * 0.6 + mod_pos * 0.2 + pron_pos * 0.2
        # TODO SR3D (5:1:1:1)/8 = 0.625: 0.125: 0.125: 0.125
        if outputs["language_dataset"][0] == "sr3d":
            tgt_weight_pos = tgt_pos * 0.625 + mod_pos * 0.125 + pron_pos * 0.125

        # mask, keep the positive term
        pos_mask = tgt_pos + mod_pos + pron_pos + other_pos
        target_mask = torch.zeros_like(logits)
        target_mask[:, :, -1] = 1
        target_mask[src_idx] = pos_mask

        target_sim = torch.zeros_like(logits)  # [B,K,L]
        target_sim[:, :, -1] = 1  # set the last position to 1
        target_sim[src_idx] = tgt_weight_pos

        # STEP Compute entropy
        entropy = torch.log(target_sim + 1e-6) * target_sim
        loss_ce = (entropy - logits * target_sim).sum(-1)

        # Weight less 'no_object'
        eos_coef = torch.full(
            loss_ce.shape, self.eos_coef,
            device=target_sim.device
        )
        eos_coef[src_idx] = 1
        loss_ce = loss_ce * eos_coef

        loss_ce = loss_ce.sum() / num_boxes

        losses = {"loss_ce": loss_ce}

        return losses

    def loss_pos_align_fourth(self, outputs, targets, indices, num_boxes, auxi_indices):
        logits = outputs["pred_logits"].log_softmax(-1)  # [B,K,L] predicted labels

        # text position label
        positive_map = torch.cat([t["positive_map"] for t in targets])  # main object [43,256]
        modify_positive_map = torch.cat([t["modify_positive_map"] for t in targets])  # attribute(modify)
        other_entity_map = torch.cat([t["other_entity_map"] for t in targets])  # other(auxi)
        rel_positive_map = torch.cat([t["rel_positive_map"] for t in targets])  # relation

        # Trick to get target indices across batches
        src_idx = self._get_src_permutation_idx(indices)

        tgt_idx = []
        offset = 0
        for i, (_, tgt) in enumerate(indices):
            tgt_idx.append(tgt + offset)
            offset += len(targets[i]["boxes"])
        tgt_idx = torch.cat(tgt_idx)  # [43]
        # NOTE constract the position label of the target object
        tgt_pos = positive_map[tgt_idx]  # [43,256]
        mod_pos = modify_positive_map[tgt_idx]
        other_pos = other_entity_map[tgt_idx]
        rel_pos = rel_positive_map[tgt_idx]

        # TODO ScanRefer & NR3D
        tgt_weight_pos = tgt_pos * 0.6 + mod_pos * 0.2 + rel_pos * 0.1
        # TODO SR3D (5:1:1:1)/8 = 0.625: 0.125: 0.125: 0.125
        if outputs["language_dataset"][0] == "sr3d":
            tgt_weight_pos = tgt_pos * 0.625 + mod_pos * 0.125 + rel_pos * 0.125

        # mask, keep the positive term
        pos_mask = tgt_pos + mod_pos + rel_pos + other_pos
        target_mask = torch.zeros_like(logits)
        target_mask[:, :, -1] = 1
        target_mask[src_idx] = pos_mask

        target_sim = torch.zeros_like(logits)  # [B,K,L]
        target_sim[:, :, -1] = 1  # set the last position to 1
        target_sim[src_idx] = tgt_weight_pos

        # STEP Compute entropy
        entropy = torch.log(target_sim + 1e-6) * target_sim
        loss_ce = (entropy - logits * target_sim).sum(-1)

        # Weight less 'no_object'
        eos_coef = torch.full(
            loss_ce.shape, self.eos_coef,
            device=target_sim.device
        )
        eos_coef[src_idx] = 1
        loss_ce = loss_ce * eos_coef

        loss_ce = loss_ce.sum() / num_boxes

        losses = {"loss_ce": loss_ce}

        return losses

    # BRIEF object detection loss.
    def loss_boxes(self, outputs, targets, indices, num_boxes, auxi_indices):
        """Compute bbox losses."""
        assert 'pred_boxes' in outputs
        idx = self._get_src_permutation_idx(indices)
        src_boxes = outputs['pred_boxes'][idx]
        target_boxes = torch.cat([
            t['boxes'][i] for t, (_, i) in zip(targets, indices)
        ], dim=0)

        loss_bbox = (
                F.l1_loss(
                    src_boxes[..., :3], target_boxes[..., :3],
                    reduction='none'
                )
                + 0.2 * F.l1_loss(
            src_boxes[..., 3:], target_boxes[..., 3:],
            reduction='none'
        )
        )
        losses = {}
        loss_giou = 1 - torch.diag(generalized_box_iou3d(
            box_cxcyczwhd_to_xyzxyz(src_boxes),
            box_cxcyczwhd_to_xyzxyz(target_boxes)))

        losses['loss_bbox'] = loss_bbox.sum() / num_boxes
        losses['loss_giou'] = loss_giou.sum() / num_boxes
        return losses

    # BRIEF object detection loss.
    def loss_masks(self, outputs, targets, indices, num_boxes, auxi_indices):
        """Compute mask losses."""
        losses = {}
        focal = 0.0
        dice = 0.0
        sp_focal = 0.0
        sp_dice = 0.0
        adaptive_weight_focal = 0.0
        adaptive_weight_dice = 0.0
        corresponding_focal = 0.0
        corresponding_dice = 0.0

        if 'pred_masks' in outputs:
            for bs in range(len(outputs['pred_masks'])):
                idx0 = indices[bs][0]  # predicted mask indices
                superpoint = outputs['superpoints'][bs]
                idx1 = indices[bs][1]  # GT mask indices
                target = targets[bs]['masks'][idx1].float()  # [len(indices), 50000] [bs,50000]
                target_masks = scatter_mean(target, superpoint, dim=-1)  # [len(indices), super_num] [bs,super_num]
                target_masks = (target_masks > 0.5).float()

                src_masks = outputs['pred_masks'][bs][0][:target.shape[0]]  # [len(indices), super_num]
                sp_src_masks = outputs['sp_pred_masks'][bs][idx0]

                focal += sigmoid_focal_loss(src_masks, target_masks, num_boxes)
                dice += dice_loss(src_masks, target_masks, num_boxes)

                sp_focal += sigmoid_focal_loss(sp_src_masks, target_masks, num_boxes)
                sp_dice += dice_loss(sp_src_masks, target_masks, num_boxes)

                sp_src_masks_2 = (sp_src_masks > 0.5).float()
                bs_super_xyz = outputs['super_xyz_list'][bs]
                if torch.sum(sp_src_masks_2) > 0:
                    selected_index = torch.nonzero(sp_src_masks_2)[:, 1]
                    selected_xyz = torch.index_select(bs_super_xyz, dim=1, index=selected_index)
                    num_points = selected_xyz.size(1)
                else:
                    num_points = 0

                if num_points > 1:
                    selected_xyz = selected_xyz.unsqueeze(2)

                    distances = torch.norm(selected_xyz - selected_xyz.permute(0, 2, 1, 3), dim=3)
                    distance_sum = distances.sum(dim=(1, 2))
                    distance_mean = distance_sum / ((num_points - 1) * num_points)
                else:
                    distance_mean = bs_super_xyz.new_tensor(0.0)

                dice_weight = 1.0 / (1 + distance_mean)
                corresponding_dice += dice_weight * dice_loss(src_masks, sp_src_masks_2, num_boxes)
                # Calculate pixel-level weight for BCE loss
                u1 = sp_src_masks.new_tensor(0.5)
                sigma1 = sp_src_masks.new_tensor(0.1)
                left = 1.0 / torch.sqrt(sp_src_masks.new_tensor(2.0 * math.pi) * sigma1)
                right = torch.exp(-((sp_src_masks.sigmoid().detach() - u1) ** 2) / (2.0 * sigma1))
                weight = 2.0 - left * right
                corresponding_focal += sigmoid_focal_loss(src_masks, sp_src_masks_2, num_boxes, weight)

                adaptive_weight = outputs['adaptive_weights'][bs]
                adaptive_weight_mask = adaptive_weight * src_masks + (1 - adaptive_weight) * sp_src_masks

                adaptive_weight_focal += sigmoid_focal_loss(adaptive_weight_mask, target_masks, num_boxes)
                adaptive_weight_dice += dice_loss(adaptive_weight_mask, target_masks, num_boxes)

        losses = {
            "loss_mask": focal,
            "loss_dice": dice,
            "sp_loss_mask": sp_focal,
            "sp_loss_dice": sp_dice,
            "corresponding_loss_mask": corresponding_focal,
            "corresponding_loss_dice": corresponding_dice,
            "adaptive_weight_loss_mask": adaptive_weight_focal,
            "adaptive_weight_loss_dice": adaptive_weight_dice,

        }

        return losses

    ############################
    # BRIEF semantic alignment #
    ############################
    def get_loss_sem_align_by_prefix(self, prefix):
        """Get the specific loss function based on the prefix."""
        if prefix == '0head_':
            return self.loss_sem_align_first
        elif prefix == '1head_':
            return self.loss_sem_align_second
        elif prefix == '2head_':
            return self.loss_sem_align_third
        elif prefix == '3head_':
            return self.loss_sem_align_fourth
        else:
            return self.loss_sem_align

    def loss_sem_align(self, outputs, targets, indices, num_boxes, auxi_indices):
        tokenized = outputs["tokenized"]

        # step 1. Contrastive logits
        norm_text_emb = outputs["proj_tokens"]  # B, num_tokens=L, dim=64
        norm_img_emb = outputs["proj_queries"]  # B, num_queries=256, dim=64
        logits = (
                torch.matmul(norm_img_emb, norm_text_emb.transpose(-1, -2))
                / self.temperature
        )  # [[B, num_queries, num_tokens]

        # step 2. positive map
        # construct a map such that positive_map[k, i, j] = True
        # iff query i is associated to token j in batch item k
        positive_map = torch.zeros(logits.shape, device=logits.device)  # ([B, 256, L])
        # handle 'not mentioned'
        inds = tokenized['attention_mask'].sum(1) - 1  # attention_mask
        positive_map[torch.arange(len(inds)), :, inds] = 0.5
        positive_map[torch.arange(len(inds)), :, inds - 1] = 0.5  # write the token sequence
        # handle true mentions
        pmap = torch.cat([
            t['positive_map'][i] for t, (_, i) in zip(targets, indices)
        ], dim=0)[..., :logits.shape[-1]]
        idx = self._get_src_permutation_idx(indices)
        positive_map[idx] = pmap
        positive_map = positive_map > 0  # obtain the true positive map from GT

        modi_positive_map = torch.zeros(logits.shape, device=logits.device)
        pron_positive_map = torch.zeros(logits.shape, device=logits.device)
        other_positive_map = torch.zeros(logits.shape, device=logits.device)
        rel_positive_map = torch.zeros(logits.shape, device=logits.device)
        # [positive, 256] --> [positive, L]
        pmap_modi = torch.cat([
            t['modify_positive_map'][i] for t, (_, i) in zip(targets, indices)
        ], dim=0)[..., :logits.shape[-1]]
        pmap_pron = torch.cat([
            t['pron_positive_map'][i] for t, (_, i) in zip(targets, indices)
        ], dim=0)[..., :logits.shape[-1]]
        pmap_other = torch.cat([
            t['other_entity_map'][i] for t, (_, i) in zip(targets, indices)
        ], dim=0)[..., :logits.shape[-1]]
        pmap_rel = torch.cat([
            t['rel_positive_map'][i] for t, (_, i) in zip(targets, indices)
        ], dim=0)[..., :logits.shape[-1]]
        modi_positive_map[idx] = pmap_modi
        pron_positive_map[idx] = pmap_pron
        other_positive_map[idx] = pmap_other
        rel_positive_map[idx] = pmap_rel

        # step object mask
        # Mask for matches <> 'not mentioned'
        mask = torch.full(
            logits.shape[:2],
            self.eos_coef,
            dtype=torch.float32, device=logits.device
        )
        mask[idx] = 1.0

        # step text mask
        # Token mask for matches <> 'not mentioned'
        tmask = torch.full(
            (len(logits), logits.shape[-1]),
            self.eos_coef,
            dtype=torch.float32, device=logits.device
        )  # [B, L]
        tmask[torch.arange(len(inds)), inds] = 1.0

        # Positive logits are those who correspond to a match
        positive_logits = -logits.masked_fill(~positive_map, 0)
        negative_logits = logits
        other_entity_neg_term = negative_logits.masked_fill(~(other_positive_map > 0), 0)

        modi_positive_logits = -logits.masked_fill(~(modi_positive_map > 0), 0)
        pron_positive_logits = -logits.masked_fill(~(pron_positive_map > 0), 0)
        rel_positive_logits = -logits.masked_fill(~(rel_positive_map > 0), 0)

        pos_modi_term = modi_positive_logits.sum(2)
        pos_pron_term = pron_positive_logits.sum(2)
        pos_rel_term = rel_positive_logits.sum(2)

        # number of the token
        nb_modi_pos_token = (modi_positive_map > 0).sum(2) + 1e-6
        nb_pron_pos_token = (pron_positive_map > 0).sum(2) + 1e-6
        nb_rel_pos_token = (rel_positive_map > 0).sum(2) + 1e-6

        ###############################
        # NOTE loss1: object --> text #
        ###############################
        boxes_with_pos = positive_map.any(2)
        pos_term = positive_logits.sum(2)
        # note negative term
        neg_term = (negative_logits + other_entity_neg_term).logsumexp(2)
        nb_pos_token = positive_map.sum(2) + 1e-6
        entropy = -torch.log(nb_pos_token + 1e-6) / nb_pos_token
        box_to_token_loss_ = (
                pos_term / nb_pos_token \
                + 0.2 * pos_modi_term / nb_modi_pos_token \
                + 0.2 * pos_pron_term / nb_pron_pos_token \
                + 0.1 * pos_rel_term / nb_rel_pos_token \
                + neg_term
        ).masked_fill(~boxes_with_pos, 0)
        box_to_token_loss = (box_to_token_loss_ * mask).sum()

        ###############################
        # NOTE loss2: text --> object #
        ###############################
        tokens_with_pos = (
                    positive_map + (modi_positive_map > 0) + (pron_positive_map > 0) + (rel_positive_map > 0)).any(1)
        tmask[positive_map.any(1)] = 1.0
        tmask[(modi_positive_map > 0).any(1)] = 0.2
        tmask[(pron_positive_map > 0).any(1)] = 0.2
        tmask[(rel_positive_map > 0).any(1)] = 0.1
        tmask[torch.arange(len(inds)), inds - 1] = 0.1

        pos_term = positive_logits.sum(1)
        pos_modi_term = modi_positive_logits.sum(1)
        pos_pron_term = pron_positive_logits.sum(1)
        pos_rel_term = rel_positive_logits.sum(1)
        # note
        pos_term = pos_term + pos_modi_term + pos_pron_term + pos_rel_term

        neg_term = negative_logits.logsumexp(1)
        nb_pos_obj = positive_map.sum(1) + modi_positive_map.sum(1) + pron_positive_map.sum(1) \
                     + rel_positive_map.sum(1) + 1e-6

        entropy = -torch.log(nb_pos_obj + 1e-6) / nb_pos_obj
        token_to_box_loss = (
            (entropy + pos_term / nb_pos_obj + neg_term)
        ).masked_fill(~tokens_with_pos, 0)
        token_to_box_loss = (token_to_box_loss * tmask).sum()

        # total loss
        tot_loss = (box_to_token_loss + token_to_box_loss) / 2
        return {"loss_sem_align": tot_loss / num_boxes}

    def loss_sem_align_first(self, outputs, targets, indices, num_boxes, auxi_indices):
        tokenized = outputs["tokenized"]

        # step 1. Contrastive logits
        norm_text_emb = outputs["proj_tokens"]  # B, num_tokens=L, dim=64
        norm_img_emb = outputs["proj_queries"]  # B, num_queries=256, dim=64
        logits = (
                torch.matmul(norm_img_emb, norm_text_emb.transpose(-1, -2))
                / self.temperature
        )  # [[B, num_queries, num_tokens]

        # step 2. positive map
        # construct a map such that positive_map[k, i, j] = True
        # iff query i is associated to token j in batch item k
        positive_map = torch.zeros(logits.shape, device=logits.device)  # ([B, 256, L])
        # handle 'not mentioned'
        inds = tokenized['attention_mask'].sum(1) - 1
        positive_map[torch.arange(len(inds)), :, inds] = 0.5
        positive_map[torch.arange(len(inds)), :, inds - 1] = 0.5
        # handle true mentions
        pmap = torch.cat([
            t['positive_map'][i] for t, (_, i) in zip(targets, indices)
        ], dim=0)[..., :logits.shape[-1]]
        idx = self._get_src_permutation_idx(indices)
        positive_map[idx] = pmap
        positive_map = positive_map > 0

        other_positive_map = torch.zeros(logits.shape, device=logits.device)

        # [positive, 256] --> [positive, L]

        pmap_other = torch.cat([
            t['other_entity_map'][i] for t, (_, i) in zip(targets, indices)
        ], dim=0)[..., :logits.shape[-1]]

        other_positive_map[idx] = pmap_other

        # step object mask
        # Mask for matches <> 'not mentioned'
        mask = torch.full(
            logits.shape[:2],
            self.eos_coef,
            dtype=torch.float32, device=logits.device
        )
        mask[idx] = 1.0

        # step text mask
        # Token mask for matches <> 'not mentioned'
        tmask = torch.full(
            (len(logits), logits.shape[-1]),
            self.eos_coef,
            dtype=torch.float32, device=logits.device
        )  # [B, L]
        tmask[torch.arange(len(inds)), inds] = 1.0

        # Positive logits are those who correspond to a match
        positive_logits = -logits.masked_fill(~positive_map, 0)
        negative_logits = logits
        other_entity_neg_term = negative_logits.masked_fill(~(other_positive_map > 0), 0)
        # number of the token

        ###############################
        # NOTE loss1: object --> text #
        ###############################
        boxes_with_pos = positive_map.any(2)
        pos_term = positive_logits.sum(2)
        # note negative term
        neg_term = (negative_logits + other_entity_neg_term).logsumexp(2)
        nb_pos_token = positive_map.sum(2) + 1e-6
        entropy = -torch.log(nb_pos_token + 1e-6) / nb_pos_token
        box_to_token_loss_ = (
                pos_term / nb_pos_token \
                + neg_term
        ).masked_fill(~boxes_with_pos, 0)
        box_to_token_loss = (box_to_token_loss_ * mask).sum()

        ###############################
        # NOTE loss2: text --> object #
        ###############################
        tokens_with_pos = (positive_map).any(1)
        tmask[positive_map.any(1)] = 1.0
        tmask[torch.arange(len(inds)), inds - 1] = 0.1

        pos_term = positive_logits.sum(1)
        # note
        pos_term = pos_term

        neg_term = negative_logits.logsumexp(1)
        nb_pos_obj = positive_map.sum(1) + 1e-6

        entropy = -torch.log(nb_pos_obj + 1e-6) / nb_pos_obj
        token_to_box_loss = (
            (entropy + pos_term / nb_pos_obj + neg_term)
        ).masked_fill(~tokens_with_pos, 0)
        token_to_box_loss = (token_to_box_loss * tmask).sum()

        # total loss
        tot_loss = (box_to_token_loss + token_to_box_loss) / 2
        return {"loss_sem_align": tot_loss / num_boxes}

    def loss_sem_align_second(self, outputs, targets, indices, num_boxes, auxi_indices):
        tokenized = outputs["tokenized"]

        # step 1. Contrastive logits
        norm_text_emb = outputs["proj_tokens"]  # B, num_tokens=L, dim=64
        norm_img_emb = outputs["proj_queries"]  # B, num_queries=256, dim=64
        logits = (
                torch.matmul(norm_img_emb, norm_text_emb.transpose(-1, -2))
                / self.temperature
        )  # [[B, num_queries, num_tokens]

        # step 2. positive map
        # construct a map such that positive_map[k, i, j] = True
        # iff query i is associated to token j in batch item k
        positive_map = torch.zeros(logits.shape, device=logits.device)  # ([B, 256, L])
        # handle 'not mentioned'
        inds = tokenized['attention_mask'].sum(1) - 1
        positive_map[torch.arange(len(inds)), :, inds] = 0.5
        positive_map[torch.arange(len(inds)), :, inds - 1] = 0.5
        # handle true mentions
        pmap = torch.cat([
            t['positive_map'][i] for t, (_, i) in zip(targets, indices)
        ], dim=0)[..., :logits.shape[-1]]
        idx = self._get_src_permutation_idx(indices)
        positive_map[idx] = pmap
        positive_map = positive_map > 0

        modi_positive_map = torch.zeros(logits.shape, device=logits.device)
        other_positive_map = torch.zeros(logits.shape, device=logits.device)

        # [positive, 256] --> [positive, L]
        pmap_modi = torch.cat([
            t['modify_positive_map'][i] for t, (_, i) in zip(targets, indices)
        ], dim=0)[..., :logits.shape[-1]]
        pmap_other = torch.cat([
            t['other_entity_map'][i] for t, (_, i) in zip(targets, indices)
        ], dim=0)[..., :logits.shape[-1]]
        modi_positive_map[idx] = pmap_modi
        other_positive_map[idx] = pmap_other

        # step object mask
        # Mask for matches <> 'not mentioned'
        mask = torch.full(
            logits.shape[:2],
            self.eos_coef,
            dtype=torch.float32, device=logits.device
        )
        mask[idx] = 1.0

        # step text mask
        # Token mask for matches <> 'not mentioned'
        tmask = torch.full(
            (len(logits), logits.shape[-1]),
            self.eos_coef,
            dtype=torch.float32, device=logits.device
        )  # [B, L]
        tmask[torch.arange(len(inds)), inds] = 1.0

        # Positive logits are those who correspond to a match
        positive_logits = -logits.masked_fill(~positive_map, 0)
        negative_logits = logits
        other_entity_neg_term = negative_logits.masked_fill(~(other_positive_map > 0), 0)

        modi_positive_logits = -logits.masked_fill(~(modi_positive_map > 0), 0)

        pos_modi_term = modi_positive_logits.sum(2)

        # number of the token
        nb_modi_pos_token = (modi_positive_map > 0).sum(2) + 1e-6

        ###############################
        # NOTE loss1: object --> text #
        ###############################
        boxes_with_pos = positive_map.any(2)
        pos_term = positive_logits.sum(2)
        # note negative term
        neg_term = (negative_logits + other_entity_neg_term).logsumexp(2)
        nb_pos_token = positive_map.sum(2) + 1e-6
        entropy = -torch.log(nb_pos_token + 1e-6) / nb_pos_token
        box_to_token_loss_ = (
                pos_term / nb_pos_token \
                + 0.2 * pos_modi_term / nb_modi_pos_token \
                + neg_term
        ).masked_fill(~boxes_with_pos, 0)
        box_to_token_loss = (box_to_token_loss_ * mask).sum()

        ###############################
        # NOTE loss2: text --> object #
        ###############################
        tokens_with_pos = (positive_map + (modi_positive_map > 0)).any(1)
        tmask[positive_map.any(1)] = 1.0
        tmask[(modi_positive_map > 0).any(1)] = 0.2
        tmask[torch.arange(len(inds)), inds - 1] = 0.1

        pos_term = positive_logits.sum(1)
        pos_modi_term = modi_positive_logits.sum(1)
        # note
        pos_term = pos_term + pos_modi_term

        neg_term = negative_logits.logsumexp(1)
        nb_pos_obj = positive_map.sum(1) + modi_positive_map.sum(1) + 1e-6

        entropy = -torch.log(nb_pos_obj + 1e-6) / nb_pos_obj
        token_to_box_loss = (
            (entropy + pos_term / nb_pos_obj + neg_term)
        ).masked_fill(~tokens_with_pos, 0)
        token_to_box_loss = (token_to_box_loss * tmask).sum()

        # total loss
        tot_loss = (box_to_token_loss + token_to_box_loss) / 2
        return {"loss_sem_align": tot_loss / num_boxes}

    def loss_sem_align_third(self, outputs, targets, indices, num_boxes, auxi_indices):
        tokenized = outputs["tokenized"]

        # step 1. Contrastive logits
        norm_text_emb = outputs["proj_tokens"]  # B, num_tokens=L, dim=64
        norm_img_emb = outputs["proj_queries"]  # B, num_queries=256, dim=64
        logits = (
                torch.matmul(norm_img_emb, norm_text_emb.transpose(-1, -2))
                / self.temperature
        )  # [[B, num_queries, num_tokens]

        # step 2. positive map
        # construct a map such that positive_map[k, i, j] = True
        # iff query i is associated to token j in batch item k
        positive_map = torch.zeros(logits.shape, device=logits.device)  # ([B, 256, L])
        # handle 'not mentioned'
        inds = tokenized['attention_mask'].sum(1) - 1
        positive_map[torch.arange(len(inds)), :, inds] = 0.5
        positive_map[torch.arange(len(inds)), :, inds - 1] = 0.5
        # handle true mentions
        pmap = torch.cat([
            t['positive_map'][i] for t, (_, i) in zip(targets, indices)
        ], dim=0)[..., :logits.shape[-1]]
        idx = self._get_src_permutation_idx(indices)
        positive_map[idx] = pmap
        positive_map = positive_map > 0

        modi_positive_map = torch.zeros(logits.shape, device=logits.device)
        pron_positive_map = torch.zeros(logits.shape, device=logits.device)
        other_positive_map = torch.zeros(logits.shape, device=logits.device)
        # [positive, 256] --> [positive, L]
        pmap_modi = torch.cat([
            t['modify_positive_map'][i] for t, (_, i) in zip(targets, indices)
        ], dim=0)[..., :logits.shape[-1]]
        pmap_pron = torch.cat([
            t['pron_positive_map'][i] for t, (_, i) in zip(targets, indices)
        ], dim=0)[..., :logits.shape[-1]]
        pmap_other = torch.cat([
            t['other_entity_map'][i] for t, (_, i) in zip(targets, indices)
        ], dim=0)[..., :logits.shape[-1]]
        modi_positive_map[idx] = pmap_modi
        pron_positive_map[idx] = pmap_pron
        other_positive_map[idx] = pmap_other

        # step object mask
        # Mask for matches <> 'not mentioned'
        mask = torch.full(
            logits.shape[:2],
            self.eos_coef,
            dtype=torch.float32, device=logits.device
        )
        mask[idx] = 1.0

        # step text mask
        # Token mask for matches <> 'not mentioned'
        tmask = torch.full(
            (len(logits), logits.shape[-1]),
            self.eos_coef,
            dtype=torch.float32, device=logits.device
        )  # [B, L]
        tmask[torch.arange(len(inds)), inds] = 1.0

        # Positive logits are those who correspond to a match
        positive_logits = -logits.masked_fill(~positive_map, 0)
        negative_logits = logits
        other_entity_neg_term = negative_logits.masked_fill(~(other_positive_map > 0), 0)

        modi_positive_logits = -logits.masked_fill(~(modi_positive_map > 0), 0)
        pron_positive_logits = -logits.masked_fill(~(pron_positive_map > 0), 0)

        pos_modi_term = modi_positive_logits.sum(2)
        pos_pron_term = pron_positive_logits.sum(2)

        # number of the token
        nb_modi_pos_token = (modi_positive_map > 0).sum(2) + 1e-6
        nb_pron_pos_token = (pron_positive_map > 0).sum(2) + 1e-6

        ###############################
        # NOTE loss1: object --> text #
        ###############################
        boxes_with_pos = positive_map.any(2)
        pos_term = positive_logits.sum(2)
        # note negative term
        neg_term = (negative_logits + other_entity_neg_term).logsumexp(2)
        nb_pos_token = positive_map.sum(2) + 1e-6
        entropy = -torch.log(nb_pos_token + 1e-6) / nb_pos_token
        box_to_token_loss_ = (
                pos_term / nb_pos_token \
                + 0.2 * pos_modi_term / nb_modi_pos_token \
                + 0.2 * pos_pron_term / nb_pron_pos_token \
                + neg_term
        ).masked_fill(~boxes_with_pos, 0)
        box_to_token_loss = (box_to_token_loss_ * mask).sum()

        ###############################
        # NOTE loss2: text --> object #
        ###############################
        tokens_with_pos = (positive_map + (modi_positive_map > 0) + (pron_positive_map > 0)).any(1)
        tmask[positive_map.any(1)] = 1.0
        tmask[(modi_positive_map > 0).any(1)] = 0.2
        tmask[(pron_positive_map > 0).any(1)] = 0.2
        tmask[torch.arange(len(inds)), inds - 1] = 0.1

        pos_term = positive_logits.sum(1)
        pos_modi_term = modi_positive_logits.sum(1)
        pos_pron_term = pron_positive_logits.sum(1)
        # note
        pos_term = pos_term + pos_modi_term + pos_pron_term

        neg_term = negative_logits.logsumexp(1)
        nb_pos_obj = positive_map.sum(1) + modi_positive_map.sum(1) + pron_positive_map.sum(1) + 1e-6

        entropy = -torch.log(nb_pos_obj + 1e-6) / nb_pos_obj
        token_to_box_loss = (
            (entropy + pos_term / nb_pos_obj + neg_term)
        ).masked_fill(~tokens_with_pos, 0)
        token_to_box_loss = (token_to_box_loss * tmask).sum()

        # total loss
        tot_loss = (box_to_token_loss + token_to_box_loss) / 2
        return {"loss_sem_align": tot_loss / num_boxes}

    def loss_sem_align_fourth(self, outputs, targets, indices, num_boxes, auxi_indices):
        tokenized = outputs["tokenized"]

        # step 1. Contrastive logits
        norm_text_emb = outputs["proj_tokens"]  # B, num_tokens=L, dim=64
        norm_img_emb = outputs["proj_queries"]  # B, num_queries=256, dim=64
        logits = (
                torch.matmul(norm_img_emb, norm_text_emb.transpose(-1, -2))
                / self.temperature
        )  # [[B, num_queries, num_tokens]

        # step 2. positive map
        # construct a map such that positive_map[k, i, j] = True
        # iff query i is associated to token j in batch item k
        positive_map = torch.zeros(logits.shape, device=logits.device)  # ([B, 256, L])
        # handle 'not mentioned'
        inds = tokenized['attention_mask'].sum(1) - 1
        positive_map[torch.arange(len(inds)), :, inds] = 0.5
        positive_map[torch.arange(len(inds)), :, inds - 1] = 0.5
        # handle true mentions
        pmap = torch.cat([
            t['positive_map'][i] for t, (_, i) in zip(targets, indices)
        ], dim=0)[..., :logits.shape[-1]]
        idx = self._get_src_permutation_idx(indices)
        positive_map[idx] = pmap
        positive_map = positive_map > 0

        modi_positive_map = torch.zeros(logits.shape, device=logits.device)
        other_positive_map = torch.zeros(logits.shape, device=logits.device)
        rel_positive_map = torch.zeros(logits.shape, device=logits.device)
        # [positive, 256] --> [positive, L]
        pmap_modi = torch.cat([
            t['modify_positive_map'][i] for t, (_, i) in zip(targets, indices)
        ], dim=0)[..., :logits.shape[-1]]
        pmap_other = torch.cat([
            t['other_entity_map'][i] for t, (_, i) in zip(targets, indices)
        ], dim=0)[..., :logits.shape[-1]]
        pmap_rel = torch.cat([
            t['rel_positive_map'][i] for t, (_, i) in zip(targets, indices)
        ], dim=0)[..., :logits.shape[-1]]
        modi_positive_map[idx] = pmap_modi
        other_positive_map[idx] = pmap_other
        rel_positive_map[idx] = pmap_rel

        # step object mask
        # Mask for matches <> 'not mentioned'
        mask = torch.full(
            logits.shape[:2],
            self.eos_coef,
            dtype=torch.float32, device=logits.device
        )
        mask[idx] = 1.0

        # step text mask
        # Token mask for matches <> 'not mentioned'
        tmask = torch.full(
            (len(logits), logits.shape[-1]),
            self.eos_coef,
            dtype=torch.float32, device=logits.device
        )  # [B, L]
        tmask[torch.arange(len(inds)), inds] = 1.0

        # Positive logits are those who correspond to a match
        positive_logits = -logits.masked_fill(~positive_map, 0)
        negative_logits = logits
        other_entity_neg_term = negative_logits.masked_fill(~(other_positive_map > 0), 0)

        modi_positive_logits = -logits.masked_fill(~(modi_positive_map > 0), 0)
        rel_positive_logits = -logits.masked_fill(~(rel_positive_map > 0), 0)

        pos_modi_term = modi_positive_logits.sum(2)
        pos_rel_term = rel_positive_logits.sum(2)

        # number of the token
        nb_modi_pos_token = (modi_positive_map > 0).sum(2) + 1e-6
        nb_rel_pos_token = (rel_positive_map > 0).sum(2) + 1e-6

        ###############################
        # NOTE loss1: object --> text #
        ###############################
        boxes_with_pos = positive_map.any(2)
        pos_term = positive_logits.sum(2)
        # note negative term
        neg_term = (negative_logits + other_entity_neg_term).logsumexp(2)
        nb_pos_token = positive_map.sum(2) + 1e-6
        entropy = -torch.log(nb_pos_token + 1e-6) / nb_pos_token
        box_to_token_loss_ = (
                pos_term / nb_pos_token \
                + 0.2 * pos_modi_term / nb_modi_pos_token \
                + 0.1 * pos_rel_term / nb_rel_pos_token \
                + neg_term
        ).masked_fill(~boxes_with_pos, 0)
        box_to_token_loss = (box_to_token_loss_ * mask).sum()

        ###############################
        # NOTE loss2: text --> object #
        ###############################
        tokens_with_pos = (positive_map + (modi_positive_map > 0) + (rel_positive_map > 0)).any(1)
        tmask[positive_map.any(1)] = 1.0
        tmask[(modi_positive_map > 0).any(1)] = 0.2
        tmask[(rel_positive_map > 0).any(1)] = 0.1
        tmask[torch.arange(len(inds)), inds - 1] = 0.1

        pos_term = positive_logits.sum(1)
        pos_modi_term = modi_positive_logits.sum(1)
        pos_rel_term = rel_positive_logits.sum(1)
        # note
        pos_term = pos_term + pos_modi_term + pos_rel_term

        neg_term = negative_logits.logsumexp(1)
        nb_pos_obj = positive_map.sum(1) + modi_positive_map.sum(1) + rel_positive_map.sum(1) + 1e-6

        entropy = -torch.log(nb_pos_obj + 1e-6) / nb_pos_obj
        token_to_box_loss = (
            (entropy + pos_term / nb_pos_obj + neg_term)
        ).masked_fill(~tokens_with_pos, 0)
        token_to_box_loss = (token_to_box_loss * tmask).sum()

        # total loss
        tot_loss = (box_to_token_loss + token_to_box_loss) / 2
        return {"loss_sem_align": tot_loss / num_boxes}

    def _get_src_permutation_idx(self, indices):
        # permute predictions following indices
        batch_idx = torch.cat([
            torch.full_like(src, i) for i, (src, _) in enumerate(indices)
        ])
        src_idx = torch.cat([src for (src, _) in indices])
        return batch_idx, src_idx

    def _get_tgt_permutation_idx(self, indices):
        # permute targets following indices
        batch_idx = torch.cat([
            torch.full_like(tgt, i) for i, (_, tgt) in enumerate(indices)
        ])
        tgt_idx = torch.cat([tgt for (_, tgt) in indices])
        return batch_idx, tgt_idx

    # BRIEF get loss.
    def get_loss(self, loss, outputs, targets, indices, num_boxes, auxi_indices, prefix='', **kwargs):
        # loss_map = {
        #     'boxes': self.loss_boxes,      # box loss
        #     'masks': self.loss_masks,      # mask loss
        #     'labels': self.loss_pos_align, # position alignment
        #     'contrastive_align': self.loss_sem_align   # semantic alignment
        # }
        # assert loss in loss_map, f'do you really want to compute {loss} loss?'
        # return loss_map[loss](outputs, targets, indices, num_boxes, auxi_indices, **kwargs)

        if loss == 'labels':
            loss_function = self.get_loss_by_prefix(prefix)
            return loss_function(outputs, targets, indices, num_boxes, auxi_indices, **kwargs)

        if loss == 'contrastive_align':
            loss_function = self.get_loss_sem_align_by_prefix(prefix)
            return loss_function(outputs, targets, indices, num_boxes, auxi_indices, **kwargs)

        if loss == 'boxes':
            loss_function = self.loss_boxes
            return loss_function(outputs, targets, indices, num_boxes, auxi_indices, **kwargs)

        if loss == 'masks':
            loss_function = self.loss_masks
            return loss_function(outputs, targets, indices, num_boxes, auxi_indices, **kwargs)

    def forward(self, outputs, targets, prefix):
        """
        Perform the loss computation.

        Parameters:
             outputs: dict of tensors
             targets: list of dicts, such that len(targets) == batch_size.
        """
        # STEP Retrieve the matching between outputs and targets
        indices = self.matcher(outputs, targets)

        # auxi object
        auxi_target = [
            {
                "labels": targets[b]["labels"],
                "boxes": targets[b]["auxi_box"],
                "positive_map": targets[b]["auxi_entity_positive_map"]
            }
            for b in range(outputs["pred_boxes"].shape[0])
        ]
        # auxi_indices = self.matcher(outputs, auxi_target)
        auxi_indices = None  # avoid bugs
        # auxi_indices = self.matcher(outputs, auxi_target)

        num_boxes = sum(len(inds[1]) for inds in indices)
        num_boxes = torch.as_tensor(
            [num_boxes], dtype=torch.float,
            device=next(iter(outputs.values())).device
        )
        if is_dist_avail_and_initialized():
            torch.distributed.all_reduce(num_boxes)
        num_boxes = torch.clamp(num_boxes, min=1.0)

        # Compute all the requested losses
        losses = {}
        # for loss in self.losses:
        #     losses.update(self.get_loss(
        #         loss, outputs, targets, indices, num_boxes, auxi_indices
        #     ))

        for loss in self.losses:
            losses.update(self.get_loss(
                loss, outputs, targets, indices, num_boxes, auxi_indices, prefix
            ))

        return losses, indices

def safe_detach(tensor):
    """Helper to safely detach tensors for logging to avoid graph breakage."""
    if torch.is_tensor(tensor):
        return tensor.detach()
    return tensor


class DGTLossModule(nn.Module):
    def __init__(
        self,
        num_decoder_layers: int = 6,
        warmup_epochs: int = 10,
        stop_dynamic_epoch: int = 30,
        diff_threshold: float = 0.1,
        momentum: float = 0.8,
        clamp_min: float = 1.0,
        clamp_max: float = 1.2,
        geom_weight: float = 0.05,
        geom_gate_floor: float = 0.0,
        geom_box_thr: float = 0.55,
        geom_mask_thr: float = 0.55,
        geom_agree_thr: float = 0.45,
        geom_cap_ratio: float = 0.05,
        harmonize_weight: float = 0.01,
        harmonize_cap_ratio: float = 0.02,
        gs_start_epoch: int = 60,
        gs_stop_epoch: int = -1,
        geom_start_epoch: int = 60,
        geom_stop_epoch: int = -1,
        start_dynamic_epoch: int = 10,
        update_interval: int = 4,
        ema_alpha: float = 0.2,
        hold_when_stable: bool = True,
        reset_on_stop: bool = True,
        patience: int = 2,
        ratio_diff_threshold: float = 0.1,
        min_epoch_gap: int = 2,
        aux_target_ratio: float = 1.0,
        aux_clamp_min: float = 0.9,
        aux_clamp_max: float = 1.02,
        aux_momentum: float = 0.9,
        main_max_step: float = 0.01,
        aux_max_step: float = 0.01,
        core_residual_scale: float = 1.0,
        aux_residual_scale: float = 1.0,
        gs_residual_scale: float = 1.0,
        geom_residual_scale: float = 1.0,
        enable_dynamic: bool = True,
        enable_geom: bool = True,
    ):
        super().__init__()

        # 1. Empirical baseline values
        self.num_decoder_layers = int(num_decoder_layers)
        self.enable_dynamic = enable_dynamic
        self.enable_geom = enable_geom
        self.enable_quality_stats = False
        self.lambdas = {'bbox': 5.0, 'mask': 10.0}
        self.dynamic_whitelist = {'bbox', 'mask'}

        # 2. DDP accumulators (0:det, 1:seg, 2:aux, 3:core, 4:count)
        self.register_buffer('epoch_acc', torch.zeros(5))

        # 3. History buffers (used to compute slopes)
        self.register_buffer('prev_epoch_losses', torch.zeros(2))

        # 4. Global multipliers
        self.register_buffer('global_multipliers', torch.ones(2))

        # 5. State control
        self.warmup_epochs = warmup_epochs
        self.start_dynamic_epoch = max(0, int(start_dynamic_epoch))
        self.stop_dynamic_epoch = max(self.start_dynamic_epoch + 1, int(stop_dynamic_epoch))
        self.task_to_idx = {'det': 0, 'seg': 1, 'bbox': 0, 'mask': 1}
        self.diff_threshold = diff_threshold
        self.momentum = momentum
        self.clamp_min = min(1.0, max(0.8, float(clamp_min)))
        self.clamp_max = max(1.0, float(clamp_max))
        self.geom_weight = geom_weight
        self.geom_gate_floor = geom_gate_floor
        self.geom_box_thr = geom_box_thr
        self.geom_mask_thr = geom_mask_thr
        self.geom_agree_thr = geom_agree_thr
        self.geom_cap_ratio = max(0.0, float(geom_cap_ratio))
        self.harmonize_weight = max(0.0, float(harmonize_weight))
        self.harmonize_cap_ratio = max(0.0, float(harmonize_cap_ratio))
        self.gs_start_epoch = max(0, int(gs_start_epoch))
        self.gs_stop_epoch = int(gs_stop_epoch)
        self.geom_start_epoch = max(0, int(geom_start_epoch))
        self.geom_stop_epoch = int(geom_stop_epoch)
        self.update_interval = max(1, int(update_interval))
        self.ema_alpha = ema_alpha
        self.hold_when_stable = hold_when_stable

        self.reset_on_stop = reset_on_stop
        self.patience = max(1, int(patience))
        self.ratio_diff_threshold = ratio_diff_threshold
        self.min_epoch_gap = max(0, int(min_epoch_gap))
        self.aux_target_ratio = max(0.1, float(aux_target_ratio))
        self.aux_clamp_min = max(0.1, float(aux_clamp_min))
        self.aux_clamp_max = max(self.aux_clamp_min, float(aux_clamp_max))
        self.aux_momentum = max(0.0, min(0.999, float(aux_momentum)))
        self.main_max_step = max(0.0, float(main_max_step))
        self.aux_max_step = max(0.0, float(aux_max_step))

        self.core_residual_scale = max(0.0, min(1.0, float(core_residual_scale)))
        self.aux_residual_scale = max(0.0, min(1.0, float(aux_residual_scale)))
        self.gs_residual_scale = max(0.0, min(1.0, float(gs_residual_scale)))
        self.geom_residual_scale = max(0.0, min(1.0, float(geom_residual_scale)))

        self.register_buffer('ema_losses', torch.zeros(2))
        self.register_buffer('task_v', torch.zeros(2))
        self.register_buffer('has_reset', torch.zeros(1))
        self.register_buffer('stagnation_counter', torch.zeros(2))
        self.register_buffer('ref_losses', torch.zeros(2))
        self.register_buffer('last_update_epoch', torch.tensor(-1.0))
        self.register_buffer('aux_ratio_ema', torch.tensor(0.0))
        self.register_buffer('aux_multiplier', torch.tensor(1.0))
        self.register_buffer('frozen_main_multipliers', torch.ones(2))
        self.register_buffer('frozen_aux_multiplier', torch.tensor(1.0))
        self.register_buffer('dynamic_ref_ready', torch.zeros(1))
        self.register_buffer('imbalance_confirm_counter', torch.zeros(1))
        self.register_buffer('last_imbalance_sign', torch.zeros(1))
        self.register_buffer('freeze_updates_flag', torch.zeros(1))
        self.register_buffer('val_core_bias', torch.tensor(0.0))
        self.register_buffer('val_gap_ema', torch.tensor(0.0))
        self.register_buffer('val_feedback_ready', torch.zeros(1))

        self.register_buffer('geom_debug_acc', torch.zeros(9))

    def _geom_debug_add(self, idx, value=1.0):
        if hasattr(self, 'geom_debug_acc'):
            self.geom_debug_acc[idx] += float(value)

    def _log_geom_debug(self, epoch):
        if not hasattr(self, 'geom_debug_acc'):
            return
        acc = self.geom_debug_acc.clone()
        if dist.is_initialized():
            dist.all_reduce(acc, op=dist.ReduceOp.SUM)
        if (not is_dist_avail_and_initialized()) or dist.get_rank() == 0:
            total = int(acc[0].item())
            if total > 0:
                msg = (
                    f"[DGTL][GEOM] Epoch {epoch} considered={int(acc[0].item())} "
                    f"called={int(acc[1].item())} skip_missing={int(acc[2].item())} "
                    f"skip_all_pseudo={int(acc[3].item())} skip_disabled={int(acc[4].item())} "
                    f"skip_zeta={int(acc[5].item())} skip_empty={int(acc[6].item())} "
                    f"computed={int(acc[7].item())} gt_anchored={int(acc[8].item())}"
                )
                print(msg)
        self.geom_debug_acc.zero_()

    def update_loss_history(self, loss_dict, epoch):
        """Phase 1: batch accumulation"""
        if not self.enable_dynamic:
            return
        if epoch >= self.stop_dynamic_epoch: return
        for idx, key in enumerate(('det', 'seg', 'aux', 'core')):
            value = loss_dict.get(key, None)
            if value is None and key == 'det':
                value = loss_dict.get('bbox', None)
            if value is None and key == 'seg':
                value = loss_dict.get('mask', None)
            if value is None:
                continue
            if torch.is_tensor(value):
                value_t = value.detach().to(
                    device=self.epoch_acc.device,
                    dtype=self.epoch_acc.dtype
                )
                if value_t.numel() != 1:
                    value_t = value_t.mean()
                else:
                    value_t = value_t.reshape(())
                self.epoch_acc[idx] += value_t
            else:
                self.epoch_acc[idx] += float(value)
        self.epoch_acc[4] += 1.0

    def step_epoch(self, epoch):
        """Phase 2: Epoch settle with contribution balancing w_i=max(exp(-v_i), lambda_i)."""
        self._log_geom_debug(epoch)
        if self.updates_frozen():
            self.epoch_acc.zero_()
            return
        if not self.enable_dynamic:
            return
        if float(self.has_reset.item()) > 0.5 and epoch < self.stop_dynamic_epoch:
            self.has_reset.zero_()
        if epoch >= self.stop_dynamic_epoch:
            if self.reset_on_stop and float(self.has_reset.item()) < 0.5:
                self.global_multipliers.fill_(1.0)
                self.task_v.zero_()
                self.aux_multiplier.fill_(1.0)
                self.dynamic_ref_ready.zero_()
                self.imbalance_confirm_counter.zero_()
                self.last_imbalance_sign.zero_()
                self.has_reset.fill_(1.0)
                if (not is_dist_avail_and_initialized()) or dist.get_rank() == 0:
                    print(
                        f"[DGTL] Epoch {epoch} stop reached -> reset multipliers "
                        f"(BBox={self.global_multipliers[0]:.3f}, "
                        f"Mask={self.global_multipliers[1]:.3f}, "
                        f"Aux={self.aux_multiplier.item():.3f})"
                    )
            self.epoch_acc.zero_()
            return

        # DDP sync
        if dist.is_initialized():
            dist.all_reduce(self.epoch_acc, op=dist.ReduceOp.SUM)

        total = self.epoch_acc[4].item()
        if total < 1: return

        # Average losses for the current epoch
        current_avg = torch.tensor([
            self.epoch_acc[0].item() / total,
            self.epoch_acc[1].item() / total
        ], device=self.epoch_acc.device)
        det_avg = float(current_avg[0].item())
        seg_avg = float(current_avg[1].item())
        aux_avg = max(0.0, self.epoch_acc[2].item() / total)
        core_avg = max(0.0, self.epoch_acc[3].item() / total)
        aux_ratio = aux_avg / max(core_avg, 1e-6)

        # Initialization phase
        if epoch <= self.warmup_epochs:
            self.prev_epoch_losses = current_avg
            self.ema_losses = current_avg
            self.aux_ratio_ema.fill_(float(aux_ratio))
            self.aux_multiplier.fill_(1.0)
            self.dynamic_ref_ready.zero_()
            self.imbalance_confirm_counter.zero_()
            self.last_imbalance_sign.zero_()
            if self.ref_losses.sum().item() == 0:
                self.ref_losses = torch.clamp(current_avg, min=1e-3)
            self.epoch_acc.zero_()
            return

        # EMA smoothing
        if self.ema_losses.sum().item() == 0:
            self.ema_losses = current_avg
        else:
            self.ema_losses = self.ema_alpha * current_avg + (1 - self.ema_alpha) * self.ema_losses

        # Dynamic adjustment start point
        if epoch < self.start_dynamic_epoch:
            self.prev_epoch_losses = self.ema_losses
            self.aux_ratio_ema.fill_(float(aux_ratio))
            self.aux_multiplier.fill_(1.0)
            self.dynamic_ref_ready.zero_()
            self.imbalance_confirm_counter.zero_()
            self.last_imbalance_sign.zero_()
            self.epoch_acc.zero_()
            return

        if float(self.dynamic_ref_ready.item()) < 0.5:
            self.ref_losses = torch.clamp(self.ema_losses.detach(), min=1e-3)
            self.prev_epoch_losses = self.ema_losses
            self.dynamic_ref_ready.fill_(1.0)
            self.last_update_epoch.fill_(float(epoch - self.update_interval))
            self.imbalance_confirm_counter.zero_()
            self.last_imbalance_sign.zero_()
            self.epoch_acc.zero_()
            return

        # Initialize reference losses
        if self.ref_losses.sum().item() == 0:
            self.ref_losses = torch.clamp(self.ema_losses, min=1e-3)
            self.prev_epoch_losses = self.ema_losses
            self.epoch_acc.zero_()
            return

        if epoch % self.update_interval != 0:
            self.prev_epoch_losses = self.ema_losses
            self.epoch_acc.zero_()
            return

        # Contribution balancing with safety gating.
        ratios = self.ema_losses / torch.clamp(self.ref_losses, min=1e-3)
        ratios = torch.clamp(ratios, min=1e-3, max=10.0)
        ratio_gap = torch.abs(ratios[0] - ratios[1]).item()
        ratio_now = float(aux_ratio)

        def _update_aux_multiplier(aux_ratio_now):
            if self.aux_ratio_ema.item() <= 1e-8:
                self.aux_ratio_ema.fill_(aux_ratio_now)
            else:
                self.aux_ratio_ema.mul_(1.0 - self.ema_alpha).add_(self.ema_alpha * aux_ratio_now)
            desired_aux = self.aux_target_ratio / max(float(self.aux_ratio_ema.item()), 1e-6)
            desired_aux = max(self.aux_clamp_min, min(self.aux_clamp_max, desired_aux))
            if aux_ratio_now <= self.aux_target_ratio * (1.0 + max(0.0, float(self.diff_threshold))):
                desired_aux = 1.0
            aux_prev = float(self.aux_multiplier.item())
            aux_new = self.aux_momentum * aux_prev + (1.0 - self.aux_momentum) * desired_aux
            if self.aux_max_step > 0.0:
                aux_delta = max(-self.aux_max_step, min(self.aux_max_step, aux_new - aux_prev))
                aux_new = aux_prev + aux_delta
            aux_new = max(self.aux_clamp_min, min(self.aux_clamp_max, aux_new))
            self.aux_multiplier.fill_(aux_new)

        epoch_gap = float(epoch) - float(self.last_update_epoch.item())
        if epoch_gap < float(self.min_epoch_gap):
            _update_aux_multiplier(ratio_now)
            self.prev_epoch_losses = self.ema_losses
            self.epoch_acc.zero_()
            return

        deadband = max(0.0, float(self.ratio_diff_threshold))
        if self.start_dynamic_epoch <= epoch < self.stop_dynamic_epoch:
            dyn_window = max(0, int(self.stop_dynamic_epoch) - int(self.start_dynamic_epoch))
            if dyn_window <= 16:
                deadband = min(deadband, 0.008)
        first_update_pending = float(self.last_update_epoch.item()) < float(self.start_dynamic_epoch)
        imbalance_sign = 0.0
        imbalance_val = float((log_ratios := torch.log(torch.clamp(ratios, min=1e-6)))[0].item() - log_ratios[1].item())
        if abs(imbalance_val) > 1e-8:
            imbalance_sign = 1.0 if imbalance_val > 0.0 else -1.0
        if (not first_update_pending) and self.hold_when_stable and ratio_gap <= deadband:
            self.imbalance_confirm_counter.zero_()
            self.last_imbalance_sign.zero_()
            _update_aux_multiplier(ratio_now)
            self.prev_epoch_losses = self.ema_losses
            self.epoch_acc.zero_()
            if (not is_dist_avail_and_initialized()) or dist.get_rank() == 0:
                print(
                    f"[DGTL] Epoch {epoch} hold: det_loss={det_avg:.4f} seg_loss={seg_avg:.4f} "
                    f"ratio_gap={ratio_gap:.4f} <= deadband={deadband:.4f}"
                )
            return

        required_confirms = max(2 if self.hold_when_stable else 1, int(self.patience))
        prev_sign = float(self.last_imbalance_sign.item())
        if prev_sign == 0.0 or imbalance_sign == prev_sign:
            self.imbalance_confirm_counter.add_(1.0)
        else:
            self.imbalance_confirm_counter.fill_(1.0)
        self.last_imbalance_sign.fill_(imbalance_sign)
        if float(self.imbalance_confirm_counter.item()) < float(required_confirms):
            _update_aux_multiplier(ratio_now)
            self.prev_epoch_losses = self.ema_losses
            self.epoch_acc.zero_()
            if (not is_dist_avail_and_initialized()) or dist.get_rank() == 0:
                print(
                    f"[DGTL] Epoch {epoch} confirm "
                    f"{int(self.imbalance_confirm_counter.item())}/{required_confirms} "
                    f"det_loss={det_avg:.4f} seg_loss={seg_avg:.4f} "
                    f"gap={ratio_gap:.4f} sign={int(imbalance_sign):+d}"
                )
            return

        imbalance = torch.clamp(log_ratios[0] - log_ratios[1], min=-3.0, max=3.0)
        delta_up = max(0.0, float(self.clamp_max - 1.0))
        delta_dn = max(0.0, float(1.0 - self.clamp_min))
        delta_cap = max(1e-6, min(max(delta_up, delta_dn), 0.1))
        delta_target = torch.tanh(imbalance) * delta_cap
        target_multipliers = torch.stack([
            ratios.new_tensor(1.0) + delta_target,
            ratios.new_tensor(1.0) - delta_target
        ], dim=0)
        target_multipliers = 2.0 * target_multipliers / target_multipliers.sum().clamp(min=1e-6)
        lower = min(float(self.clamp_min), float(2.0 - self.clamp_max))
        upper = max(float(self.clamp_max), float(2.0 - self.clamp_min))
        target_multipliers = torch.clamp(target_multipliers, min=lower, max=upper)
        target_v = -torch.log(torch.clamp(target_multipliers, min=1e-6))
        momentum = self.momentum
        self.task_v = momentum * self.task_v + (1 - momentum) * target_v
        target_multipliers = torch.exp(-self.task_v)
        target_multipliers = 2.0 * target_multipliers / target_multipliers.sum().clamp(min=1e-6)
        target_multipliers = torch.clamp(target_multipliers, min=lower, max=upper)
        if self.main_max_step > 0.0:
            delta = torch.clamp(target_multipliers - self.global_multipliers,
                                min=-self.main_max_step, max=self.main_max_step)
            self.global_multipliers = torch.clamp(
                self.global_multipliers + delta, self.clamp_min, self.clamp_max
            )
        else:
            self.global_multipliers = target_multipliers
        _update_aux_multiplier(ratio_now)
        self.imbalance_confirm_counter.zero_()

        self.last_update_epoch.fill_(float(epoch))

        # Update history
        self.prev_epoch_losses = self.ema_losses
        self.epoch_acc.zero_()

        if (not is_dist_avail_and_initialized()) or dist.get_rank() == 0:
            print(
                f"[DGTL] Epoch {epoch} Weights: "
                f"BBox={self.global_multipliers[0]:.3f}, "
                f"Mask={self.global_multipliers[1]:.3f}, "
                f"Aux={self.aux_multiplier.item():.3f}, "
                f"v=({self.task_v[0]:.3f},{self.task_v[1]:.3f}) "
                f"det_loss={det_avg:.4f} seg_loss={seg_avg:.4f} aux_ratio={ratio_now:.3f}"
            )

    @staticmethod
    def _clone_snapshot_tensor(value):
        if not torch.is_tensor(value):
            return value
        return value.detach().cpu().clone()

    def updates_frozen(self):
        return bool(float(self.freeze_updates_flag.item()) > 0.5)

    def freeze_updates(self):
        self.frozen_main_multipliers.copy_(self.global_multipliers.detach())
        self.frozen_aux_multiplier.copy_(self.aux_multiplier.detach())
        self.freeze_updates_flag.fill_(1.0)

    def unfreeze_updates(self):
        self.freeze_updates_flag.zero_()

    def has_dynamic_adjustment(self, eps=1e-6):
        if not self.enable_dynamic:
            return False
        main_shift = torch.max(torch.abs(self.global_multipliers.detach() - 1.0)).item()
        val_shift = abs(float(self.val_core_bias.detach().item()))
        return max(main_shift, val_shift) > float(eps)

    def get_validation_core_bias(self):
        if not self.enable_dynamic:
            return self.val_core_bias.new_tensor(0.0)
        return self.val_core_bias

    def apply_validation_feedback(
        self,
        det_score,
        seg_score,
        record_score,
        best_score=None,
        drop_tolerance=0.0,
        gap_threshold=0.02,
        epoch=0,
    ):
        if not self.enable_dynamic or self.updates_frozen():
            return False
        if det_score is None or seg_score is None or record_score is None:
            return False

        det = float(det_score)
        seg = float(seg_score)
        record = float(record_score)
        if not (math.isfinite(det) and math.isfinite(seg) and math.isfinite(record)):
            return False

        signed_gap = seg - det
        if float(self.val_feedback_ready.item()) < 0.5:
            self.val_gap_ema.fill_(signed_gap)
            self.val_feedback_ready.fill_(1.0)
        else:
            alpha = max(0.0, min(1.0, float(self.ema_alpha)))
            self.val_gap_ema.mul_(1.0 - alpha).add_(alpha * signed_gap)

        gap_ema = float(self.val_gap_ema.item())
        min_gap = max(0.02, float(gap_threshold))
        best_drop = 0.0
        if best_score is not None and math.isfinite(float(best_score)):
            best_drop = max(0.0, float(best_score) - record)

        target_bias = 0.0
        if best_drop <= max(float(drop_tolerance), 1e-6) and record >= 0.35 and gap_ema > min_gap:
            target_bias = min(0.006, 0.15 * (gap_ema - min_gap))

        prev_bias = float(self.val_core_bias.item())
        max_step = max(0.0005, min(0.0015, max(float(self.main_max_step), 0.0010)))
        delta = max(-max_step, min(max_step, target_bias - prev_bias))
        new_bias = max(0.0, min(0.006, prev_bias + delta))
        self.val_core_bias.fill_(new_bias)

        changed = abs(new_bias - prev_bias) > 1e-8
        if changed and ((not is_dist_avail_and_initialized()) or dist.get_rank() == 0):
            print(
                f"[DGTL][DynWeight] val_bias@e{epoch} "
                f"bias={new_bias:.5f} target={target_bias:.5f} "
                f"gap_ema={gap_ema:.4f} det={det:.4f} seg={seg:.4f} "
                f"best_drop={best_drop:.4f}"
            )
        return changed

    def export_state_snapshot(self):
        return {
            'state_dict': {
                key: self._clone_snapshot_tensor(value)
                for key, value in self.state_dict().items()
            },
            'attrs': {
                'enable_dynamic': bool(self.enable_dynamic),
                'main_max_step': float(self.main_max_step),
                'aux_max_step': float(self.aux_max_step),
                'geom_weight': float(self.geom_weight),
                'harmonize_weight': float(self.harmonize_weight),
                'core_residual_scale': float(self.core_residual_scale),
                'aux_residual_scale': float(self.aux_residual_scale),
                'gs_residual_scale': float(self.gs_residual_scale),
                'geom_residual_scale': float(self.geom_residual_scale),
            }
        }

    def load_state_snapshot(self, snapshot):
        if not isinstance(snapshot, dict):
            return False
        state_dict = snapshot.get('state_dict', None)
        if isinstance(state_dict, dict):
            cur_state = self.state_dict()
            compat_state = {}
            for key, value in state_dict.items():
                if key not in cur_state or not torch.is_tensor(value):
                    continue
                target = cur_state[key]
                if target.shape != value.shape:
                    continue
                compat_state[key] = value.to(device=target.device, dtype=target.dtype)
            if compat_state:
                self.load_state_dict(compat_state, strict=False)
        attrs = snapshot.get('attrs', {})
        if isinstance(attrs, dict):
            self.enable_dynamic = bool(attrs.get('enable_dynamic', self.enable_dynamic))
            self.main_max_step = float(attrs.get('main_max_step', self.main_max_step))
            self.aux_max_step = float(attrs.get('aux_max_step', self.aux_max_step))
            self.geom_weight = float(attrs.get('geom_weight', self.geom_weight))
            self.harmonize_weight = float(attrs.get('harmonize_weight', self.harmonize_weight))
            self.core_residual_scale = float(attrs.get('core_residual_scale', self.core_residual_scale))
            self.aux_residual_scale = float(attrs.get('aux_residual_scale', self.aux_residual_scale))
            self.gs_residual_scale = float(attrs.get('gs_residual_scale', self.gs_residual_scale))
            self.geom_residual_scale = float(attrs.get('geom_residual_scale', self.geom_residual_scale))
        return True

    def get_multiplier(self, key):
        """Return a scalar multiplier used to adjust a group of losses together."""
        if not self.enable_dynamic:
            return torch.tensor(1.0, device=self.global_multipliers.device)
        if key == 'bbox':
            if self.updates_frozen():
                return self.frozen_main_multipliers[0]
            return self.global_multipliers[0]
        elif key == 'mask':
            if self.updates_frozen():
                return self.frozen_main_multipliers[1]
            return self.global_multipliers[1]
        elif key == 'aux':
            if self.updates_frozen():
                return self.frozen_aux_multiplier
            return self.aux_multiplier
        return torch.tensor(1.0, device=self.global_multipliers.device)

    def get_weight_stats(self, epoch):
        # Return current contribution-balance multipliers.
        return {
            'bbox': {'base': 5.0, 'final_weight': 5.0 * self.global_multipliers[0].item()},
            'mask': {'base': 10.0, 'final_weight': 10.0 * self.global_multipliers[1].item()}
        }

    def get_zeta(self, epoch, total_epochs=100):
        # Linear warm-up, then hold (theory-aligned zeta(t)).
        start = int(self.start_dynamic_epoch)
        warm = max(1, int(self.warmup_epochs))
        if epoch < start:
            return 0.0
        if self.stop_dynamic_epoch > start and epoch >= int(self.stop_dynamic_epoch):
            return 0.0
        if epoch < start + warm:
            return float(epoch - start) / float(warm)
        return 1.0

    def get_geom_factor(self, epoch, total_epochs=100):
        start = int(getattr(self, 'geom_start_epoch', self.start_dynamic_epoch))
        stop = int(getattr(self, 'geom_stop_epoch', -1))
        if stop <= start:
            stop = int(total_epochs) + 1
        if epoch < start or epoch >= stop:
            return 0.0
        return 1.0

    def get_eta(self, epoch, total_epochs=100):
        start = int(getattr(self, 'gs_start_epoch', self.start_dynamic_epoch))
        stop = int(getattr(self, 'gs_stop_epoch', -1))
        if stop <= start:
            stop = int(total_epochs) + 1
        if epoch < start:
            return 0.0
        if epoch >= stop:
            return 0.0
        return float(stop - epoch) / max(float(stop - start), 1e-6)

    def compute_grad_harmonization_penalty(self, cos_sim, det_loss, seg_loss,
                                           epoch, total_epochs=100, rho=1.0, tau=-0.05):
        if cos_sim is None:
            return det_loss.new_tensor(0.0)
        eta = self.get_eta(epoch, total_epochs)
        if eta <= 0.0:
            return det_loss.new_tensor(0.0)

        cos_val = torch.clamp(cos_sim.detach(), min=-1.0, max=1.0)
        tau_t = det_loss.new_tensor(max(float(tau), 0.0))
        if torch.is_tensor(rho):
            rho_t = rho.detach().to(device=det_loss.device, dtype=det_loss.dtype).clamp(min=0.0, max=1.0)
        else:
            rho_t = det_loss.new_tensor(float(rho)).clamp(min=0.0, max=1.0)
        indicator = (cos_val < tau_t).float()
        gap = torch.relu(tau_t - cos_val)
        conflict_scale = gap / (torch.abs(tau_t) + 0.1)
        conflict_scale = torch.clamp(conflict_scale, min=0.0, max=2.0)
        base = 0.5 * (det_loss + seg_loss)
        harmonize_weight = max(float(self.harmonize_weight), 1e-6)
        cap_ratio = max(float(self.harmonize_cap_ratio), 1e-6)
        raw = eta * rho_t * indicator * conflict_scale * harmonize_weight * base
        cap = cap_ratio * base.detach()
        return torch.minimum(raw, cap)


    def compute_geom_consistency(self, pred_boxes, pred_masks, gt_boxes, gt_masks,
                                 pts, epoch, total_epochs, valid_q=None,
                                 return_stats=False):
        device_tensor = pred_boxes[0] if isinstance(pred_boxes, list) and len(pred_boxes) > 0 else pred_boxes

        def _zero_geom(quality_valid=False):
            zero = device_tensor.new_tensor(0.0)
            if return_stats:
                return zero, {
                    "agreement": zero,
                    "box_iou": zero,
                    "mask_iou": zero,
                    "quality_valid": zero.new_tensor(1.0 if quality_valid else 0.0),
                }
            return zero

        if (not self.enable_geom or self.geom_weight <= 0) and not return_stats:
            self._geom_debug_add(4)
            return _zero_geom(False)

        geom_factor = self.get_geom_factor(epoch, total_epochs) if self.enable_geom else 0.0
        geom_active = self.enable_geom and self.geom_weight > 0 and geom_factor > 1e-4
        if not geom_active:
            self._geom_debug_add(5)

        if isinstance(pred_boxes, list):
            max_q = max(b.shape[0] for b in pred_boxes) if len(pred_boxes) > 0 else 0
            if max_q == 0:
                self._geom_debug_add(6)
                return _zero_geom(False)
            padded_boxes = []
            local_valid = []
            for b in pred_boxes:
                q = b.shape[0]
                pad_q = max_q - q
                if pad_q > 0:
                    b = torch.cat([b, b.new_zeros(pad_q, b.shape[-1])], dim=0)
                padded_boxes.append(b)
                v = b.new_zeros(max_q)
                if q > 0:
                    v[:q] = 1.0
                local_valid.append(v)
            pred_boxes = torch.stack(padded_boxes, dim=0)
            local_valid_q = torch.stack(local_valid, dim=0)
            if isinstance(valid_q, list):
                padded_v = []
                for v in valid_q:
                    pad_q = max_q - v.shape[0]
                    if pad_q > 0:
                        v = torch.cat([v, v.new_zeros(pad_q)], dim=0)
                    padded_v.append(v)
                valid_q = torch.stack(padded_v, dim=0)
            valid_q = local_valid_q if valid_q is None else (valid_q * local_valid_q)

        valid_pts = None
        if isinstance(pts, list):
            max_n = max(p.shape[1] for p in pts)
            padded_pts = []
            valid_pts_list = []
            for p in pts:
                n = p.shape[1]
                pad_len = max_n - n
                if pad_len > 0:
                    p = torch.cat([p, p.new_zeros(p.shape[0], pad_len, p.shape[2])], dim=1)
                padded_pts.append(p)
                v = p.new_zeros(max_n)
                if n > 0:
                    v[:n] = 1.0
                valid_pts_list.append(v)
            pts = torch.cat(padded_pts, dim=0)
            valid_pts = torch.stack(valid_pts_list, dim=0)

        if isinstance(pred_masks, list):
            norm_masks = []
            for m in pred_masks:
                if m.dim() == 3:
                    m = m.squeeze(0)
                if m.dim() != 2:
                    m = m.view(m.shape[0], -1)
                norm_masks.append(m)
            max_q = max(m.shape[0] for m in norm_masks) if len(norm_masks) > 0 else 0
            max_n = max(m.shape[1] for m in norm_masks) if len(norm_masks) > 0 else 0
            if max_q == 0 or max_n == 0:
                self._geom_debug_add(6)
                return _zero_geom(False)
            padded_masks = []
            for m in norm_masks:
                pad_n = max_n - m.shape[1]
                if pad_n > 0:
                    m = torch.cat([m, m.new_zeros(m.shape[0], pad_n)], dim=-1)
                pad_q = max_q - m.shape[0]
                if pad_q > 0:
                    m = torch.cat([m, m.new_zeros(pad_q, max_n)], dim=0)
                padded_masks.append(m)
            pred_masks = torch.stack(padded_masks, dim=0)

        if pred_masks.dim() < 2 or pts.dim() < 2:
            self._geom_debug_add(6)
            return _zero_geom(False)

        if isinstance(gt_boxes, list):
            padded_gt = []
            for g in gt_boxes:
                if g.dim() != 2:
                    g = g.view(g.shape[0], -1)
                pad_q = pred_boxes.shape[1] - g.shape[0]
                if pad_q > 0:
                    g = torch.cat([g, g.new_zeros(pad_q, g.shape[1])], dim=0)
                padded_gt.append(g)
            gt_boxes = torch.stack(padded_gt, dim=0)

        if isinstance(gt_masks, list):
            norm_gt = []
            for m in gt_masks:
                if m.dim() == 3:
                    m = m.squeeze(0)
                if m.dim() != 2:
                    m = m.view(m.shape[0], -1)
                norm_gt.append(m)
            max_q = pred_masks.shape[1]
            max_n = pred_masks.shape[2]
            padded_gt = []
            for m in norm_gt:
                pad_n = max_n - m.shape[1]
                if pad_n > 0:
                    m = torch.cat([m, m.new_zeros(m.shape[0], pad_n)], dim=-1)
                pad_q = max_q - m.shape[0]
                if pad_q > 0:
                    m = torch.cat([m, m.new_zeros(pad_q, max_n)], dim=0)
                padded_gt.append(m)
            gt_masks = torch.stack(padded_gt, dim=0)

        B, Q, _ = pred_boxes.shape
        N = pts.shape[1]
        pts_expanded = pts.unsqueeze(1).expand(-1, Q, -1, -1)
        if valid_pts is not None:
            valid_pts = valid_pts.to(device=pred_masks.device, dtype=pred_masks.dtype)
            valid_pts_exp = valid_pts.unsqueeze(1)
        else:
            valid_pts_exp = pred_masks.new_ones((B, 1, N))

        def _soft_mask_from_boxes(boxes):
            boxes_expanded = boxes.unsqueeze(2).expand(-1, -1, N, -1)
            centers = boxes_expanded[..., :3]
            sizes = torch.clamp(boxes_expanded[..., 3:], min=1e-4)
            rel_dist = (pts_expanded - centers).abs() / sizes
            max_dist = rel_dist.max(dim=-1)[0]
            return torch.sigmoid(10.0 * (0.5 - max_dist)) * valid_pts_exp

        def _box_from_masks(mask_prob):
            masked = mask_prob * valid_pts_exp
            mass = masked.sum(dim=-1, keepdim=True)
            weights = masked / mass.clamp(min=1e-6)
            center = (weights.unsqueeze(-1) * pts_expanded).sum(dim=2)
            diff = pts_expanded - center.unsqueeze(2)
            var = (weights.unsqueeze(-1) * diff.pow(2)).sum(dim=2)
            size = 2.0 * torch.sqrt(var + 1e-6)
            return torch.cat([center, size], dim=-1), mass.squeeze(-1)

        def _dice_prob_loss(inputs, targets):
            inputs = inputs * valid_pts_exp
            targets = targets * valid_pts_exp
            inter = (inputs * targets).sum(dim=-1)
            denom = inputs.sum(dim=-1) + targets.sum(dim=-1)
            return 1.0 - (2.0 * inter + 1.0) / (denom + 1.0)

        def _aligned_box_iou(box_a, box_b):
            xyz_a = box_cxcyczwhd_to_xyzxyz(box_a)
            xyz_b = box_cxcyczwhd_to_xyzxyz(box_b)
            mins = torch.max(xyz_a[..., :3], xyz_b[..., :3])
            maxs = torch.min(xyz_a[..., 3:], xyz_b[..., 3:])
            inter_dims = (maxs - mins).clamp(min=0.0)
            inter_vol = inter_dims.prod(dim=-1)
            vol_a = (xyz_a[..., 3:] - xyz_a[..., :3]).clamp(min=0.0).prod(dim=-1)
            vol_b = (xyz_b[..., 3:] - xyz_b[..., :3]).clamp(min=0.0).prod(dim=-1)
            return inter_vol / (vol_a + vol_b - inter_vol + 1e-6)

        pred_box_mask = _soft_mask_from_boxes(pred_boxes.detach()).detach()
        gt_box_mask = _soft_mask_from_boxes(gt_boxes.detach()).detach()
        mask_probs = torch.sigmoid(pred_masks) * valid_pts_exp
        gt_masks = gt_masks.float().detach() * valid_pts_exp

        pred_mask_box, _ = _box_from_masks(mask_probs)
        gt_mask_box, gt_mask_mass = _box_from_masks(gt_masks)
        gt_mask_center = torch.where(
            (gt_mask_mass > 1e-4).unsqueeze(-1),
            gt_mask_box[..., :3].detach(),
            gt_boxes[..., :3].detach()
        )

        box_iou = _aligned_box_iou(pred_boxes.detach(), gt_boxes.detach()).detach()
        pred_mask_inter = (mask_probs * gt_masks).sum(dim=-1)
        pred_mask_union = mask_probs.sum(dim=-1) + gt_masks.sum(dim=-1) - pred_mask_inter
        mask_iou = (pred_mask_inter / (pred_mask_union + 1e-6)).detach()
        agreement_inter = (mask_probs * pred_box_mask).sum(dim=-1)
        agreement_union = mask_probs.sum(dim=-1) + pred_box_mask.sum(dim=-1) - agreement_inter
        agreement = (agreement_inter / (agreement_union + 1e-6)).detach()

        box_gate = (box_iou >= float(self.geom_box_thr)).float()
        mask_gate = (mask_iou >= float(self.geom_mask_thr)).float()
        agree_gate = (agreement >= float(self.geom_agree_thr)).float()
        valid_gate = pred_boxes.new_ones((B, Q))
        if valid_q is not None:
            valid_gate = valid_q.to(device=box_gate.device, dtype=box_gate.dtype)
        gate = box_gate * mask_gate * agree_gate * valid_gate

        mask_center_l1 = F.smooth_l1_loss(
            pred_mask_box[..., :3], gt_mask_center, reduction='none'
        ).sum(dim=-1)
        mask_mass = (mask_probs * valid_pts_exp).sum(dim=-1).clamp(min=1e-6)
        mask_outside_gt_box = (
            mask_probs * (1.0 - gt_box_mask).clamp(min=0.0)
        ).sum(dim=-1) / mask_mass
        geom_core = (0.5 * mask_center_l1 + mask_outside_gt_box) * gate

        denom = gate.sum().clamp(min=1.0)
        valid_denom = valid_gate.sum().clamp(min=1.0)
        valid_ratio = (gate.sum() / valid_denom).detach()
        geom_loss = geom_core.sum() / denom
        mean_box_iou = (box_iou * gate).sum() / denom
        mean_mask_iou = (mask_iou * gate).sum() / denom
        mean_agree = (agreement * gate).sum() / denom
        quality = 0.5 * (box_iou + mask_iou)
        mean_quality = (quality * gate).sum() / denom

        self._geom_debug_add(7)
        self._geom_debug_add(8)
        geom_res_eff = max(0.0, min(1.0, float(getattr(self, 'geom_residual_scale', 1.0))))
        quality_gate = torch.clamp(mean_quality, min=float(self.geom_gate_floor), max=1.0).detach()
        geom_scale = self.geom_weight * geom_res_eff * (geom_factor if geom_active else 0.0) * quality_gate
        quality_valid = (
            float(valid_ratio.item()) >= 0.25
            and float(mean_box_iou.detach().item()) >= max(0.25, float(self.geom_box_thr) - 0.1)
            and float(mean_mask_iou.detach().item()) >= max(0.25, float(self.geom_mask_thr) - 0.1)
            and float(mean_agree.detach().item()) >= max(0.25, float(self.geom_agree_thr) - 0.05)
        )
        if return_stats:
            return geom_scale * geom_loss, {
                "agreement": mean_agree.detach(),
                "box_iou": mean_box_iou.detach(),
                "mask_iou": mean_mask_iou.detach(),
                "quality_valid": mean_agree.detach().new_tensor(1.0 if quality_valid else 0.0),
            }
        return geom_scale * geom_loss



def compute_hungarian_loss(end_points, num_decoder_layers, set_criterion,
                           query_points_obj_topk=5,
                           dgt_module=None, epoch=0, total_epochs=100):
    """
    Compute Hungarian matching loss with DGTL Logic:
    L = Sum(w_i * L_i) + P + L_geom
    """
    prefixes = ['last_'] + [f'{i}head_' for i in range(num_decoder_layers - 1)]
    prefixes = ['proposal_'] + prefixes
    is_multi_mask = "proposal_pred_masks" in end_points

    gt_center = end_points['center_label'][:, :, 0:3]
    gt_size = end_points['size_gts']
    gt_labels = end_points['sem_cls_label']
    gt_bbox = torch.cat([gt_center, gt_size], dim=-1)
    gt_masks = end_points['gt_masks']

    positive_map = end_points['positive_map']
    modify_positive_map = end_points['modify_positive_map']
    pron_positive_map = end_points['pron_positive_map']
    other_entity_map = end_points['other_entity_map']
    rel_positive_map = end_points['rel_positive_map']
    box_label_mask = end_points['box_label_mask']
    auxi_entity_positive_map = end_points['auxi_entity_positive_map']
    auxi_box = end_points['auxi_box']

    target = [
        {
            "labels": gt_labels[b, box_label_mask[b].bool()],
            "boxes": gt_bbox[b, box_label_mask[b].bool()],
            "masks": gt_masks[b, box_label_mask[b].bool()],
            "positive_map": positive_map[b, box_label_mask[b].bool()],
            "modify_positive_map": modify_positive_map[b, box_label_mask[b].bool()],
            "pron_positive_map": pron_positive_map[b, box_label_mask[b].bool()],
            "other_entity_map": other_entity_map[b, box_label_mask[b].bool()],
            "rel_positive_map": rel_positive_map[b, box_label_mask[b].bool()],
            "auxi_entity_positive_map": auxi_entity_positive_map[b, 0].unsqueeze(0),
            "auxi_box": auxi_box[b]
        }
        for b in range(gt_labels.shape[0])
    ]

    # Initialize accumulators
    loss_ce, loss_bbox, loss_giou, loss_sem_align = 0, 0, 0, 0
    loss_mask, loss_dice = 0, 0
    sp_loss_mask, sp_loss_dice = 0, 0
    corresponding_loss_mask, corresponding_loss_dice = 0, 0
    adaptive_weight_loss_mask, adaptive_weight_loss_dice = 0, 0

    # --- [Loop Over Layers] ---
    geom_indices = None
    for prefix in prefixes:
        output = {}
        if 'proj_tokens' in end_points:
            output['proj_tokens'] = end_points['proj_tokens']
            output['proj_queries'] = end_points[f'{prefix}proj_queries']
            output['tokenized'] = end_points['tokenized']

        pred_center = end_points[f'{prefix}center']
        pred_size = end_points[f'{prefix}pred_size']
        pred_bbox = torch.cat([pred_center, pred_size], dim=-1)
        pred_logits = end_points[f'{prefix}sem_cls_scores']

        output['pred_logits'] = pred_logits
        output["pred_boxes"] = pred_bbox
        output["superpoints"] = end_points["superpoints"]
        output["language_dataset"] = end_points["language_dataset"]

        if is_multi_mask:
            output["pred_masks"] = end_points[f"{prefix}pred_masks"]
        else:
            if prefix == 'last_':
                output["pred_masks"] = end_points["last_pred_masks"]
                output["sp_pred_masks"] = end_points["sp_last_pred_masks"]
                output["adaptive_weights"] = end_points['adaptive_weights']
                output['super_xyz_list'] = end_points['super_xyz_list']

        losses, indices = set_criterion(output, target, prefix)
        if prefix == 'last_':
            geom_indices = indices
        elif geom_indices is None and prefix == 'proposal_':
            geom_indices = indices

        for loss_key in losses.keys():
            end_points[f'{prefix}_{loss_key}'] = losses[loss_key]

        loss_ce += losses.get('loss_ce', 0)
        loss_bbox += losses['loss_bbox']
        loss_giou += losses.get('loss_giou', 0)
        loss_mask += losses.get('loss_mask', 0)
        loss_dice += losses.get('loss_dice', 0)
        sp_loss_mask += losses.get('sp_loss_mask', 0)
        sp_loss_dice += losses.get('sp_loss_dice', 0)
        corresponding_loss_mask += losses.get('corresponding_loss_mask', 0)
        corresponding_loss_dice += losses.get('corresponding_loss_dice', 0)
        adaptive_weight_loss_mask += losses.get('adaptive_weight_loss_mask', 0)
        adaptive_weight_loss_dice += losses.get('adaptive_weight_loss_dice', 0)

        if 'proj_tokens' in end_points:
            loss_sem_align += losses['loss_sem_align']

    if 'seeds_obj_cls_logits' in end_points.keys():
        query_points_generation_loss = compute_points_obj_cls_loss_hard_topk(
            end_points, query_points_obj_topk
        )
    else:
        query_points_generation_loss = 0.0

    geom_loss = gt_center.new_tensor(0.0)
    geom_stats = None
    if dgt_module is not None and dgt_module.enable_geom:
        need_quality_stats = bool(getattr(dgt_module, 'enable_quality_stats', False))
        need_geom_loss = dgt_module.get_geom_factor(epoch, total_epochs) > 1e-4
        if not need_geom_loss and not need_quality_stats:
            dgt_module._geom_debug_add(5)
        else:
            dgt_module._geom_debug_add(0)
            geom_boxes = None
            if 'last_center' in end_points and 'last_pred_size' in end_points:
                geom_boxes = torch.cat([end_points['last_center'], end_points['last_pred_size']], dim=-1)
            elif 'proposal_center' in end_points and 'proposal_pred_size' in end_points:
                geom_boxes = torch.cat([end_points['proposal_center'], end_points['proposal_pred_size']], dim=-1)


            geom_masks = end_points.get('sp_last_pred_masks', None)
            if geom_masks is None:
                geom_masks = end_points.get('last_pred_masks', None)
            if geom_masks is None:
                geom_masks = end_points.get('proposal_pred_masks', None)

            geom_pts = None
            use_super = False
            if geom_masks is not None:
                if 'super_xyz_list' in end_points:
                    geom_pts = end_points['super_xyz_list']
                    use_super = True
                elif 'coords' in end_points and 'superpoints' in end_points:
                    per_point_masks = []
                    for bs in range(len(geom_masks)):
                        mask_bs = geom_masks[bs]
                        if mask_bs.dim() == 3:
                            mask_bs = mask_bs.squeeze(0)
                        superpoint = end_points['superpoints'][bs]
                        mask_pts = mask_bs.gather(1, superpoint.unsqueeze(0).expand(mask_bs.size(0), -1))
                        per_point_masks.append(mask_pts)
                    geom_masks = per_point_masks
                    geom_pts = end_points['coords']

            if geom_boxes is not None and geom_masks is not None and geom_pts is not None and geom_indices is not None:
                B = gt_bbox.shape[0]
                gt_masks_geom = []
                valid_q = []
                for bs in range(B):
                    gt_mask_bs = gt_masks[bs, box_label_mask[bs].bool()].float()
                    if use_super:
                        superpoints = end_points['superpoints'][bs]
                        if superpoints.device != gt_mask_bs.device:
                            superpoints = superpoints.to(gt_mask_bs.device)
                        superpoints = superpoints.long()
                        gt_mask_bs = scatter_mean(gt_mask_bs, superpoints, dim=1)
                    gt_mask_bs = (gt_mask_bs > 0.5).float()
                    gt_masks_geom.append(gt_mask_bs)

                boxes_list = []
                masks_list = []
                gt_boxes_list = []
                gt_masks_list = []
                for bs in range(B):
                    idx_pred, idx_tgt = geom_indices[bs]
                    mask_bs = geom_masks[bs] if isinstance(geom_masks, list) else geom_masks[bs]
                    if mask_bs.dim() == 3:
                        mask_bs = mask_bs.squeeze(0)
                    if idx_pred.device != mask_bs.device:
                        idx_pred = idx_pred.to(mask_bs.device)
                    if idx_tgt.device != mask_bs.device:
                        idx_tgt = idx_tgt.to(mask_bs.device)

                    if idx_pred.numel() == 0 or idx_tgt.numel() == 0:
                        boxes_list.append(geom_boxes.new_zeros((0, geom_boxes.shape[-1])))
                        masks_list.append(mask_bs.new_zeros((0, mask_bs.shape[-1])))
                        gt_boxes_list.append(geom_boxes.new_zeros((0, geom_boxes.shape[-1])))
                        gt_masks_list.append(mask_bs.new_zeros((0, mask_bs.shape[-1])))
                        valid_q.append(mask_bs.new_zeros((0,)))
                        continue

                    boxes_sel = geom_boxes[bs, idx_pred]
                    masks_sel = mask_bs.index_select(0, idx_pred)
                    gt_boxes_valid = gt_bbox[bs, box_label_mask[bs].bool()].to(boxes_sel.device)
                    gt_masks_valid = gt_masks_geom[bs].to(masks_sel.device)
                    gt_boxes_sel = gt_boxes_valid.index_select(0, idx_tgt.to(gt_boxes_valid.device))
                    gt_masks_sel = gt_masks_valid.index_select(0, idx_tgt.to(gt_masks_valid.device))

                    boxes_list.append(boxes_sel)
                    masks_list.append(masks_sel)
                    gt_boxes_list.append(gt_boxes_sel)
                    gt_masks_list.append(gt_masks_sel)
                    valid_q.append(masks_sel.new_ones((masks_sel.shape[0],)))
                if valid_q and sum(v.sum().item() for v in valid_q) <= 0:
                    dgt_module._geom_debug_add(3)
                else:
                    dgt_module._geom_debug_add(1)
                    if need_quality_stats and not need_geom_loss:
                        with torch.no_grad():
                            geom_out = dgt_module.compute_geom_consistency(
                                boxes_list, masks_list, gt_boxes_list, gt_masks_list,
                                geom_pts, epoch, total_epochs, valid_q=valid_q,
                                return_stats=True
                            )
                        _, geom_stats = geom_out
                    else:
                        geom_out = dgt_module.compute_geom_consistency(
                            boxes_list, masks_list, gt_boxes_list, gt_masks_list,
                            geom_pts, epoch, total_epochs, valid_q=valid_q,
                            return_stats=need_quality_stats
                        )
                        if need_quality_stats:
                            geom_loss, geom_stats = geom_out
                        else:
                            geom_loss = geom_out
            else:
                dgt_module._geom_debug_add(2)

    # =========================================================================
    # [DGTL INTEGRATION] Retrieve grouped multipliers
    # =========================================================================
    m_bbox = gt_center.new_tensor(1.0)
    m_mask = gt_center.new_tensor(1.0)
    m_aux = gt_center.new_tensor(1.0)
    m_bbox_eff = m_bbox
    m_mask_eff = m_mask
    m_aux_eff = m_aux
    val_core_bias = gt_center.new_tensor(0.0)

    if dgt_module is not None and dgt_module.enable_dynamic:
        m_bbox = dgt_module.get_multiplier('bbox')
        m_mask = dgt_module.get_multiplier('mask')
        m_aux = dgt_module.get_multiplier('aux')
        if not torch.is_tensor(m_bbox):
            m_bbox = gt_center.new_tensor(float(m_bbox))
        else:
            m_bbox = m_bbox.to(device=gt_center.device, dtype=gt_center.dtype)
        if not torch.is_tensor(m_mask):
            m_mask = gt_center.new_tensor(float(m_mask))
        else:
            m_mask = m_mask.to(device=gt_center.device, dtype=gt_center.dtype)
        if not torch.is_tensor(m_aux):
            m_aux = gt_center.new_tensor(float(m_aux))
        else:
            m_aux = m_aux.to(device=gt_center.device, dtype=gt_center.dtype)

        core_res = max(0.0, min(1.0, float(getattr(dgt_module, 'core_residual_scale', 1.0))))
        m_bbox_eff = 1.0 + core_res * (m_bbox - 1.0)
        m_mask_eff = 1.0 + core_res * (m_mask - 1.0)
        if hasattr(dgt_module, 'get_validation_core_bias'):
            val_core_bias = dgt_module.get_validation_core_bias()
            if not torch.is_tensor(val_core_bias):
                val_core_bias = gt_center.new_tensor(float(val_core_bias))
            else:
                val_core_bias = val_core_bias.to(device=gt_center.device, dtype=gt_center.dtype)
            m_bbox_eff = torch.clamp(m_bbox_eff + val_core_bias, min=0.98, max=1.02)
            m_mask_eff = torch.clamp(m_mask_eff - val_core_bias, min=0.98, max=1.02)
        m_aux_eff = gt_center.new_tensor(1.0)

    if dgt_module is not None:
        # Logging
        if not hasattr(compute_hungarian_loss, '_last_printed_epoch'):
            compute_hungarian_loss._last_printed_epoch = -1
        if epoch != compute_hungarian_loss._last_printed_epoch and epoch % 1 == 0:
            if not is_dist_avail_and_initialized() or dist.get_rank() == 0:
                print(
                    f"[DGTL] Epoch {epoch} Core(B={m_bbox_eff.item():.3f},M={m_mask_eff.item():.3f}) "
                    f"Aux={m_aux_eff.item():.3f}"
                )
            compute_hungarian_loss._last_printed_epoch = epoch

    # =========================================================================
    # [FINAL LOSS CALCULATION]
    # =========================================================================

    weight = 1
    if end_points["language_dataset"][0] == "scanrefer":
        weight = 0.5
    BASE_BBOX = 5.0
    BASE_GIOU = 1.0
    BASE_MASK = 10.0
    BASE_DICE = 2.0
    BASE_SP_MASK = 5.0
    BASE_SP_DICE = 1.0
    BASE_CORR_MASK = 10.0
    BASE_CORR_DICE = 2.0
    BASE_ADA_MASK = 10.0
    BASE_ADA_DICE = 2.0
    core_det_base = 1.0 / (num_decoder_layers + 1) * (
        BASE_BBOX * loss_bbox + BASE_GIOU * loss_giou
    )
    core_seg_base = BASE_MASK * loss_mask + BASE_DICE * loss_dice
    aux_base = (
        BASE_SP_MASK * sp_loss_mask
        + BASE_SP_DICE * sp_loss_dice
        + BASE_CORR_MASK * corresponding_loss_mask
        + BASE_CORR_DICE * corresponding_loss_dice
        + BASE_ADA_MASK * adaptive_weight_loss_mask
        + BASE_ADA_DICE * adaptive_weight_loss_dice
    )
    core_base = core_det_base + core_seg_base
    core_dyn_renorm = gt_center.new_tensor(1.0)

    if dgt_module is not None and dgt_module.enable_dynamic:
        core_dyn = m_bbox_eff * core_det_base + m_mask_eff * core_seg_base
        if bool(torch.isfinite(core_base).all()) and bool(torch.isfinite(core_dyn).all()):
            denom = core_dyn.detach().abs().clamp(min=1e-6)
            core_dyn_renorm = torch.clamp(core_base.detach() / denom, min=0.9, max=1.1)
            m_bbox_eff = m_bbox_eff * core_dyn_renorm
            m_mask_eff = m_mask_eff * core_dyn_renorm

    geom_loss_raw = geom_loss
    if torch.is_tensor(geom_loss):
        geom_loss = torch.nan_to_num(geom_loss, nan=0.0, posinf=0.0, neginf=0.0).clamp(min=0.0)
        if dgt_module is not None and float(getattr(dgt_module, 'geom_cap_ratio', 0.0)) > 0.0:
            geom_cap = float(getattr(dgt_module, 'geom_cap_ratio', 0.0)) * core_base.detach().clamp(min=0.0)
            geom_loss = torch.minimum(geom_loss, geom_cap)
    else:
        geom_loss = gt_center.new_tensor(0.0)
        geom_loss_raw = geom_loss

    loss = (
            8 * query_points_generation_loss
            + 1.0 / (num_decoder_layers + 1) * (
                    weight * loss_ce
                    + (BASE_BBOX * m_bbox_eff) * loss_bbox
                    + (BASE_GIOU * m_bbox_eff) * loss_giou
                    + weight * loss_sem_align
            )
            + (BASE_MASK * m_mask_eff) * loss_mask
            + (BASE_DICE * m_mask_eff) * loss_dice
            + m_aux_eff * aux_base
            + geom_loss
    )

    # -------------------------------------------------------------------------
    # Gradient Consistency Support (for Gradient Surgery)
    # -------------------------------------------------------------------------
    loss_det_tensor = core_det_base
    loss_seg_tensor = core_seg_base
    loss_det_weighted_tensor = (
            1.0 / (num_decoder_layers + 1) * (
                    (BASE_BBOX * m_bbox_eff) * loss_bbox
                    + (BASE_GIOU * m_bbox_eff) * loss_giou
            )
    )
    loss_seg_weighted_tensor = (
            (BASE_MASK * m_mask_eff) * loss_mask
            + (BASE_DICE * m_mask_eff) * loss_dice
    )

    # Logging
    end_points['loss_ce'] = loss_ce
    end_points['loss_bbox'] = loss_bbox
    end_points['loss_giou'] = loss_giou
    end_points['query_points_generation_loss'] = query_points_generation_loss
    end_points['loss_sem_align'] = loss_sem_align
    end_points['loss'] = loss
    end_points['loss_geom'] = geom_loss
    end_points['loss_geom_raw'] = geom_loss_raw.detach() if torch.is_tensor(geom_loss_raw) else gt_center.new_tensor(0.0)
    end_points['loss_mask'] = loss_mask
    end_points['loss_dice'] = loss_dice
    end_points['sp_loss_mask'] = sp_loss_mask
    end_points['sp_loss_dice'] = sp_loss_dice
    end_points['loss_det_tensor'] = loss_det_tensor
    end_points['loss_seg_tensor'] = loss_seg_tensor
    end_points['loss_det_weighted_tensor'] = loss_det_weighted_tensor
    end_points['loss_seg_weighted_tensor'] = loss_seg_weighted_tensor
    end_points['dgt_m_bbox_raw'] = m_bbox.detach()
    end_points['dgt_m_mask_raw'] = m_mask.detach()
    end_points['dgt_m_bbox_eff'] = m_bbox_eff.detach()
    end_points['dgt_m_mask_eff'] = m_mask_eff.detach()
    end_points['dgt_core_renorm'] = core_dyn_renorm.detach()
    end_points['dgt_val_core_bias'] = val_core_bias.detach()
    end_points['dgt_m_aux_raw'] = m_aux.detach()
    end_points['dgt_m_aux_eff'] = m_aux_eff.detach()
    end_points['dgt_m_aux'] = m_aux_eff.detach()
    if dgt_module is not None:
        end_points['dgt_res_core'] = gt_center.new_tensor(float(getattr(dgt_module, 'core_residual_scale', 1.0)))
        end_points['dgt_res_aux'] = gt_center.new_tensor(0.0)
        end_points['dgt_res_gs'] = gt_center.new_tensor(float(getattr(dgt_module, 'gs_residual_scale', 1.0)))
        end_points['dgt_res_geom'] = gt_center.new_tensor(float(getattr(dgt_module, 'geom_residual_scale', 1.0)))
    end_points['corresponding_loss_mask'] = corresponding_loss_mask
    end_points['corresponding_loss_dice'] = corresponding_loss_dice
    end_points['adaptive_weight_loss_mask'] = adaptive_weight_loss_mask
    end_points['adaptive_weight_loss_dice'] = adaptive_weight_loss_dice
    if geom_stats is not None:
        end_points['gs_det_quality'] = geom_stats['box_iou']
        end_points['gs_seg_quality'] = geom_stats['mask_iou']
        end_points['gs_agreement'] = geom_stats['agreement']
        end_points['gs_quality_valid'] = geom_stats.get('quality_valid', gt_center.new_tensor(0.0))
    else:
        end_points['gs_quality_valid'] = gt_center.new_tensor(0.0)

    if dgt_module is not None and dgt_module.enable_dynamic:
        det_group = core_det_base
        seg_group = core_seg_base
        aux_group = aux_base
        core_group = core_base

        if det_group is not None and seg_group is not None and aux_group is not None and core_group is not None:
            if (
                (not bool(torch.isfinite(det_group).all()))
                or (not bool(torch.isfinite(seg_group).all()))
                or (not bool(torch.isfinite(aux_group).all()))
                or (not bool(torch.isfinite(core_group).all()))
            ):
                return loss, end_points
            dgt_module.update_loss_history({
                'det': det_group,
                'seg': seg_group,
                'aux': aux_group,
                'core': core_group,
            }, epoch)

    return loss, end_points
