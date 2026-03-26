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
"""A class to collect and evaluate language grounding results."""

import torch
import torch.distributed as dist

from models.losses import _iou3d_par, box_cxcyczwhd_to_xyzxyz
import utils.misc as misc
import numpy as np
try:
    import wandb
    _WANDB_IMPORT_ERROR = None
except Exception as exc:
    wandb = None
    _WANDB_IMPORT_ERROR = exc

def box2points(box):
    """Convert box center/hwd coordinates to vertices (8x3)."""
    x_min, y_min, z_min = (box[:, :3] - (box[:, 3:] / 2)).transpose(1, 0)
    x_max, y_max, z_max = (box[:, :3] + (box[:, 3:] / 2)).transpose(1, 0)
    return np.stack((
        np.concatenate((x_min[:, None], y_min[:, None], z_min[:, None]), 1),
        np.concatenate((x_min[:, None], y_max[:, None], z_min[:, None]), 1),
        np.concatenate((x_max[:, None], y_min[:, None], z_min[:, None]), 1),
        np.concatenate((x_max[:, None], y_max[:, None], z_min[:, None]), 1),
        np.concatenate((x_min[:, None], y_min[:, None], z_max[:, None]), 1),
        np.concatenate((x_min[:, None], y_max[:, None], z_max[:, None]), 1),
        np.concatenate((x_max[:, None], y_min[:, None], z_max[:, None]), 1),
        np.concatenate((x_max[:, None], y_max[:, None], z_max[:, None]), 1)
    ), axis=1)

def softmax(x):
    """Numpy function for softmax."""
    shape = x.shape
    probs = np.exp(x - np.max(x, axis=len(shape) - 1, keepdims=True))
    probs /= np.sum(probs, axis=len(shape) - 1, keepdims=True)
    return probs

# BRIEF Evaluator
class GroundingEvaluator:
    """
    Evaluate language grounding.

    Args:
        only_root (bool): detect only the root noun
        thresholds (list): IoU thresholds to check
        topks (list): k to evaluate top--k accuracy
        prefixes (list): names of layers to evaluate
    """

    def __init__(self, only_root=True, thresholds=[0.25, 0.5],
                 topks=[1, 5, 10], prefixes=[], filter_non_gt_boxes=False, logger=None, model=None):
        """Initialize accumulators."""
        self.only_root = only_root
        self.thresholds = thresholds
        self.topks = topks
        self.prefixes = prefixes
        self.filter_non_gt_boxes = filter_non_gt_boxes
        self.reset()
        self.logger = logger
        self.model = model
        self.visualization_pred = False
        self.visualization_gt = False
        self.bad_case_visualization = False
        self.kps_points_visualization = False
        self.bad_case_threshold = 0.15
        self._wandb_warned = False

    def _ensure_wandb(self):
        if wandb is not None:
            return True
        if not self._wandb_warned and self.logger is not None:
            self.logger.warning(f"wandb disabled: {_WANDB_IMPORT_ERROR}")
            self._wandb_warned = True
        return False

    def reset(self):
        """Reset accumulators to empty."""
        self.dets = {
            (prefix, t, k, mode): 0
            for prefix in self.prefixes
            for t in self.thresholds
            for k in self.topks
            for mode in ['bbs', 'bbf']
        }
        self.gts = dict(self.dets)

        self.dets.update({'vd': 0, 'vid': 0})
        self.dets.update({'hard': 0, 'easy': 0})
        self.dets.update({'multi': 0, 'unique': 0})
        self.gts.update({'vd': 1e-14, 'vid': 1e-14})
        self.gts.update({'hard': 1e-14, 'easy': 1e-14})
        self.gts.update({'multi': 1e-14, 'unique': 1e-14})
        self.dets.update({'vd50': 0, 'vid50': 0})
        self.dets.update({'hard50': 0, 'easy50': 0})
        self.dets.update({'multi50': 0, 'unique50': 0})
        self.gts.update({'vd50': 1e-14, 'vid50': 1e-14})
        self.gts.update({'hard50': 1e-14, 'easy50': 1e-14})
        self.gts.update({'multi50': 1e-14, 'unique50': 1e-14})
        self.dets.update({'mask_pos': 0})
        self.gts.update({'mask_pos': 1e-14})
        self.dets.update({'mask_sem': 0})
        self.gts.update({'mask_sem': 1e-14})
        self.dets.update({'vd_mask': 0})
        self.dets.update({'vid_mask': 0})
        self.dets.update({'hard_mask': 0})
        self.dets.update({'easy_mask': 0})
        self.dets.update({'unique_mask': 0})
        self.dets.update({'multi_mask': 0})
        self.dets.update({'vd50_mask': 0})
        self.dets.update({'vid50_mask': 0})
        self.dets.update({'hard50_mask': 0})
        self.dets.update({'easy50_mask': 0})
        self.dets.update({'unique50_mask': 0})
        self.dets.update({'multi50_mask': 0})
        self.dets.update({'overall_mask': 0})
        self.dets.update({'overall50_mask': 0})
        self.gts.update({'vd_num': 0})
        self.gts.update({'vid_num': 0})
        self.gts.update({'easy_num': 0})
        self.gts.update({'hard_num': 0})
        self.gts.update({'unique_num': 0})
        self.gts.update({'multi_num': 0})
        # Dataset name for overall metric selection (scanrefer vs nr3d/sr3d)
        self.dataset_name = None

    def _infer_dataset_name(self, end_points):
        if self.dataset_name is not None:
            return
        dataset = end_points.get('language_dataset', None)
        if isinstance(dataset, (list, tuple)):
            dataset = dataset[0] if len(dataset) > 0 else None
        if dataset is None:
            return
        self.dataset_name = str(dataset).lower()

    def _overall_group(self):
        ds = self.dataset_name or ""
        if "scanrefer" in ds:
            return "unique_multi"
        if "nr3d" in ds or "sr3d" in ds:
            return "easy_hard"
        return "auto"

    def _overall_det(self, suffix=""):
        group = self._overall_group()
        if group == "unique_multi":
            return (
                (self.dets['unique' + suffix] + self.dets['multi' + suffix])
                / max(self.gts['unique' + suffix] + self.gts['multi' + suffix], 1)
            ), "unique+multi"
        if group == "easy_hard":
            return (
                (self.dets['easy' + suffix] + self.dets['hard' + suffix])
                / max(self.gts['easy' + suffix] + self.gts['hard' + suffix], 1)
            ), "easy+hard"
        # auto: prefer unique/multi if available, else easy/hard, else vd/vid
        if (self.gts['unique' + suffix] + self.gts['multi' + suffix]) > 0:
            return (
                (self.dets['unique' + suffix] + self.dets['multi' + suffix])
                / max(self.gts['unique' + suffix] + self.gts['multi' + suffix], 1)
            ), "unique+multi"
        if (self.gts['easy' + suffix] + self.gts['hard' + suffix]) > 0:
            return (
                (self.dets['easy' + suffix] + self.dets['hard' + suffix])
                / max(self.gts['easy' + suffix] + self.gts['hard' + suffix], 1)
            ), "easy+hard"
        return (
            (self.dets['vd' + suffix] + self.dets['vid' + suffix])
            / max(self.gts['vd' + suffix] + self.gts['vid' + suffix], 1)
        ), "vd+vid"

    def _overall_mask(self, suffix=""):
        group = self._overall_group()
        if group == "unique_multi":
            return (
                (self.dets['unique' + suffix + '_mask'] + self.dets['multi' + suffix + '_mask'])
                / max(self.gts['unique_num'] + self.gts['multi_num'], 1)
            ), "unique+multi"
        if group == "easy_hard":
            return (
                (self.dets['easy' + suffix + '_mask'] + self.dets['hard' + suffix + '_mask'])
                / max(self.gts['easy_num'] + self.gts['hard_num'], 1)
            ), "easy+hard"
        if (self.gts['unique_num'] + self.gts['multi_num']) > 0:
            return (
                (self.dets['unique' + suffix + '_mask'] + self.dets['multi' + suffix + '_mask'])
                / max(self.gts['unique_num'] + self.gts['multi_num'], 1)
            ), "unique+multi"
        if (self.gts['easy_num'] + self.gts['hard_num']) > 0:
            return (
                (self.dets['easy' + suffix + '_mask'] + self.dets['hard' + suffix + '_mask'])
                / max(self.gts['easy_num'] + self.gts['hard_num'], 1)
            ), "easy+hard"
        return (
            (self.dets['vd' + suffix + '_mask'] + self.dets['vid' + suffix + '_mask'])
            / max(self.gts['vd_num'] + self.gts['vid_num'], 1)
        ), "vd+vid"

    @staticmethod
    def _safe_ratio(num, denom):
        return num / max(denom, 1)

    def print_stats(self):
        """Print accumulated accuracies."""
        mode_str = {
            'bbs': 'position alignment',
            'bbf': 'semantic alignment'
        }
        ds = (self.dataset_name or "").lower()
        is_scanrefer = "scanrefer" in ds
        is_nr3d_sr3d = ("nr3d" in ds) or ("sr3d" in ds)
        for prefix in self.prefixes:
            for mode in ['bbs', 'bbf']:
                for t in self.thresholds:
                    self.logger.info(
                    # print(
                        prefix + ' ' + mode_str[mode] + ' ' +  'Acc%.2f:' % t + ' ' + 
                        ', '.join([
                            'Top-%d: %.5f' % (
                                k,
                                self.dets[(prefix, t, k, mode)]
                                / max(self.gts[(prefix, t, k, mode)], 1)
                            )
                            for k in self.topks
                        ])
                    )
        self.logger.info('\nAnalysis')
        self.logger.info('iou@0.25')
        for field in ['easy', 'hard', 'vd', 'vid', 'unique', 'multi']:
            self.logger.info(field + ' ' +  str(self.dets[field] / self.gts[field]))
        overall_um_25 = self._safe_ratio(
            self.dets['unique'] + self.dets['multi'],
            self.gts['unique'] + self.gts['multi']
        )
        overall_eh_25 = self._safe_ratio(
            self.dets['easy'] + self.dets['hard'],
            self.gts['easy'] + self.gts['hard']
        )
        overall_vv_25 = self._safe_ratio(
            self.dets['vd'] + self.dets['vid'],
            self.gts['vd'] + self.gts['vid']
        )
        overall_25, overall_group = self._overall_det("")
        self.logger.info('Overall(' + overall_group + ') @0.25' + ' ' +  str(overall_25))
        if overall_group != "unique+multi" and not is_nr3d_sr3d:
            self.logger.info('Overall_unique+multi @0.25' + ' ' +  str(overall_um_25))
        if overall_group != "easy+hard" and not is_scanrefer:
            self.logger.info('Overall_easy+hard @0.25' + ' ' +  str(overall_eh_25))
        if overall_group != "vd+vid" and not is_scanrefer:
            self.logger.info('Overall_vd+vid @0.25' + ' ' +  str(overall_vv_25))

        self.logger.info('iou@0.50')
        for field in ['easy50', 'hard50', 'vd50', 'vid50', 'unique50', 'multi50']:
            self.logger.info(field + ' ' +  str(self.dets[field] / self.gts[field]))
        overall_um_50 = self._safe_ratio(
            self.dets['unique50'] + self.dets['multi50'],
            self.gts['unique50'] + self.gts['multi50']
        )
        overall_eh_50 = self._safe_ratio(
            self.dets['easy50'] + self.dets['hard50'],
            self.gts['easy50'] + self.gts['hard50']
        )
        overall_vv_50 = self._safe_ratio(
            self.dets['vd50'] + self.dets['vid50'],
            self.gts['vd50'] + self.gts['vid50']
        )
        overall_50, overall_group_50 = self._overall_det("50")
        self.logger.info('Overall(' + overall_group_50 + ') @0.50' + ' ' +  str(overall_50))
        if overall_group_50 != "unique+multi" and not is_nr3d_sr3d:
            self.logger.info('Overall_unique+multi @0.50' + ' ' +  str(overall_um_50))
        if overall_group_50 != "easy+hard" and not is_scanrefer:
            self.logger.info('Overall_easy+hard @0.50' + ' ' +  str(overall_eh_50))
        if overall_group_50 != "vd+vid" and not is_scanrefer:
            self.logger.info('Overall_vd+vid @0.50' + ' ' +  str(overall_vv_50))

        # cal overall
        self.logger.info('detection statistics by overall')
        self.logger.info('overall_det_25(' + overall_group + ')' + ' ' +  str(overall_25))
        self.logger.info('overall_det_50(' + overall_group_50 + ')' + ' ' +  str(overall_50))
        
#         self.logger.info('total gt unique+mutil' + ' ' +  str( self.gts['unique']+self.gts['multi']))
#         self.logger.info('total gt unique+mutil 50'+ ' ' +  str(self.gts['unique50']+self.gts['multi50']))

        self.logger.info('mask@mean iou')
        self.logger.info('mask_pos' + ' ' +  str(self.dets['mask_pos'] / self.gts['mask_sem']))
        self.logger.info('mask_sem' + ' ' +  str(self.dets['mask_sem'] / self.gts['mask_sem']))
        self.logger.info('mask@kiou')
        if self.gts['unique_num'] != 0:
            self.logger.info('unique25' + ' ' +  str(self.dets['unique_mask'] / self.gts['unique_num']))
            self.logger.info('unique50' + ' ' +  str(self.dets['unique50_mask'] / self.gts['unique_num']))
            self.logger.info('multi25' + ' ' +  str(self.dets['multi_mask'] / self.gts['multi_num']))
            self.logger.info('multi50' + ' ' +  str(self.dets['multi50_mask'] / self.gts['multi_num']))
        self.logger.info('overall25_all' + ' ' +  str(self.dets['overall_mask'] / self.gts['mask_sem']))
        self.logger.info('overall50_all' + ' ' +  str(self.dets['overall50_mask'] / self.gts['mask_sem']))
        self.logger.info('mask@identity')
        self.logger.info('vd25' + ' ' +  str(self.dets['vd_mask'] / self.gts['vd_num']))
        self.logger.info('vd50' + ' ' +  str(self.dets['vd50_mask'] / self.gts['vd_num']))
        self.logger.info('vid25' + ' ' +  str(self.dets['vid_mask'] / self.gts['vid_num']))
        self.logger.info('vid50' + ' ' +  str(self.dets['vid50_mask'] / self.gts['vid_num']))
        self.logger.info('easy25' + ' ' +  str(self.dets['easy_mask'] / self.gts['easy_num']))
        self.logger.info('easy50' + ' ' +  str(self.dets['easy50_mask'] / self.gts['easy_num']))
        self.logger.info('hard25' + ' ' +  str(self.dets['hard_mask'] / self.gts['hard_num']))
        self.logger.info('hard50' + ' ' +  str(self.dets['hard50_mask'] / self.gts['hard_num']))
        overall_mask_25, overall_mask_group = self._overall_mask("")
        overall_mask_50, overall_mask_group_50 = self._overall_mask("50")
        self.logger.info('Overall_mask(' + overall_mask_group + ') @0.25' + ' ' +  str(overall_mask_25))
        self.logger.info('Overall_mask(' + overall_mask_group_50 + ') @0.50' + ' ' +  str(overall_mask_50))
        overall_mask_um_25 = self._safe_ratio(
            self.dets['unique_mask'] + self.dets['multi_mask'],
            self.gts['unique_num'] + self.gts['multi_num']
        )
        overall_mask_eh_25 = self._safe_ratio(
            self.dets['easy_mask'] + self.dets['hard_mask'],
            self.gts['easy_num'] + self.gts['hard_num']
        )
        overall_mask_vv_25 = self._safe_ratio(
            self.dets['vd_mask'] + self.dets['vid_mask'],
            self.gts['vd_num'] + self.gts['vid_num']
        )
        overall_mask_um_50 = self._safe_ratio(
            self.dets['unique50_mask'] + self.dets['multi50_mask'],
            self.gts['unique_num'] + self.gts['multi_num']
        )
        overall_mask_eh_50 = self._safe_ratio(
            self.dets['easy50_mask'] + self.dets['hard50_mask'],
            self.gts['easy_num'] + self.gts['hard_num']
        )
        overall_mask_vv_50 = self._safe_ratio(
            self.dets['vd50_mask'] + self.dets['vid50_mask'],
            self.gts['vd_num'] + self.gts['vid_num']
        )
        if overall_mask_group != "unique+multi" and not is_nr3d_sr3d:
            self.logger.info('Overall_mask_unique+multi @0.25' + ' ' +  str(overall_mask_um_25))
        if overall_mask_group != "easy+hard" and not is_scanrefer:
            self.logger.info('Overall_mask_easy+hard @0.25' + ' ' +  str(overall_mask_eh_25))
        if overall_mask_group != "vd+vid" and not is_scanrefer:
            self.logger.info('Overall_mask_vd+vid @0.25' + ' ' +  str(overall_mask_vv_25))
        if overall_mask_group_50 != "unique+multi" and not is_nr3d_sr3d:
            self.logger.info('Overall_mask_unique+multi @0.50' + ' ' +  str(overall_mask_um_50))
        if overall_mask_group_50 != "easy+hard" and not is_scanrefer:
            self.logger.info('Overall_mask_easy+hard @0.50' + ' ' +  str(overall_mask_eh_50))
        if overall_mask_group_50 != "vd+vid" and not is_scanrefer:
            self.logger.info('Overall_mask_vd+vid @0.50' + ' ' +  str(overall_mask_vv_50))



    def synchronize_between_processes(self):
        if not misc.is_dist_avail_and_initialized():
            return
        if dist.get_world_size() == 1:
            return

        # Sync after all ranks finish eval loop to avoid mismatched collectives.
        dist.barrier()

        def _sync_sum_dict(data_dict):
            keys = list(data_dict.keys())
            if not keys:
                return data_dict
            tuple_keys = sorted([k for k in keys if isinstance(k, tuple)])
            str_keys = sorted([k for k in keys if isinstance(k, str)])
            other_keys = [k for k in keys if not isinstance(k, (tuple, str))]
            if other_keys:
                other_keys = sorted(other_keys, key=lambda x: repr(x))
            keys = tuple_keys + str_keys + other_keys
            device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
            values = torch.tensor([float(data_dict[k]) for k in keys], device=device)
            dist.all_reduce(values, op=dist.ReduceOp.SUM)
            return {k: float(v.item()) for k, v in zip(keys, values)}

        self.dets = _sync_sum_dict(self.dets)
        self.gts = _sync_sum_dict(self.gts)

    # BRIEF Evaluation
    def evaluate(self, end_points, prefix):
        """
        Evaluate all accuracies.

        Args:
            end_points (dict): contains predictions and gt
            prefix (str): layer name
        """
        self._infer_dataset_name(end_points)
        # NOTE Two Evaluation Ways: position alignment, semantic alignment
        self.evaluate_bbox_by_pos_align(end_points, prefix)
        self.evaluate_bbox_by_sem_align(end_points, prefix)
        self.evaluate_masks_by_pos_align(end_points, prefix)
        self.evaluate_masks_by_sem_align(end_points, prefix)
    
    # BRIEF position alignment
    def evaluate_bbox_by_pos_align(self, end_points, prefix):
        """
        Evaluate bounding box IoU by position alignment

        Args:
            end_points (dict): contains predictions and gt
            prefix (str): layer name
        """
        # step get the position label and GT box 
        positive_map, modify_positive_map, pron_positive_map, other_entity_map, \
            auxi_entity_positive_map, rel_positive_map, gt_bboxes = self._parse_gt(end_points)    
        
        # Parse predictions
        sem_scores = end_points[f'{prefix}sem_cls_scores'].softmax(-1)

        if sem_scores.shape[-1] != positive_map.shape[-1]:
            sem_scores_ = torch.zeros(
                sem_scores.shape[0], sem_scores.shape[1],
                positive_map.shape[-1]).to(sem_scores.device)
            sem_scores_[:, :, :sem_scores.shape[-1]] = sem_scores
            sem_scores = sem_scores_

        # Parse predictions
        pred_center = end_points[f'{prefix}center']  # B, Q=256, 3
        pred_size = end_points[f'{prefix}pred_size']  # (B,Q,3) (l,w,h)
        assert (pred_size < 0).sum() == 0
        pred_bbox = torch.cat([pred_center, pred_size], dim=-1) # ([B, 256, 6])

        # Highest scoring box -> iou
        for bid in range(len(positive_map)):
            is_correct = None
            if self.filter_non_gt_boxes:  # this works only for the target box
                ious, _ = _iou3d_par(
                    box_cxcyczwhd_to_xyzxyz(
                        end_points['all_detected_boxes'][bid][
                            end_points['all_detected_bbox_label_mask'][bid]
                        ]
                    ),  # (gt, 6)
                    box_cxcyczwhd_to_xyzxyz(pred_bbox[bid])  # (Q, 6)
                )  # (gt, Q)
                is_correct = (ious.max(0)[0] > 0.25) * 1.0
            
            # Keep scores for annotated objects only
            num_obj = int(end_points['box_label_mask'][bid].sum())
            pmap = positive_map[bid, :num_obj]
            scores_main = (
                sem_scores[bid].unsqueeze(0)    
                * pmap.unsqueeze(1)             
            ).sum(-1)

            # score
            pmap_modi = modify_positive_map[bid, :1]
            pmap_pron = pron_positive_map[bid, :1]
            pmap_other = other_entity_map[bid, :1]
            pmap_rel = rel_positive_map[bid, :1]    # num_obj
            scores_modi = (
                sem_scores[bid].unsqueeze(0)    
                * pmap_modi.unsqueeze(1)             
            ).sum(-1)
            scores_pron = (
                sem_scores[bid].unsqueeze(0)    
                * pmap_pron.unsqueeze(1)             
            ).sum(-1)
            scores_other = (
                sem_scores[bid].unsqueeze(0)    
                * pmap_other.unsqueeze(1)             
            ).sum(-1)
            scores_rel = (
                sem_scores[bid].unsqueeze(0)    
                * pmap_rel.unsqueeze(1)             
            ).sum(-1)

            scores = scores_main + scores_modi + scores_pron + scores_rel - scores_other

            if is_correct is not None:
                scores = scores * is_correct[None]

            top = scores.argsort(1, True)[:, :10]
            pbox = pred_bbox[bid, top.reshape(-1)]

            ious, _ = _iou3d_par(
                box_cxcyczwhd_to_xyzxyz(gt_bboxes[bid][:num_obj]),  # (obj, 6)
                box_cxcyczwhd_to_xyzxyz(pbox)  # (obj*10, 6)
            )  # (obj, obj*10)
            ious = ious.reshape(top.size(0), top.size(0), top.size(1))
            ious = ious[torch.arange(len(ious)), torch.arange(len(ious))]   # ([1, 10])

            # step Measure IoU>threshold, ious are (obj, 10)
            topks = self.topks
            for t in self.thresholds:
                thresholded = ious > t
                for k in topks:
                    found = thresholded[:, :k].any(1)
                    self.dets[(prefix, t, k, 'bbs')] += found.sum().item()
                    self.gts[(prefix, t, k, 'bbs')] += len(thresholded)

    # BRIEF semantic alignment
    def evaluate_bbox_by_sem_align(self, end_points, prefix):
        """
        Evaluate bounding box IoU by semantic alignment.

        Args:
            end_points (dict): contains predictions and gt
            prefix (str): layer name
        """
        # step get the position label and GT box 
        positive_map, modify_positive_map, pron_positive_map, other_entity_map, \
            auxi_entity_positive_map, rel_positive_map, gt_bboxes = self._parse_gt(end_points)    
        
        # Parse predictions
        pred_center = end_points[f'{prefix}center']  # B, Q, 3
        pred_size = end_points[f'{prefix}pred_size']  # (B,Q,3) (l,w,h)

        assert (pred_size < 0).sum() == 0
        pred_bbox = torch.cat([pred_center, pred_size], dim=-1)
        
        # step compute similarity between vision and text
        proj_tokens = end_points['proj_tokens']             # text feature   (B, 256, 64)
        proj_queries = end_points[f'{prefix}proj_queries']  # vision feature (B, 256, 64)
        sem_scores = torch.matmul(proj_queries, proj_tokens.transpose(-1, -2))  # similarity ([B, 256, L]) 
        sem_scores_ = (sem_scores / 0.07).softmax(-1)                           # softmax ([B, 256, L])
        sem_scores = torch.zeros(sem_scores_.size(0), sem_scores_.size(1), 256) # ([B, 256, 256])
        sem_scores = sem_scores.to(sem_scores_.device)
        sem_scores[:, :sem_scores_.size(1), :sem_scores_.size(2)] = sem_scores_ # ([B, P=256, L=256])

        # Highest scoring box -> iou
        for bid in range(len(positive_map)):
            is_correct = None
            if self.filter_non_gt_boxes:  # this works only for the target box
                ious, _ = _iou3d_par(
                    box_cxcyczwhd_to_xyzxyz(
                        end_points['all_detected_boxes'][bid][
                            end_points['all_detected_bbox_label_mask'][bid]
                        ]
                    ),  # (gt, 6)
                    box_cxcyczwhd_to_xyzxyz(pred_bbox[bid])  # (Q, 6)
                )  # (gt, Q)
                is_correct = (ious.max(0)[0] > 0.25) * 1.0
            
            # Keep scores for annotated objects only
            num_obj = int(end_points['box_label_mask'][bid].sum())
            pmap = positive_map[bid, :num_obj]
            scores_main = (
                sem_scores[bid].unsqueeze(0)  # (1, Q, 256)
                * pmap.unsqueeze(1)  # (obj, 1, 256)
            ).sum(-1)  # (obj, Q)
            
            # score
            pmap_modi = modify_positive_map[bid, :1]
            pmap_pron = pron_positive_map[bid, :1]
            pmap_other = other_entity_map[bid, :1]
            pmap_auxi = auxi_entity_positive_map[bid, :1]
            pmap_rel = rel_positive_map[bid, :1]
            scores_modi = (
                sem_scores[bid].unsqueeze(0)    
                * pmap_modi.unsqueeze(1)             
            ).sum(-1)
            scores_pron = (
                sem_scores[bid].unsqueeze(0)    
                * pmap_pron.unsqueeze(1)             
            ).sum(-1)
            scores_other = (
                sem_scores[bid].unsqueeze(0)    
                * pmap_other.unsqueeze(1)             
            ).sum(-1)
            scores_auxi = (
                sem_scores[bid].unsqueeze(0)    
                * pmap_auxi.unsqueeze(1)             
            ).sum(-1)
            scores_rel = (
                sem_scores[bid].unsqueeze(0)    
                * pmap_rel.unsqueeze(1)             
            ).sum(-1)

            # total score
            scores = scores_main + scores_modi + scores_pron + scores_rel - scores_other

            if is_correct is not None:
                scores = scores * is_correct[None]

            # 10 predictions per gt box
            top = scores.argsort(1, True)[:, :10]  # (obj, 10)
            pbox = pred_bbox[bid, top.reshape(-1)]

            # IoU
            ious, _ = _iou3d_par(
                box_cxcyczwhd_to_xyzxyz(gt_bboxes[bid][:num_obj]),  # (obj, 6)
                box_cxcyczwhd_to_xyzxyz(pbox)  # (obj*10, 6)
            )  # (obj, obj*10)
            ious = ious.reshape(top.size(0), top.size(0), top.size(1))
            ious = ious[torch.arange(len(ious)), torch.arange(len(ious))]

            # Check for bad cases
            if self.bad_case_visualization and self._ensure_wandb():
                wandb.init(project="vis", name="badcase")
                bad_cases = ious < self.bad_case_threshold  # Here you set your bad_case_threshold
                if bad_cases.any():
                    # Get point cloud and original color
                    point_cloud = end_points['point_clouds'][bid]
                    og_color = end_points['og_color'][bid]
                    point_cloud[:, 3:] = (og_color + torch.tensor([109.8, 97.2, 83.8]).cuda() / 256) * 256
                    target_name = end_points['target_name'][bid]
                    utterances = end_points['utterances'][bid]

                    # Get all boxes and predicted boxes
                    topk_boxes = 0
                    all_bboxes = end_points['all_bboxes'][bid].cpu()
                    pbox_bad_cases = pbox[bad_cases[0]].cpu()[topk_boxes].unsqueeze(0)  # top 1
                    gt_box = gt_bboxes[bid].cpu()

                    # Convert boxes to points for visualization
                    all_boxes_points = box2points(all_bboxes[..., :6])  # all boxes
                    gt_box = box2points(gt_box[..., :6])  # gt boxes
                    pbox_bad_cases_points = box2points(pbox_bad_cases[..., :6])

                    # Log bad case visualization to wandb
                    wandb.log({
                        "bad_case_point_scene": wandb.Object3D({
                            "type": "lidar/beta",
                            "points": point_cloud,
                            "boxes": np.array(
                                [  # actual boxes
                                    {
                                        "corners": c.tolist(),
                                        "label": "actual",
                                        "color": [0, 255, 0]
                                    }
                                    for c in gt_box
                                ]
                                + [  # predicted boxes
                                    {
                                        "corners": c.tolist(),
                                        "label": "predicted",
                                        "color": [255, 0, 0]
                                    }
                                    for c in pbox_bad_cases_points
                                ]
                            )
                        }),
                        "target_name": wandb.Html(target_name),
                        "utterance": wandb.Html(utterances),
                    })

            # Check for kps points
            if self.kps_points_visualization and self._ensure_wandb():
                wandb.init(project="vis", name="kps_points")
                point_cloud = end_points['point_clouds'][bid]
                og_color = end_points['og_color'][bid]
                point_cloud[:, 3:] = (og_color + torch.tensor([109.8, 97.2, 83.8]).cuda() / 256) * 256
                kps_points = end_points['query_points_xyz'][bid]
                red = torch.zeros((256, 3)).cuda()
                red[:, 0] = 255.0
                kps_points = torch.cat([kps_points, red], dim=1)
                total_point = torch.cat([point_cloud, kps_points], dim=0)
                utterances = end_points['utterances'][bid]
                gt_box = gt_bboxes[bid].cpu()
                gt_box = box2points(gt_box[..., :6])

                wandb.log({
                        "kps_point_scene": wandb.Object3D({
                            "type": "lidar/beta",
                            "points": total_point,
                            "boxes": np.array(
                                [
                                    {
                                        "corners": c.tolist(),
                                        "label": "target",
                                        "color": [0, 255, 0]
                                    }
                                    for c in gt_box
                                ]
                            )
                        }),
                        "utterance": wandb.Html(utterances),
                    })

            # step Measure IoU>threshold, ious are (obj, 10)
            for t in self.thresholds:
                thresholded = ious > t
                for k in self.topks:
                    found = thresholded[:, :k].any(1)
                    self.dets[(prefix, t, k, 'bbf')] += found.sum().item()
                    self.gts[(prefix, t, k, 'bbf')] += len(thresholded)
                    if prefix == 'last_':
                        found = found[0].item()
                        if k == 1 and t == self.thresholds[0]:
                            if end_points['is_view_dep'][bid]:
                                self.gts['vd'] += 1
                                self.dets['vd'] += found
                            else:
                                self.gts['vid'] += 1
                                self.dets['vid'] += found
                            if end_points['is_hard'][bid]:
                                self.gts['hard'] += 1
                                self.dets['hard'] += found
                            else:
                                self.gts['easy'] += 1
                                self.dets['easy'] += found
                            if end_points['is_unique'][bid]:
                                self.gts['unique'] += 1
                                self.dets['unique'] += found
                            else:
                                self.gts['multi'] += 1
                                self.dets['multi'] += found
                        if k == 1 and t == self.thresholds[1]:
                            if end_points['is_view_dep'][bid]:
                                self.gts['vd50'] += 1
                                self.dets['vd50'] += found
                            else:
                                self.gts['vid50'] += 1
                                self.dets['vid50'] += found
                            if end_points['is_hard'][bid]:
                                self.gts['hard50'] += 1
                                self.dets['hard50'] += found
                            else:
                                self.gts['easy50'] += 1
                                self.dets['easy50'] += found
                            if end_points['is_unique'][bid]:
                                self.gts['unique50'] += 1
                                self.dets['unique50'] += found
                            else:
                                self.gts['multi50'] += 1
                                self.dets['multi50'] += found


    # BRIEF Get the postion label of the decoupled text component.
    def _parse_gt(self, end_points):
        positive_map = torch.clone(end_points['positive_map'])                  # main
        modify_positive_map = torch.clone(end_points['modify_positive_map'])    # attribute
        pron_positive_map = torch.clone(end_points['pron_positive_map'])        # pron
        other_entity_map = torch.clone(end_points['other_entity_map'])          # other(including auxi)
        auxi_entity_positive_map = torch.clone(end_points['auxi_entity_positive_map'])  # auxi
        rel_positive_map = torch.clone(end_points['rel_positive_map'])

        positive_map[positive_map > 0] = 1                      
        gt_center = end_points['center_label'][:, :, 0:3]       
        gt_size = end_points['size_gts']                        
        gt_bboxes = torch.cat([gt_center, gt_size], dim=-1)     # GT box cxcyczwhd
        
        if self.only_root:
            positive_map = positive_map[:, :1]  # (B, 1, 256)
            gt_bboxes = gt_bboxes[:, :1]        # (B, 1, 6)
        
        return positive_map, modify_positive_map, pron_positive_map, other_entity_map, auxi_entity_positive_map, \
            rel_positive_map, gt_bboxes
    

    # BRIEF position alignment
    def evaluate_masks_by_pos_align(self, end_points, prefix):
        """
        Evaluate masks IoU by position alignment

        Args:
            end_points (dict): contains predictions and gt
            prefix (str): layer name
        """
        # step get the position label and GT box 
        positive_map, modify_positive_map, pron_positive_map, other_entity_map, \
            auxi_entity_positive_map, rel_positive_map, gt_masks = self._parse_gt_mask(end_points)    
        
        # Parse predictions
        sem_scores = end_points[f'{prefix}sem_cls_scores'].softmax(-1)  # [B, 256 256]

        if sem_scores.shape[-1] != positive_map.shape[-1]:
            sem_scores_ = torch.zeros(
                sem_scores.shape[0], sem_scores.shape[1],
                positive_map.shape[-1]).to(sem_scores.device)
            sem_scores_[:, :, :sem_scores.shape[-1]] = sem_scores
            sem_scores = sem_scores_

        # Parse predictions
        if self.model == "ThreeDRefTR_SP":
            pred_masks = []
            for bs in range(len(end_points['last_pred_masks'])):
                pred_masks_ = end_points['last_pred_masks'][bs].unsqueeze(0)  # ([1, 256, super_num])
                pred_masks_ = (pred_masks_.sigmoid() > 0.5).int()
                superpoints = end_points['superpoints'][bs].unsqueeze(0)  # (1, 50000)
                pred_masks_ = torch.gather(pred_masks_, 2, superpoints.unsqueeze(1).expand(-1, 256, -1))  # (1, 256, 50000)
                pred_masks.append(pred_masks_.squeeze(0))

            pred_masks = torch.stack(pred_masks, dim=0)  # (B, 256, 50000)
        elif self.model == "ThreeDRefTR_HR":
            pred_masks = end_points['last_pred_masks'] # ([B, 256, 50000])
            pred_masks = (pred_masks.sigmoid() > 0.5).int()
        
        else:# default: same format as ThreeDRefTR_SP
            pred_masks = []
            for bs in range(len(end_points['last_pred_masks'])):
                pred_masks_ = end_points['last_pred_masks'][bs] # ([1, 256, super_num])
                adaptive_weight=end_points['adaptive_weights'][bs]
                sp_pred_masks_=end_points['sp_last_pred_masks'][bs].unsqueeze(0)
                pred_masks_=(adaptive_weight*pred_masks_+(1-adaptive_weight)*sp_pred_masks_).sigmoid()
                pred_masks_ = (pred_masks_ > 0.5).int()
                superpoints = end_points['superpoints'][bs].unsqueeze(0)  # (1, 50000)
                pred_masks_ = torch.gather(pred_masks_, 2, superpoints.unsqueeze(1).expand(-1, 256, -1))  # (1, 256, 50000)
                pred_masks.append(pred_masks_.squeeze(0))
            pred_masks = torch.stack(pred_masks, dim=0)  # (B, 256, 50000)

        # Highest scoring box -> iou
        for bid in range(len(positive_map)):
            is_correct = None
            
            # Keep scores for annotated objects only
            num_obj = int(end_points['box_label_mask'][bid].sum())
            pmap = positive_map[bid, :num_obj]
            scores_main = (
                sem_scores[bid].unsqueeze(0)    
                * pmap.unsqueeze(1)             
            ).sum(-1)

            # score
            pmap_modi = modify_positive_map[bid, :1]
            pmap_pron = pron_positive_map[bid, :1]
            pmap_other = other_entity_map[bid, :1]
            pmap_rel = rel_positive_map[bid, :1]    # num_obj
            scores_modi = (
                sem_scores[bid].unsqueeze(0)    
                * pmap_modi.unsqueeze(1)             
            ).sum(-1)
            scores_pron = (
                sem_scores[bid].unsqueeze(0)    
                * pmap_pron.unsqueeze(1)             
            ).sum(-1)
            scores_other = (
                sem_scores[bid].unsqueeze(0)    
                * pmap_other.unsqueeze(1)             
            ).sum(-1)
            scores_rel = (
                sem_scores[bid].unsqueeze(0)    
                * pmap_rel.unsqueeze(1)             
            ).sum(-1)

            scores = scores_main + scores_modi + scores_pron + scores_rel - scores_other  # [1, 256]

            if is_correct is not None:
                scores = scores * is_correct[None]

            top = scores.argsort(1, True)[:, :1]  # top-1 mask
            pmasks = pred_masks[bid, top.reshape(-1)]

            # comupte mask iou
            iou_score_pos = self.calculate_masks_iou(pmasks, gt_masks[bid])
            # print("{:.14f}".format(iou_score_pos))
            self.gts['mask_pos'] += 1
            self.dets['mask_pos'] += iou_score_pos


    # BRIEF semantic alignment
    def evaluate_masks_by_sem_align(self, end_points, prefix):
        """
        Evaluate masks IoU by semantic alignment.

        Args:
            end_points (dict): contains predictions and gt
            prefix (str): layer name
        """
        # step get the position label and GT box 
        positive_map, modify_positive_map, pron_positive_map, other_entity_map, \
            auxi_entity_positive_map, rel_positive_map, gt_masks = self._parse_gt_mask(end_points)    
        
        # Parse predictions
        if self.model == "ThreeDRefTR_SP":
            pred_masks = []
            for bs in range(len(end_points['last_pred_masks'])):
                pred_masks_ = end_points['last_pred_masks'][bs].unsqueeze(0)  # ([1, 256, super_num])
                pred_masks_ = (pred_masks_.sigmoid() > 0.5).int()
                superpoints = end_points['superpoints'][bs].unsqueeze(0)  # (1, 50000)
                pred_masks_ = torch.gather(pred_masks_, 2, superpoints.unsqueeze(1).expand(-1, 256, -1))  # (1, 256, 50000)
                pred_masks.append(pred_masks_.squeeze(0))

            pred_masks = torch.stack(pred_masks, dim=0)  # (B, 256, 50000)
        elif self.model == "ThreeDRefTR_HR":
            pred_masks = end_points['last_pred_masks'] # ([B, 256, 50000])
            pred_masks = (pred_masks.sigmoid() > 0.5).int()
        else:
            pred_masks = []
            for bs in range(len(end_points['last_pred_masks'])):
                pred_masks_ = end_points['last_pred_masks'][bs]  # ([1, 256, super_num])
                sp_pred_masks_=end_points['sp_last_pred_masks'][bs].unsqueeze(0)
                adaptive_weight=end_points['adaptive_weights'][bs]
                pred_masks_=(adaptive_weight*pred_masks_+(1-adaptive_weight)*sp_pred_masks_).sigmoid()
                pred_masks_ = (pred_masks_> 0.5).int()
                superpoints = end_points['superpoints'][bs].unsqueeze(0)  # (1, 50000)
                pred_masks_ = torch.gather(pred_masks_, 2, superpoints.unsqueeze(1).expand(-1, 256, -1))  # (1, 256, 50000)
                pred_masks.append(pred_masks_.squeeze(0))
            pred_masks = torch.stack(pred_masks, dim=0)  # (B, 256, 50000)

        
        # step compute similarity between vision and text
        proj_tokens = end_points['proj_tokens']             # text feature   (B, 256, 64)
        proj_queries = end_points[f'{prefix}proj_queries']  # vision feature (B, 256, 64)
        sem_scores = torch.matmul(proj_queries, proj_tokens.transpose(-1, -2))  # similarity ([B, 256, L]) 
        sem_scores_ = (sem_scores / 0.07).softmax(-1)                           # softmax ([B, 256, L])
        sem_scores = torch.zeros(sem_scores_.size(0), sem_scores_.size(1), 256) # ([B, 256, 256])
        sem_scores = sem_scores.to(sem_scores_.device)
        sem_scores[:, :sem_scores_.size(1), :sem_scores_.size(2)] = sem_scores_ # ([B, P=256, L=256])

        # Highest scoring box -> iou
        for bid in range(len(positive_map)):
            is_correct = None
            
            # Keep scores for annotated objects only
            num_obj = int(end_points['box_label_mask'][bid].sum())
            pmap = positive_map[bid, :num_obj]
            scores_main = (
                sem_scores[bid].unsqueeze(0)  # (1, Q, 256)
                * pmap.unsqueeze(1)  # (obj, 1, 256)
            ).sum(-1)  # (obj, Q)
            
            # score
            pmap_modi = modify_positive_map[bid, :1]
            pmap_pron = pron_positive_map[bid, :1]
            pmap_other = other_entity_map[bid, :1]
            pmap_auxi = auxi_entity_positive_map[bid, :1]
            pmap_rel = rel_positive_map[bid, :1]
            scores_modi = (
                sem_scores[bid].unsqueeze(0)    
                * pmap_modi.unsqueeze(1)             
            ).sum(-1)
            scores_pron = (
                sem_scores[bid].unsqueeze(0)    
                * pmap_pron.unsqueeze(1)             
            ).sum(-1)
            scores_other = (
                sem_scores[bid].unsqueeze(0)    
                * pmap_other.unsqueeze(1)             
            ).sum(-1)
            scores_auxi = (
                sem_scores[bid].unsqueeze(0)    
                * pmap_auxi.unsqueeze(1)             
            ).sum(-1)
            scores_rel = (
                sem_scores[bid].unsqueeze(0)    
                * pmap_rel.unsqueeze(1)             
            ).sum(-1)

            # total score
            scores = scores_main + scores_modi + scores_pron + scores_rel - scores_other

            if is_correct is not None:
                scores = scores * is_correct[None]

            top = scores.argsort(1, True)[:, :1]
            pmasks = pred_masks[bid, top.reshape(-1)]

            # compute IoU
            iou_score_sem = self.calculate_masks_iou(pmasks, gt_masks[bid])
            # print("{:.14f}".format(iou_score_sem))
            self.gts['mask_sem'] += 1
            self.dets['mask_sem'] += iou_score_sem

            if end_points['is_view_dep'][bid]:
                self.gts['vd_num'] += 1
            else:
                self.gts['vid_num'] += 1
            if end_points['is_unique'][bid]:
                self.gts['unique_num'] += 1
            else:
                self.gts['multi_num'] += 1
            if end_points['is_hard'][bid]:
                self.gts['hard_num'] += 1
            else:
                self.gts['easy_num'] += 1

            if iou_score_sem > 0.25:
                self.dets['overall_mask'] += 1
                if end_points['is_view_dep'][bid]:
                    self.dets['vd_mask'] += 1
                else:
                    self.dets['vid_mask'] += 1
                if end_points['is_hard'][bid]:
                    self.dets['hard_mask'] += 1
                else:
                    self.dets['easy_mask'] += 1
                if end_points['is_unique'][bid]:
                    self.dets['unique_mask'] += 1
                else:
                    self.dets['multi_mask'] += 1
            if iou_score_sem > 0.5:
                self.dets['overall50_mask'] += 1
                if end_points['is_view_dep'][bid]:
                    self.dets['vd50_mask'] += 1
                else:
                    self.dets['vid50_mask'] += 1
                if end_points['is_hard'][bid]:
                    self.dets['hard50_mask'] += 1
                else:
                    self.dets['easy50_mask'] += 1
                if end_points['is_unique'][bid]:
                    self.dets['unique50_mask'] += 1
                else:
                    self.dets['multi50_mask'] += 1

            # visualization for pres mask and box
            if self.visualization_pred and self._ensure_wandb():
                wandb.init(project="vis", name="pred")
                point_cloud = end_points['point_clouds'][bid]
                og_color = end_points['og_color'][bid]
                point_cloud[:, 3:] = (og_color + torch.tensor([109.8, 97.2, 83.8]).cuda() / 256) * 256
                red = torch.tensor([255.0, 0.0, 0.0]).cuda()

                pred_center = end_points[f'{prefix}center'][bid]
                pred_size = end_points[f'{prefix}pred_size'][bid]
                pred_bbox = torch.cat([pred_center, pred_size], dim=-1).cpu()[top.reshape(-1)]

                utterances = end_points['utterances'][bid]
                pred_bbox = box2points(pred_bbox[..., :6])

                mask_idx = pmasks[0] == 1
                pred_cloud = point_cloud
                pred_cloud[mask_idx, 3:] = red

                wandb.log({
                        "point_scene": wandb.Object3D({
                            "type": "lidar/beta",
                            "points": pred_cloud,
                            "boxes": np.array(
                                [ 
                                    {
                                        "corners": c.tolist(),
                                        "label": "predicted",
                                        "color": [0, 0, 255]
                                    }
                                    for c in pred_bbox
                                ]
                            )
                        }),
                        "utterance": wandb.Html(utterances),
                    })
                

            # visualization for gt mask and box
            if self.visualization_gt and self._ensure_wandb():
                wandb.init(project="vis", name="gt")
                point_cloud = end_points['point_clouds'][bid]
                og_color = end_points['og_color'][bid]
                point_cloud[:, 3:] = (og_color + torch.tensor([109.8, 97.2, 83.8]).cuda() / 256) * 256
                blue = torch.tensor([0.0, 0.0, 255.0]).cuda()

                gt_center = end_points['center_label'][bid, :, 0:3]       
                gt_size = end_points['size_gts'][bid]                        
                gt_box = torch.cat([gt_center, gt_size], dim=-1).cpu()

                utterances = end_points['utterances'][bid]
                gt_box = box2points(gt_box[..., :6])

                gt_cloud = point_cloud
                gt_mask_idx = gt_masks[bid][0] == 1
                gt_cloud[gt_mask_idx, 3:] = blue

                wandb.log({
                        "point_scene": wandb.Object3D({
                            "type": "lidar/beta",
                            "points": gt_cloud,
                            "boxes": np.array(
                                [
                                    {
                                        "corners": c.tolist(),
                                        "label": "target",
                                        "color": [0, 255, 0]
                                    }
                                    for c in gt_box
                                ]
                            )
                        }),
                        "utterance": wandb.Html(utterances),
                    })
            

    # BRIEF Get the postion label of the decoupled text component.
    def _parse_gt_mask(self, end_points):
        positive_map = torch.clone(end_points['positive_map'])                  # main
        modify_positive_map = torch.clone(end_points['modify_positive_map'])    # attribute
        pron_positive_map = torch.clone(end_points['pron_positive_map'])        # pron
        other_entity_map = torch.clone(end_points['other_entity_map'])          # other(including auxi)
        auxi_entity_positive_map = torch.clone(end_points['auxi_entity_positive_map'])  # auxi
        rel_positive_map = torch.clone(end_points['rel_positive_map'])

        positive_map[positive_map > 0] = 1                      
        gt_masks = end_points['gt_masks']   
        
        if self.only_root:
            positive_map = positive_map[:, :1]  # (B, 1, 256)
            gt_masks = gt_masks[:, :1]        # (B, 1, 50000)
        
        return positive_map, modify_positive_map, pron_positive_map, other_entity_map, auxi_entity_positive_map, \
            rel_positive_map, gt_masks
    
    def calculate_masks_iou(self, mask1, mask2):
        mask1, mask2 = mask1.cpu().numpy(), mask2.cpu().numpy()
        intersection = np.logical_and(mask1, mask2)
        union = np.logical_or(mask1, mask2)
        iou_score = np.sum(intersection) / np.sum(union)
        return iou_score
