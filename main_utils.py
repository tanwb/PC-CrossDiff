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
"""Shared utilities for all main scripts."""

import argparse
import logging
import re
import json
import os
import random
import time

import numpy as np
import torch
import torch.optim as optim
from torch.utils.data import DataLoader, Sampler
from torch.utils.data.distributed import DistributedSampler
import torch.distributed as dist
from torch.nn.parallel import DistributedDataParallel

from models import HungarianMatcher, SetCriterion, compute_hungarian_loss
from utils import get_scheduler, setup_logger

from utils import record_tensorboard

from tqdm import tqdm
from models.losses import DGTLossModule


class DistributedEvalSampler(Sampler):
    """Split evaluation data across ranks without padding or duplication."""

    def __init__(self, dataset, num_replicas=None, rank=None):
        if num_replicas is None:
            if dist.is_available() and dist.is_initialized():
                num_replicas = dist.get_world_size()
            else:
                num_replicas = 1
        if rank is None:
            if dist.is_available() and dist.is_initialized():
                rank = dist.get_rank()
            else:
                rank = 0
        self.dataset = dataset
        self.num_replicas = int(num_replicas)
        self.rank = int(rank)

    def __iter__(self):
        return iter(range(self.rank, len(self.dataset), self.num_replicas))

    def __len__(self):
        if len(self.dataset) <= self.rank:
            return 0
        return (len(self.dataset) - 1 - self.rank) // self.num_replicas + 1

    def set_epoch(self, epoch):
        return None


def parse_option():
    """Parse cmd arguments."""
    parser = argparse.ArgumentParser()
    # Model
    parser.add_argument('--num_target', type=int, default=256,
                        help='Proposal number')
    parser.add_argument('--sampling', default='kps', type=str,
                        help='Query points sampling method (kps, fps)')

    # Transformer
    parser.add_argument('--num_encoder_layers', default=3, type=int)
    parser.add_argument('--num_decoder_layers', default=6, type=int)  # 6
    parser.add_argument('--self_position_embedding', default='loc_learned',
                        type=str, help='(none, xyz_learned, loc_learned)')
    parser.add_argument('--self_attend', action='store_true')
    parser.add_argument('--model', type=str, default='BeaUTyDETR')

    # Loss
    parser.add_argument('--query_points_obj_topk', default=4, type=int)
    parser.add_argument('--use_contrastive_align', action='store_true')
    parser.add_argument('--use_soft_token_loss', action='store_true')
    parser.add_argument('--detect_intermediate', action='store_true')
    parser.add_argument('--joint_det', action='store_true')
    parser.add_argument('--dgt_warmup_epochs', type=int, default=5,
                        help='DGTL warmup epochs (use fixed weights)')
    parser.add_argument('--dgt_start_epoch', type=int, default=24,
                        help='DGTL start epoch (enable dynamic weights)')
    parser.add_argument('--dgt_stop_epoch', type=int, default=42,
                        help='DGTL stop epoch (freeze dynamic weights)')
    parser.add_argument('--dgt_diff_threshold', type=float, default=0.03,
                        help='DGTL slope diff threshold (lower = easier to trigger updates)')
    parser.add_argument('--dgt_update_interval', type=int, default=2,
                        help='DGTL update interval (epochs)')
    parser.add_argument('--dgt_ema_alpha', type=float, default=0.2,
                        help='DGTL EMA alpha for loss smoothing')
    parser.add_argument('--dgt_hold_when_stable', type=int, default=1,
                        help='DGTL hold when stable (1=on, 0=off)')
    parser.add_argument('--dgt_patience', type=int, default=2,
                        help='DGTL patience for stagnation counter')
    parser.add_argument('--dgt_min_epoch_gap', type=int, default=1,
                        help='DGTL minimum epochs between weight updates')
    parser.add_argument('--dgt_ratio_diff_threshold', type=float, default=0.03,
                        help='DGTL ratio diff threshold')
    parser.add_argument('--disable_dynamic_weights', action='store_true',
                        help='Disable DGTL dynamic weights (ablation)')
    # Geometry consistency is enabled by default in DGTLossModule (do not expose CLI flags).
    parser.add_argument('--dgt_clamp_min', type=float, default=0.99,
                        help='DGTL core multiplier lower bound (paired balancing keeps total core budget stable)')
    parser.add_argument('--dgt_clamp_max', type=float, default=1.01,
                        help='DGTL multiplier upper bound for core losses (stricter to reduce oscillation)')
    parser.add_argument('--dgt_main_max_step', type=float, default=0.0015,
                        help='Maximum per-update step for core dynamic multipliers')
    parser.add_argument('--dgt_geom_weight', type=float, default=0.008,
                        help='Dual-geometry consistency weight')
    parser.add_argument('--dgt_geom_cap_ratio', type=float, default=0.02,
                        help='Cap geometry loss by this ratio of detached main-core loss')
    parser.add_argument('--dgt_geom_residual_scale', type=float, default=0.05,
                        help='Resource-anchor residual scale for geometry consistency (0=off, 1=full)')
    parser.add_argument('--dgt_geom_gate_floor', type=float, default=0.0,
                        help='Lower bound of detached quality gate for geometry consistency (0 disables low-quality forcing)')
    parser.add_argument('--dgt_aux_target_ratio', dest='_deprecated_dgt_aux_target_ratio',
                        type=float, default=None, help=argparse.SUPPRESS)
    parser.add_argument('--dgt_aux_clamp_min', dest='_deprecated_dgt_aux_clamp_min',
                        type=float, default=None, help=argparse.SUPPRESS)
    parser.add_argument('--dgt_aux_clamp_max', dest='_deprecated_dgt_aux_clamp_max',
                        type=float, default=None, help=argparse.SUPPRESS)
    parser.add_argument('--dgt_aux_momentum', dest='_deprecated_dgt_aux_momentum',
                        type=float, default=None, help=argparse.SUPPRESS)
    parser.add_argument('--dgt_aux_max_step', dest='_deprecated_dgt_aux_max_step',
                        type=float, default=None, help=argparse.SUPPRESS)
    parser.add_argument('--dgt_core_residual_scale', type=float, default=0.1,
                        help='Resource-anchor residual scale for DGTL core multipliers (0=baseline resource, 1=full DGTL)')
    parser.add_argument('--dgt_aux_residual_scale', dest='_deprecated_dgt_aux_residual_scale',
                        type=float, default=None, help=argparse.SUPPRESS)
    parser.add_argument('--gs_residual_scale', type=float, default=0.1,
                        help='Resource-anchor residual scale for GS penalties (0=off, 1=full)')
    parser.add_argument('--main_score_mode', type=str, default='default',
                        choices=['default', 'det50', 'seg50', 'mix'],
                        help='Legacy validation score view used for logging and late-stage heuristics')
    parser.add_argument('--main_score_det_weight', type=float, default=0.5,
                        help='Detection weight when main_score_mode=mix')
    parser.add_argument('--gs_ema_alpha', type=float, default=0.1,
                        help='EMA alpha for gradient consistency trust gate')
    parser.add_argument('--gs_trust_margin', type=float, default=0.5,
                        help='Loss ratio margin for trust gate (>=1.0 means no reward)')
    parser.add_argument('--enable_grad_consistency', action='store_true',
                        help='Enable gradient consistency / GS (ablation)')
    parser.add_argument('--disable_grad_consistency', dest='enable_grad_consistency', action='store_false',
                        help='Disable gradient consistency / GS')
    parser.add_argument('--gs_late_start_epoch', type=int, default=60,
                        help='Late-stage epoch to consider enabling gradient consistency')
    parser.add_argument('--gs_late_stop_epoch', type=int, default=-1,
                        help='Late-stage stop epoch for gradient consistency (<=0 means train end)')
    parser.add_argument('--gs_activation_drop', type=float, default=0.004,
                        help='Validation-score drop from running best needed to activate late GS')
    parser.add_argument('--gs_det_activation_drop', type=float, default=-1.0,
                        help='Detection-score drop from best needed to activate late GS (<=0 uses gs_activation_drop)')
    parser.add_argument('--gs_bad_margin', type=float, default=0.2,
                        help='Both losses exceed EMA by this margin -> treat as bad-both')
    parser.add_argument('--gs_tau', type=float, default=-0.1,
                        help='Gradient conflict threshold tau in harmonization penalty')
    parser.add_argument('--gs_harmonize_weight', type=float, default=0.006,
                        help='Scale of gradient harmonization penalty P')
    parser.add_argument('--gs_harmonize_cap_ratio', type=float, default=0.015,
                        help='Cap P by this ratio of (L_det+L_seg)/2')
    parser.add_argument('--gs_anchor_max_params', type=int, default=12,
                        help='Maximum number of anchor parameters used by GS autograd.grad')
    parser.add_argument('--gs_anchor_max_elems', type=int, default=800000,
                        help='Maximum total parameter elements used by GS anchors')
    parser.add_argument('--gs_use_text_anchors', action='store_true',
                        help='Also allow small text-encoder parameters as GS anchors (off by default for stability)')
    parser.add_argument('--gs_det_good_thr', type=float, default=0.4,
                        help='Det quality threshold to treat as good (0-1)')
    parser.add_argument('--gs_seg_good_thr', type=float, default=0.4,
                        help='Seg quality threshold to treat as good (0-1)')
    parser.add_argument('--gs_det_bad_thr', type=float, default=0.1,
                        help='Det quality threshold to treat as bad (0-1)')
    parser.add_argument('--gs_seg_bad_thr', type=float, default=0.1,
                        help='Seg quality threshold to treat as bad (0-1)')
    parser.add_argument('--gs_agree_good_thr', type=float, default=0.4,
                        help='Agreement threshold to treat det/seg as GT-consistent good (0-1)')
    parser.add_argument('--gs_agree_bad_thr', type=float, default=0.2,
                        help='Agreement threshold to treat det/seg as GT-consistent bad (0-1)')
    parser.add_argument('--gs_quality_blend', type=float, default=0.5,
                        help='Blend ratio for quality-based trust (0=EMA only, 1=quality only)')
    parser.add_argument('--dgt_val_patience', type=int, default=1,
                        help='Validation patience before attenuating DGTL auxiliary effects')
    parser.add_argument('--dgt_val_min_delta', type=float, default=0.0,
                        help='Minimum validation score gain to reset DGTL bad-count')
    parser.add_argument('--dgt_val_drop_tolerance', type=float, default=0.003,
                        help='Ignore tiny validation-score drops within this tolerance')
    parser.add_argument('--dgt_val_severe_drop', type=float, default=0.01,
                        help='Severe validation drop threshold that triggers aggressive attenuation')
    parser.add_argument('--dgt_val_cooldown_epochs', type=int, default=3,
                        help='Cooldown epochs between two DGTL attenuations')
    parser.add_argument('--dgt_val_max_attenuations', type=int, default=2,
                        help='Maximum DGTL attenuation events before stop epoch')
    parser.add_argument('--geom_late_start_epoch', type=int, default=60,
                        help='Late-stage epoch to consider enabling geometry consistency')
    parser.add_argument('--geom_late_stop_epoch', type=int, default=-1,
                        help='Late-stage stop epoch for geometry consistency (<=0 means train end)')
    parser.add_argument('--geom_gap_threshold', type=float, default=0.05,
                        help='Enable geometry consistency only when det/seg score gap exceeds this threshold')
    parser.add_argument('--geom_gap_worsen_threshold', type=float, default=0.01,
                        help='Require det/seg gap to worsen by at least this amount from its running best before late geometry can activate')
    parser.add_argument('--geom_quality_floor', type=float, default=0.35,
                        help='Require both det/seg validation scores above this floor before enabling geometry consistency')
    parser.add_argument('--geom_seg_stable_drop', type=float, default=0.005,
                        help='Maximum segmentation-score drop from best while considering late geometry rescue')
    parser.add_argument('--geom_gap_ema_alpha', type=float, default=0.3,
                        help='EMA alpha for validation det/seg gap tracking')
    parser.add_argument('--late_stage_confirm_epochs', type=int, default=2,
                        help='Consecutive late-stage validation confirmations required before GS or geometry can activate')
    parser.add_argument('--late_stage_cooldown_epochs', type=int, default=2,
                        help='Cooldown epochs after a late-stage controller hurts validation before it can re-arm')
    parser.add_argument('--late_stage_kill_drop', type=float, default=0.004,
                        help='If an active late-stage controller causes at least this det drop from its activation reference, disable it immediately')
    parser.add_argument('--gs_late_ramp_min', type=float, default=0.12,
                        help='Minimum fraction of base GS weight when late GS first activates')
    parser.add_argument('--gs_late_ramp_max', type=float, default=0.30,
                        help='Maximum fraction of base GS weight used by the late-stage controller')
    parser.add_argument('--geom_late_ramp_min', type=float, default=0.08,
                        help='Minimum fraction of base geometry weight when late geometry first activates')
    parser.add_argument('--geom_late_ramp_max', type=float, default=0.25,
                        help='Maximum fraction of base geometry weight used by the late-stage controller')
    parser.add_argument('--dgt_reset_on_stop', dest='dgt_reset_on_stop', action='store_true', default=True,
                        help='Reset DGTL multipliers to 1.0 once stop epoch is reached (recommended)')
    parser.add_argument('--dgt_keep_stop_weights', dest='dgt_reset_on_stop', action='store_false',
                        help='Keep DGTL multipliers after stop epoch (may increase late-epoch oscillation)')

    # Data
    parser.add_argument('--batch_size', type=int, default=8,
                        help='Batch Size during training')
    parser.add_argument('--dataset', type=str, default=['scanrefer'],
                        nargs='+', help='list of datasets to train on')
    parser.add_argument('--test_dataset', default='scanrefer')
    parser.add_argument('--data_root', default='./')
    parser.add_argument('--use_height', action='store_true',
                        help='Use height signal in input.')
    parser.add_argument('--use_color', action='store_true',
                        help='Use RGB color in input.')  # color
    parser.add_argument('--use_multiview', action='store_true')
    parser.add_argument('--wo_obj_name', default='None')  # grounding without object name
    parser.add_argument('--butd', action='store_true')
    parser.add_argument('--butd_gt', action='store_true')
    parser.add_argument('--butd_cls', action='store_true')
    parser.add_argument('--augment_det', action='store_true')
    parser.add_argument('--num_workers', type=int, default=4)

    # Training
    parser.add_argument('--start_epoch', type=int, default=1)
    parser.add_argument('--max_epoch', type=int, default=400)
    parser.add_argument('--optimizer', type=str, default='adamW')
    parser.add_argument('--weight_decay', type=float, default=0.0005)
    parser.add_argument("--lr", default=1e-3, type=float)
    parser.add_argument("--lr_backbone", default=1e-4, type=float)
    parser.add_argument("--text_encoder_lr", default=1e-5, type=float)
    parser.add_argument('--lr-scheduler', type=str, default='step',
                        choices=["step", "cosine"])
    parser.add_argument('--lr_decay_epochs', type=int, default=[280, 340],
                        nargs='+', help='when to decay lr, can be a list')
    parser.add_argument('--lr_decay_rate', type=float, default=0.1,
                        help='for step scheduler. decay rate for lr')
    parser.add_argument('--clip_norm', default=0.1, type=float,
                        help='gradient clipping max norm')
    parser.add_argument('--bn_momentum', type=float, default=0.1)
    parser.add_argument('--syncbn', action='store_true')
    parser.add_argument('--warmup-epoch', type=int, default=-1)
    parser.add_argument('--warmup-multiplier', type=int, default=100)
    parser.add_argument('--frozen', action='store_true')
    parser.add_argument('--small_lr', default=False, action='store_true')

    # io
    parser.add_argument('--checkpoint_path', default=None,
                        help='Model checkpoint path')
    parser.add_argument('--log_dir', default='log',
                        help='Dump dir to save model checkpoint')
    parser.add_argument('--exp', default='exp',
                        help='exp name to save model checkpoint')
    parser.add_argument('--print_freq', type=int, default=10)  # batch-wise
    parser.add_argument('--save_freq', type=int, default=10)  # epoch-wise
    parser.add_argument('--val_freq', type=int, default=5)  # epoch-wise

    # others
    env_local_rank = int(os.environ.get("LOCAL_RANK", os.environ.get("SLURM_LOCALID", 0)))
    parser.add_argument("--local_rank", type=int, default=env_local_rank,
                        help='local rank for DistributedDataParallel')  # note
    parser.add_argument('--ap_iou_thresholds', type=float, default=[0.25, 0.5],
                        nargs='+', help='A list of AP IoU thresholds')
    parser.add_argument("--rng_seed", type=int, default=0, help='manual seed')
    parser.add_argument("--debug", action='store_true',
                        help="try to overfit few samples")
    parser.add_argument('--eval', default=False, action='store_true')
    parser.add_argument('--eval_train', action='store_true')
    parser.add_argument('--eval_all_ranks', dest='_deprecated_eval_all_ranks',
                        action='store_true', default=False, help=argparse.SUPPRESS)
    parser.add_argument('--eval_rank0_only', dest='_deprecated_eval_rank0_only',
                        action='store_true', default=False, help=argparse.SUPPRESS)
    parser.add_argument('--pp_checkpoint', default=None)  # pointnet checkpoint
    parser.add_argument('--reduce_lr', action='store_true')

    # Keep baseline behavior safe; GS is enabled only when explicitly requested.
    parser.set_defaults(enable_grad_consistency=False)

    args, _ = parser.parse_known_args()
    deprecated_aux_args = [
        '_deprecated_dgt_aux_target_ratio', '_deprecated_dgt_aux_clamp_min',
        '_deprecated_dgt_aux_clamp_max', '_deprecated_dgt_aux_momentum',
        '_deprecated_dgt_aux_max_step', '_deprecated_dgt_aux_residual_scale',
    ]
    if getattr(args, '_deprecated_eval_all_ranks', False):
        print('WARNING: --eval_all_ranks is deprecated and ignored; evaluation always aggregates all ranks.')
    if getattr(args, '_deprecated_eval_rank0_only', False):
        print('WARNING: --eval_rank0_only is deprecated and ignored; evaluation always aggregates all ranks.')
    if any(getattr(args, name, None) is not None for name in deprecated_aux_args):
        print('WARNING: DGTL auxiliary dynamic-weight arguments are deprecated and ignored; auxiliary losses keep baseline weight.')
    if hasattr(args, '_deprecated_eval_all_ranks'):
        delattr(args, '_deprecated_eval_all_ranks')
    if hasattr(args, '_deprecated_eval_rank0_only'):
        delattr(args, '_deprecated_eval_rank0_only')
    for name in deprecated_aux_args:
        if hasattr(args, name):
            delattr(args, name)

    args.eval = args.eval or args.eval_train
    args.log_root = os.path.normpath(args.log_dir)
    args.parse_result_dir = os.path.join(args.log_root, args.exp, 'parse_result') if args.exp else os.path.join(args.log_root, 'parse_result')

    args.main_score_det_weight = max(0.0, min(1.0, float(args.main_score_det_weight)))
    args.dgt_start_epoch = max(0, int(args.dgt_start_epoch))
    args.dgt_stop_epoch = max(args.dgt_start_epoch + 1, int(args.dgt_stop_epoch))
    args.dgt_reset_on_stop = False
    args.gs_late_start_epoch = max(60, int(getattr(args, 'gs_late_start_epoch', 60)))
    gs_stop = int(getattr(args, 'gs_late_stop_epoch', -1))
    if gs_stop > 0:
        args.gs_late_stop_epoch = max(args.gs_late_start_epoch + 1, gs_stop)
    else:
        args.gs_late_stop_epoch = -1
    args.geom_late_start_epoch = max(
        60, int(getattr(args, 'geom_late_start_epoch', args.gs_late_start_epoch))
    )
    geom_stop = int(getattr(args, 'geom_late_stop_epoch', -1))
    if geom_stop > 0:
        args.geom_late_stop_epoch = max(args.geom_late_start_epoch + 1, geom_stop)
    else:
        args.geom_late_stop_epoch = -1
    return args


def load_checkpoint(args, model, optimizer, scheduler,
                    loss_module=None, trainer=None, n_iter_per_epoch=None):
    """Load from checkpoint."""
    print("=> loading checkpoint '{}'".format(args.checkpoint_path))
    logger = getattr(trainer, 'logger', None) if trainer is not None else None

    def _emit(msg):
        print(msg)
        if logger is not None and dist.get_rank() == 0:
            logger.info(msg)

    def _cfg_get(cfg, key, default=None):
        if cfg is None:
            return default
        if isinstance(cfg, dict):
            return cfg.get(key, default)
        return getattr(cfg, key, default)

    def _filter_state_dict(ckpt_state, cur_state, allow_module_prefix=False):
        compat_state = {}
        skipped_shape = 0
        for key, value in ckpt_state.items():
            target_key = None
            if key in cur_state:
                target_key = key
            elif allow_module_prefix and key.startswith('module.') and key[7:] in cur_state:
                target_key = key[7:]
            elif allow_module_prefix and f'module.{key}' in cur_state:
                target_key = f'module.{key}'
            if target_key is None:
                continue
            if cur_state[target_key].shape != value.shape:
                skipped_shape += 1
                continue
            compat_state[target_key] = value
        return compat_state, skipped_shape

    checkpoint = torch.load(args.checkpoint_path, map_location='cpu')
    try:
        args.start_epoch = int(checkpoint['epoch']) + 1
    except Exception:
        args.start_epoch = 0
    ckpt_state = checkpoint.get('model', {})
    cur_state = model.state_dict()
    compat_state, skipped_shape = _filter_state_dict(
        ckpt_state, cur_state, allow_module_prefix=True
    )
    msg = model.load_state_dict(compat_state, strict=False)
    _emit(
        f"=> loaded model params: {len(compat_state)}/{len(cur_state)} "
        f"(missing={len(getattr(msg, 'missing_keys', []))}, "
        f"unexpected={len(getattr(msg, 'unexpected_keys', []))}, "
        f"shape_skip={skipped_shape})"
    )

    if loss_module is not None and 'loss_module' in checkpoint:
        try:
            ckpt_state = checkpoint['loss_module']
            cur_state = loss_module.state_dict()
            compat_state, skipped_shape = _filter_state_dict(ckpt_state, cur_state)
            msg = loss_module.load_state_dict(compat_state, strict=False)
            _emit(
                f"=> loaded DGTLossModule params: {len(compat_state)}/{len(cur_state)} "
                f"(missing={len(getattr(msg, 'missing_keys', []))}, "
                f"unexpected={len(getattr(msg, 'unexpected_keys', []))}, "
                f"shape_skip={skipped_shape})"
            )
            # Only clear per-epoch accumulators; keep learned DGTL running state intact.
            if hasattr(loss_module, 'epoch_acc'):
                loss_module.epoch_acc.zero_()
            if hasattr(loss_module, 'geom_debug_acc'):
                loss_module.geom_debug_acc.zero_()
            _emit("=> preserved DGTL running states after resume")
        except Exception as e:
            _emit(f"=> Warning: Could not load loss module state ({e})")

    if trainer is not None and 'trainer_state' in checkpoint:
        try:
            trainer.load_trainer_state(checkpoint['trainer_state'])
            if hasattr(trainer, '_load_best_checkpoint_snapshot'):
                trainer._load_best_checkpoint_snapshot()
            _emit("=> loaded trainer control state")
        except Exception as e:
            _emit(f"=> Warning: Could not load trainer state ({e})")
    if not args.eval and not args.reduce_lr:
        try:
            optimizer.load_state_dict(checkpoint['optimizer'])
            scheduler.load_state_dict(checkpoint['scheduler'])
            _emit("=> loaded optimizer and scheduler state")
        except (KeyError, ValueError) as e:
            _emit(f"=> Warning: Could not load optimizer state ({e})")
            _emit("=> Starting with fresh optimizer state")

    _emit("=> loaded successfully '{}' (epoch {})".format(
        args.checkpoint_path, checkpoint['epoch']
    ))

    del checkpoint
    torch.cuda.empty_cache()


def save_checkpoint(args, epoch, model, optimizer, scheduler,
                    loss_module=None, trainer_state=None, save_cur=False):
    """Save checkpoint if requested."""
    if save_cur or epoch % args.save_freq == 0:
        state = {
            'config': args,
            'save_path': '',
            'model': model.state_dict(),
            'optimizer': optimizer.state_dict(),
            'scheduler': scheduler.state_dict(),
            'epoch': epoch
        }

        if loss_module is not None:
            state['loss_module'] = loss_module.state_dict()
        if trainer_state is not None:
            state['trainer_state'] = trainer_state

        spath = os.path.join(args.log_dir, f'ckpt_epoch_{epoch}.pth')
        state['save_path'] = spath
        torch.save(state, spath)
        print("Saved in {}".format(spath))
    else:
        print("not saving checkpoint")


def save_best_checkpoint(args, epoch, model, optimizer, scheduler,
                         loss_module=None, trainer_state=None):
    state = {
        'config': args,
        'save_path': '',
        'model': model.state_dict(),
        'optimizer': optimizer.state_dict(),
        'scheduler': scheduler.state_dict(),
        'epoch': epoch
    }
    if loss_module is not None:
        state['loss_module'] = loss_module.state_dict()
    if trainer_state is not None:
        state['trainer_state'] = trainer_state

    spath = os.path.join(args.log_dir, 'ckpt_best.pth')
    state['save_path'] = spath
    torch.save(state, spath)
    print("Saved in {}".format(spath))
    return spath




class BaseTrainTester:
    """Basic train/test class to be inherited."""

    # logger.
    def __init__(self, args):
        """Initialize."""
        self.args = args
        name = args.log_dir.split('/')[-1]  # log_dir: './logs/eda', name: eda

        # Create log dir
        args.log_dir = os.path.join(
            args.log_dir,
            args.exp,
            ','.join(args.dataset),
            f'{int(time.time())}'
        )
        os.makedirs(args.log_dir, exist_ok=True)

        # Create logger
        self.logger = setup_logger(
            output=args.log_dir, distributed_rank=dist.get_rank(),
            name=name
        )
        # Suppress spammy alignment and GS formula logs if they appear
        class _SuppressGSFormula(logging.Filter):
            _align_re = re.compile(
                r"Top-1:\s*([0-9.]+),\s*Top-5:\s*([0-9.]+),\s*Top-10:\s*([0-9.]+)"
            )

            def filter(self, record):
                msg = record.getMessage()
                if "scale = 1 + reward * max(cos, 0) - suppress * max(-cos, 0)" in msg:
                    return False
                if ("position alignment Acc" in msg) or ("semantic alignment Acc" in msg):
                    m = self._align_re.search(msg)
                    if m:
                        t1, t5, t10 = (float(m.group(1)), float(m.group(2)), float(m.group(3)))
                        if t1 == 0.0 and t5 == 0.0 and t10 == 0.0:
                            return False
                return True
        self.logger.addFilter(_SuppressGSFormula())
        self._best_val_score = None
        self._best_legacy_main_score = None
        self._prev_val_score = None
        self._dyn_best_val_score = None
        self._dyn_prev_val_score = None
        self._dgt_last_freeze_epoch = -10 ** 9
        self._best_dynamic_state = None
        self._best_model_state = None
        self._best_ckpt_path = None
        self._dgt_consecutive_drop_count = 0
        self._dgt_freeze_reason = 'off'
        self._best_det_score = None
        self._best_seg_score = None
        self._best_det_seg_gap = None
        self._det_seg_gap_ema = None
        self._latest_eval_stats = {}
        self._gs_stage_limit = 1.0
        self._geom_stage_limit = 1.0
        self._late_gs_confirm_count = 0
        self._late_geom_confirm_count = 0
        self._late_stage_mode = 'off'
        self._late_stage_last_switch_epoch = -10 ** 9
        self._late_stage_cooldown_until = -10 ** 9
        self._late_stage_ref_main_score = None
        self._late_stage_ref_det_score = None
        self._late_stage_model_state = None
        self._late_stage_snapshot_mode = 'off'
        self._late_stage_snapshot_det_score = None
        self._last_stage_log_epoch = -1
        # Gradient consistency gating (EMA-based trust)
        self.gs_loss_ema = None
        self.gs_ema_alpha = getattr(args, 'gs_ema_alpha', 0.1)
        self.gs_trust_margin = getattr(args, 'gs_trust_margin', 0.5)
        self.gs_bad_margin = getattr(args, 'gs_bad_margin', 0.2)
        self.gs_det_good_thr = getattr(args, 'gs_det_good_thr', 0.4)
        self.gs_seg_good_thr = getattr(args, 'gs_seg_good_thr', 0.4)
        self.gs_det_bad_thr = getattr(args, 'gs_det_bad_thr', 0.1)
        self.gs_seg_bad_thr = getattr(args, 'gs_seg_bad_thr', 0.1)
        self.gs_agree_good_thr = getattr(args, 'gs_agree_good_thr', 0.4)
        self.gs_agree_bad_thr = getattr(args, 'gs_agree_bad_thr', 0.2)
        self.gs_quality_blend = getattr(args, 'gs_quality_blend', 0.5)
        self._base_geom_weight = max(0.0, float(getattr(args, 'dgt_geom_weight', 0.008)))
        self._base_gs_weight = max(0.0, float(getattr(args, 'gs_harmonize_weight', 0.006)))
        self._base_main_max_step = max(0.0, float(getattr(args, 'dgt_main_max_step', 0.0015)))
        self._base_aux_max_step = 0.0
        self.global_loss_module = DGTLossModule(
            args.num_decoder_layers,
            warmup_epochs=getattr(args, 'dgt_warmup_epochs', 5),
            stop_dynamic_epoch=getattr(args, 'dgt_stop_epoch', 50),
            start_dynamic_epoch=getattr(args, 'dgt_start_epoch', 24),
            diff_threshold=getattr(args, 'dgt_diff_threshold', 0.03),
            update_interval=getattr(args, 'dgt_update_interval', 2),
            ema_alpha=getattr(args, 'dgt_ema_alpha', 0.2),
            hold_when_stable=bool(getattr(args, 'dgt_hold_when_stable', 1)),
            patience=getattr(args, 'dgt_patience', 2),
            ratio_diff_threshold=getattr(args, 'dgt_ratio_diff_threshold', 0.03),
            min_epoch_gap=getattr(args, 'dgt_min_epoch_gap', 1),
            clamp_min=getattr(args, 'dgt_clamp_min', 0.99),
            clamp_max=getattr(args, 'dgt_clamp_max', 1.01),
            main_max_step=getattr(args, 'dgt_main_max_step', 0.0015),
            geom_weight=getattr(args, 'dgt_geom_weight', 0.008),
            geom_cap_ratio=getattr(args, 'dgt_geom_cap_ratio', 0.02),
            geom_gate_floor=getattr(args, 'dgt_geom_gate_floor', 0.0),
            aux_target_ratio=1.0,
            aux_clamp_min=1.0,
            aux_clamp_max=1.0,
            aux_momentum=1.0,
            aux_max_step=0.0,
            harmonize_weight=getattr(args, 'gs_harmonize_weight', 0.006),
            harmonize_cap_ratio=getattr(args, 'gs_harmonize_cap_ratio', 0.015),
            gs_start_epoch=getattr(args, 'gs_late_start_epoch', 60),
            gs_stop_epoch=getattr(args, 'gs_late_stop_epoch', -1),
            geom_start_epoch=getattr(args, 'geom_late_start_epoch', 60),
            geom_stop_epoch=getattr(args, 'geom_late_stop_epoch', -1),
            core_residual_scale=getattr(args, 'dgt_core_residual_scale', 0.1),
            aux_residual_scale=0.0,
            gs_residual_scale=getattr(args, 'gs_residual_scale', 0.1),
            geom_residual_scale=getattr(args, 'dgt_geom_residual_scale', 0.05),
            reset_on_stop=bool(getattr(args, 'dgt_reset_on_stop', True)),
            enable_dynamic=not getattr(args, 'disable_dynamic_weights', False),
            enable_geom=True,
        )
        self.global_loss_module.enable_quality_stats = bool(
            getattr(args, 'enable_grad_consistency', False)
        )

        # tensorboard
        self.tensorboard = record_tensorboard.TensorBoard(args.log_dir, distributed_rank=dist.get_rank())

        # Save config file and initialize tb writer
        if dist.get_rank() == 0:
            path = os.path.join(args.log_dir, "config.json")
            with open(path, 'w') as f:
                json.dump(vars(args), f, indent=2)
            self.logger.info("Full config saved to {}".format(path))
            self.logger.info(str(vars(args)))

    @staticmethod
    def _to_scalar_or_none(value):
        if value is None:
            return None
        if torch.is_tensor(value):
            if value.numel() == 0:
                return None
            return float(value.detach().reshape(-1)[0].item())
        try:
            return float(value)
        except (TypeError, ValueError):
            return None

    def _dynamic_updates_frozen(self):
        if not hasattr(self, 'global_loss_module'):
            return False
        return bool(getattr(self.global_loss_module, 'updates_frozen', lambda: False)())

    def _compose_main_score(self, stats, fallback=None):
        det_score = self._to_scalar_or_none(stats.get('det_score', None))
        seg_score = self._to_scalar_or_none(stats.get('seg_score', None))
        fallback = self._to_scalar_or_none(fallback)
        mode = str(getattr(self.args, 'main_score_mode', 'default'))
        if mode == 'det50' and det_score is not None:
            return det_score
        if mode == 'seg50' and seg_score is not None:
            return seg_score
        if mode == 'mix' and det_score is not None and seg_score is not None:
            det_w = float(getattr(self.args, 'main_score_det_weight', 0.5))
            return det_w * det_score + (1.0 - det_w) * seg_score
        if fallback is not None:
            return fallback
        if det_score is not None and seg_score is not None:
            return 0.5 * (det_score + seg_score)
        return det_score if det_score is not None else seg_score

    def _compose_record_score(self, stats=None, main_score=None, seg_score=None):
        if main_score is None and stats is not None:
            main_score = self._compose_main_score(stats)
        else:
            main_score = self._to_scalar_or_none(main_score)

        if seg_score is None and stats is not None:
            seg_score = self._to_scalar_or_none(stats.get('seg_score', None))
        else:
            seg_score = self._to_scalar_or_none(seg_score)

        mode = str(getattr(self.args, 'main_score_mode', 'default'))
        if mode in ('det50', 'mix'):
            return main_score if main_score is not None else seg_score
        if mode == 'seg50':
            return seg_score if seg_score is not None else main_score
        if seg_score is not None:
            return seg_score
        return main_score

    def _best_record_score_for_current_mode(self):
        mode = str(getattr(self.args, 'main_score_mode', 'default'))
        best_det = self._to_scalar_or_none(self._best_det_score)
        best_seg = self._to_scalar_or_none(self._best_seg_score)
        if mode == 'det50':
            return best_det
        if mode in ('default', 'seg50'):
            return best_seg if best_seg is not None else best_det
        if mode == 'mix':
            if best_det is not None and best_seg is not None:
                det_w = float(getattr(self.args, 'main_score_det_weight', 0.5))
                return det_w * best_det + (1.0 - det_w) * best_seg
            return best_det if best_det is not None else best_seg
        return best_seg if best_seg is not None else best_det

    def _migrate_loaded_record_scores(self):
        latest = dict(self._latest_eval_stats or {})
        if not latest:
            return
        old_record = self._to_scalar_or_none(latest.get('record_score', None))
        main_score = self._compose_main_score(
            latest,
            fallback=latest.get('legacy_main_score', latest.get('main_score', None))
        )
        new_record = self._compose_record_score(
            stats=latest,
            main_score=main_score,
            seg_score=latest.get('seg_score', None)
        )
        if new_record is None:
            return
        latest['legacy_main_score'] = main_score
        latest['main_score'] = main_score
        latest['record_score'] = new_record
        self._latest_eval_stats = latest
        if old_record is not None and abs(float(old_record) - float(new_record)) <= 1e-8:
            return

        self._prev_val_score = new_record
        best_record = self._best_record_score_for_current_mode()
        self._best_val_score = best_record if best_record is not None else new_record
        self._dyn_prev_val_score = None
        self._dyn_best_val_score = None
        self._best_dynamic_state = None
        self._best_ckpt_path = None
        if (not dist.is_available()) or (not dist.is_initialized()) or dist.get_rank() == 0:
            print(
                f"=> migrated trainer record_score for main_score_mode="
                f"{getattr(self.args, 'main_score_mode', 'default')} "
                f"({0.0 if old_record is None else float(old_record):.6f} -> {float(new_record):.6f})"
            )

    def _capture_dynamic_state(self):
        if not hasattr(self, 'global_loss_module'):
            return None
        return self.global_loss_module.export_state_snapshot()

    def _restore_dynamic_state(self, snapshot):
        if not hasattr(self, 'global_loss_module') or snapshot is None:
            return False
        return bool(self.global_loss_module.load_state_snapshot(snapshot))

    def _freeze_dynamic_updates(self, epoch, reason, restore_best=True, model=None):
        if not hasattr(self, 'global_loss_module'):
            return False
        restored_model = False
        if restore_best and self._best_dynamic_state is not None:
            self._restore_dynamic_state(self._best_dynamic_state)
        if restore_best and model is not None and self._best_model_state is not None:
            restored_model = bool(self._restore_model_state(model, self._best_model_state))
        self.global_loss_module.freeze_updates()
        self._dgt_last_freeze_epoch = int(epoch)
        self._dgt_consecutive_drop_count = 0
        self._dgt_freeze_reason = str(reason)
        if self._late_stage_log_enabled():
            best_dyn_score = self._dyn_best_val_score
            best_score = self._best_val_score
            self.logger.info(
                f"[DGTL][DynWeight] lock@e{epoch} reason={reason} "
                f"best_record={0.0 if best_score is None else float(best_score):.4f} "
                f"best_dyn_record={0.0 if best_dyn_score is None else float(best_dyn_score):.4f} "
                f"restore_model={'yes' if restored_model else 'no'}"
            )
        return restored_model

    def _thaw_dynamic_updates(self, epoch, reason='resume'):
        if not hasattr(self, 'global_loss_module'):
            return
        self.global_loss_module.main_max_step = min(
            float(self.global_loss_module.main_max_step),
            0.25 * float(self._base_main_max_step)
        )
        self.global_loss_module.aux_max_step = min(
            float(self.global_loss_module.aux_max_step),
            0.25 * float(self._base_aux_max_step)
        )
        self.global_loss_module.unfreeze_updates()
        self._dgt_freeze_reason = str(reason)
        self._dgt_consecutive_drop_count = 0
        if self._late_stage_log_enabled():
            self.logger.info(
                f"[DGTL][DynWeight] unlock@e{epoch} reason={reason} "
                f"step=({float(self.global_loss_module.main_max_step):.5f},"
                f"{float(self.global_loss_module.aux_max_step):.5f})"
            )

    @staticmethod
    def _clone_model_state(model):
        return {
            key: value.detach().cpu().clone()
            for key, value in model.state_dict().items()
        }

    def _restore_model_state(self, model, state_dict):
        if state_dict is None:
            return False
        cur_state = model.state_dict()
        compat_state = {}
        for key, value in state_dict.items():
            if key not in cur_state or cur_state[key].shape != value.shape:
                continue
            compat_state[key] = value.to(device=cur_state[key].device, dtype=cur_state[key].dtype)
        if not compat_state:
            return False
        model.load_state_dict(compat_state, strict=False)
        return True

    def _clear_late_stage_model_state(self):
        self._late_stage_model_state = None
        self._late_stage_snapshot_mode = 'off'
        self._late_stage_snapshot_det_score = None

    def _capture_late_stage_model_state(self, model, mode, det_score=None):
        if model is None:
            return False
        self._late_stage_model_state = self._clone_model_state(model)
        self._late_stage_snapshot_mode = str(mode)
        self._late_stage_snapshot_det_score = self._to_scalar_or_none(det_score)
        return True

    def _restore_late_stage_model_state(self, model, epoch, reason):
        restored = False
        if model is not None and self._late_stage_model_state is not None:
            restored = bool(self._restore_model_state(model, self._late_stage_model_state))
        self._clear_late_stage_model_state()
        if self._late_stage_log_enabled():
            self.logger.info(
                f"[StageCtrl] restore@e{epoch} reason={reason} "
                f"restore_model={'yes' if restored else 'no'}"
            )
        return restored

    def _restore_best_model_state(self, model, epoch, reason):
        restored = False
        if model is not None and self._best_model_state is not None:
            restored = bool(self._restore_model_state(model, self._best_model_state))
        self._clear_late_stage_model_state()
        if self._late_stage_log_enabled():
            self.logger.info(
                f"[StageCtrl] restore_best@e{epoch} reason={reason} "
                f"restore_model={'yes' if restored else 'no'}"
            )
        return restored

    def _save_global_best_checkpoint(self, epoch, args, model, optimizer, scheduler):
        self._best_ckpt_path = os.path.join(args.log_dir, 'ckpt_best.pth')
        return save_best_checkpoint(
            args, epoch, model, optimizer, scheduler,
            loss_module=self.global_loss_module,
            trainer_state=self.get_trainer_state()
        )

    def _load_best_checkpoint_snapshot(self):
        path = self._best_ckpt_path
        if not path or not os.path.isfile(path):
            return False
        checkpoint = torch.load(path, map_location='cpu')
        model_state = checkpoint.get('model', None)
        if isinstance(model_state, dict):
            self._best_model_state = {
                key: value.detach().cpu().clone()
                for key, value in model_state.items()
                if torch.is_tensor(value)
            }
        if self._best_dynamic_state is None and 'loss_module' in checkpoint and hasattr(self, 'global_loss_module'):
            snapshot = self.global_loss_module.export_state_snapshot()
            snapshot['state_dict'] = {
                key: value.detach().cpu().clone()
                for key, value in checkpoint['loss_module'].items()
                if torch.is_tensor(value)
            }
            self._best_dynamic_state = snapshot
        del checkpoint
        return self._best_model_state is not None

    @staticmethod
    def _broadcast_model_state(model, src=0):
        if not dist.is_available() or not dist.is_initialized() or dist.get_world_size() == 1:
            return
        for value in model.state_dict().values():
            if torch.is_tensor(value):
                dist.broadcast(value, src=src)

    @staticmethod
    def _dynamic_window_bounds(args):
        start = int(getattr(args, 'dgt_start_epoch', 30))
        stop = int(getattr(args, 'dgt_stop_epoch', 45))
        return start, stop

    def _dynamic_window_active(self, epoch, args):
        start, stop = self._dynamic_window_bounds(args)
        epoch = int(epoch)
        return start <= epoch < stop

    def _dynamic_window_started(self, epoch, args):
        start, _ = self._dynamic_window_bounds(args)
        return int(epoch) >= start

    def _active_stage_tags(self, epoch, args, dgt_module=None):
        if dgt_module is None:
            dgt_module = getattr(self, 'global_loss_module', None)
        if dgt_module is None:
            return ""
        tags = []
        if (
            self._dynamic_window_active(epoch, args)
            and not self._dynamic_updates_frozen()
            and bool(getattr(dgt_module, 'enable_dynamic', True))
        ):
            dyn_applied = False
            if hasattr(dgt_module, 'has_dynamic_adjustment'):
                dyn_applied = bool(dgt_module.has_dynamic_adjustment())
            tags.append("Dyn=applied" if dyn_applied else "Dyn=armed")
        if float(getattr(dgt_module, 'harmonize_weight', 0.0)) > 0.0:
            tags.append("GS=active")
        if float(getattr(dgt_module, 'geom_weight', 0.0)) > 0.0:
            tags.append("Geom=active")
        return " ".join(tags)

    @staticmethod
    def _active_stage_weights_text(dgt_module):
        if dgt_module is None:
            return ""
        parts = []
        gs_weight = float(getattr(dgt_module, 'harmonize_weight', 0.0))
        geom_weight = float(getattr(dgt_module, 'geom_weight', 0.0))
        val_bias = getattr(dgt_module, 'val_core_bias', None)
        if torch.is_tensor(val_bias):
            val_bias = float(val_bias.detach().item())
        elif val_bias is None:
            val_bias = 0.0
        else:
            val_bias = float(val_bias)
        if abs(val_bias) > 1e-8:
            parts.append(f"DynBias={val_bias:.5f}")
        if gs_weight > 0.0:
            parts.append(f"GSw={gs_weight:.5f}")
        if geom_weight > 0.0:
            parts.append(f"GeoW={geom_weight:.5f}")
        return " ".join(parts)

    def _handle_validation_feedback(self, epoch, args, model, optimizer, scheduler,
                                    main_score, det_score=None, seg_score=None, gap_score=None):
        if not hasattr(self, 'global_loss_module'):
            return False

        main_score = self._to_scalar_or_none(main_score)
        record_score = self._compose_record_score(
            stats=self._latest_eval_stats,
            main_score=main_score,
            seg_score=seg_score
        )
        if record_score is None:
            return False

        global_improved = (
            self._best_val_score is None
            or record_score > float(self._best_val_score)
        )
        if global_improved:
            self._best_val_score = record_score

        if main_score is not None and (
            self._best_legacy_main_score is None
            or float(main_score) > float(self._best_legacy_main_score)
        ):
            self._best_legacy_main_score = float(main_score)
        if det_score is not None and (
            self._best_det_score is None or float(det_score) > float(self._best_det_score)
        ):
            self._best_det_score = float(det_score)
        if seg_score is not None and (
            self._best_seg_score is None or float(seg_score) > float(self._best_seg_score)
        ):
            self._best_seg_score = float(seg_score)
        if gap_score is not None and (
            self._best_det_seg_gap is None or float(gap_score) < float(self._best_det_seg_gap)
        ):
            self._best_det_seg_gap = float(gap_score)

        def _maybe_save_global_best():
            if global_improved:
                self._save_global_best_checkpoint(epoch, args, model, optimizer, scheduler)

        dyn_started = self._dynamic_window_started(epoch, args)
        dyn_active = self._dynamic_window_active(epoch, args)
        if not dyn_started:
            self._dyn_prev_val_score = None
            self._dgt_consecutive_drop_count = 0
            self._prev_val_score = record_score
            _maybe_save_global_best()
            return False

        if not dyn_active:
            if not self._dynamic_updates_frozen():
                restore_best = (
                    (not global_improved)
                    and (
                        self._best_dynamic_state is not None
                        or self._best_model_state is not None
                    )
                )
                restored_model = self._freeze_dynamic_updates(
                    epoch,
                    'window_end',
                    restore_best=restore_best,
                    model=model
                )
                self._dyn_prev_val_score = self._dyn_best_val_score
                self._prev_val_score = (
                    self._dyn_best_val_score
                    if self._dyn_best_val_score is not None else record_score
                )
                if global_improved and not restored_model:
                    self._save_global_best_checkpoint(epoch, args, model, optimizer, scheduler)
                return restored_model
            self._prev_val_score = record_score
            _maybe_save_global_best()
            return False

        min_delta = max(0.0, float(getattr(args, 'dgt_val_min_delta', 0.0)))
        drop_tol = max(0.0, float(getattr(args, 'dgt_val_drop_tolerance', 0.0)))
        patience = max(1, int(getattr(args, 'dgt_val_patience', 1)))
        severe_drop = max(
            drop_tol,
            float(getattr(args, 'dgt_val_severe_drop', drop_tol))
        )
        thaw_delta = max(drop_tol, severe_drop, min_delta)

        prev_best = self._dyn_best_val_score
        prev_score = self._dyn_prev_val_score

        improved = prev_best is None or record_score > float(prev_best) + min_delta
        step_drop = (
            prev_score is not None and record_score < float(prev_score) - drop_tol
        )
        best_drop = 0.0 if prev_best is None else max(0.0, float(prev_best) - record_score)
        has_applied_dynamic = True
        if hasattr(self.global_loss_module, 'has_dynamic_adjustment'):
            has_applied_dynamic = bool(self.global_loss_module.has_dynamic_adjustment())

        if improved:
            self.global_loss_module.main_max_step = float(self._base_main_max_step)
            self.global_loss_module.aux_max_step = float(self._base_aux_max_step)
            self._dyn_best_val_score = record_score
            self._best_dynamic_state = self._capture_dynamic_state()
            self._best_model_state = self._clone_model_state(model)
            self._dgt_consecutive_drop_count = 0
            if (
                self._dynamic_updates_frozen()
                and prev_best is not None
                and record_score > float(prev_best) + thaw_delta
            ):
                self._thaw_dynamic_updates(epoch, reason='best_recovered')
        else:
            can_restore_best = (
                self._best_dynamic_state is not None
                or self._best_model_state is not None
            )
            if (
                not self._dynamic_updates_frozen()
                and has_applied_dynamic
                and can_restore_best
                and best_drop > severe_drop
            ):
                restored_model = self._freeze_dynamic_updates(
                    epoch, 'severe_drop', restore_best=True, model=model
                )
                self._dyn_prev_val_score = self._dyn_best_val_score
                self._prev_val_score = self._dyn_best_val_score
                return restored_model

            if step_drop and best_drop > drop_tol:
                self._dgt_consecutive_drop_count += 1
            else:
                self._dgt_consecutive_drop_count = 0

            if (
                not self._dynamic_updates_frozen()
                and has_applied_dynamic
                and self._dgt_consecutive_drop_count >= patience
                and can_restore_best
            ):
                restored_model = self._freeze_dynamic_updates(
                    epoch, 'consecutive_drop', restore_best=True, model=model
                )
                self._dyn_prev_val_score = self._dyn_best_val_score
                self._prev_val_score = self._dyn_best_val_score
                return restored_model

        if gap_score is not None:
            gap_ema_alpha = max(
                0.0, min(1.0, float(getattr(args, 'geom_gap_ema_alpha', 0.3)))
            )
            if self._det_seg_gap_ema is None:
                self._det_seg_gap_ema = gap_score
            else:
                self._det_seg_gap_ema = (
                    gap_ema_alpha * float(gap_score)
                    + (1.0 - gap_ema_alpha) * float(self._det_seg_gap_ema)
                )

        if (
            hasattr(self.global_loss_module, 'apply_validation_feedback')
            and not self._dynamic_updates_frozen()
        ):
            self.global_loss_module.apply_validation_feedback(
                det_score=det_score,
                seg_score=seg_score,
                record_score=record_score,
                best_score=self._dyn_best_val_score,
                drop_tolerance=drop_tol,
                gap_threshold=getattr(args, 'dgt_ratio_diff_threshold', 0.02),
                epoch=epoch,
            )

        self._dyn_prev_val_score = record_score
        self._prev_val_score = record_score
        _maybe_save_global_best()
        return False

    def get_trainer_state(self):
        return {
            'best_val_score': self._best_val_score,
            'best_legacy_main_score': self._best_legacy_main_score,
            'prev_val_score': self._prev_val_score,
            'dyn_best_val_score': self._dyn_best_val_score,
            'dyn_prev_val_score': self._dyn_prev_val_score,
            'dgt_last_freeze_epoch': int(self._dgt_last_freeze_epoch),
            'best_dynamic_state': self._best_dynamic_state,
            'best_ckpt_path': self._best_ckpt_path,
            'dgt_consecutive_drop_count': int(self._dgt_consecutive_drop_count),
            'dgt_freeze_reason': self._dgt_freeze_reason,
            'best_det_score': self._best_det_score,
            'best_seg_score': self._best_seg_score,
            'best_det_seg_gap': self._best_det_seg_gap,
            'det_seg_gap_ema': self._det_seg_gap_ema,
            'latest_eval_stats': dict(self._latest_eval_stats),
            'gs_stage_limit': float(self._gs_stage_limit),
            'geom_stage_limit': float(self._geom_stage_limit),
            'late_gs_confirm_count': int(self._late_gs_confirm_count),
            'late_geom_confirm_count': int(self._late_geom_confirm_count),
            'late_stage_mode': self._late_stage_mode,
            'late_stage_last_switch_epoch': int(self._late_stage_last_switch_epoch),
            'late_stage_cooldown_until': int(self._late_stage_cooldown_until),
            'late_stage_ref_main_score': self._late_stage_ref_main_score,
            'late_stage_ref_det_score': self._late_stage_ref_det_score,
            'late_stage_snapshot_mode': self._late_stage_snapshot_mode,
            'late_stage_snapshot_det_score': self._late_stage_snapshot_det_score,
            'gs_loss_ema': (
                self.gs_loss_ema.detach().cpu().tolist()
                if torch.is_tensor(self.gs_loss_ema) else self.gs_loss_ema
            ),
        }

    def load_trainer_state(self, state):
        if not isinstance(state, dict):
            return
        self._best_val_score = state.get('best_val_score', self._best_val_score)
        self._best_legacy_main_score = state.get(
            'best_legacy_main_score', self._best_legacy_main_score
        )
        self._prev_val_score = state.get('prev_val_score', self._prev_val_score)
        self._dyn_best_val_score = state.get('dyn_best_val_score', self._dyn_best_val_score)
        self._dyn_prev_val_score = state.get('dyn_prev_val_score', self._dyn_prev_val_score)
        self._dgt_last_freeze_epoch = int(state.get('dgt_last_freeze_epoch', self._dgt_last_freeze_epoch))
        self._best_dynamic_state = state.get('best_dynamic_state', self._best_dynamic_state)
        self._best_ckpt_path = state.get('best_ckpt_path', self._best_ckpt_path)
        self._dgt_consecutive_drop_count = int(
            state.get('dgt_consecutive_drop_count', self._dgt_consecutive_drop_count)
        )
        self._dgt_freeze_reason = str(state.get('dgt_freeze_reason', self._dgt_freeze_reason))
        self._best_det_score = state.get('best_det_score', self._best_det_score)
        self._best_seg_score = state.get('best_seg_score', self._best_seg_score)
        self._best_det_seg_gap = state.get('best_det_seg_gap', self._best_det_seg_gap)
        self._det_seg_gap_ema = state.get('det_seg_gap_ema', self._det_seg_gap_ema)
        self._latest_eval_stats = dict(state.get('latest_eval_stats', self._latest_eval_stats or {}))
        self._migrate_loaded_record_scores()
        self._gs_stage_limit = float(state.get('gs_stage_limit', self._gs_stage_limit))
        self._geom_stage_limit = float(state.get('geom_stage_limit', self._geom_stage_limit))
        self._late_gs_confirm_count = int(state.get('late_gs_confirm_count', self._late_gs_confirm_count))
        self._late_geom_confirm_count = int(state.get('late_geom_confirm_count', self._late_geom_confirm_count))
        self._late_stage_mode = str(state.get('late_stage_mode', self._late_stage_mode))
        self._late_stage_last_switch_epoch = int(
            state.get('late_stage_last_switch_epoch', self._late_stage_last_switch_epoch)
        )
        self._late_stage_cooldown_until = int(
            state.get('late_stage_cooldown_until', self._late_stage_cooldown_until)
        )
        self._late_stage_ref_main_score = state.get(
            'late_stage_ref_main_score', self._late_stage_ref_main_score
        )
        self._late_stage_ref_det_score = state.get(
            'late_stage_ref_det_score', self._late_stage_ref_det_score
        )
        self._late_stage_snapshot_mode = str(
            state.get('late_stage_snapshot_mode', self._late_stage_snapshot_mode)
        )
        self._late_stage_snapshot_det_score = state.get(
            'late_stage_snapshot_det_score', self._late_stage_snapshot_det_score
        )
        # Runtime model snapshots are intentionally not serialized; capture a fresh
        # rollback point the next time late-stage geometry is armed.
        self._late_stage_model_state = None
        gs_loss_ema = state.get('gs_loss_ema', None)
        if gs_loss_ema is not None:
            device = self.global_loss_module.global_multipliers.device
            self.gs_loss_ema = torch.as_tensor(gs_loss_ema, device=device, dtype=torch.float32)

    def _late_stage_log_enabled(self):
        return dist.is_available() and dist.is_initialized() and dist.get_rank() == 0

    def _set_late_stage_mode(self, mode, epoch, main_score=None, det_score=None, args=None, reason=None):
        prev_mode = self._late_stage_mode
        next_mode = str(mode)
        self._late_stage_mode = next_mode
        self._late_stage_last_switch_epoch = int(epoch)
        active_modes = ('gs', 'geom', 'both')
        if prev_mode in active_modes and next_mode in active_modes:
            main_ref = self._to_scalar_or_none(self._late_stage_ref_main_score)
            det_ref = self._to_scalar_or_none(self._late_stage_ref_det_score)
            main_cur = self._to_scalar_or_none(main_score)
            det_cur = self._to_scalar_or_none(det_score)
            self._late_stage_ref_main_score = (
                main_cur if main_ref is None else (main_ref if main_cur is None else max(main_ref, main_cur))
            )
            self._late_stage_ref_det_score = (
                det_cur if det_ref is None else (det_ref if det_cur is None else max(det_ref, det_cur))
            )
        else:
            self._late_stage_ref_main_score = main_score
            self._late_stage_ref_det_score = det_score
        if mode == 'gs':
            self._late_geom_confirm_count = 0
        elif mode == 'geom':
            self._late_gs_confirm_count = 0
        elif mode == 'both':
            self._late_gs_confirm_count = 0
            self._late_geom_confirm_count = 0
        else:
            cooldown = max(0, int(getattr(args, 'late_stage_cooldown_epochs', 2))) if args is not None else 0
            self._late_stage_cooldown_until = int(epoch) + cooldown
            self._late_gs_confirm_count = 0
            self._late_geom_confirm_count = 0
            self._late_stage_ref_main_score = None
            self._late_stage_ref_det_score = None
        if self._late_stage_log_enabled() and prev_mode != self._late_stage_mode:
            reason_txt = "" if not reason else f" reason={reason}"
            self.logger.info(
                f"[StageCtrl] switch@e{epoch} {prev_mode}->{self._late_stage_mode}"
                f"{reason_txt}"
            )

    def _update_late_stage_controller(self, epoch, args, main_score, det_score, seg_score, gap_score, model=None):
        gs_start = max(0, int(getattr(args, 'gs_late_start_epoch', 60)))
        gs_stop = int(getattr(args, 'gs_late_stop_epoch', -1))
        if gs_stop <= 0:
            gs_stop = int(getattr(args, 'max_epoch', 100)) + 1
        geom_start = max(gs_start, int(getattr(args, 'geom_late_start_epoch', gs_start)))
        geom_stop = int(getattr(args, 'geom_late_stop_epoch', -1))
        if geom_stop <= 0:
            geom_stop = int(getattr(args, 'max_epoch', 100)) + 1

        late_window_open = (gs_start <= int(epoch) < gs_stop) or (geom_start <= int(epoch) < geom_stop)
        if not late_window_open or det_score is None or main_score is None:
            if self._late_stage_mode != 'off':
                self._set_late_stage_mode('off', epoch, args=args, reason='late_window_closed')
            else:
                self._late_gs_confirm_count = 0
                self._late_geom_confirm_count = 0
            return

        best_main = (
            main_score if self._best_legacy_main_score is None
            else float(self._best_legacy_main_score)
        )
        best_det = det_score if self._best_det_score is None else float(self._best_det_score)
        best_seg = seg_score if self._best_seg_score is None or seg_score is None else float(self._best_seg_score)
        best_gap = gap_score if self._best_det_seg_gap is None or gap_score is None else float(self._best_det_seg_gap)

        main_drop = max(0.0, best_main - float(main_score))
        det_drop = max(0.0, best_det - float(det_score))
        seg_drop = 0.0 if seg_score is None else max(0.0, best_seg - float(seg_score))
        gap_worsen = 0.0 if gap_score is None else max(0.0, float(gap_score) - best_gap)

        confirm_epochs = max(2, int(getattr(args, 'late_stage_confirm_epochs', 2)))
        # Geometry is a rollback-protected probe; requiring two bad validations
        # often arms it only after det has already degraded too far.
        geom_confirm_epochs = 1
        cooldown_until = int(self._late_stage_cooldown_until)
        gs_drop_thr = float(getattr(args, 'gs_det_activation_drop', -1.0))
        if gs_drop_thr <= 0.0:
            gs_drop_thr = float(getattr(args, 'gs_activation_drop', 0.004))
        gs_drop_thr = max(1e-6, gs_drop_thr)
        geom_gap_thr = max(0.0, float(getattr(args, 'geom_gap_threshold', 0.05)))
        geom_worsen_thr = max(1e-6, float(getattr(args, 'geom_gap_worsen_threshold', 0.01)))
        geom_worsen_eff = min(geom_worsen_thr, max(0.002, 0.25 * geom_worsen_thr))
        best_gap_val = 0.0 if best_gap is None else float(best_gap)
        geom_gap_eff = min(
            geom_gap_thr,
            max(0.045, best_gap_val + geom_worsen_eff)
        )
        geom_quality_floor = max(0.0, float(getattr(args, 'geom_quality_floor', 0.35)))
        seg_stable_drop = max(0.0, float(getattr(args, 'geom_seg_stable_drop', 0.005)))
        kill_drop = float(getattr(args, 'late_stage_kill_drop', gs_drop_thr))
        if kill_drop <= 0.0:
            kill_drop = gs_drop_thr
        kill_drop = max(0.0, kill_drop)
        val_drop_tol = max(0.0, float(getattr(args, 'dgt_val_drop_tolerance', 0.0)))
        geom_guard_drop = min(kill_drop, max(0.0010, val_drop_tol if val_drop_tol > 0.0 else 0.0015))
        geom_close_det_drop = max(geom_guard_drop, min(gs_drop_thr, 1.5 * kill_drop))

        severe_late_drop = max(kill_drop, float(getattr(args, 'dgt_val_severe_drop', kill_drop)))
        late_best_restore_drop = max(gs_drop_thr, severe_late_drop)
        late_probe_allowed = det_drop <= late_best_restore_drop
        if det_drop > late_best_restore_drop:
            reason = f'late_severe_det_drop={det_drop:.4f}'
            if self._late_stage_mode != 'off':
                self._set_late_stage_mode('off', epoch, args=args, reason=reason)
            else:
                self._late_stage_cooldown_until = int(epoch) + max(0, int(getattr(args, 'late_stage_cooldown_epochs', 2)))
                self._late_gs_confirm_count = 0
                self._late_geom_confirm_count = 0
            if self._best_model_state is not None:
                return self._restore_best_model_state(model, epoch, reason)

        det_regress = (
            bool(getattr(args, 'enable_grad_consistency', False))
            and late_probe_allowed
            and gs_start <= int(epoch) < gs_stop
            and det_drop >= gs_drop_thr
            and main_drop >= min(gs_drop_thr, 0.5 * gs_drop_thr + 0.001)
        )
        geom_regress = (
            late_probe_allowed
            and geom_start <= int(epoch) < geom_stop
            and gap_score is not None
            and seg_score is not None
            and float(gap_score) >= geom_gap_eff
            and det_drop <= geom_close_det_drop
            and seg_drop <= seg_stable_drop
            and min(float(det_score), float(seg_score)) >= geom_quality_floor
        )

        self._late_gs_confirm_count = (
            self._late_gs_confirm_count + 1 if det_regress else 0
        )
        self._late_geom_confirm_count = (
            self._late_geom_confirm_count + 1 if geom_regress else 0
        )

        has_gs_mode = self._late_stage_mode in ('gs', 'both')
        has_geom_mode = self._late_stage_mode in ('geom', 'both')

        if has_gs_mode and self._late_stage_ref_det_score is not None:
            det_hurt = max(0.0, float(self._late_stage_ref_det_score) - float(det_score))
            if det_hurt >= kill_drop:
                reason = f'gs_hurt_det={det_hurt:.4f}'
                self._set_late_stage_mode('off', epoch, args=args, reason=reason)
                return self._restore_late_stage_model_state(model, epoch, reason)
        if has_geom_mode and self._late_stage_ref_det_score is not None:
            ref_det = self._to_scalar_or_none(self._late_stage_ref_det_score)
            snap_det = self._to_scalar_or_none(self._late_stage_snapshot_det_score)
            if snap_det is not None:
                ref_det = snap_det if ref_det is None else max(float(ref_det), float(snap_det))
            det_hurt = max(0.0, float(ref_det) - float(det_score)) if ref_det is not None else 0.0
            if det_hurt >= geom_guard_drop:
                reason = f'geom_no_regret_drop={det_hurt:.4f}'
                self._set_late_stage_mode('off', epoch, args=args, reason=reason)
                return self._restore_late_stage_model_state(model, epoch, reason)

        gs_ready = self._late_gs_confirm_count >= confirm_epochs
        geom_ready = self._late_geom_confirm_count >= geom_confirm_epochs
        if self._late_stage_mode == 'gs' and geom_ready and geom_regress:
            self._set_late_stage_mode(
                'both' if det_regress else 'geom',
                epoch, main_score=main_score, det_score=det_score,
                args=args, reason=f'add_geom_gap={float(gap_score):.4f}'
            )
            return
        if self._late_stage_mode == 'geom' and gs_ready and det_regress:
            self._set_late_stage_mode(
                'both' if geom_regress else 'gs',
                epoch, main_score=main_score, det_score=det_score,
                args=args, reason=f'add_gs_det_drop={det_drop:.4f}'
            )
            return

        if self._late_stage_mode == 'both':
            if not det_regress and not geom_regress:
                self._set_late_stage_mode('off', epoch, args=args, reason='both_recovered')
                return
            if not det_regress:
                self._set_late_stage_mode(
                    'geom', epoch, main_score=main_score, det_score=det_score,
                    args=args, reason='gs_recovered'
                )
                return
            if not geom_regress:
                self._set_late_stage_mode(
                    'gs', epoch, main_score=main_score, det_score=det_score,
                    args=args, reason='geom_recovered'
                )
                return

        if self._late_stage_mode == 'gs' and not det_regress:
            self._set_late_stage_mode('off', epoch, args=args, reason='gs_recovered')
            return
        if self._late_stage_mode == 'geom' and not geom_regress:
            self._set_late_stage_mode('off', epoch, args=args, reason='geom_recovered')
            return

        if int(epoch) < cooldown_until or self._late_stage_mode != 'off':
            return

        if gs_ready and geom_ready:
            self._set_late_stage_mode(
                'both', epoch, main_score=main_score, det_score=det_score,
                args=args,
                reason=f'det_drop={det_drop:.4f},gap={float(gap_score):.4f}'
            )
            return

        if geom_ready:
            self._set_late_stage_mode(
                'geom', epoch, main_score=main_score, det_score=det_score,
                args=args, reason=f'gap={float(gap_score):.4f},det_drop={det_drop:.4f}'
            )
            return

        if gs_ready:
            self._set_late_stage_mode(
                'gs', epoch, main_score=main_score, det_score=det_score,
                args=args, reason=f'det_drop={det_drop:.4f}'
            )

    def _apply_staged_loss_controls(self, epoch, args, model=None):
        if not hasattr(self, 'global_loss_module'):
            return
        dgt_module = self.global_loss_module
        latest = dict(self._latest_eval_stats or {})
        main_score = self._to_scalar_or_none(
            latest.get('legacy_main_score', latest.get('main_score', None))
        )
        det_score = self._to_scalar_or_none(latest.get('det_score', None))
        seg_score = self._to_scalar_or_none(latest.get('seg_score', None))
        record_score = self._to_scalar_or_none(latest.get('record_score', self._prev_val_score))
        if record_score is None:
            record_score = self._compose_record_score(
                stats=latest, main_score=main_score, seg_score=seg_score
            )
        gap_score = self._to_scalar_or_none(latest.get('det_seg_gap', None))
        if gap_score is None and det_score is not None and seg_score is not None:
            gap_score = abs(det_score - seg_score)

        prev_mode = str(self._late_stage_mode)
        restored_model = bool(self._update_late_stage_controller(
            epoch, args, main_score, det_score, seg_score, gap_score, model=model
        ))
        mode_after_update = str(self._late_stage_mode)
        prev_has_geom = prev_mode in ('geom', 'both')
        has_geom_after_update = mode_after_update in ('geom', 'both')
        if model is not None and has_geom_after_update and not restored_model:
            snap_det = self._to_scalar_or_none(self._late_stage_snapshot_det_score)
            should_capture = (
                self._late_stage_model_state is None
                or not prev_has_geom
                or (det_score is not None and (snap_det is None or float(det_score) >= float(snap_det)))
            )
            if should_capture:
                self._capture_late_stage_model_state(model, mode_after_update, det_score)
        elif prev_has_geom and not has_geom_after_update and not restored_model:
            self._clear_late_stage_model_state()

        dgt_module.geom_weight = 0.0
        dgt_module.harmonize_weight = 0.0

        det_drop = None
        if self._best_det_score is not None and det_score is not None:
            det_drop = max(0.0, float(self._best_det_score) - float(det_score))

        mode = str(self._late_stage_mode)
        epoch_i = int(epoch)
        gs_start = max(0, int(getattr(args, 'gs_late_start_epoch', 60)))
        gs_stop = int(getattr(args, 'gs_late_stop_epoch', -1))
        if gs_stop <= 0:
            gs_stop = int(getattr(args, 'max_epoch', 100)) + 1
        geom_start = max(0, int(getattr(args, 'geom_late_start_epoch', gs_start)))
        geom_stop = int(getattr(args, 'geom_late_stop_epoch', -1))
        if geom_stop <= 0:
            geom_stop = int(getattr(args, 'max_epoch', 100)) + 1

        if self._base_geom_weight > 0.0 and mode in ('geom', 'both') and geom_start <= epoch_i < geom_stop:
            geom_min = max(0.0, min(1.0, float(getattr(args, 'geom_late_ramp_min', 0.08))))
            geom_max = max(geom_min, min(1.0, float(getattr(args, 'geom_late_ramp_max', 0.25))))
            dgt_module.geom_weight = self._base_geom_weight * geom_max * float(self._geom_stage_limit)

        if (
            bool(getattr(args, 'enable_grad_consistency', False))
            and self._base_gs_weight > 0.0
            and mode in ('gs', 'both')
            and gs_start <= epoch_i < gs_stop
        ):
            gs_min = max(0.0, min(1.0, float(getattr(args, 'gs_late_ramp_min', 0.12))))
            gs_max = max(gs_min, min(1.0, float(getattr(args, 'gs_late_ramp_max', 0.30))))
            dgt_module.harmonize_weight = self._base_gs_weight * gs_max * float(self._gs_stage_limit)

        active_tags = self._active_stage_tags(epoch, args, dgt_module)
        active_weights = self._active_stage_weights_text(dgt_module)

        if dist.get_rank() == 0 and self._last_stage_log_epoch != int(epoch):
            best_txt = 0.0 if self._best_val_score is None else float(self._best_val_score)
            best_dyn_txt = 0.0 if self._dyn_best_val_score is None else float(self._dyn_best_val_score)
            self.logger.info(
                f"[StageCtrl] e{epoch} best_record={best_txt:.4f} "
                f"best_dyn_record={best_dyn_txt:.4f} "
                f"ctrl={0.0 if record_score is None else float(record_score):.4f} "
                f"legacy_main={0.0 if main_score is None else float(main_score):.4f} "
                f"det={0.0 if det_score is None else float(det_score):.4f} "
                f"seg={0.0 if seg_score is None else float(seg_score):.4f} "
                f"gap={0.0 if gap_score is None else float(gap_score):.4f} "
                f"dd={0.0 if det_drop is None else float(det_drop):.4f} "
                f"freeze={str(getattr(self, '_dgt_freeze_reason', 'off'))}@e{int(getattr(self, '_dgt_last_freeze_epoch', -1))} "
                f"DGTL[{int(getattr(args, 'dgt_start_epoch', 0))},{int(getattr(args, 'dgt_stop_epoch', 0))}) "
                f"{'' if not active_weights else ' ' + active_weights}"
                f"{'' if not active_tags else ' ' + active_tags}"
            )
            self._last_stage_log_epoch = int(epoch)
        return restored_model

    def _broadcast_loss_module_state(self, src=0):
        """Keep DGTLossModule state identical on all ranks."""
        if not hasattr(self, 'global_loss_module'):
            return
        if not dist.is_available() or not dist.is_initialized() or dist.get_world_size() == 1:
            return

        module_state = self.global_loss_module.state_dict()
        tensor_device = None
        for value in module_state.values():
            if torch.is_tensor(value):
                dist.broadcast(value, src=src)
                if tensor_device is None:
                    tensor_device = value.device

        if tensor_device is None:
            tensor_device = torch.device(
                f"cuda:{torch.cuda.current_device()}"
            ) if torch.cuda.is_available() else torch.device("cpu")

        stop_epoch = torch.tensor(
            [float(self.global_loss_module.stop_dynamic_epoch)],
            device=tensor_device
        )
        dist.broadcast(stop_epoch, src=src)
        self.global_loss_module.stop_dynamic_epoch = int(stop_epoch.item())

        extra_state = torch.tensor(
            [
                float(getattr(self.global_loss_module, 'main_max_step', 0.0)),
                float(getattr(self.global_loss_module, 'aux_max_step', 0.0)),
                float(getattr(self.global_loss_module, 'geom_weight', 0.0)),
                float(getattr(self.global_loss_module, 'harmonize_weight', 0.0)),
                1.0 if bool(getattr(self.global_loss_module, 'enable_dynamic', True)) else 0.0,
                float(getattr(self.global_loss_module, 'core_residual_scale', 1.0)),
                float(getattr(self.global_loss_module, 'aux_residual_scale', 1.0)),
                float(getattr(self.global_loss_module, 'gs_residual_scale', 1.0)),
                float(getattr(self.global_loss_module, 'geom_residual_scale', 1.0)),
            ],
            device=tensor_device
        )
        dist.broadcast(extra_state, src=src)
        self.global_loss_module.main_max_step = float(extra_state[0].item())
        self.global_loss_module.aux_max_step = float(extra_state[1].item())
        self.global_loss_module.geom_weight = float(extra_state[2].item())
        self.global_loss_module.harmonize_weight = float(extra_state[3].item())
        self.global_loss_module.enable_dynamic = bool(extra_state[4].item() > 0.5)
        self.global_loss_module.core_residual_scale = float(extra_state[5].item())
        self.global_loss_module.aux_residual_scale = float(extra_state[6].item())
        self.global_loss_module.gs_residual_scale = float(extra_state[7].item())
        self.global_loss_module.geom_residual_scale = float(extra_state[8].item())

    @staticmethod
    def get_datasets(args):
        """Initialize datasets."""
        train_dataset = None
        test_dataset = None
        return train_dataset, test_dataset

    # BRIEF dataloader.
    def get_loaders(self, args):
        """Initialize data loaders."""

        def seed_worker(worker_id):
            worker_seed = torch.initial_seed() % 2 ** 32
            np.random.seed(worker_seed)
            random.seed(worker_seed)
            np.random.seed(np.random.get_state()[1][0] + worker_id)

        # Datasets
        train_dataset, test_dataset = self.get_datasets(args)

        # Samplers and loaders
        g = torch.Generator()
        g.manual_seed(0)

        if args.eval:
            train_loader = None
        else:
            train_sampler = DistributedSampler(train_dataset)
            train_loader = DataLoader(
                train_dataset,
                batch_size=args.batch_size,
                shuffle=False,  # TODO
                num_workers=args.num_workers,
                worker_init_fn=seed_worker,
                pin_memory=True,
                sampler=train_sampler,
                drop_last=True,
                generator=g
            )

        test_sampler = DistributedEvalSampler(test_dataset)
        test_loader = DataLoader(
            test_dataset,
            batch_size=args.batch_size,
            shuffle=False,
            num_workers=args.num_workers,
            worker_init_fn=seed_worker,
            pin_memory=True,
            sampler=test_sampler,
            drop_last=False,
            generator=g
        )
        return train_loader, test_loader

    @staticmethod
    def get_model(args):
        """Initialize the model."""
        return None

    @staticmethod
    def get_criterion(args):
        """Get loss criterion for training."""
        losses = ['boxes', 'labels', 'masks']
        if args.use_contrastive_align:
            losses.append('contrastive_align')
        matcher = HungarianMatcher(1, 0, 2, args.use_soft_token_loss)
        set_criterion = SetCriterion(
            matcher=matcher,
            losses=losses, eos_coef=0.1, temperature=0.07
        )
        criterion = compute_hungarian_loss

        return criterion, set_criterion

    @staticmethod
    def get_optimizer(args, model):
        """Initialize optimizer."""
        if args.frozen:
            print("-------------------------------frozen EDA parameters------------------------------------")
            for n, p in model.named_parameters():
                if "x_mask" not in n and "x_query" not in n and "seed_decoder" not in n:
                    p.requires_grad = False
            param_dicts = [
                {
                    "params": [
                        p for n, p in model.named_parameters()
                        if "x_mask" in n or "x_query" in n or "seed_decoder" in n
                    ]
                },
                {
                    "params": [],
                    "lr": args.lr_backbone
                },
                {
                    "params": [],
                    "lr": args.text_encoder_lr
                }
            ]
        elif args.small_lr:
            param_dicts = [
                {
                    "params": [
                        p for n, p in model.named_parameters()
                        if "x_mask" in n or "x_query" in n or "seed_decoder" in n
                    ],
                    "lr": args.lr
                },
                {
                    "params": [
                        p for n, p in model.named_parameters()
                        if "backbone_net" not in n and "text_encoder" not in n
                           and "x_mask" not in n and "x_query" not in n and "seed_decoder" not in n
                           and p.requires_grad
                    ],
                    "lr": args.lr * 0.01
                },
                {
                    "params": [
                        p for n, p in model.named_parameters()
                        if "backbone_net" in n and p.requires_grad
                    ],
                    "lr": args.lr_backbone * 0.01
                },
                {
                    "params": [
                        p for n, p in model.named_parameters()
                        if "text_encoder" in n and p.requires_grad
                    ],
                    "lr": args.text_encoder_lr * 0.01
                }
            ]
        else:
            param_dicts = [
                {
                    "params": [
                        p for n, p in model.named_parameters()
                        if "backbone_net" not in n and "text_encoder" not in n
                           and p.requires_grad
                    ]
                },
                {
                    "params": [
                        p for n, p in model.named_parameters()
                        if "backbone_net" in n and p.requires_grad
                    ],
                    "lr": args.lr_backbone
                },
                {
                    "params": [
                        p for n, p in model.named_parameters()
                        if "text_encoder" in n and p.requires_grad
                    ],
                    "lr": args.text_encoder_lr
                }
            ]
        optimizer = optim.AdamW(param_dicts,
                                lr=args.lr,
                                weight_decay=args.weight_decay)
        return optimizer

    def _flatten_grads(self, grads):
        """
        Flatten a list of gradients into a single 1D tensor.
        Returns None if all gradients are None.
        """
        if grads is None:
            return None

        valid_grads = [g for g in grads if g is not None]
        if len(valid_grads) == 0:
            return None

        return torch.cat([g.flatten() for g in valid_grads])

    @staticmethod
    def _select_gs_anchor_params(shared_params, max_params=12, max_elems=800_000):
        """
        Use a bounded subset of shared parameters for GS.
        Full-backbone/full-text autograd.grad is too expensive and can stall DDP.
        """
        if not shared_params:
            return []
        max_params = max(1, int(max_params))
        max_elems = max(1, int(max_elems))
        max_single_param_elems = max(1, max_elems // 2)
        candidates = []
        for param in shared_params:
            if param is None or (not param.requires_grad):
                continue
            if int(param.numel()) > max_single_param_elems:
                continue
            candidates.append(param)
        if not candidates:
            fallback = [p for p in shared_params if p is not None and p.requires_grad]
            if not fallback:
                return []
            smallest = min(fallback, key=lambda p: int(p.numel()))
            return [smallest]
        chosen = []
        total_elems = 0
        for param in reversed(candidates):
            n_elem = int(param.numel())
            if total_elems + n_elem > max_elems:
                continue
            chosen.append(param)
            total_elems += n_elem
            if len(chosen) >= max_params or total_elems >= max_elems:
                break
        if not chosen:
            for param in sorted(candidates, key=lambda p: int(p.numel())):
                n_elem = int(param.numel())
                if total_elems + n_elem > max_elems and chosen:
                    break
                chosen.append(param)
                total_elems += n_elem
                if len(chosen) >= max_params or total_elems >= max_elems:
                    break
        chosen.reverse()
        return chosen

    def _compute_gs_trust(self, det_loss, seg_loss, det_quality=None, seg_quality=None, agree_quality=None):
        """
        Compute trust/good/bad signals from det/seg losses (EMA based).
        - good_both: both losses not worse than EMA within a small margin
        - bad_both: both losses worse than EMA beyond bad_margin
        """
        det_val = det_loss.detach()
        seg_val = seg_loss.detach()
        cur_losses = torch.stack([det_val, seg_val]).to(dtype=torch.float32)
        if self.gs_loss_ema is None:
            self.gs_loss_ema = cur_losses.clone()
            return cur_losses.new_tensor(0.0), False, False, cur_losses.new_tensor(0.0)

        ema_losses = self.gs_loss_ema.to(device=cur_losses.device, dtype=cur_losses.dtype)
        det_ratio = cur_losses[0] / ema_losses[0].clamp(min=1e-6)
        seg_ratio = cur_losses[1] / ema_losses[1].clamp(min=1e-6)
        max_ratio = torch.maximum(det_ratio, seg_ratio)

        margin = max(self.gs_trust_margin, 1e-3)
        good_margin = max(0.0, 0.5 * margin)
        good_both = bool(((det_ratio <= (1.0 + good_margin)) & (seg_ratio <= (1.0 + good_margin))).item())
        if good_both:
            ema_trust = cur_losses.new_tensor(1.0)
        else:
            ema_trust = 1.0 - torch.relu(max_ratio - 1.0) / margin
            ema_trust = torch.clamp(ema_trust, min=0.0, max=1.0)

        bad_margin = max(self.gs_bad_margin, 0.0)
        bad_both = bool(((det_ratio > (1.0 + bad_margin)) & (seg_ratio > (1.0 + bad_margin))).item())
        if bad_margin <= 1e-6:
            ema_bad_strength = cur_losses.new_tensor(1.0 if bad_both else 0.0)
        else:
            ema_bad_strength = (
                torch.relu(det_ratio - 1.0) + torch.relu(seg_ratio - 1.0)
            ) / (2.0 * bad_margin)
            ema_bad_strength = torch.clamp(ema_bad_strength, min=0.0, max=1.0)

        alpha = self.gs_ema_alpha
        self.gs_loss_ema = alpha * cur_losses + (1 - alpha) * ema_losses
        # GT-guided strict mode:
        # if GT-quality/alignment is unavailable, disable GS influence to avoid non-GT drift.
        if det_quality is None or seg_quality is None or agree_quality is None:
            return cur_losses.new_tensor(0.0), False, False, cur_losses.new_tensor(0.0)

        if not torch.is_tensor(det_quality):
            det_quality = cur_losses.new_tensor(float(det_quality))
        else:
            det_quality = det_quality.detach().to(device=cur_losses.device, dtype=cur_losses.dtype)
        if not torch.is_tensor(seg_quality):
            seg_quality = cur_losses.new_tensor(float(seg_quality))
        else:
            seg_quality = seg_quality.detach().to(device=cur_losses.device, dtype=cur_losses.dtype)
        if not torch.is_tensor(agree_quality):
            agree_quality = cur_losses.new_tensor(float(agree_quality))
        else:
            agree_quality = agree_quality.detach().to(device=cur_losses.device, dtype=cur_losses.dtype)
        good_both_q = (
            (det_quality >= self.gs_det_good_thr)
            & (seg_quality >= self.gs_seg_good_thr)
            & (agree_quality >= self.gs_agree_good_thr)
        )
        trusted_q = (
            (det_quality >= self.gs_det_bad_thr)
            & (seg_quality >= self.gs_seg_bad_thr)
            & (agree_quality >= self.gs_agree_bad_thr)
        )

        det_denom = max(self.gs_det_good_thr - self.gs_det_bad_thr, 1e-6)
        seg_denom = max(self.gs_seg_good_thr - self.gs_seg_bad_thr, 1e-6)
        agree_denom = max(self.gs_agree_good_thr - self.gs_agree_bad_thr, 1e-6)
        det_q_norm = (det_quality - self.gs_det_bad_thr) / det_denom
        seg_q_norm = (seg_quality - self.gs_seg_bad_thr) / seg_denom
        agree_q_norm = (agree_quality - self.gs_agree_bad_thr) / agree_denom
        trust_q = torch.clamp(torch.minimum(torch.minimum(det_q_norm, seg_q_norm), agree_q_norm), min=0.0, max=1.0)

        regress_margin = max(0.02, 0.25 * bad_margin)
        ema_regress = bool((max_ratio > (1.0 + regress_margin)).item())
        regress_strength = torch.relu(max_ratio - 1.0) / max(bad_margin, regress_margin, 1e-6)
        regress_strength = torch.clamp(regress_strength, min=0.0, max=1.0)

        blend = max(0.0, min(1.0, float(self.gs_quality_blend)))
        trust = (1.0 - blend) * ema_trust + blend * trust_q
        trust = torch.clamp(trust, min=0.0, max=1.0)
        # GT quality is a trust gate, not a low-quality trigger. This keeps GS
        # anchored to GT-derived quality and avoids pred-pred soft supervision.
        trusted_q_bool = bool(trusted_q.item())
        good_both = bool(good_both and bool(good_both_q.item()))
        bad_both = bool(ema_regress and trusted_q_bool)
        bad_strength = (1.0 - blend) * torch.maximum(ema_bad_strength, regress_strength) + blend * trust_q
        bad_strength = torch.clamp(bad_strength, min=0.0, max=1.0)
        if not bad_both:
            bad_strength = bad_strength.new_tensor(0.0)

        return trust, good_both, bad_both, bad_strength

    def _project_conflicting_grads(
            self, g_det, g_seg, eta=1.0, threshold=-0.25, trust=1.0,
            good_both=False, bad_both=False, bad_strength=0.0
    ):
        """
        [Safety Critical] Asymmetric Gradient Surgery with Consistency Gate.
        - cos_sim > 0: mildly encourage aligned directions (scale > 1)
        - cos_sim < 0: mildly suppress conflicts (scale < 1)
        - apply projection correction only for severe conflicts, with reduced strength
        """
        g_det_flat = self._flatten_grads(g_det)
        g_seg_flat = self._flatten_grads(g_seg)

        # 1. Basic checks
        if g_det_flat is None or g_seg_flat is None:
            base = g_det_flat if g_det_flat is not None else g_seg_flat
            if base is None:
                return g_det, g_seg, torch.tensor(0.0)
            return g_det, g_seg, base.new_tensor(0.0)

        # 2. Norm computation and numerical-stability guards
        norm_det = torch.norm(g_det_flat)
        norm_seg = torch.norm(g_seg_flat)

        # If gradients are too small, treat them as converged/dead zones; skip projection to avoid divide-by-zero.
        if norm_det < 1e-6 or norm_seg < 1e-6:
            return g_det, g_seg, g_det_flat.new_tensor(0.0)

        # 3. Compute cosine similarity
        dot_product = torch.dot(g_det_flat, g_seg_flat)
        cos_sim = dot_product / (norm_det * norm_seg + 1e-8)
        cos_sim = torch.nan_to_num(cos_sim, nan=0.0)
        cos_sim = cos_sim.detach()

        # Keep good/uncertain samples on the original optimization trajectory.
        if not bad_both:
            return g_det, g_seg, cos_sim

        # 4. [CRITICAL] Magnitude-ratio guard
        # If the norm ratio is too large (>100x), skip projection and only apply consistency gating.
        ratio = norm_det / (norm_seg + 1e-8)
        if ratio > 100.0 or ratio < 0.01:
            return g_det, g_seg, cos_sim

        # 5. Conflict correction (only intervene when both tasks are poor and cos < threshold)
        # Strict rule: only treat it as a severe conflict when cos_sim is below the threshold (e.g. -0.25).
        if bad_both and cos_sim < threshold:
            # Limit projection strength to avoid numerical instability.
            dot_product = torch.nan_to_num(dot_product, nan=0.0)

            proj_scale_d = dot_product / (norm_seg ** 2 + 1e-8)
            proj_scale_s = dot_product / (norm_det ** 2 + 1e-8)

            eta_eff = max(0.0, float(eta)) * (0.15 + 0.25 * float(bad_strength))
            if eta_eff <= 1e-6:
                return g_det, g_seg, cos_sim

            g_det_new = []
            g_seg_new = []

            for g_d, g_s in zip(g_det, g_seg):
                if g_d is not None and g_s is not None:
                    # Safely compute correction terms
                    correction_d = eta_eff * proj_scale_d * g_s
                    correction_s = eta_eff * proj_scale_s * g_d

                    # Avoid NaNs
                    correction_d = torch.nan_to_num(correction_d, nan=0.0)
                    correction_s = torch.nan_to_num(correction_s, nan=0.0)

                    g_det_new.append(g_d - correction_d)
                    g_seg_new.append(g_s - correction_s)
                else:
                    g_det_new.append(g_d)
                    g_seg_new.append(g_s)

            return g_det_new, g_seg_new, cos_sim

        return g_det, g_seg, cos_sim

    def _rewrite_shared_grads(self, shared_params, g_det_raw, g_seg_raw, g_det_proj, g_seg_proj):
        """Replace the det/seg component of shared grads with the projected version."""
        for param, gd_raw, gs_raw, gd_proj, gs_proj in zip(
                shared_params, g_det_raw, g_seg_raw, g_det_proj, g_seg_proj):
            total_grad = param.grad
            ref_grad = total_grad
            if ref_grad is None:
                for cand in (gd_raw, gs_raw, gd_proj, gs_proj):
                    if cand is not None:
                        ref_grad = cand
                        break
            if ref_grad is None:
                continue

            other_grad = torch.zeros_like(ref_grad)
            if total_grad is not None:
                other_grad = total_grad.detach().clone()

            if gd_raw is not None:
                other_grad = other_grad - gd_raw.detach()
            if gs_raw is not None:
                other_grad = other_grad - gs_raw.detach()

            new_grad = other_grad
            if gd_proj is not None:
                new_grad = new_grad + gd_proj.detach()
            if gs_proj is not None:
                new_grad = new_grad + gs_proj.detach()

            if param.grad is None:
                param.grad = new_grad
            else:
                param.grad.copy_(new_grad)

    def train_one_epoch(self, epoch, train_loader, model, criterion, set_criterion, optimizer, scheduler, args):
        stat_dict = {}
        model.train()

        # --- [1. Fix: remove optimizer injection] ---
        # DGTL V3 is statistics-based and does not require gradient updates, so it should not be added to the optimizer.
        if hasattr(self, 'global_loss_module'):
            # Ensure the module is on the correct device
            device = next(model.parameters()).device
            self.global_loss_module.to(device)
            # Set train mode (V3 has no dropout/bn, but this keeps the usual convention)
            self.global_loss_module.train()
            controls_restored_model = False
            if dist.get_rank() == 0:
                controls_restored_model = bool(self._apply_staged_loss_controls(epoch, args, model=model))
            if dist.is_initialized():
                restored_flag = torch.zeros(1, device=device)
                if dist.get_rank() == 0 and controls_restored_model:
                    restored_flag.fill_(1.0)
                dist.broadcast(restored_flag, src=0)
                if bool(restored_flag.item() > 0.5):
                    self._broadcast_model_state(model, src=0)
                    # Late-stage no-regret restore invalidates Adam moments from the rejected probe.
                    optimizer.state.clear()
                self._broadcast_loss_module_state(src=0)
            elif controls_restored_model:
                optimizer.state.clear()

        # --- [2. Gradient surgery policy control] ---
        GS_START_EPOCH = max(0, int(getattr(args, 'gs_late_start_epoch', 60)))
        GS_END_EPOCH = int(getattr(args, 'gs_late_stop_epoch', -1))
        if GS_END_EPOCH <= 0:
            GS_END_EPOCH = int(getattr(args, 'max_epoch', 100)) + 1

        # Determine the active time window
        enable_gs = getattr(args, 'enable_grad_consistency', False)
        if enable_gs and GS_START_EPOCH <= epoch < GS_END_EPOCH:
            do_gradient_surgery = True
        else:
            do_gradient_surgery = False
        if hasattr(self, 'global_loss_module'):
            if float(getattr(self.global_loss_module, 'harmonize_weight', 0.0)) <= 0.0:
                do_gradient_surgery = False

        # GS uses low-cost anchors on the shared backbone by default, instead of covering the whole text encoder.
        real_model = model.module if hasattr(model, 'module') else model
        shared_params = []
        if hasattr(real_model, 'backbone_net'):
            shared_params += [p for p in real_model.backbone_net.parameters() if p.requires_grad]
        if hasattr(real_model, 'text_encoder'):
            shared_params += [p for p in real_model.text_encoder.parameters() if p.requires_grad]

        gs_param_pool = []
        gs_anchor_source = "none"
        if hasattr(real_model, 'backbone_net'):
            gs_param_pool += [p for p in real_model.backbone_net.parameters() if p.requires_grad]
            if len(gs_param_pool) > 0:
                gs_anchor_source = "backbone"
        if getattr(args, 'gs_use_text_anchors', False) and hasattr(real_model, 'text_encoder'):
            small_text_params = [
                p for p in real_model.text_encoder.parameters()
                if p.requires_grad and int(p.numel()) <= max(1, int(getattr(args, 'gs_anchor_max_elems', 800000)) // 4)
            ]
            if small_text_params:
                gs_param_pool += small_text_params
                gs_anchor_source = "backbone+text" if gs_anchor_source != "none" else "text"
        if len(gs_param_pool) == 0 and len(shared_params) > 0:
            gs_param_pool = shared_params
            gs_anchor_source = "shared_fallback"

        gs_anchor_params = self._select_gs_anchor_params(
            gs_param_pool,
            max_params=getattr(args, 'gs_anchor_max_params', 12),
            max_elems=getattr(args, 'gs_anchor_max_elems', 800000)
        )
        if len(gs_anchor_params) == 0:
            do_gradient_surgery = False
        elif dist.get_rank() == 0 and not getattr(self, '_gs_anchor_logged', False):
            gs_anchor_elems = sum(int(p.numel()) for p in gs_anchor_params)
            self.logger.info(
                f"[GS] anchor_source={gs_anchor_source} anchor_params={len(gs_anchor_params)} "
                f"anchor_elems={gs_anchor_elems}"
            )
            self._gs_anchor_logged = True

        if dist.get_rank() == 0:
            train_loader = tqdm(train_loader, ascii=True)

        for batch_idx, batch_data in enumerate(train_loader):
            batch_data = self._to_gpu(batch_data)
            inputs = self._get_inputs(batch_data)

            end_points = model(inputs)
            if 'point_clouds' not in end_points and 'point_clouds' in inputs:
                end_points['point_clouds'] = inputs['point_clouds']
            for key in batch_data:
                if key not in end_points: end_points[key] = batch_data[key]

            # 3. Compute Loss
            # Note: pass dgt_module so compute_hungarian_loss can access the current weights.
            loss, end_points = self._compute_loss(
                end_points, criterion, set_criterion, args,
                current_epoch=epoch,
                total_epochs=args.max_epoch
                # If _compute_loss supports passing the module, pass it here; otherwise set it externally.
            )

            # --- [3. Fix: core statistics collection hook] ---
            # Statistics are already collected inside compute_hungarian_loss; avoid double counting here.

            # 4. Gradient Surgery & Backward
            optimizer.zero_grad()

            has_gs_tensors = (do_gradient_surgery and
                              'loss_det_tensor' in end_points and
                              'loss_seg_tensor' in end_points)
            if dist.is_available() and dist.is_initialized():
                gs_ready_t = loss.new_tensor(1.0 if has_gs_tensors else 0.0)
                dist.all_reduce(gs_ready_t, op=dist.ReduceOp.MIN)
                has_gs_tensors = bool(gs_ready_t.item() > 0.5)

            gs_trust = 0.0
            gs_good = False
            gs_bad = False
            gs_penalty_val = 0.0
            gs_penalty_bad_val = 0.0
            quality_valid = False
            det_q = None
            seg_q = None
            agree_q = None
            cos_sim = None
            bad_strength = 0.0
            projected_shared_grads = None

            gs_penalty_tensor = loss.new_tensor(0.0)

            if has_gs_tensors:
                loss_det = end_points.get('loss_det_weighted_tensor', end_points['loss_det_tensor'])
                loss_seg = end_points.get('loss_seg_weighted_tensor', end_points['loss_seg_tensor'])

                quality_valid = end_points.get('gs_quality_valid', None)
                if torch.is_tensor(quality_valid):
                    quality_valid = bool((quality_valid.detach() > 0.5).item())
                elif quality_valid is not None:
                    quality_valid = bool(quality_valid)
                else:
                    quality_valid = False
                if dist.is_available() and dist.is_initialized():
                    # Every rank must make the same GS decision before autograd.grad.
                    # Otherwise SyncBN/DDP collectives can be entered on only a subset
                    # of ranks, causing NCCL watchdog timeouts.
                    quality_valid_t = loss.new_tensor(1.0 if quality_valid else 0.0)
                    dist.all_reduce(quality_valid_t, op=dist.ReduceOp.MIN)
                    quality_valid = bool(quality_valid_t.item() > 0.5)

                det_q = end_points.get('gs_det_quality', None)
                seg_q = end_points.get('gs_seg_quality', None)
                agree_q = end_points.get('gs_agreement', None)
                if quality_valid:
                    gs_trust, gs_good, gs_bad, bad_strength = self._compute_gs_trust(
                        loss_det, loss_seg,
                        det_quality=det_q,
                        seg_quality=seg_q,
                        agree_quality=agree_q
                    )
                    if dist.is_available() and dist.is_initialized():
                        gs_bad_t = loss.new_tensor(1.0 if gs_bad else 0.0)
                        dist.all_reduce(gs_bad_t, op=dist.ReduceOp.MIN)
                        gs_bad = bool(gs_bad_t.item() > 0.5)
                        if not gs_bad:
                            bad_strength = loss.new_tensor(0.0)

                    if gs_bad:
                        g_shared_det = torch.autograd.grad(
                            loss_det, gs_anchor_params, retain_graph=True, allow_unused=True
                        )
                        g_shared_seg = torch.autograd.grad(
                            loss_seg, gs_anchor_params, retain_graph=True, allow_unused=True
                        )
                        g_det_flat = self._flatten_grads(g_shared_det)
                        g_seg_flat = self._flatten_grads(g_shared_seg)

                        if g_det_flat is not None and g_seg_flat is not None:
                            norm_det = torch.norm(g_det_flat)
                            norm_seg = torch.norm(g_seg_flat)
                            if norm_det >= 1e-6 and norm_seg >= 1e-6:
                                tau = float(getattr(args, 'gs_tau', -0.1))
                                tau = max(-1.0, min(1.0, tau))
                                project_threshold = min(max(tau, -0.15), -0.05)
                                gs_eta = float(getattr(self.global_loss_module, 'gs_residual_scale', 0.0))
                                g_det_proj, g_seg_proj, cos_sim = self._project_conflicting_grads(
                                    g_shared_det,
                                    g_shared_seg,
                                    eta=gs_eta,
                                    threshold=project_threshold,
                                    trust=gs_trust,
                                    good_both=gs_good,
                                    bad_both=gs_bad,
                                    bad_strength=bad_strength
                                )
                                projected_shared_grads = (
                                    g_shared_det,
                                    g_shared_seg,
                                    g_det_proj,
                                    g_seg_proj,
                                )

                        del g_shared_det, g_shared_seg

                    if gs_bad and cos_sim is not None and hasattr(self, 'global_loss_module'):
                        dgt_module = self.global_loss_module
                        bad_penalty = dgt_module.compute_grad_harmonization_penalty(
                            cos_sim, loss_det, loss_seg,
                            epoch, args.max_epoch,
                            rho=bad_strength,
                            tau=float(getattr(args, 'gs_tau', -0.1))
                        )
                        gs_penalty_tensor = gs_penalty_tensor + bad_penalty
                        gs_penalty_bad_val = float(bad_penalty.detach().item())

            if gs_penalty_tensor is not None and torch.is_tensor(gs_penalty_tensor):
                loss = loss + gs_penalty_tensor
                end_points['loss_gs'] = gs_penalty_tensor.detach()
                gs_penalty_val = float(gs_penalty_tensor.detach().item())
            else:
                end_points['loss_gs'] = loss.new_tensor(0.0)

            can_run_gs = bool(has_gs_tensors and quality_valid and cos_sim is not None)
            loss.backward()

            if projected_shared_grads is not None:
                self._rewrite_shared_grads(
                    gs_anchor_params,
                    projected_shared_grads[0],
                    projected_shared_grads[1],
                    projected_shared_grads[2],
                    projected_shared_grads[3]
                )
                del projected_shared_grads

            # Track GS stats (per-batch accumulation)
            if has_gs_tensors and quality_valid:
                stat_dict['gs_count'] = stat_dict.get('gs_count', 0) + 1
                stat_dict['gs_trust'] = stat_dict.get('gs_trust', 0.0) + float(gs_trust.detach().item())
                stat_dict['gs_good'] = stat_dict.get('gs_good', 0.0) + (1.0 if gs_good else 0.0)
                stat_dict['gs_bad'] = stat_dict.get('gs_bad', 0.0) + (1.0 if gs_bad else 0.0)
                stat_dict['gs_conflict'] = stat_dict.get('gs_conflict', 0.0) + (
                    1.0 if cos_sim is not None and cos_sim.item() < 0 else 0.0
                )
                stat_dict['gs_penalty'] = stat_dict.get('gs_penalty', 0.0) + gs_penalty_val
                stat_dict['gs_penalty_bad'] = stat_dict.get('gs_penalty_bad', 0.0) + gs_penalty_bad_val
                if det_q is not None and seg_q is not None and agree_q is not None:
                    stat_dict['gs_q_count'] = stat_dict.get('gs_q_count', 0) + 1
                    det_q_val = float(det_q.detach().item()) if torch.is_tensor(det_q) else float(det_q)
                    seg_q_val = float(seg_q.detach().item()) if torch.is_tensor(seg_q) else float(seg_q)
                    agree_q_val = float(agree_q.detach().item()) if torch.is_tensor(agree_q) else float(agree_q)
                    stat_dict['gs_det_q'] = stat_dict.get('gs_det_q', 0.0) + det_q_val
                    stat_dict['gs_seg_q'] = stat_dict.get('gs_seg_q', 0.0) + seg_q_val
                    stat_dict['gs_agree_q'] = stat_dict.get('gs_agree_q', 0.0) + agree_q_val

                if (batch_idx + 1) % args.print_freq == 0:
                    stat_dict['grad_cos_sim'] = float(cos_sim.item()) if cos_sim is not None else 0.0

            # 5. Clip & Step
            if args.clip_norm > 0:
                torch.nn.utils.clip_grad_norm_(model.parameters(), args.clip_norm)
                # The DGTL module does not need gradient clipping because it has no gradients.

            optimizer.step()
            scheduler.step()

            # 6. Logging
            stat_dict = self._accumulate_stats(stat_dict, end_points)

            if (batch_idx + 1) % args.print_freq == 0:
                if dist.get_rank() == 0:  # Main process only
                    self.logger.info(f'Train: [{epoch}][{batch_idx + 1}/{len(train_loader)}]  ')

                    # Log Dynamic Weights (multipliers)
                    w_info = ""
                    if hasattr(self, 'global_loss_module'):
                        w_b = self.global_loss_module.get_multiplier('bbox')
                        w_m = self.global_loss_module.get_multiplier('mask')
                        w_a = self.global_loss_module.get_multiplier('aux')
                        w_b_val = w_b.item() if torch.is_tensor(w_b) else float(w_b)
                        w_m_val = w_m.item() if torch.is_tensor(w_m) else float(w_m)
                        w_a_val = w_a.item() if torch.is_tensor(w_a) else float(w_a)
                        w_be = end_points.get('dgt_m_bbox_eff', None)
                        w_me = end_points.get('dgt_m_mask_eff', None)
                        w_ae = end_points.get('dgt_m_aux', None)
                        w_rn = end_points.get('dgt_core_renorm', None)
                        w_vb = end_points.get('dgt_val_core_bias', None)
                        if torch.is_tensor(w_be) and torch.is_tensor(w_me) and torch.is_tensor(w_ae):
                            rn_txt = ""
                            if torch.is_tensor(w_rn):
                                rn_txt = f" Rn:{float(w_rn.item()):.5f}"
                            vb_txt = ""
                            if torch.is_tensor(w_vb) and abs(float(w_vb.item())) > 1e-8:
                                vb_txt = f" VB:{float(w_vb.item()):.5f}"
                            rs_txt = (
                                f" Rs(C:{float(getattr(self.global_loss_module, 'core_residual_scale', 1.0)):.5f},"
                                f"A:0.00000,"
                                f"G:{float(getattr(self.global_loss_module, 'gs_residual_scale', 1.0)):.5f},"
                                f"Geo:{float(getattr(self.global_loss_module, 'geom_residual_scale', 1.0)):.5f})"
                            )
                            w_info = (
                                f"[DGT] Raw(B:{w_b_val:.6f},M:{w_m_val:.6f}) "
                                f"Eff(B:{float(w_be.item()):.6f},M:{float(w_me.item()):.6f},A:{float(w_ae.item()):.6f})"
                                f"{rn_txt}{vb_txt}{rs_txt} "
                            )
                        else:
                            w_info = f"[DGT] M_box:{w_b_val:.6f} M_msk:{w_m_val:.6f} M_aux:{w_a_val:.6f} "

                    # Log GS Info
                    eta_info = ""
                    if can_run_gs:
                        eta_info = " GS:ON"
                    elif do_gradient_surgery and float(getattr(self.global_loss_module, 'harmonize_weight', 0.0)) > 0.0:
                        eta_info = " GS:ARM"
                    if can_run_gs and 'grad_cos_sim' in stat_dict:
                        eta_info += f" Cos:{stat_dict['grad_cos_sim']:.2f}"
                    if self._dynamic_updates_frozen():
                        eta_info += " DGTL-Dyn:LOCK"
                    gsq_info = ""
                    if can_run_gs and stat_dict.get('gs_count', 0) > 0:
                        gs_cnt = max(1.0, float(stat_dict.get('gs_count', 0)))
                        trust_avg = stat_dict.get('gs_trust', 0.0) / gs_cnt
                        good_rate = stat_dict.get('gs_good', 0.0) / gs_cnt
                        bad_rate = stat_dict.get('gs_bad', 0.0) / gs_cnt
                        conflict_rate = stat_dict.get('gs_conflict', 0.0) / gs_cnt
                        pen_avg = stat_dict.get('gs_penalty', 0.0) / gs_cnt
                        pen_bad_avg = stat_dict.get('gs_penalty_bad', 0.0) / gs_cnt
                        if stat_dict.get('gs_q_count', 0) > 0:
                            q_cnt = max(1.0, float(stat_dict.get('gs_q_count', 0)))
                            det_q_avg = stat_dict.get('gs_det_q', 0.0) / q_cnt
                            seg_q_avg = stat_dict.get('gs_seg_q', 0.0) / q_cnt
                            agree_avg = stat_dict.get('gs_agree_q', 0.0) / q_cnt
                            gsq_info = (
                                f" GSQ(det:{det_q_avg:.2f} seg:{seg_q_avg:.2f} "
                                f"agr:{agree_avg:.2f} trust:{trust_avg:.2f} "
                                f"good:{good_rate:.2f} bad:{bad_rate:.2f} "
                                f"conf:{conflict_rate:.2f} pen:{pen_avg:.4f} "
                                f"badP:{pen_bad_avg:.4f})"
                            )
                        else:
                            gsq_info = (
                                f" GSQ(trust:{trust_avg:.2f} good:{good_rate:.2f} "
                                f"bad:{bad_rate:.2f} conf:{conflict_rate:.2f} pen:{pen_avg:.4f} "
                                f"badP:{pen_bad_avg:.4f})"
                            )

                    log_str = ''.join([
                        f'{key} {stat_dict[key] / (batch_idx + 1):.4f} \t'
                        for key in sorted(stat_dict.keys())
                        if 'loss' in key and 'proposal_' not in key
                           and 'last_' not in key and 'head_' not in key
                           and 'tensor' not in key
                    ])
                    self.logger.info(log_str + w_info + eta_info + gsq_info)

                    for key in self.tensorboard.item["train_loss"]:
                        if key in stat_dict:
                            self.tensorboard.item["train_loss"][key] = stat_dict[key] / (batch_idx + 1)
                    self.tensorboard.dump_tensorboard("train_loss", (epoch - 1) * len(train_loader) + batch_idx + 1)

        # --- [4. Fix: core epoch-finalization hook] ---
        # Must be called at the end of the epoch loop.
        # This triggers DDP synchronization and computes weights for the next round.
        if hasattr(self, 'global_loss_module'):
            self.global_loss_module.step_epoch(epoch)

        # End of epoch logging
        if dist.get_rank() == 0:
            for key in self.tensorboard.item["train_loss"]:
                if key in stat_dict:
                    self.tensorboard.item["train_loss"][key] = stat_dict[key] / len(train_loader)
            self.tensorboard.dump_tensorboard("train_loss", epoch * len(train_loader))
            self.tensorboard.item["train_lr"]["lr_base"] = optimizer.param_groups[0]['lr']
            if len(optimizer.param_groups) > 1:
                self.tensorboard.item["train_lr"]["lr_pointnet"] = optimizer.param_groups[1]['lr']
            self.tensorboard.dump_tensorboard("train_lr", epoch)

    def _compute_loss(self, end_points, criterion, set_criterion, args, current_epoch=0, total_epochs=100):
        """
        Wrapper to handle Loss computation and ensure tensors are retained for Gradient Surgery.
        """
        is_eval_mode = getattr(args, 'eval', False)

        if is_eval_mode:
            dummy_loss = torch.tensor(0.0, device=next(iter(end_points.values())).device)
            return dummy_loss, end_points
        else:
            # Pass global_loss_module to enable dynamic weights
            loss, end_points = criterion(
                end_points, args.num_decoder_layers,
                set_criterion,
                query_points_obj_topk=args.query_points_obj_topk,
                epoch=current_epoch,
                total_epochs=total_epochs,
                dgt_module=getattr(self, 'global_loss_module', None)  # Keep compatibility
            )

            # [CRITICAL ADDITION]
            # For Gradient Surgery, ensure end_points contains 'loss_det_tensor' and 'loss_seg_tensor'.
            # If the upstream criterion (compute_hungarian_loss) does not explicitly return these tensors,
            # we attempt a fallback here using loss_bbox and loss_mask.
            # Note: this requires loss_bbox and related terms to be gradient-carrying tensors.

            if 'loss_det_tensor' not in end_points and 'loss_bbox' in end_points:
                # Try to build a detection-loss group (approximate for GS)
                # Note: the weights here must stay consistent with the criterion; if that is too complex, handle it inside the criterion.
                # For safety, if the criterion does not provide them, do not fabricate them aggressively; just skip GS later.
                pass

            return loss, end_points

    def main(self, args):
        """Run main training/testing pipeline."""
        # Get loaders
        train_loader, test_loader = self.get_loaders(args)
        if not args.eval:
            n_data = len(train_loader.dataset)
            self.logger.info(f"length of training dataset: {n_data}")
        n_data = len(test_loader.dataset)
        rank = dist.get_rank() if dist.is_initialized() else 0
        world_size = dist.get_world_size() if dist.is_initialized() else 1
        local_eval = len(test_loader.sampler) if hasattr(test_loader, 'sampler') else n_data
        self.logger.info(f"length of testing dataset: {n_data}")
        self.logger.info(
            f"evaluation scope: full dataset, non-overlapping shards, "
            f"world_size={world_size}, rank={rank}, local_eval_samples={local_eval}"
        )

        # Get model
        model = self.get_model(args)

        # Get criterion
        criterion, set_criterion = self.get_criterion(args)

        # Get optimizer
        optimizer = self.get_optimizer(args, model)

        # Get scheduler
        if not args.eval:
            scheduler = get_scheduler(optimizer, len(train_loader), args)
        else:
            scheduler = None

        # Move model to devices
        if torch.cuda.is_available():
            if torch.cuda.device_count() > 1:
                # synBN
                model = torch.nn.SyncBatchNorm.convert_sync_batchnorm(model).cuda()
            else:
                model = model.cuda()

        # note Distributed Data-Parallel Training (DDP)
        model = DistributedDataParallel(
            model, device_ids=[args.local_rank],
            broadcast_buffers=False, find_unused_parameters=True
        )
        # Check for a checkpoint
        if args.checkpoint_path is not None and args.checkpoint_path != 'none':
            if not os.path.isfile(args.checkpoint_path):
                if args.eval:
                    raise FileNotFoundError(
                        "Checkpoint '{}' not found".format(args.checkpoint_path)
                    )
                if dist.get_rank() == 0:
                    print("=> Warning: checkpoint '{}' not found, training from scratch".format(
                        args.checkpoint_path
                    ))
                args.checkpoint_path = None
            else:
                # [Update] Pass in self.global_loss_module
                load_checkpoint(
                    args, model, optimizer, scheduler,
                    loss_module=self.global_loss_module, trainer=self,
                    n_iter_per_epoch=(len(train_loader) if train_loader is not None else None)
                )
                if self._best_model_state is None and self._dyn_best_val_score is not None:
                    self._best_model_state = self._clone_model_state(model)
        # ##############################################
        # NOTE [eval-only] Just eval and end execution #
        # ##############################################
        if args.eval:
            if dist.is_initialized():
                dist.barrier()
            if dist.get_rank() == 0:
                print("Testing evaluation.....................")
            if hasattr(test_loader, "sampler") and hasattr(test_loader.sampler, "set_epoch"):
                test_loader.sampler.set_epoch(args.start_epoch)
            self.evaluate_one_epoch(
                args.start_epoch, test_loader, model, args
            )
            if dist.is_initialized():
                dist.barrier()
            return

        # ##############################
        # NOTE Training and Validation #
        # ##############################
        for epoch in range(args.start_epoch, args.max_epoch + 1):
            train_loader.sampler.set_epoch(epoch)
            tic = time.time()

            # train *
            self.train_one_epoch(
                epoch, train_loader, model,
                criterion, set_criterion,
                optimizer, scheduler, args
            )

            # log (rank 0 only to avoid duplicates)
            if dist.get_rank() == 0:
                self.logger.info(
                    'epoch {}, total time {:.2f}, '
                    'lr_base {:.5f}, lr_pointnet {:.5f}'.format(
                        epoch, (time.time() - tic),
                        optimizer.param_groups[0]['lr'],
                        optimizer.param_groups[1]['lr']
                    )
                )

            # save model and validate
            if epoch % args.val_freq == 0:
                if dist.get_rank() == 0:
                    save_checkpoint(
                        args, epoch, model, optimizer, scheduler,
                        loss_module=self.global_loss_module,
                        trainer_state=self.get_trainer_state()
                    )
                if dist.is_initialized():
                    rank = dist.get_rank()
                    dist.barrier()

                # validate *
                if dist.get_rank() == 0:
                    print("Test evaluation.......")
                if hasattr(test_loader, "sampler") and hasattr(test_loader.sampler, "set_epoch"):
                    test_loader.sampler.set_epoch(epoch)
                val_score = self.evaluate_one_epoch(
                    epoch, test_loader, model, args
                )
                if dist.is_initialized():
                    rank = dist.get_rank()
                    dist.barrier()
                if dist.get_rank() == 0 and val_score is not None:
                    val_score = float(val_score)
                    eval_stats = dict(self._latest_eval_stats or {})
                    det_score = self._to_scalar_or_none(eval_stats.get('det_score', None))
                    seg_score = self._to_scalar_or_none(eval_stats.get('seg_score', None))
                    gap_score = self._to_scalar_or_none(eval_stats.get('det_seg_gap', None))
                    if gap_score is None and det_score is not None and seg_score is not None:
                        gap_score = abs(det_score - seg_score)
                    main_score = self._compose_main_score(eval_stats, fallback=val_score)
                    record_score = self._compose_record_score(
                        stats=eval_stats, main_score=main_score, seg_score=seg_score
                    )
                    eval_stats['legacy_main_score'] = main_score
                    eval_stats['main_score'] = main_score
                    eval_stats['record_score'] = record_score
                    self._latest_eval_stats = eval_stats
                    model_restored = self._handle_validation_feedback(
                        epoch, args, model, optimizer, scheduler, main_score,
                        det_score=det_score, seg_score=seg_score, gap_score=gap_score
                    )
                else:
                    model_restored = False
                if dist.is_initialized():
                    sync_device = next(model.parameters()).device
                    model_restored_flag = torch.zeros(1, device=sync_device)
                    if dist.get_rank() == 0 and model_restored:
                        model_restored_flag.fill_(1.0)
                    dist.broadcast(model_restored_flag, src=0)
                    if bool(model_restored_flag.item() > 0.5):
                        self._broadcast_model_state(model, src=0)
                    self._broadcast_loss_module_state(src=0)

        # Training is over
        save_checkpoint(
            args, 'last', model, optimizer, scheduler,
            self.global_loss_module, self.get_trainer_state(), True
        )
        saved_path = os.path.join(args.log_dir, 'ckpt_epoch_last.pth')
        self.logger.info("Saved in {}".format(saved_path))
        if dist.is_initialized():
            rank = dist.get_rank()
            dist.barrier()
        if hasattr(test_loader, "sampler") and hasattr(test_loader.sampler, "set_epoch"):
            test_loader.sampler.set_epoch(args.max_epoch)
        self.evaluate_one_epoch(
            args.max_epoch, test_loader, model, args
        )
        if dist.is_initialized():
            rank = dist.get_rank()
            dist.barrier()
        return saved_path

    @staticmethod
    def _to_gpu(data_dict):
        if torch.cuda.is_available():
            for key in data_dict:
                if isinstance(data_dict[key], torch.Tensor):
                    data_dict[key] = data_dict[key].cuda(non_blocking=True)
        return data_dict

    @staticmethod
    def _get_inputs(batch_data):
        return {
            'point_clouds': batch_data['point_clouds'].float(),
            'text': batch_data['utterances']

            # Attribute-target strings
            , 'modified_entity_bs': batch_data['modified_entity_bs']

        }

    @staticmethod
    def _accumulate_stats(stat_dict, end_points):
        for key in end_points:
            if 'loss' in key or 'acc' in key or 'ratio' in key:
                if key not in stat_dict:
                    stat_dict[key] = 0
                if isinstance(end_points[key], (float, int)):
                    stat_dict[key] += end_points[key]
                else:
                    stat_dict[key] += end_points[key].item()
        return stat_dict

    # BRIEF eval
    @torch.no_grad()
    def _main_eval_branch(self, batch_idx, batch_data, test_loader, model,
                          stat_dict, args):
        # Move to GPU
        batch_data = self._to_gpu(batch_data)
        inputs = self._get_inputs(batch_data)
        if "train" not in inputs:
            inputs.update({"train": False})
        else:
            inputs["train"] = False

        # STEP Forward pass (model prediction; this is the critical path)
        end_points = model(inputs)

        # STEP Merge batch data into end_points
        for key in batch_data:
            assert (key not in end_points)
            end_points[key] = batch_data[key]

        for key in end_points:
            if 'pred_size' in key:
                end_points[key] = torch.clamp(end_points[key], min=1e-6)

        return stat_dict, end_points

    @torch.no_grad()
    def evaluate_one_epoch(self, epoch, test_loader, model, args):
        """
        Eval grounding after a single epoch.

        Some of the args:
            model: a nn.Module that returns end_points (dict)
        """
        return None
