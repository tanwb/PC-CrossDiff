import os
import sys
import time
import numpy as np

import torch

from tensorboardX import SummaryWriter  # tensorboard --logdir=./output/tensorboard --port 3090


class TensorBoard():
    def __init__(self, out_dir, distributed_rank):
        self.init_log_item()

        self.best = {
            "epoch": 0,
            "train_loss": float("inf"),
            "val_loss": float("inf"),
            "iou": float("inf"),
            "recall": -float("inf"),
            "acc": -float("inf")
        }

        if distributed_rank == 0:
            self.tensorboard_writer = {
                "train": SummaryWriter(os.path.join('tensorboard_output/', "tensorboard/train")),
                "val": SummaryWriter(os.path.join('tensorboard_output/', "tensorboard/val"))
            }

    def dump_tensorboard(self, phase, timestamp):
        """
        ✅ 修复版：动态处理所有键
        """
        # ═══════════════════════════════════════════════════════
        # 训练损失
        # ═══════════════════════════════════════════════════════
        if phase == "train_loss":
            for key, value in self.item[phase].items():
                if isinstance(value, (int, float, list)):
                    # 处理列表（取最后一个值）或标量
                    scalar_value = value[-1] if isinstance(value, list) and len(value) > 0 else value
                    if isinstance(scalar_value, (int, float)):
                        self.tensorboard_writer["train"].add_scalar(
                            f"train_loss/{key}",
                            scalar_value,
                            timestamp
                        )

        # ═══════════════════════════════════════════════════════
        # 学习率
        # ═══════════════════════════════════════════════════════
        elif phase == "train_lr":
            for key, value in self.item[phase].items():
                if isinstance(value, (int, float, list)):
                    scalar_value = value[-1] if isinstance(value, list) and len(value) > 0 else value
                    if isinstance(scalar_value, (int, float)):
                        self.tensorboard_writer["train"].add_scalar(
                            f"learning_rate/{key}",
                            scalar_value,
                            timestamp
                        )

         # ═══════════════════════════════════════════════════════
        # 验证分数
        # ═══════════════════════════════════════════════════════
        elif phase == "val_score":
            for key, value in self.item[phase].items():
                if isinstance(value, (int, float, list)):
                    scalar_value = value[-1] if isinstance(value, list) and len(value) > 0 else value
                    if isinstance(scalar_value, (int, float)):
                        self.tensorboard_writer["val"].add_scalar(
                            f"score/{key}",
                            scalar_value,
                            timestamp
                        )

    def init_log_item(self):
        self.item = {
            "train_lr": {
                "lr_base": [],
                "lr_pointnet": [],
            },

            "train_loss": {
                "loss": [],
                "loss_bbox": [],
                "loss_ce": [],
                "loss_sem_align": [],
                "loss_giou": [],
                "query_points_generation_loss": [],
                "loss_mask": [],
                "loss_dice": [],

            },

            "val_score": {
                "soft_token_0.25": [],
                "soft_token_0.5": [],
                "contrastive_0.25": [],
                "contrastive_0.5": [],
            },

        }

    def update(self, phase, epoch):
        if phase == "train":
            self.dump_tensorboard("train_loss", epoch)
            self.dump_tensorboard("train_lr", epoch)

