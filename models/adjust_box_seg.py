import numpy as np
import torch
from torch import nn
from scipy.optimize import linear_sum_assignment


def mask_to_bbox(points, mask):
    masked_points = points[mask]
    if masked_points.shape[0] == 0:
        return None
    x_min, y_min, z_min = masked_points.min(axis=0)
    x_max, y_max, z_max = masked_points.max(axis=0)
    return (x_min, y_min, z_min, x_max, y_max, z_max)


def bbox_to_mask(points, bbox, epsilon=1e-6):
    x_min, y_min, z_min, x_max, y_max, z_max = bbox
    mask = (
            (points[:, 0] >= x_min - epsilon) & (points[:, 0] <= x_max + epsilon) &
            (points[:, 1] >= y_min - epsilon) & (points[:, 1] <= y_max + epsilon) &
            (points[:, 2] >= z_min - epsilon) & (points[:, 2] <= z_max + epsilon)
    )
    return mask


class LinearWeightScheduler:
    def __init__(self, start_weight, end_weight, total_steps):
        self.start_weight = start_weight
        self.end_weight = end_weight
        self.total_steps = total_steps
        self.current_step = 0

    def step(self):
        self.current_step += 1
        return self.get_weight()

    def get_weight(self):
        if self.current_step >= self.total_steps:
            return self.end_weight
        return self.start_weight + (self.end_weight - self.start_weight) * (self.current_step / self.total_steps)

