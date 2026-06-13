import math
import torch
import torch.nn.functional as F
from torch import nn
import numpy as np
import einops

class MappingPointDim0(nn.Module):
    def __init__(self, text_dim=288, output_dim=256):
        super(MappingPointDim0, self).__init__()

        self.text_dim = text_dim
        self.output_dim = output_dim
        self.conv1 = nn.Conv1d(self.text_dim, self.output_dim, kernel_size=1)
        self.bn1 = nn.BatchNorm1d(self.output_dim,eps=1e-6)

        self.upsample_first = nn.ConvTranspose1d(self.output_dim, self.output_dim, kernel_size=2, stride=2)
        self.interpolate = nn.functional.interpolate

    def forward(self, x):
        x = x.transpose(1, 2)
        x = nn.SiLU()(self.bn1(self.conv1(x)))
        x = self.upsample_first(x)
        x = self.interpolate(x, size=(self.output_dim,), mode='linear', align_corners=False)
        x = x.transpose(1, 2)
        return x


class MappingPointDim(nn.Module):
    def __init__(self, text_dim=768, output_dim=256):
        super(MappingPointDim, self).__init__()

        self.text_dim = text_dim
        self.output_dim = output_dim
        self.intermediate_dim = 512

        # Use a smaller convolution kernel to capture more contextual information
        self.conv1 = nn.Conv1d(self.text_dim, self.intermediate_dim, kernel_size=1)
        self.bn1 = nn.BatchNorm1d(self.intermediate_dim,eps=1e-6)

        self.conv2 = nn.Conv1d(self.intermediate_dim, self.output_dim, kernel_size=1)
        self.bn2 = nn.BatchNorm1d(self.output_dim,eps=1e-6)

        self.upsample_first = nn.ConvTranspose1d(self.output_dim, self.output_dim, kernel_size=2, stride=2)
        self.interpolate = nn.functional.interpolate

    def forward(self, x):
        x = x.transpose(1, 2)
        x = nn.SiLU()(self.bn1(self.conv1(x)))
        if self.output_dim != self.intermediate_dim:
            x = nn.SiLU()(self.bn2(self.conv2(x)))
        x = self.upsample_first(x)
        x = self.interpolate(x, size=(self.output_dim,), mode='linear', align_corners=False)
        x = x.transpose(1, 2)
        return x

# resource
def calc_pairwise_locs(obj_centers, eps=1e-10):
    pairwise_locs = einops.repeat(obj_centers, 'b l d -> b l 1 d') \
                    - einops.repeat(obj_centers, 'b l d -> b 1 l d')
    pairwise_dists = torch.sqrt(torch.sum(pairwise_locs ** 2, 3) + eps)  # (b, l, l)
    # if self.config.spatial_dist_norm:
    #     max_dists = torch.max(pairwise_dists.view(pairwise_dists.size(0), -1), dim=1)[0]
    #     norm_pairwise_dists = pairwise_dists / einops.repeat(max_dists, 'b -> b 1 1')
    # else:
    norm_pairwise_dists = pairwise_dists

    pairwise_dists_2d = torch.sqrt(torch.sum(pairwise_locs[..., :2] ** 2, 3) + eps)

    pairwise_locs = torch.stack(
        [norm_pairwise_dists, pairwise_locs[..., 2] / pairwise_dists,
         pairwise_dists_2d / pairwise_dists, pairwise_locs[..., 1] / pairwise_dists_2d,
         pairwise_locs[..., 0] / pairwise_dists_2d],
        dim=3
    )
    return pairwise_locs



class PositionEmbeddingLearned(nn.Module):
    """Absolute pos embedding, learned."""

    def __init__(self, input_channel, num_pos_feats=288):
        super().__init__()
        self.position_embedding_head = nn.Sequential(
            nn.Conv1d(input_channel, num_pos_feats, kernel_size=1),
            nn.BatchNorm1d(num_pos_feats),
            nn.SiLU(),
            nn.Conv1d(num_pos_feats, num_pos_feats, kernel_size=1))

    def forward(self, xyz):
        """Forward pass, xyz is (B, N, 3or6), output (B, F, N)."""
        xyz = xyz.transpose(1, 2).contiguous()
        position_embedding = self.position_embedding_head(xyz)
        return position_embedding

