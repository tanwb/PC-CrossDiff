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

import torch.nn as nn
import sys
import os
import  numpy as np
import torch
import  re
import torch.nn.functional as F

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
ROOT_DIR = os.path.dirname(BASE_DIR)
sys.path.append(ROOT_DIR)
sys.path.append(os.path.join(ROOT_DIR, 'utils'))
sys.path.append(os.path.join(ROOT_DIR, 'pointnet2'))
sys.path.append(os.path.join(ROOT_DIR, 'ops', 'pt_custom_ops'))

from pointnet2.pointnet2_modules import PointnetSAModuleVotes, PointnetFPModule

from .pcda import PointEnhancedByKeyText

class Pointnet2Backbone(nn.Module):
    """
    Backbone network for point cloud feature learning.

    Based on Pointnet++ single-scale grouping network.

    Parameters
    ----------
    input_feature_dim: int
        Number of input channels in the feature descriptor for each point.
        e.g. 3 for RGB.
    """

    def __init__(self, input_feature_dim=0, width=1, depth=2, output_dim=288):
        super().__init__()
        self.depth = depth
        self.width = width

        self.sa1 = PointnetSAModuleVotes(
            npoint=2048,
            radius=0.2,
            nsample=64,
            mlp=[input_feature_dim] + [64 * width for i in range(depth)] + [128 * width],
            use_xyz=True,
            normalize_xyz=True
        )

        self.sa2 = PointnetSAModuleVotes(
            npoint=1024,
            radius=0.4,
            nsample=32,
            mlp=[128 * width] + [128 * width for i in range(depth)] + [256 * width],
            use_xyz=True,
            normalize_xyz=True
        )

        self.sa3 = PointnetSAModuleVotes(
            npoint=512,
            radius=0.8,
            nsample=16,
            mlp=[256 * width] + [128 * width for i in range(depth)] + [256 * width],
            use_xyz=True,
            normalize_xyz=True
        )

        self.sa4 = PointnetSAModuleVotes(
            npoint=256,
            radius=1.2,
            nsample=16,
            mlp=[256 * width] + [128 * width for i in range(depth)] + [256 * width],
            use_xyz=True,
            normalize_xyz=True
        )


        self.key_text_enhance_point= PointEnhancedByKeyText(point_dim=256,text_dim=768,dim_feedforward=256)

        self.fp1 = PointnetFPModule(mlp=[256 * width + 256 * width, 256 * width, 256 * width])
        self.fp2 = PointnetFPModule(mlp=[256 * width + 256 * width, 256 * width, output_dim])



    def _break_up_pc(self, pc):
        xyz = pc[..., 0:3].contiguous()
        features = (
            pc[..., 3:].transpose(1, 2).contiguous()
            if pc.size(-1) > 3 else None
        )

        return xyz, features


    def forward(self, point_clouds,  text_feats, attr_obj_feats):

        end_points = {}

        xyz, features = self._break_up_pc(point_clouds)       # ([B, 50000, 3]), [B, 3, 50000])

        # --------- 4 SET ABSTRACTION LAYERS ---------
        xyz, features, fps_inds = self.sa1(xyz, features)
        end_points['sa1_inds'] = fps_inds       # ([B, 2048])
        end_points['sa1_xyz'] = xyz             # ([B, 2048, 3])
        end_points['sa1_features'] = features   # ([B, 128, 2048])
        # print(f"end_points['sa1_features']{features.shape}") #torch.Size([4, 128, 2048])

        xyz, features, fps_inds = self.sa2(xyz, features)  # this fps_inds is just 0,1,...,1023
        end_points['sa2_inds'] = fps_inds       # 1024
        end_points['sa2_xyz'] = xyz
        end_points['sa2_features'] = features


        xyz, features, fps_inds = self.sa3(xyz, features)
        end_points['sa3_xyz'] = xyz
        end_points['sa3_features'] = features


        xyz, features, fps_inds = self.sa4(xyz, features)  # this fps_inds is just 0,1,...,255
        end_points['sa4_xyz'] = xyz



        features,text_point_enhanced=self.key_text_enhance_point( # point_feats #need in : bsz, tgt_len, embed_dim
            point_feats=features,
            text_feats=text_feats,
            attr_obj_feats=attr_obj_feats

        )

        end_points['text_point_enhanced'] = text_point_enhanced


        end_points['sa4_features'] = features #   features.shape  torch.Size([B, 256, 256])

        # --------- 2 FEATURE UPSAMPLING LAYERS --------
        features = self.fp1(end_points['sa3_xyz'], end_points['sa4_xyz'], end_points['sa3_features'],
                            end_points['sa4_features'])
        features = self.fp2(end_points['sa2_xyz'], end_points['sa3_xyz'], end_points['sa2_features'], features)

        #
        end_points['fp2_features'] = features           # ([B, 288, 1024])
        end_points['fp2_xyz'] = end_points['sa2_xyz']   # ([B, 1024, 3])
        num_seed = end_points['fp2_xyz'].shape[1]       # 1024
        end_points['fp2_inds'] = end_points['sa1_inds'][:, 0:num_seed]  # ([B, 1024]) indices among the entire input point clouds

        return end_points