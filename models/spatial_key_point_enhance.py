import math
import torch
import torch.nn.functional as F
from torch import nn
import numpy as np

from   .diff.diff_attn_dim import TextPointMultiHeadDiffAttn
from  .spatial_related_tools import MappingPointDim


class PointEnhancedByKeyText(nn.Module):
    def __init__(self, point_dim=256, text_dim=768, dropout=0.10, activation="relu", n_heads=8,
                 dim_feedforward=256, diff_attend_lang=True, diff_attend_vis=True, self_attend_vis=True):
        super().__init__()
        self.text_dim = text_dim
        self.output_dim = dim_feedforward
        self.encoder_dims=dim_feedforward
        self.num_tokens=dim_feedforward
        self.text_dim_adjust = MappingPointDim(text_dim=768, output_dim=256)
        self.text_attr_obj_dim_adjust = MappingPointDim(text_dim=768, output_dim=256)

        self.text_attr_obj_enhance_point = TextPointMultiHeadDiffAttn(embed_dim=self.output_dim, dropout=0.1,
                                                                      num_heads=8, depth=1) #in bsz, tgt_len, embed_dim
        self.text_enhance_point = TextPointMultiHeadDiffAttn(embed_dim=self.output_dim, dropout=0.1, num_heads=8,
                                                             depth=1)
        # self.dgcnn_1 = DGCNN(encoder_channel = self.encoder_dims, output_channel = self.num_tokens)
        # grouper
        # self.group_divider = Group(num_group=self.num_group, group_size=self.group_size)
        # define the encoder
        # self.encoder = Encoder(encoder_channel=self.encoder_dims)

        self.conv1 = nn.Conv1d(self.output_dim * 2, self.output_dim, kernel_size=1)
        self.bn1 = nn.BatchNorm1d(self.output_dim, eps=1e-6)

        self.conv2 = nn.Conv1d(self.output_dim * 2, self.output_dim, kernel_size=1)
        self.bn2 = nn.BatchNorm1d(self.output_dim, eps=1e-6)

    def forward(self, point_feats, pos_feats, point_mask,xyz, text_feats, text_padding_mask, attr_obj_feats,
                obj_padding_mask, spatial_point_xyz=None, end_points={}, detected_feats=None, detected_mask=None):

        text_features_adjusted=self.text_dim_adjust(text_feats)
        text_attr_obj_features_adjusted = self.text_attr_obj_dim_adjust(attr_obj_feats)

        text_features_adjusted=F.relu(self.bn1(self.conv1(torch.cat((text_features_adjusted,text_attr_obj_features_adjusted),dim=1))))


        text_point_enhanced = self.text_enhance_point(text_features_adjusted,point_feats) ##in bsz, tgt_len, embed_dim
        text_attr_obj_point_enhanced = self.text_attr_obj_enhance_point(point_feats,text_features_adjusted) #in bsz, tgt_len, embed_dim

        key_point_enhanced=F.relu(self.bn2(self.conv2(torch.cat((text_point_enhanced,text_attr_obj_point_enhanced),dim=1))))#+point_feats

        return key_point_enhanced,text_attr_obj_point_enhanced,text_attr_obj_point_enhanced

