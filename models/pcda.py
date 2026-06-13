import math
import torch
import torch.nn.functional as F
from torch import nn
import numpy as np

from  .spatial_related_tools import MappingPointDim
from  .cross_model_diff_atten import TextPointMultiHeadDiffAttn

class PointEnhancedByKeyText(nn.Module):
    def __init__(self, point_dim=256,text_dim=768,dim_feedforward=256):
        super().__init__()
        self.output_dim = dim_feedforward
        self.text_dim_adjust = MappingPointDim(text_dim=768, output_dim=256)
        self.text_attr_dim_adjust = MappingPointDim(text_dim=768, output_dim=256)

        self.point_enhance_text = TextPointMultiHeadDiffAttn(embed_dim=self.output_dim, dropout=0.1,
                                                                      num_heads=8, depth=1) #in bsz, tgt_len, embed_dim
        self.text_enhance_point = TextPointMultiHeadDiffAttn(embed_dim=self.output_dim, dropout=0.1, num_heads=8,
                                                             depth=1)

        self.conv1 = nn.Conv1d(self.output_dim * 2, self.output_dim, kernel_size=1)
        self.bn1 = nn.BatchNorm1d(self.output_dim, eps=1e-6)

        self.conv2 = nn.Conv1d(self.output_dim * 2, self.output_dim, kernel_size=1)
        self.bn2 = nn.BatchNorm1d(self.output_dim, eps=1e-6)

    def forward(self, point_feats, text_feats,  attr_obj_feats):

        text_features_adjusted=self.text_dim_adjust(text_feats)
        text_attr_features_adjusted = self.text_attr_dim_adjust(attr_obj_feats)

        text_features_enhance=F.relu(self.bn1(self.conv1(torch.cat((text_features_adjusted,text_attr_features_adjusted),dim=1))))


        point_enhanced = self.point_enhance_text(text_features_enhance,point_feats) ##in bsz, tgt_len, embed_dim
        text_enhanced = self.text_enhance_point(point_feats,text_features_enhance) #in bsz, tgt_len, embed_dim

        key_point_enhanced=F.relu(self.bn2(self.conv2(torch.cat((point_enhanced,text_enhanced),dim=1))))#+point_feats

        return key_point_enhanced,text_enhanced

