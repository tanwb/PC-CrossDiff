import torch
import torch.nn as nn
import torch.nn.functional as F
from   .encoder_decoder_layers import (
    BiEncoder, BiEncoderLayer
)
from  .spatial_related_tools import PositionEmbeddingLearned,calc_pairwise_locs
from  .cross_model_diff_atten import TextPointMultiHeadDiffAttn,PointTextMultiHeadDiffAttn
from timm.models.layers import DropPath, trunc_normal_
from  .dgcnn import Group,DGCNN

class Encoder(nn.Module):
    def __init__(self, encoder_channel):
        super().__init__()
        self.encoder_channel = encoder_channel
        self.first_conv = nn.Sequential(
            nn.Conv1d(3, 128, 1),
            nn.BatchNorm1d(128),
            nn.ReLU(inplace=True),
            nn.Conv1d(128, 256, 1)
        )
        self.second_conv = nn.Sequential(
            nn.Conv1d(512, 512, 1),
            nn.BatchNorm1d(512),
            nn.ReLU(inplace=True),
            nn.Conv1d(512, self.encoder_channel, 1)
        )
    def forward(self, point_groups):
        '''
            point_groups : B G N 3
            -----------------
            feature_global : B G C
        '''
        bs, g, n , _ = point_groups.shape
        point_groups = point_groups.reshape(bs * g, n, 3)
        # encoder
        feature = self.first_conv(point_groups.transpose(2,1))  # BG 256 n
        feature_global = torch.max(feature,dim=2,keepdim=True)[0]  # BG 256 1
        feature = torch.cat([feature_global.expand(-1,-1,n), feature], dim=1)# BG 512 n
        feature = self.second_conv(feature) # BG 1024 n
        feature_global = torch.max(feature, dim=2, keepdim=False)[0] # BG 1024
        return feature_global.reshape(bs, g, self.encoder_channel)

class clda(nn.Module):

    def __init__(self, num_class=256, num_obj_class=485,
                 input_feature_dim=3,
                 num_queries=256,
                 num_decoder_layers=6, self_position_embedding='loc_learned',
                 contrastive_align_loss=True,
                 d_model=288, butd=True, pointnet_ckpt=None, data_path=None,
                 self_attend=True):
        """Initialize layers."""
        super().__init__()

        self.encoder_dims = 288
        self.num_tokens = 288  # N
        self.trans_dim = 288
        self.num_group = 128  # G
        self.group_size = 8
        self.depth = 3
        self.dropout = 0.1
        self.num_heads = 8
        self.dgcnn_1 = DGCNN(encoder_channel=self.encoder_dims,
                             output_channel=self.trans_dim)  # in f: B C N   # coor: B N 3, N stays unchanged throughout; the first input argument is f's channel C. Output: [bs, N, output_channel]

        # grouper
        self.group_divider = Group(num_group=self.num_group, group_size=self.group_size)
        # define the encoder
        self.encoder = Encoder(encoder_channel=self.encoder_dims)
        self.pos_embed_mask = nn.Sequential(
            nn.Linear(3, 128),
            nn.GELU(),
            nn.Linear(128, self.trans_dim)
        )
        self.pos_embed = PositionEmbeddingLearned(3, d_model)

        bi_layer = BiEncoderLayer(
            d_model, dropout=0.1, activation="relu",
            n_heads=8, dim_feedforward=256,
            self_attend_lang=self_attend, self_attend_vis=self_attend,
            use_butd_enc_attn=butd
        )
        self.cross_encoder_3 = BiEncoder(bi_layer, 3)

        self.blocks = TransformerEncoder(
            embed_dim=self.trans_dim,
            depth=self.depth,
            drop_path_rate=self.dropout,
            num_heads=self.num_heads
        )
        self.blocks2 = TransformerEncoder(
            embed_dim=self.trans_dim,
            depth=self.depth,
            drop_path_rate=self.dropout,
            num_heads=self.num_heads
        )
        self.norm = nn.LayerNorm(self.trans_dim)

        self.text_projector = nn.Sequential(
            nn.Linear(256, d_model),
            nn.LayerNorm(d_model, eps=1e-12),
            nn.Dropout(0.1)
        )
        self.diff_key_point = PointTextMultiHeadDiffAttn(embed_dim=288, dropout=0.1, num_heads=8,
                                                         depth=1)  # in bsz, tgt_len, embed_dim
        self.diff_key_text = TextPointMultiHeadDiffAttn(embed_dim=288, dropout=0.1, num_heads=8,
                                                        depth=1)  # in bsz, tgt_len, embed_dim



    # BRIEF forward.
    def forward(self, attr_obj_point=None, vis_feats=None, points_xyz=None, padding_mask=None, text_feats=None,
                text_padding_mask=None, superpoint=None, source_xzy=None, end_points={}, detected_feats=None,
                detected_mask=None, spatial_point_xyz=None):
        # 1. Alignment
        features, text_feats = self.cross_encoder_3(
            vis_feats=vis_feats.transpose(1, 2).contiguous(),
            # resource vis_feats ([B,288, 1024])   -- ([B, 1024, 288])
            pos_feats=self.pos_embed(points_xyz).transpose(1, 2).contiguous(),
            # xyz ([B, 1024, 3]) -- ([B, 1024, 288])  -- ([B,  288,1024])
            padding_mask=torch.zeros(
                len(points_xyz), points_xyz.size(1)
            ).to(points_xyz.device).bool(),
            text_feats=text_feats,
            text_padding_mask=text_padding_mask,
            end_points=end_points,
            detected_feats=detected_feats,
            detected_mask=detected_mask,
            spatial_point_xyz=calc_pairwise_locs(points_xyz.contiguous())  # torch.Size([4, N, N, 5])
        )
        # 3. Build relative positional relationships
        # input: B N 3  #output: B num_group group_size 3  #center : B num_group 3
        neighborhood, center = self.group_divider(points_xyz.contiguous())
        # in point_groups : B num_group group_size 3   #out feature_global : B num_group encoder_channel
        group_input_tokens = self.encoder(neighborhood)  # out B G C  # output [bs, 64, 288]
        dgcnn_out = self.dgcnn_1(group_input_tokens,
                                 center)  # in need f: B N C   # coor: B N 3, N stays unchanged throughout; the first input argument is f's channel C. Output: [bs, N, output_channel]
        # N 128   dgcnn_out # output [bs, 64, 288]
        anchor_feat = end_points.get('text_point_enhanced')
        cls_pos = torch.max(anchor_feat, dim=1, keepdim=True)[0]  # [B, 1, C]
        if cls_pos.shape[-1] != 256:
            cls_pos = F.interpolate(cls_pos, size=256, mode='linear', align_corners=False)
        cls_pos = self.text_projector(cls_pos)  # torch.Size([8, 1, 288])
        # # add pos embedding
        # pos = calc_pairwise_locs(center.contiguous()) # out torch.Size([8, 64, 64,5])
        # pos=self.pairwise_locs_encoder(pos)
        pos = self.pos_embed_mask(center)

        cat_cls_pos = torch.cat((cls_pos, pos), dim=1)
        diff_key_text = self.diff_key_text(text_feats, cat_cls_pos) + text_feats
        text_feats = self.blocks(diff_key_text) + text_feats
        text_feats = self.norm(text_feats)

        # 4. Pass dgcnn_out through differential attention and then BERT # dgcnn_out output [bs, 64, 288]
        diff_key_feats = self.diff_key_point(features, dgcnn_out) + features
        features = self.blocks2(diff_key_feats) + features
        features = self.norm(features)

        features = features.transpose(1, 2).contiguous()  # out f: B C N

        return features, text_feats, text_padding_mask


class Mlp(nn.Module):
    def __init__(self, in_features, hidden_features=None, out_features=None, act_layer=nn.GELU, drop=0.):
        super().__init__()
        out_features = out_features or in_features
        hidden_features = hidden_features or in_features
        self.fc1 = nn.Linear(in_features, hidden_features)
        self.act = act_layer()
        self.fc2 = nn.Linear(hidden_features, out_features)
        self.drop = nn.Dropout(drop)

    def forward(self, x):
        x = self.fc1(x)
        x = self.act(x)
        x = self.drop(x)
        x = self.fc2(x)
        x = self.drop(x)
        return x


class Attention(nn.Module):
    def __init__(self, dim, num_heads=8, qkv_bias=False, qk_scale=None, attn_drop=0., proj_drop=0.):
        super().__init__()
        self.num_heads = num_heads
        head_dim = dim // num_heads
        # NOTE scale factor was wrong in my original version, can set manually to be compat with prev weights
        self.scale = qk_scale or head_dim ** -0.5

        self.qkv = nn.Linear(dim, dim * 3, bias=qkv_bias)
        self.attn_drop = nn.Dropout(attn_drop)
        self.proj = nn.Linear(dim, dim)
        self.proj_drop = nn.Dropout(proj_drop)

    def forward(self, x):
        B, N, C = x.shape
        qkv = self.qkv(x).reshape(B, N, 3, self.num_heads, C // self.num_heads).permute(2, 0, 3, 1, 4)
        q, k, v = qkv[0], qkv[1], qkv[2]  # make torchscript happy (cannot use tensor as tuple)

        attn = (q @ k.transpose(-2, -1)) * self.scale
        attn = attn.softmax(dim=-1)
        attn = self.attn_drop(attn)

        x = (attn @ v).transpose(1, 2).reshape(B, N, C)
        x = self.proj(x)
        x = self.proj_drop(x)
        return x


class Block(nn.Module):
    def __init__(self, dim, num_heads, mlp_ratio=4., qkv_bias=False, qk_scale=None, drop=0., attn_drop=0.,
                 drop_path=0., act_layer=nn.GELU, norm_layer=nn.LayerNorm):
        super().__init__()
        self.norm1 = norm_layer(dim)

        # NOTE: drop path for stochastic depth, we shall see if this is better than dropout here
        self.drop_path = DropPath(drop_path) if drop_path > 0. else nn.Identity()
        self.norm2 = norm_layer(dim)
        mlp_hidden_dim = int(dim * mlp_ratio)
        self.mlp = Mlp(in_features=dim, hidden_features=mlp_hidden_dim, act_layer=act_layer, drop=drop)

        self.attn = Attention(
            dim, num_heads=num_heads, qkv_bias=qkv_bias, qk_scale=qk_scale, attn_drop=attn_drop, proj_drop=drop)

    def forward(self, x):
        x = x + self.drop_path(self.attn(self.norm1(x)))
        x = x + self.drop_path(self.mlp(self.norm2(x)))
        return x


class TransformerEncoder(nn.Module):
    """ Transformer Encoder without hierarchical structure
    """

    def __init__(self, embed_dim=768, depth=4, num_heads=12, mlp_ratio=4., qkv_bias=False, qk_scale=None,
                 drop_rate=0., attn_drop_rate=0., drop_path_rate=0.):
        super().__init__()

        self.blocks = nn.ModuleList([
            Block(
                dim=embed_dim, num_heads=num_heads, mlp_ratio=mlp_ratio, qkv_bias=qkv_bias, qk_scale=qk_scale,
                drop=drop_rate, attn_drop=attn_drop_rate,
                drop_path=drop_path_rate[i] if isinstance(drop_path_rate, list) else drop_path_rate
            )
            for i in range(depth)])

    def forward(self, x):
        for _, block in enumerate(self.blocks):
            x = block(x)
        return x
