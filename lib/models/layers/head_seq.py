import torch.nn as nn
import torch
import torch.nn.functional as F
from typing import Optional
from torch import Tensor
from torch.nn import Identity
from timm.models.layers import trunc_normal_
from timm.models.layers import DropPath
from lib.models.layers.frozen_bn import FrozenBatchNorm2d
import copy


def top_k_top_p_filtering_batch(logits, top_k=0, top_p=0.0, filter_value=-float('Inf')):
    """ Filter a distribution of logits using top-k and/or nucleus (top-p) filtering
        Args:
            logits: logits distribution shape (vocabulary size)
            top_k > 0: keep only top k tokens with highest probability (top-k filtering).
            top_p > 0.0: keep the top tokens with cumulative probability >= top_p (nucleus filtering).
                Nucleus filtering is described in Holtzman et al. (http://arxiv.org/abs/1904.09751)
        From: https://gist.github.com/thomwolf/1a5a29f6962089e871b94cbd09daf317
    """
    top_k = min(top_k, logits.size(-1))  # Safety check
    if top_k > 0:
        # Remove all tokens with a probability less than the last token of the top-k
        # torch.topk()返回最后一维最大的top_k个元素，返回值为二维(values,indices)
        # ...表示其他维度由计算机自行推断
        for i in range(logits.shape[0]):
            indices_to_remove = logits[i] < torch.topk(logits[i], top_k)[0][..., -1, None]
            logits[i][indices_to_remove] = filter_value  # 对于topk之外的其他元素的logits值设为负无穷

    if top_p > 0.0:
        for i in range(logits.shape[0]):
            sorted_logits, sorted_indices = torch.sort(logits[i], descending=True)  # 对logits进行递减排序
            cumulative_probs = torch.cumsum(F.softmax(sorted_logits, dim=-1), dim=-1)

            # Remove tokens with cumulative probability above the threshold
            sorted_indices_to_remove = cumulative_probs > top_p
            # Shift the indices to the right to keep also the first token above the threshold
            sorted_indices_to_remove[..., 1:] = sorted_indices_to_remove[..., :-1].clone()
            sorted_indices_to_remove[..., 0] = 0

            indices_to_remove = sorted_indices[sorted_indices_to_remove]
            logits[i][indices_to_remove] = filter_value
    return logits


def conv(in_planes, out_planes, kernel_size=3, stride=1, padding=1, dilation=1,
         freeze_bn=False):
    if freeze_bn:
        return nn.Sequential(
            nn.Conv2d(in_planes, out_planes, kernel_size=kernel_size, stride=stride,
                      padding=padding, dilation=dilation, bias=True),
            FrozenBatchNorm2d(out_planes),
            nn.ReLU(inplace=True))
    else:
        return nn.Sequential(
            nn.Conv2d(in_planes, out_planes, kernel_size=kernel_size, stride=stride,
                      padding=padding, dilation=dilation, bias=True),
            nn.BatchNorm2d(out_planes),
            nn.ReLU(inplace=True))


class Corner_Predictor(nn.Module):
    """ Corner Predictor module"""

    def __init__(self, inplanes=64, channel=256, feat_sz=20, stride=16, freeze_bn=False):
        super(Corner_Predictor, self).__init__()
        self.feat_sz = feat_sz
        self.stride = stride
        self.img_sz = self.feat_sz * self.stride
        '''top-left corner'''
        self.conv1_tl = conv(inplanes, channel, freeze_bn=freeze_bn)
        self.conv2_tl = conv(channel, channel // 2, freeze_bn=freeze_bn)
        self.conv3_tl = conv(channel // 2, channel // 4, freeze_bn=freeze_bn)
        self.conv4_tl = conv(channel // 4, channel // 8, freeze_bn=freeze_bn)
        self.conv5_tl = nn.Conv2d(channel // 8, 1, kernel_size=1)

        '''bottom-right corner'''
        self.conv1_br = conv(inplanes, channel, freeze_bn=freeze_bn)
        self.conv2_br = conv(channel, channel // 2, freeze_bn=freeze_bn)
        self.conv3_br = conv(channel // 2, channel // 4, freeze_bn=freeze_bn)
        self.conv4_br = conv(channel // 4, channel // 8, freeze_bn=freeze_bn)
        self.conv5_br = nn.Conv2d(channel // 8, 1, kernel_size=1)

        '''about coordinates and indexs'''
        with torch.no_grad():
            self.indice = torch.arange(0, self.feat_sz).view(-1, 1) * self.stride
            # generate mesh-grid
            self.coord_x = self.indice.repeat((self.feat_sz, 1)) \
                .view((self.feat_sz * self.feat_sz,)).float().cuda()
            self.coord_y = self.indice.repeat((1, self.feat_sz)) \
                .view((self.feat_sz * self.feat_sz,)).float().cuda()

    def forward(self, x, return_dist=False, softmax=True):
        """ Forward pass with input x. """
        score_map_tl, score_map_br = self.get_score_map(x)
        if return_dist:
            coorx_tl, coory_tl, prob_vec_tl = self.soft_argmax(score_map_tl, return_dist=True, softmax=softmax)
            coorx_br, coory_br, prob_vec_br = self.soft_argmax(score_map_br, return_dist=True, softmax=softmax)
            return torch.stack((coorx_tl, coory_tl, coorx_br, coory_br), dim=1) / self.img_sz, prob_vec_tl, prob_vec_br
        else:
            coorx_tl, coory_tl = self.soft_argmax(score_map_tl)
            coorx_br, coory_br = self.soft_argmax(score_map_br)
            return torch.stack((coorx_tl, coory_tl, coorx_br, coory_br), dim=1) / self.img_sz

    def get_score_map(self, x):
        # top-left branch
        x_tl1 = self.conv1_tl(x)
        x_tl2 = self.conv2_tl(x_tl1)
        x_tl3 = self.conv3_tl(x_tl2)
        x_tl4 = self.conv4_tl(x_tl3)
        score_map_tl = self.conv5_tl(x_tl4)

        # bottom-right branch
        x_br1 = self.conv1_br(x)
        x_br2 = self.conv2_br(x_br1)
        x_br3 = self.conv3_br(x_br2)
        x_br4 = self.conv4_br(x_br3)
        score_map_br = self.conv5_br(x_br4)
        return score_map_tl, score_map_br

    def soft_argmax(self, score_map, return_dist=False, softmax=True):
        """ get soft-argmax coordinate for a given heatmap """
        score_vec = score_map.view((-1, self.feat_sz * self.feat_sz))  # (batch, feat_sz * feat_sz)
        prob_vec = nn.functional.softmax(score_vec, dim=1)
        exp_x = torch.sum((self.coord_x * prob_vec), dim=1)
        exp_y = torch.sum((self.coord_y * prob_vec), dim=1)
        if return_dist:
            if softmax:
                return exp_x, exp_y, prob_vec
            else:
                return exp_x, exp_y, score_vec
        else:
            return exp_x, exp_y


class CenterPredictor(nn.Module, ):
    def __init__(self, inplanes=64, channel=256, feat_sz=20, stride=16, freeze_bn=False):
        super(CenterPredictor, self).__init__()
        self.feat_sz = feat_sz
        self.stride = stride
        self.img_sz = self.feat_sz * self.stride

        # corner predict
        self.conv1_ctr = conv(inplanes, channel, freeze_bn=freeze_bn)
        self.conv2_ctr = conv(channel, channel // 2, freeze_bn=freeze_bn)
        self.conv3_ctr = conv(channel // 2, channel // 4, freeze_bn=freeze_bn)
        self.conv4_ctr = conv(channel // 4, channel // 8, freeze_bn=freeze_bn)
        self.conv5_ctr = nn.Conv2d(channel // 8, 1, kernel_size=1)

        # size regress
        self.conv1_offset = conv(inplanes, channel, freeze_bn=freeze_bn)
        self.conv2_offset = conv(channel, channel // 2, freeze_bn=freeze_bn)
        self.conv3_offset = conv(channel // 2, channel // 4, freeze_bn=freeze_bn)
        self.conv4_offset = conv(channel // 4, channel // 8, freeze_bn=freeze_bn)
        self.conv5_offset = nn.Conv2d(channel // 8, 2, kernel_size=1)

        # size regress
        self.conv1_size = conv(inplanes, channel, freeze_bn=freeze_bn)
        self.conv2_size = conv(channel, channel // 2, freeze_bn=freeze_bn)
        self.conv3_size = conv(channel // 2, channel // 4, freeze_bn=freeze_bn)
        self.conv4_size = conv(channel // 4, channel // 8, freeze_bn=freeze_bn)
        self.conv5_size = nn.Conv2d(channel // 8, 2, kernel_size=1)

        for p in self.parameters():
            if p.dim() > 1:
                nn.init.xavier_uniform_(p)

    def forward(self, x, gt_score_map=None):
        """ Forward pass with input x. """
        score_map_ctr, size_map, offset_map = self.get_score_map(x)

        # assert gt_score_map is None
        if gt_score_map is None:
            bbox = self.cal_bbox(score_map_ctr, size_map, offset_map)
        else:
            bbox = self.cal_bbox(gt_score_map.unsqueeze(1), size_map, offset_map)

        return score_map_ctr, bbox, size_map, offset_map

    def cal_bbox(self, score_map_ctr, size_map, offset_map, return_score=False):
        max_score, idx = torch.max(score_map_ctr.flatten(1), dim=1, keepdim=True)
        idx_y = idx // self.feat_sz
        idx_x = idx % self.feat_sz

        idx = idx.unsqueeze(1).expand(idx.shape[0], 2, 1)
        size = size_map.flatten(2).gather(dim=2, index=idx)
        offset = offset_map.flatten(2).gather(dim=2, index=idx).squeeze(-1)

        # bbox = torch.cat([idx_x - size[:, 0] / 2, idx_y - size[:, 1] / 2,
        #                   idx_x + size[:, 0] / 2, idx_y + size[:, 1] / 2], dim=1) / self.feat_sz
        # cx, cy, w, h
        bbox = torch.cat([(idx_x.to(torch.float) + offset[:, :1]) / self.feat_sz,
                          (idx_y.to(torch.float) + offset[:, 1:]) / self.feat_sz,
                          size.squeeze(-1)], dim=1)

        if return_score:
            return bbox, max_score
        return bbox

    def get_pred(self, score_map_ctr, size_map, offset_map):
        max_score, idx = torch.max(score_map_ctr.flatten(1), dim=1, keepdim=True)
        idx_y = idx // self.feat_sz
        idx_x = idx % self.feat_sz

        idx = idx.unsqueeze(1).expand(idx.shape[0], 2, 1)
        size = size_map.flatten(2).gather(dim=2, index=idx)
        offset = offset_map.flatten(2).gather(dim=2, index=idx).squeeze(-1)

        # bbox = torch.cat([idx_x - size[:, 0] / 2, idx_y - size[:, 1] / 2,
        #                   idx_x + size[:, 0] / 2, idx_y + size[:, 1] / 2], dim=1) / self.feat_sz
        return size * self.feat_sz, offset

    def get_score_map(self, x):

        def _sigmoid(x):
            y = torch.clamp(x.sigmoid_(), min=1e-4, max=1 - 1e-4)
            return y

        # ctr branch
        x_ctr1 = self.conv1_ctr(x)
        x_ctr2 = self.conv2_ctr(x_ctr1)
        x_ctr3 = self.conv3_ctr(x_ctr2)
        x_ctr4 = self.conv4_ctr(x_ctr3)
        score_map_ctr = self.conv5_ctr(x_ctr4)

        # offset branch
        x_offset1 = self.conv1_offset(x)
        x_offset2 = self.conv2_offset(x_offset1)
        x_offset3 = self.conv3_offset(x_offset2)
        x_offset4 = self.conv4_offset(x_offset3)
        score_map_offset = self.conv5_offset(x_offset4)

        # size branch
        x_size1 = self.conv1_size(x)
        x_size2 = self.conv2_size(x_size1)
        x_size3 = self.conv3_size(x_size2)
        x_size4 = self.conv4_size(x_size3)
        score_map_size = self.conv5_size(x_size4)
        return _sigmoid(score_map_ctr), _sigmoid(score_map_size), score_map_offset


class MLP(nn.Module):
    """ Very simple multi-layer perceptron (also called FFN)"""

    def __init__(self, input_dim, hidden_dim, output_dim, num_layers, BN=False):
        super().__init__()
        self.num_layers = num_layers
        h = [hidden_dim] * (num_layers - 1)
        if BN:
            self.layers = nn.ModuleList(nn.Sequential(nn.Linear(n, k), nn.BatchNorm1d(k))
                                        for n, k in zip([input_dim] + h, h + [output_dim]))
        else:
            self.layers = nn.ModuleList(nn.Linear(n, k)
                                        for n, k in zip([input_dim] + h, h + [output_dim]))

    def forward(self, x):
        for i, layer in enumerate(self.layers):
            x = F.relu(layer(x)) if i < self.num_layers - 1 else layer(x)
        return x


class SelfAttention(nn.Module):
    def __init__(self, dim, num_heads=8, qkv_bias=False, qk_scale=None, attn_drop=0., proj_drop=0.,
                 attn_pos_encoding_only=False):
        super(SelfAttention, self).__init__()
        assert dim % num_heads == 0, f"dim {dim} should be divided by num_heads {num_heads}."

        self.dim = dim
        self.num_heads = num_heads
        head_dim = dim // num_heads
        self.scale = qk_scale or head_dim ** -0.5

        if attn_pos_encoding_only:
            self.qkv = nn.Linear(dim, 3 * dim, bias=qkv_bias)
        else:
            self.q = nn.Linear(dim, dim, bias=qkv_bias)
            self.k = nn.Linear(dim, dim, bias=qkv_bias)
            self.v = nn.Linear(dim, dim, bias=qkv_bias)
        self.attn_drop = nn.Dropout(attn_drop)
        self.proj = nn.Linear(dim, dim)
        self.proj_drop = nn.Dropout(proj_drop)

        self.attn_pos_encoding_only = attn_pos_encoding_only

    def forward(self, x, q_ape, k_ape, attn_pos):
        '''
            Args:
                x (torch.Tensor): (B, L, C)
                q_ape (torch.Tensor | None): (1 or B, L, C), absolute positional encoding for q
                k_ape (torch.Tensor | None): (1 or B, L, C), absolute positional encoding for k
                attn_pos (torch.Tensor | None): (1 or B, num_heads, L, L), untied positional encoding
            Returns:
                torch.Tensor: (B, L, C)
        '''
        B, N, C = x.shape

        if self.attn_pos_encoding_only:
            assert q_ape is None and k_ape is None
            qkv = self.qkv(x).reshape(B, N, 3, self.num_heads, C // self.num_heads).permute(2, 0, 3, 1, 4)
            q, k, v = qkv[0], qkv[1], qkv[2]
        else:
            q = x + q_ape if q_ape is not None else x
            q = self.q(q).reshape(B, N, self.num_heads, C // self.num_heads).permute(0, 2, 1, 3)

            k = x + k_ape if k_ape is not None else x
            k = self.k(k).reshape(B, -1, self.num_heads, C // self.num_heads).permute(0, 2, 1, 3)
            v = self.v(x).reshape(B, -1, self.num_heads, C // self.num_heads).permute(0, 2, 1, 3)

        attn = q @ k.transpose(-2, -1)
        attn = attn * self.scale
        if attn_pos is not None:
            attn = attn + attn_pos
        attn = attn.softmax(dim=-1)
        attn = self.attn_drop(attn)

        x = attn @ v
        x = x.transpose(1, 2).reshape(B, N, C)
        x = self.proj(x)
        x = self.proj_drop(x)

        return x


class CrossAttention(nn.Module):
    def __init__(self, dim, num_heads=8, qkv_bias=False, qk_scale=None, attn_drop=0., proj_drop=0.,
                 attn_pos_encoding_only=False):
        super(CrossAttention, self).__init__()
        assert dim % num_heads == 0, f"dim {dim} should be divided by num_heads {num_heads}."

        self.dim = dim
        self.num_heads = num_heads
        head_dim = dim // num_heads
        self.scale = qk_scale or head_dim ** -0.5

        if attn_pos_encoding_only:
            self.q = nn.Linear(dim, dim, bias=qkv_bias)
            self.kv = nn.Linear(dim, 2 * dim, bias=qkv_bias)
        else:
            self.q = nn.Linear(dim, dim, bias=qkv_bias)
            self.k = nn.Linear(dim, dim, bias=qkv_bias)
            self.v = nn.Linear(dim, dim, bias=qkv_bias)
        self.attn_drop = nn.Dropout(attn_drop)
        self.proj = nn.Linear(dim, dim)
        self.proj_drop = nn.Dropout(proj_drop)

        self.attn_pos_encoding_only = attn_pos_encoding_only

    def forward(self, q, kv, q_ape, k_ape, attn_pos):
        '''
            Args:
                q (torch.Tensor): (B, L_q, C)
                kv (torch.Tensor): (B, L_kv, C)
                q_ape (torch.Tensor | None): (1 or B, L_q, C), absolute positional encoding for q
                k_ape (torch.Tensor | None): (1 or B, L_kv, C), absolute positional encoding for k
                attn_pos (torch.Tensor | None): (1 or B, num_heads, L_q, L_kv), untied positional encoding
            Returns:
                torch.Tensor: (B, L_q, C)
        '''
        B, q_N, C = q.shape
        kv_N = kv.shape[1]

        if self.attn_pos_encoding_only:
            assert q_ape is None and k_ape is None
            q = self.q(q).reshape(B, q_N, self.num_heads, C // self.num_heads).permute(0, 2, 1, 3)
            kv = self.kv(kv).reshape(B, kv_N, 2, self.num_heads, C // self.num_heads).permute(2, 0, 3, 1, 4)
            k, v = kv[0], kv[1]
        else:
            q = q + q_ape if q_ape is not None else q
            q = self.q(q).reshape(B, q_N, self.num_heads, C // self.num_heads).permute(0, 2, 1, 3)
            k = kv + k_ape if k_ape is not None else kv
            k = self.k(k).reshape(B, -1, self.num_heads, C // self.num_heads).permute(0, 2, 1, 3)
            v = self.v(kv).reshape(B, -1, self.num_heads, C // self.num_heads).permute(0, 2, 1, 3)

        attn = q @ k.transpose(-2, -1)
        attn = attn * self.scale
        if attn_pos is not None:
            attn = attn + attn_pos
        attn = attn.softmax(dim=-1)
        attn = self.attn_drop(attn)
        x = attn @ v
        x = x.transpose(1, 2).reshape(B, q_N, C)
        x = self.proj(x)
        x = self.proj_drop(x)

        return x


class Mlp(nn.Module):
    """ Multilayer perceptron."""

    def __init__(self, in_features, hidden_features=None, out_features=None, act_layer=nn.GELU, drop=0.):
        super().__init__()
        out_features = out_features or in_features
        hidden_features = hidden_features or in_features
        self.fc1 = nn.Linear(in_features, hidden_features)
        self.act = act_layer()
        self.fc2 = nn.Linear(hidden_features, out_features)
        self.drop = nn.Dropout(drop)

    def forward(self, x):
        '''
            Args:
                x (torch.Tensor): (B, L, C), input tensor
            Returns:
                torch.Tensor: (B, L, C), output tensor
        '''
        x = self.fc1(x)
        x = self.act(x)
        x = self.drop(x)
        x = self.fc2(x)
        x = self.drop(x)
        return x


class FeatureFusion(nn.Module):
    def __init__(self,
                 dim, num_heads, mlp_ratio=2., qkv_bias=False, qk_scale=None, drop=0., attn_drop=0.,
                 drop_path=nn.Identity(), act_layer=nn.GELU, norm_layer=nn.LayerNorm, attn_pos_encoding_only=False):
        super(FeatureFusion, self).__init__()
        self.z_norm1 = norm_layer(dim)
        self.x_norm1 = norm_layer(dim)
        self.z_self_attn = SelfAttention(dim, num_heads, qkv_bias, qk_scale, attn_drop, drop, attn_pos_encoding_only)
        self.x_self_attn = SelfAttention(dim, num_heads, qkv_bias, qk_scale, attn_drop, drop, attn_pos_encoding_only)

        self.z_norm2_1 = norm_layer(dim)
        self.z_norm2_2 = norm_layer(dim)
        self.x_norm2_1 = norm_layer(dim)
        self.x_norm2_2 = norm_layer(dim)

        self.z_x_cross_attention = CrossAttention(dim, num_heads, qkv_bias, qk_scale, attn_drop, drop,
                                                  attn_pos_encoding_only)
        self.x_z_cross_attention = CrossAttention(dim, num_heads, qkv_bias, qk_scale, attn_drop, drop,
                                                  attn_pos_encoding_only)

        mlp_hidden_dim = int(dim * mlp_ratio)
        self.z_norm3 = norm_layer(dim)
        self.x_norm3 = norm_layer(dim)
        print(mlp_ratio)
        self.z_mlp = Mlp(in_features=dim, hidden_features=mlp_hidden_dim, act_layer=act_layer, drop=drop)
        self.x_mlp = Mlp(in_features=dim, hidden_features=mlp_hidden_dim, act_layer=act_layer, drop=drop)

        self.drop_path = drop_path

    def forward(self, z, x, z_self_attn_pos, x_self_attn_pos, z_x_cross_attn_pos, x_z_cross_attn_pos):
        z = z + self.drop_path(self.z_self_attn(self.z_norm1(z), None, None, z_self_attn_pos))
        x = x + self.drop_path(self.x_self_attn(self.x_norm1(x), None, None, x_self_attn_pos))

        z = z + self.drop_path(
            self.z_x_cross_attention(self.z_norm2_1(z), self.x_norm2_1(x), None, None, z_x_cross_attn_pos))
        x = x + self.drop_path(
            self.x_z_cross_attention(self.x_norm2_2(x), self.z_norm2_2(z), None, None, x_z_cross_attn_pos))

        z = z + self.drop_path(self.z_mlp(self.z_norm3(z)))
        x = x + self.drop_path(self.x_mlp(self.x_norm3(x)))
        return z, x


class FeatureFusionEncoder(nn.Module):
    def __init__(self, feature_fusion_layers, z_pos_enc, x_pos_enc,
                 z_rel_pos_index, x_rel_pos_index, z_x_rel_pos_index, x_z_rel_pos_index,
                 z_rel_pos_bias_table, x_rel_pos_bias_table, z_x_rel_pos_bias_table, x_z_rel_pos_bias_table):
        super(FeatureFusionEncoder, self).__init__()
        self.layers = nn.ModuleList(feature_fusion_layers)
        self.z_pos_enc = z_pos_enc
        self.x_pos_enc = x_pos_enc
        self.register_buffer('z_rel_pos_index', z_rel_pos_index, False)
        self.register_buffer('x_rel_pos_index', x_rel_pos_index, False)
        self.register_buffer('z_x_rel_pos_index', z_x_rel_pos_index, False)
        self.register_buffer('x_z_rel_pos_index', x_z_rel_pos_index, False)
        self.z_rel_pos_bias_table = z_rel_pos_bias_table
        self.x_rel_pos_bias_table = x_rel_pos_bias_table
        self.z_x_rel_pos_bias_table = z_x_rel_pos_bias_table
        self.x_z_rel_pos_bias_table = x_z_rel_pos_bias_table

    def forward(self, z, x, z_pos, x_pos):
        '''
            Args:
                z (torch.Tensor): (B, L_z, C), template image feature tokens
                x (torch.Tensor): (B, L_x, C), search image feature tokens
                z_pos (torch.Tensor | None): (1 or B, L_z, C), optional positional encoding for z
                x_pos (torch.Tensor | None): (1 or B, L_x, C), optional positional encoding for x
            Returns:
                Tuple[torch.Tensor, torch.Tensor]:
                    (B, L_z, C): template image feature tokens
                    (B, L_x, C): search image feature tokens
        '''
        # Support untied positional encoding only for simplicity
        assert z_pos is None and x_pos is None

        # untied positional encoding
        z_q_pos, z_k_pos = self.z_pos_enc()
        x_q_pos, x_k_pos = self.x_pos_enc()
        z_self_attn_pos = (z_q_pos @ z_k_pos.transpose(-2, -1)).unsqueeze(0)
        x_self_attn_pos = (x_q_pos @ x_k_pos.transpose(-2, -1)).unsqueeze(0)

        z_x_cross_attn_pos = (z_q_pos @ x_k_pos.transpose(-2, -1)).unsqueeze(0)
        x_z_cross_attn_pos = (x_q_pos @ z_k_pos.transpose(-2, -1)).unsqueeze(0)

        # relative positional encoding
        z_self_attn_pos = z_self_attn_pos + self.z_rel_pos_bias_table(self.z_rel_pos_index)
        x_self_attn_pos = x_self_attn_pos + self.x_rel_pos_bias_table(self.x_rel_pos_index)
        z_x_cross_attn_pos = z_x_cross_attn_pos + self.z_x_rel_pos_bias_table(self.z_x_rel_pos_index)
        x_z_cross_attn_pos = x_z_cross_attn_pos + self.x_z_rel_pos_bias_table(self.x_z_rel_pos_index)

        for layer in self.layers:
            z, x = layer(z, x, z_self_attn_pos, x_self_attn_pos, z_x_cross_attn_pos, x_z_cross_attn_pos)

        return z, x


class Learned2DPositionalEncoder(nn.Module):
    def __init__(self, dim, w, h):
        super(Learned2DPositionalEncoder, self).__init__()
        self.w_pos = nn.Parameter(torch.empty(w, dim))
        self.h_pos = nn.Parameter(torch.empty(h, dim))
        trunc_normal_(self.w_pos, std=0.02)
        trunc_normal_(self.h_pos, std=0.02)

    def forward(self):
        w = self.w_pos.shape[0]
        h = self.h_pos.shape[0]
        return (self.w_pos[None, :, :] + self.h_pos[:, None, :]).view(h * w, -1)


class Untied2DPositionalEncoder(nn.Module):
    def __init__(self, dim, num_heads, w, h, scale=None, with_q=True, with_k=True):
        super(Untied2DPositionalEncoder, self).__init__()
        assert dim % num_heads == 0, f"dim {dim} should be divided by num_heads {num_heads}."
        self.pos = Learned2DPositionalEncoder(dim, w, h)
        self.norm = nn.LayerNorm(dim)
        self.pos_q_linear = None
        self.pos_k_linear = None
        if with_q:
            self.pos_q_linear = nn.Linear(dim, dim)
        if with_k:
            self.pos_k_linear = nn.Linear(dim, dim)

        self.num_heads = num_heads
        head_dim = dim // num_heads
        self.scale = scale or head_dim ** -0.5

    def forward(self):
        pos = self.norm(self.pos())
        seq_len = pos.shape[0]
        if self.pos_q_linear is not None and self.pos_k_linear is not None:
            pos_q = self.pos_q_linear(pos).view(seq_len, self.num_heads, -1).transpose(0, 1) * self.scale
            pos_k = self.pos_k_linear(pos).view(seq_len, self.num_heads, -1).transpose(0, 1)
            return pos_q, pos_k
        elif self.pos_q_linear is not None:
            pos_q = self.pos_q_linear(pos).view(seq_len, self.num_heads, -1).transpose(0, 1) * self.scale
            return pos_q
        elif self.pos_k_linear is not None:
            pos_k = self.pos_k_linear(pos).view(seq_len, self.num_heads, -1).transpose(0, 1)
            return pos_k
        else:
            raise RuntimeError


def generate_2d_relative_positional_encoding_index(z_shape, x_shape):
    '''
        z_shape: (z_h, z_w)
        x_shape: (x_h, x_w)
    '''
    z_2d_index_h, z_2d_index_w = torch.meshgrid(torch.arange(z_shape[0]), torch.arange(z_shape[1]))
    x_2d_index_h, x_2d_index_w = torch.meshgrid(torch.arange(x_shape[0]), torch.arange(x_shape[1]))

    z_2d_index_h = z_2d_index_h.flatten(0)
    z_2d_index_w = z_2d_index_w.flatten(0)
    x_2d_index_h = x_2d_index_h.flatten(0)
    x_2d_index_w = x_2d_index_w.flatten(0)

    diff_h = z_2d_index_h[:, None] - x_2d_index_h[None, :]
    diff_w = z_2d_index_w[:, None] - x_2d_index_w[None, :]

    diff = torch.stack((diff_h, diff_w), dim=-1)
    _, indices = torch.unique(diff.view(-1, 2), return_inverse=True, dim=0)
    return indices.view(z_shape[0] * z_shape[1], x_shape[0] * x_shape[1])


class RelativePosition2DEncoder(nn.Module):
    def __init__(self, num_heads, embed_size):
        super(RelativePosition2DEncoder, self).__init__()
        self.relative_position_bias_table = nn.Parameter(torch.empty((num_heads, embed_size)))
        trunc_normal_(self.relative_position_bias_table, std=0.02)

    def forward(self, attn_rpe_index):
        '''
            Args:
                attn_rpe_index (torch.Tensor): (*), any shape containing indices, max(attn_rpe_index) < embed_size
            Returns:
                torch.Tensor: (1, num_heads, *)
        '''
        return self.relative_position_bias_table[:, attn_rpe_index].unsqueeze(0)


class DropPathAllocator:
    def __init__(self, max_drop_path_rate, stochastic_depth_decay=True):
        self.max_drop_path_rate = max_drop_path_rate
        self.stochastic_depth_decay = stochastic_depth_decay
        self.allocated = []
        self.allocating = []

    def __enter__(self):
        self.allocating = []

    def __exit__(self, exc_type, exc_val, exc_tb):
        if len(self.allocating) != 0:
            self.allocated.append(self.allocating)
        self.allocating = None
        if not self.stochastic_depth_decay:
            for depth_module in self.allocated:
                for module in depth_module:
                    if isinstance(module, DropPath):
                        module.drop_prob = self.max_drop_path_rate
        else:
            depth = self.get_depth()
            dpr = [x.item() for x in torch.linspace(0, self.max_drop_path_rate, depth)]
            assert len(dpr) == len(self.allocated)
            for drop_path_rate, depth_modules in zip(dpr, self.allocated):
                for module in depth_modules:
                    if isinstance(module, DropPath):
                        module.drop_prob = drop_path_rate

    def __len__(self):
        length = 0

        for depth_modules in self.allocated:
            length += len(depth_modules)

        return length

    def increase_depth(self):
        self.allocated.append(self.allocating)
        self.allocating = []

    def get_depth(self):
        return len(self.allocated)

    def allocate(self):
        if self.max_drop_path_rate == 0 or (self.stochastic_depth_decay and self.get_depth() == 0):
            drop_path_module = Identity()
        else:
            drop_path_module = DropPath()
        self.allocating.append(drop_path_module)
        return drop_path_module

    def get_all_allocated(self):
        allocated = []
        for depth_module in self.allocated:
            for module in depth_module:
                allocated.append(module)
        return allocated


def build_encoder(encoder_layer, num_heads, mlp_ratio, qkv_bias, drop_rate, attn_drop, dim, z_size, x_size, drop_path):
    z_shape = [z_size, z_size]
    x_shape = [x_size, x_size]
    encoder_layers = []
    for i in range(encoder_layer):
        encoder_layers.append(
            FeatureFusion(dim, num_heads, mlp_ratio, qkv_bias, drop=drop_rate, attn_drop=attn_drop,
                          drop_path=drop_path.allocate(),
                          attn_pos_encoding_only=True)
        )
    z_abs_encoder = Untied2DPositionalEncoder(dim, num_heads, z_shape[0], z_shape[1])
    x_abs_encoder = Untied2DPositionalEncoder(dim, num_heads, x_shape[0], x_shape[1])

    z_self_attn_rel_pos_index = generate_2d_relative_positional_encoding_index(z_shape, z_shape)
    x_self_attn_rel_pos_index = generate_2d_relative_positional_encoding_index(x_shape, x_shape)

    z_x_cross_attn_rel_pos_index = generate_2d_relative_positional_encoding_index(z_shape, x_shape)
    x_z_cross_attn_rel_pos_index = generate_2d_relative_positional_encoding_index(x_shape, z_shape)

    z_self_attn_rel_pos_bias_table = RelativePosition2DEncoder(num_heads, z_self_attn_rel_pos_index.max() + 1)
    x_self_attn_rel_pos_bias_table = RelativePosition2DEncoder(num_heads, x_self_attn_rel_pos_index.max() + 1)
    z_x_cross_attn_rel_pos_bias_table = RelativePosition2DEncoder(num_heads, z_x_cross_attn_rel_pos_index.max() + 1)
    x_z_cross_attn_rel_pos_bias_table = RelativePosition2DEncoder(num_heads, x_z_cross_attn_rel_pos_index.max() + 1)

    return FeatureFusionEncoder(encoder_layers, z_abs_encoder, x_abs_encoder, z_self_attn_rel_pos_index,
                                x_self_attn_rel_pos_index,
                                z_x_cross_attn_rel_pos_index, x_z_cross_attn_rel_pos_index,
                                z_self_attn_rel_pos_bias_table,
                                x_self_attn_rel_pos_bias_table, z_x_cross_attn_rel_pos_bias_table,
                                x_z_cross_attn_rel_pos_bias_table)


class TargetQueryDecoderLayer(nn.Module):
    def __init__(self, dim, num_heads, mlp_ratio=4., qkv_bias=False, qk_scale=None, drop=0., attn_drop=0.,
                 drop_path=nn.Identity(), act_layer=nn.GELU, norm_layer=nn.LayerNorm):
        super(TargetQueryDecoderLayer, self).__init__()
        self.norm_1 = norm_layer(dim)
        self.self_attn1 = nn.MultiheadAttention(dim, num_heads, dropout=drop)
        self.norm_2_query = norm_layer(dim)
        self.norm_2_memory = norm_layer(dim)
        self.multihead_attn = nn.MultiheadAttention(dim, num_heads, dropout=drop)
        self.norm_3 = norm_layer(dim)
        mlp_hidden_dim = int(dim * mlp_ratio)
        self.mlpz = Mlp(dim, hidden_features=mlp_hidden_dim, act_layer=act_layer, drop=drop)

        self.drop_path = drop_path

    def forward(self, query, memoryz, memoryx, query_pos, pos_z, pos_x, identity, identity_search,
                tgt_mask: Optional[Tensor] = None,
                memory_mask: Optional[Tensor] = None,
                tgt_key_padding_mask: Optional[Tensor] = None,
                memory_key_padding_mask: Optional[Tensor] = None,
                ):
        '''
            Args:
                query (torch.Tensor): (B, num_queries, C)
                memory (torch.Tensor): (B, L, C)
                query_pos (torch.Tensor): (1 or B, num_queries, C)
                memory_pos (torch.Tensor): (1 or B, L, C)
            Returns:
                torch.Tensor: (B, num_queries, C)
        '''

        tgt = query
        q = k = self.norm_1(query) + query_pos
        query = query + self.drop_path(self.self_attn1(q, k, value=tgt, attn_mask=tgt_mask,
                                                       key_padding_mask=tgt_key_padding_mask)[0])
        q2 = self.norm_2_query(query) + query_pos
        memory = torch.cat((memoryz, memoryx), dim=1)

        pos = torch.cat((pos_z, pos_x), dim=1)

        ide = torch.cat(
            (identity[:, 0, :].repeat(1, pos_z.shape[1], 1), identity[:, 1, :].repeat(1, pos_x.shape[1], 1)), dim=1)

        k2 = (self.norm_2_memory(memory) + pos + ide).permute(1, 0, 2)
        memory_in = memory.permute(1, 0, 2)
        query = query + self.drop_path(
            self.multihead_attn(query=q2, key=k2, value=memory_in, attn_mask=memory_mask,
                                key_padding_mask=memory_key_padding_mask)[0])
        query = query + self.drop_path(self.mlpz(self.norm_3(query)))

        return query


def _get_clones(module, N):
    return nn.ModuleList([copy.deepcopy(module) for i in range(N)])


class TargetQueryDecoderBlock(nn.Module):
    def __init__(self, dim, decoder_layers, num_layer):
        super(TargetQueryDecoderBlock, self).__init__()
        self.layers = nn.ModuleList(decoder_layers)
        self.num_layers = num_layer
        self.norm = nn.LayerNorm(dim)

    def forward(self, tgt, z, x, pos_z, pos_x, identity, identity_search, query_pos: Optional[Tensor] = None,
                tgt_mask: Optional[Tensor] = None,
                memory_mask: Optional[Tensor] = None,
                tgt_key_padding_mask: Optional[Tensor] = None,
                memory_key_padding_mask: Optional[Tensor] = None):
        '''
            Args:
                z (torch.Tensor): (B, L_z, C)
                x (torch.Tensor): (B, L_x, C)
            Returns:
                torch.Tensor: (B, num_queries, C)
        '''
        output = tgt
        for layer in self.layers:
            output = layer(output, z, x, query_pos, pos_z, pos_x, identity, identity_search,
                           tgt_mask=tgt_mask,
                           memory_mask=memory_mask,
                           tgt_key_padding_mask=tgt_key_padding_mask,
                           memory_key_padding_mask=memory_key_padding_mask)
        output = self.norm(output)

        return output


def build_decoder(decoder_layer, drop_path, dim, num_heads, mlp_ratio, qkv_bias, drop_rate, attn_drop_rate, z_size,
                  x_size):
    z_shape = [z_size, z_size]
    x_shape = [x_size, x_size]
    num_layers = decoder_layer
    decoder_layers = []
    for _ in range(num_layers):
        decoder_layers.append(
            TargetQueryDecoderLayer(dim, num_heads, mlp_ratio, qkv_bias, drop=drop_rate, attn_drop=attn_drop_rate,
                                    drop_path=drop_path.allocate()))
        drop_path.increase_depth()

    decoder = TargetQueryDecoderBlock(dim, decoder_layers, num_layers)
    return decoder


def generate_square_subsequent_mask(sz):
    r"""Generate a square mask for the sequence. The masked positions are filled with float('-inf').
        Unmasked positions are filled with float(0.0).
    """
    mask = (torch.triu(torch.ones(sz, sz)) == 1).transpose(0, 1)
    mask = mask.float().masked_fill(mask == 0, float(
        '-inf')).masked_fill(mask == 1, float(0.0))
    return mask


class Pix2Track(nn.Module):
    def __init__(self, in_channel=64, feat_sz=20, feat_tz=10, range=2, pre_num=7, stride=16, encoder_layer=3, decoder_layer=3,
                 bins=400, num_heads=12, mlp_ratio=2, qkv_bias=True, drop_rate=0.0, attn_drop=0.0,
                 drop_path=nn.Identity):
        super(Pix2Track, self).__init__()
        self.bins = bins
        self.range = range
        self.pre_num = pre_num
        self.word_embeddings = nn.Embedding(self.bins * self.range + 2, in_channel, padding_idx=self.bins * self.range, max_norm=1,
                                            norm_type=2.0)

        self.position_embeddings = nn.Embedding(
            5, in_channel)
        self.prev_position_embeddings = nn.Embedding(self.pre_num * 4, in_channel)
        self.output_bias = torch.nn.Parameter(torch.zeros(self.bins * self.range + 2))

        self.momentum_param = 0.25
        self.identity_search = torch.nn.Parameter(torch.zeros(1, 1, 768))
        self.identity_search = trunc_normal_(self.identity_search, std=.02)
        self.encoder_layer = encoder_layer
        self.drop_path = drop_path
        self.tz = feat_tz * feat_tz
        self.sz = feat_sz * feat_sz
        trunc_normal_(self.word_embeddings.weight, std=.02)
        if self.encoder_layer > 0:
            self.encoder = build_encoder(encoder_layer, num_heads, mlp_ratio, qkv_bias,
                                         drop_rate, attn_drop, in_channel, feat_tz, feat_sz, self.drop_path)
        else:
            self.encoder = None
        self.decoder = build_decoder(decoder_layer, self.drop_path, in_channel, num_heads,
                                     mlp_ratio, qkv_bias, drop_rate, attn_drop, feat_tz, feat_sz)
        self.magic_num = (self.range-1) * 0.5

    def forward(self, zx_feat, pos_z, pos_x, identity, seqs_input=None, stage=None):
        emb_weight = self.word_embeddings.weight.clone()
        share_weight = emb_weight.T

        z_feat = zx_feat[:, :self.tz]
        x_feat = zx_feat[:, self.tz:]

        out_list = []
        bs = zx_feat.shape[0]
        if self.encoder != None:
            z_feat, x_feat = self.encoder(z_feat, x_feat, None, None)
        output_x_feat = x_feat.clone()

        if stage == None:
            seqs_input = seqs_input.to(torch.int64).to(zx_feat.device)
            tgt = self.word_embeddings(seqs_input).permute(1, 0, 2)
            query_embed_ = self.position_embeddings.weight.unsqueeze(1)
            prev_embed = self.prev_position_embeddings.weight.unsqueeze(1)
            query_embed = torch.cat([prev_embed, query_embed_], dim=0)
            query_embed = query_embed.repeat(1, bs, 1)

            decoder_feat_cls = self.decoder(tgt, z_feat, x_feat, pos_z, pos_x, identity, self.identity_search,
                                            query_embed[:len(tgt)],
                                            tgt_mask=generate_square_subsequent_mask(len(tgt)).to(tgt.device))

            at = torch.matmul(decoder_feat_cls, share_weight)
            at = at + self.output_bias
            output = {'feat': at, "state": "train"}

        else:
            seqs_origin = seqs_input
            start_token = torch.ones(bs, 1) * self.bins * self.range
            start_token = start_token.to(seqs_origin)

            real_seq = torch.cat([seqs_origin, start_token], dim=1)

            seqs_input = real_seq.to(zx_feat.device).to(torch.int32)

            for i in range(4):
                tgt = self.word_embeddings(seqs_input).permute(1, 0, 2)
                query_embed_ = self.position_embeddings.weight.unsqueeze(1)
                prev_embed = self.prev_position_embeddings.weight.unsqueeze(1)
                query_embed = torch.cat([prev_embed, query_embed_], dim=0)
                query_embed = query_embed.repeat(1, bs, 1)

                decoder_feat_cls = self.decoder(tgt, z_feat, x_feat, pos_z, pos_x, identity, self.identity_search,
                                                query_embed[:len(tgt)],
                                                tgt_mask=generate_square_subsequent_mask(len(seqs_input[0])).to(
                                                    tgt.device))

                out = torch.matmul(decoder_feat_cls.transpose(0, 1)[:, -1, :],
                                   share_weight) + self.output_bias

                out_list.append(out.unsqueeze(0))
                out = out.softmax(-1)

                value, extra_seq = out.topk(dim=-1, k=1)[0], out.topk(dim=-1, k=1)[1]
                seqs_input = torch.cat([seqs_input, extra_seq], dim=-1)
                if i == 0:
                    seqs_output = extra_seq
                    values = value
                else:
                    seqs_output = torch.cat([seqs_output, extra_seq], dim=-1)
                    values = torch.cat([values, value], dim=-1)
            if not (not out_list):
                feat = torch.cat(out_list)
            output = {'seqs': seqs_output, 'class': values, 'feat': feat, "state": "val/test",
                      "x_feat": output_x_feat.detach()}
        return output


def build_pix_head(cfg, hidden_dim):
    stride = cfg.MODEL.BACKBONE.STRIDE
    in_channel = hidden_dim
    feat_sz = int(cfg.DATA.SEARCH.SIZE / stride)
    feat_tz = int(cfg.DATA.TEMPLATE.SIZE / stride)
    decoder_layer = cfg.MODEL.DECODER_LAYER
    encoder_layer = cfg.MODEL.ENCODER_LAYER
    pre_num = cfg.MODEL.PRENUM
    bins = cfg.MODEL.BINS
    range = cfg.MODEL.RANGE
    num_heads = cfg.MODEL.NUM_HEADS
    mlp_ratio = cfg.MODEL.MLP_RATIO
    qkv_bias = cfg.MODEL.QKV_BIAS
    drop_rate = cfg.MODEL.DROP_RATE
    attn_drop = cfg.MODEL.ATTN_DROP
    drop_path = cfg.MODEL.DROP_PATH
    drop_path_allocator = DropPathAllocator(drop_path)
    pix_head = Pix2Track(in_channel=in_channel, feat_sz=feat_sz, feat_tz=feat_tz, range=range, pre_num=pre_num,
                         stride=stride, encoder_layer=encoder_layer, decoder_layer=decoder_layer, bins=bins,
                         num_heads=num_heads, mlp_ratio=mlp_ratio, qkv_bias=qkv_bias, drop_rate=drop_rate,
                         attn_drop=attn_drop, drop_path=drop_path_allocator)
    return pix_head

