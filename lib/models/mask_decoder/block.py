# -*- coding:utf-8 -*-
# author  : Skye Song
# file    : block.py
# Copyright (c) Skye-Song. All Rights Reserved

from .attention import *
from .drop import DropPath
from .mlp import Mlp

class Block(nn.Module):
    def __init__(self, dim, num_heads, mlp_ratio=4., qkv_bias=False, drop=0., attn_drop=0.,
                 drop_path=0., attention = "Attention", act_layer=nn.GELU, norm_layer=nn.LayerNorm):
        super().__init__()
        if norm_layer is None:
            norm_layer = nn.LayerNorm
        self.norm1 = norm_layer(dim)

        self.attn = globals()[attention](dim, num_heads=num_heads, qkv_bias=qkv_bias, attn_drop=attn_drop,
                                             proj_drop=drop)

        # NOTE: drop path for stochastic depth, we shall see if this is better than dropout here
        self.drop_path = DropPath(drop_path) if drop_path > 0. else nn.Identity()
        self.norm2 = norm_layer(dim)
        mlp_hidden_dim = int(dim * mlp_ratio)
        self.mlp = Mlp(in_features=dim, hidden_features=mlp_hidden_dim, act_layer=act_layer, drop=drop)

    def forward(self, x, **kwargs):
        x = x + self.drop_path(self.attn(self.norm1(x), **kwargs))
        x = x + self.drop_path(self.mlp(self.norm2(x)))
        return x