"""
Basic OSTrack model.
"""
import math
import os
from typing import List

import torch
from torch import nn
from torch.nn.modules.transformer import _get_clones
from timm.models.layers import DropPath, to_2tuple, trunc_normal_

from lib.models.layers.head_seq import build_pix_head
from lib.models.artrack_seq.vit import vit_base_patch16_224, vit_large_patch16_224
from lib.utils.box_ops import box_xyxy_to_cxcywh

import time


class ARTrackSeq(nn.Module):
    """ This is the base class for ARTrackSeq """

    def __init__(self, transformer, pix_head, hidden_dim):
        """ Initializes the model.
        Parameters:
            transformer: torch module of the transformer architecture.
            aux_loss: True if auxiliary decoding losses (loss at each decoder layer) are to be used.
        """
        super().__init__()
        self.backbone = transformer
        self.pix_head = pix_head
            
        self.identity = torch.nn.Parameter(torch.zeros(1, 2, hidden_dim))
        self.identity = trunc_normal_(self.identity, std=.02)

    def forward(self, template: torch.Tensor,
                search: torch.Tensor,
                seq_input=None,
                head_type=None,
                stage=None,
                search_feature=None,
                update=None
                ):
        x, aux_dict = self.backbone(z=template, x=search, identity=self.identity)

        # Forward head
        feat_last = x
        if isinstance(x, list):
            feat_last = x[-1]
            
        pos_z = self.backbone.pos_embed_z
        pos_x = self.backbone.pos_embed_x
        out = self.forward_head(feat_last, pos_z, pos_x, self.identity, seq_input, stage)

        out.update(aux_dict)
        out['backbone_feat'] = x
        return out

    def forward_head(self, cat_feature, pos_z, pos_x, identity, seq_input=None, stage=None):
        """
        cat_feature: output embeddings of the backbone, it can be (HW1+HW2, B, C) or (HW2, B, C)
        """

        output_dict = self.pix_head(cat_feature, pos_z, pos_x, identity, seq_input, stage)
        return output_dict



def build_artrack_seq(cfg, training=True):
    current_dir = os.path.dirname(os.path.abspath(__file__))  # This is your Project Root
    pretrained_path = os.path.join(current_dir, '../../../pretrained_models')
    if cfg.MODEL.PRETRAIN_FILE and ('ARTrack' not in cfg.MODEL.PRETRAIN_FILE) and training:
        pretrained = os.path.join(pretrained_path, cfg.MODEL.PRETRAIN_FILE)
    else:
        pretrained = ''

    if cfg.MODEL.BACKBONE.TYPE == 'vit_base_patch16_224':
        backbone = vit_base_patch16_224(pretrained, drop_path_rate=cfg.TRAIN.DROP_PATH_RATE)
        hidden_dim = backbone.embed_dim
        patch_start_index = 1

    elif cfg.MODEL.BACKBONE.TYPE == 'vit_large_patch16_224':
        print("i use vit_large")
        backbone = vit_large_patch16_224(pretrained, drop_path_rate=cfg.TRAIN.DROP_PATH_RATE)
        hidden_dim = backbone.embed_dim
        patch_start_index = 1

    else:
        raise NotImplementedError

    backbone.finetune_track(cfg=cfg, patch_start_index=patch_start_index)

    pix_head = build_pix_head(cfg, hidden_dim)

    model = ARTrackSeq(
        backbone,
        pix_head,
        hidden_dim,
    )
    load_from = cfg.MODEL.PRETRAIN_PTH
    checkpoint = torch.load(load_from, map_location="cpu")
    missing_keys, unexpected_keys = model.load_state_dict(checkpoint["net"], strict=False)
    print('Load pretrained model from: ' + load_from)
    if 'sequence' in cfg.MODEL.PRETRAIN_FILE and training:
        print("i change myself")
        checkpoint = torch.load(cfg.MODEL.PRETRAIN_FILE, map_location="cpu")
        missing_keys, unexpected_keys = model.load_state_dict(checkpoint["net"], strict=False)
        print('Load pretrained model from: ' + cfg.MODEL.PRETRAIN_FILE)

    return model
