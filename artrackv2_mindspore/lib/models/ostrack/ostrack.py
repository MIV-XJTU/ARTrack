"""
Basic OSTrack model.
"""
import sys
from copy import deepcopy
import math
import os
from typing import List

import mindspore as ms
from mindspore import nn
from lib.models.timm import *

from lib.models.ostrack.vit import vit_base_patch16_224, vit_large_patch16_224
from lib.models.ostrack.vit_ce import vit_large_patch16_224_ce, vit_base_patch16_224_ce
from lib.models.layers.mask_decoder import build_maskdecoder
from lib.models.layers.head import build_decoder, MLP, DropPathAllocator


class OSTrack(nn.Cell):
    """ This is the base class for OSTrack """

    def __init__(self, transformer,
                 #decoder,
                 cross_2_decoder,
                 score_mlp,
                 #cover_mlp,
                 ):
        """ Initializes the model.
        Parameters:
            transformer: torch module of the transformer architecture.
            aux_loss: True if auxiliary decoding losses (loss at each decoder layer) are to be used.
        """
        super().__init__()
        self.backbone = transformer
        self.score_mlp = score_mlp
            
        self.identity = ms.Parameter(ops.zeros((1, 3, 768)))
        self.identity = trunc_normal_(self.identity, std=.02)

        self.cross_2_decoder = cross_2_decoder


    def construct(self, template: ms.Tensor,
                dz_feat: ms.Tensor,
                search: ms.Tensor,
                ce_template_mask=None,
                ce_keep_rate=None,
                return_last_attn=False,
                seq_input=None,
                head_type=None,
                stage=None,
                search_feature=None,
                target_in_search_img=None,
                gt_bboxes=None,
                ):
        template_0 = template[:, 0]
        out, z_0_feat, z_1_feat, x_feat, score_feat = self.backbone(z_0=template_0, z_1_feat=dz_feat, x=search, identity=self.identity, seqs_input=seq_input,
                                    ce_template_mask=ce_template_mask,
                                    ce_keep_rate=ce_keep_rate,
                                    return_last_attn=return_last_attn,)
        seq_feat = out['seq_feat'].permute(1, 0 ,2)
        share_weight = self.backbone.word_embeddings.embedding_table[:800, :].unsqueeze(0).tile((seq_feat.shape[1], 1, 1))
        pos = self.backbone.position_embeddings.embedding_table.unsqueeze(0).tile((seq_feat.shape[1], 1, 1)).permute(1, 0 ,2)
        score = self.score_mlp(score_feat)
        ops.clamp(score, min=0.0, max=1.0)
        out['score'] = score

        loss = ms.tensor(0.0, dtype=ms.float32)
        if target_in_search_img != None:
            target_in_search_gt = self.backbone.patch_embed(target_in_search_img)
            z_1_feat = z_1_feat.reshape(z_1_feat.shape[0], int(z_1_feat.shape[1] ** 0.5), int(z_1_feat.shape[1] ** 0.5),
                                        z_1_feat.shape[2]).permute(0, 3, 1, 2)
            target_in_search_gt = self.cross_2_decoder.unpatchify(target_in_search_gt)               

            update_img, loss_temp = self.cross_2_decoder(z_1_feat, target_in_search_gt)
            update_feat = self.cross_2_decoder.patchify(update_img)
            out['dz_feat'] = update_feat
            loss += loss_temp

            out['renew_loss'] = loss

        else:
            z_1_feat = z_1_feat.reshape(z_1_feat.shape[0], int(z_1_feat.shape[1] ** 0.5), int(z_1_feat.shape[1] ** 0.5),
                                        z_1_feat.shape[2]).permute(0, 3, 1, 2)
            update_feat = self.cross_2_decoder(z_1_feat, eval=True)
            update_feat = self.cross_2_decoder.patchify(update_feat)
            out['dz_feat'] = update_feat

        return out

    def forward_head(self, cat_feature, pos_z, pos_x, identity, seq_input=None, gt_score_map=None, head_type=None, stage=None, search_feature=None):
        """
        cat_feature: output embeddings of the backbone, it can be (HW1+HW2, B, C) or (HW2, B, C)
        """
        if self.head_type == "CORNER":
            # run the corner head
            pred_box, score_map = self.box_head(opt_feat, True)
            outputs_coord = box_xyxy_to_cxcywh(pred_box)
            outputs_coord_new = outputs_coord.view(bs, Nq, 4)
            out = {'pred_boxes': outputs_coord_new,
                   'score_map': score_map,
                   }
            return out

        elif self.head_type == "CENTER":
            # run the center head
            score_map_ctr, bbox, size_map, offset_map = self.box_head(opt_feat, gt_score_map)
            # outputs_coord = box_xyxy_to_cxcywh(bbox)
            outputs_coord = bbox
            outputs_coord_new = outputs_coord.view(bs, Nq, 4)
            out = {'pred_boxes': outputs_coord_new,
                   'score_map': score_map_ctr,
                   'size_map': size_map,
                   'offset_map': offset_map}
            return out
        elif self.head_type == "PIX":
            output_dict = self.box_head(cat_feature, pos_z, pos_x, identity, seq_input, head_type, stage, search_feature)
            return output_dict
        else:
            raise NotImplementedError

class MlpScoreDecoder(nn.Cell):
    def __init__(self, in_dim, hidden_dim, num_layers, bn=False):
        super().__init__()
        self.num_layers = num_layers
        h = [hidden_dim] * (num_layers - 1)
        out_dim = 1 # score
        if bn:
            self.layers = nn.SequentialCell(*[nn.SequentialCell(nn.Dense(n, k), nn.BatchNorm1d(k), nn.ReLU())
                                          if i < num_layers - 1
                                          else nn.SequentialCell(nn.Dense(n, k), nn.BatchNorm1d(k))
                                          for i, (n, k) in enumerate(zip([in_dim] + h, h + [out_dim]))])
        else:
            self.layers = nn.SequentialCell(*[nn.SequentialCell(nn.Dense(n, k), nn.ReLU())
                                          if i < num_layers - 1
                                          else nn.Dense(n, k)
                                          for i, (n, k) in enumerate(zip([in_dim] + h, h + [out_dim]))])

    def construct(self, reg_tokens):
        """
        reg tokens shape: (b, 4, embed_dim)
        """
        x = self.layers(reg_tokens) # (b, 4, 1)
        x = ops.mean(x,axis=1)   # (b, 1)
        return x

def build_score_decoder(cfg):
    return MlpScoreDecoder(
        in_dim=cfg.MODEL.BACKBONE.EMBEDDIM,
        hidden_dim=cfg.MODEL.BACKBONE.EMBEDDIM,
        num_layers=2,
        bn=False
    )

def build_ostrack(cfg, training=True):
    current_dir = os.path.dirname(os.path.abspath(__file__))  # This is your Project Root
    pretrained_path = "/home/baiyifan/code/vitrack/"
    if cfg.MODEL.PRETRAIN_FILE and ('OSTrack' not in cfg.MODEL.PRETRAIN_FILE) and training:
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

    elif cfg.MODEL.BACKBONE.TYPE == 'vit_base_patch16_224_ce':
        backbone = vit_base_patch16_224_ce(pretrained, drop_path_rate=cfg.TRAIN.DROP_PATH_RATE,
                                           ce_loc=cfg.MODEL.BACKBONE.CE_LOC,
                                           ce_keep_ratio=cfg.MODEL.BACKBONE.CE_KEEP_RATIO,
                                           )
        hidden_dim = backbone.embed_dim
        patch_start_index = 1

    elif cfg.MODEL.BACKBONE.TYPE == 'vit_large_patch16_224_ce':
        backbone = vit_large_patch16_224_ce(pretrained, drop_path_rate=cfg.TRAIN.DROP_PATH_RATE,
                                            ce_loc=cfg.MODEL.BACKBONE.CE_LOC,
                                            ce_keep_ratio=cfg.MODEL.BACKBONE.CE_KEEP_RATIO,
                                            )

        hidden_dim = backbone.embed_dim
        patch_start_index = 1

    else:
        raise NotImplementedError

    backbone.finetune_track(cfg=cfg, patch_start_index=patch_start_index)

    #decoder = build_maskdecoder(cfg)
    cross_2_decoder = build_maskdecoder(cfg)

    drop_path = cfg.MODEL.DROP_PATH
    drop_path_allocator = DropPathAllocator(drop_path)
    num_heads = cfg.MODEL.NUM_HEADS
    mlp_ratio = cfg.MODEL.MLP_RATIO
    qkv_bias = cfg.MODEL.QKV_BIAS
    drop_rate = cfg.MODEL.DROP_RATE
    attn_drop = cfg.MODEL.ATTN_DROP
    score_mlp = build_score_decoder(cfg)
    cover_mlp = build_score_decoder(cfg)

    model = OSTrack(
        backbone,
        #decoder,
        cross_2_decoder,
        score_mlp,
        #cover_mlp,
    )

    from mindspore.amp import auto_mixed_precision
    model = auto_mixed_precision(model, 'O0')
    load_from = cfg.MODEL.PRETRAIN_FILE
    param_dict = ms.load_checkpoint(load_from)
    param_not_load, _ = ms.load_param_into_net(model, param_dict)
    print("未加载权重：",param_not_load)
    print('Load pretrained model from: ' + load_from)
    model.backbone.pos_embed_z0 = model.backbone.pos_embed_z1

    return model
