from functools import partial

import mindspore as ms
import mindspore.nn as nn
import mindspore.ops as ops
import sys
sys.path.append("/home/baiyifan/code/AR2_mindspore_cp/2stage")
from lib.models.timm import *

from lib.models.layers.patch_embed import PatchEmbed
from lib.models.ostrack.utils import combine_tokens, recover_tokens


import time

def generate_square_subsequent_mask(sz, sx, ss):
    r"""Generate a square mask for the sequence. The masked positions are filled with float('-inf').
        Unmasked positions are filled with float(0.0).
    """
    # 0 means mask, 1 means visible
    sum = sz + sx + ss
    mask = (ops.triu(ops.ones((sum, sum))) == 1).swapaxes(0, 1)
    mask[:, :] = 0
    mask[:int(sz/2), :int(sz/2)] = 1 #template self
    mask[int(sz/2):sz, int(sz/2):sz] = 1 # dt self
    mask[int(sz/2):sz, sz:sz+sx] = 1 # dt search
    mask[int(sz / 2):sz, -1] = 1  # dt search
    mask[sz:sz+sx, :sz+sx] = 1 # sr dt-t-sr
    mask[sz+sx:, :] = 1 # co dt-t-sr-co
    # mask[sz+sx:, :sz] = 0
    return ~mask

class BaseBackbone(nn.Cell):
    def __init__(self):
        super().__init__()

        # for original ViT
        self.pos_embed = None
        self.img_size = [224, 224]
        self.patch_size = 16
        self.embed_dim = 384

        self.cat_mode = 'direct'

        self.pos_embed_z = None
        self.pos_embed_x = None

        self.bins = 400
        in_channel = 768
        self.range = 2
        self.word_embeddings = nn.Embedding(self.bins * self.range + 6, in_channel, padding_idx=self.bins * self.range+4)
        # mindspore的nn.Embedding中没有max_norm,norm_type，所以只能删去参数max_norm=1, norm_type=2.0
        print(self.bins)
        self.position_embeddings = nn.Embedding(
            5, in_channel)
        self.output_bias = ms.Parameter(ops.zeros(self.bins * self.range + 6))
        self.prev_position_embeddings = nn.Embedding(7 * 4, in_channel)

        self.template_segment_pos_embed = None
        self.search_segment_pos_embed = None

        self.return_inter = False
        self.return_stage = [2, 5, 8, 11]

        self.add_cls_token = False
        self.add_sep_seg = False

    def finetune_track(self, cfg, patch_start_index=1):

        search_size = to_2tuple(cfg.DATA.SEARCH.SIZE)
        template_size = to_2tuple(cfg.DATA.TEMPLATE.SIZE)
        new_patch_size = cfg.MODEL.BACKBONE.STRIDE

        self.cat_mode = cfg.MODEL.BACKBONE.CAT_MODE
        self.return_inter = cfg.MODEL.RETURN_INTER
        self.add_sep_seg = cfg.MODEL.BACKBONE.SEP_SEG

        # resize patch embedding
        if new_patch_size != self.patch_size:
            print('Inconsistent Patch Size With The Pretrained Weights, Interpolate The Weight!')
            old_patch_embed = {}
            for name, param in self.patch_embed.named_parameters():
                if 'weight' in name:
                    param = ops.interpolate(param, size=(new_patch_size, new_patch_size),
                                                      mode='bicubic', align_corners=False)
                    param = ms.Parameter(param)
                old_patch_embed[name] = param
            print("Attention:old_patch_embed:",old_patch_embed)
            self.patch_embed = PatchEmbed(img_size=self.img_size, patch_size=new_patch_size, in_chans=3,
                                          embed_dim=self.embed_dim)
            self.patch_embed.proj.bias = old_patch_embed['proj.bias']
            self.patch_embed.proj.weight = old_patch_embed['proj.weight']

        # for patch embedding
        patch_pos_embed = self.pos_embed[:, patch_start_index:, :]
        patch_pos_embed = patch_pos_embed.swapaxes(1, 2)
        B, E, Q = patch_pos_embed.shape
        P_H, P_W = self.img_size[0] // self.patch_size, self.img_size[1] // self.patch_size
        patch_pos_embed = patch_pos_embed.view(B, E, P_H, P_W)

        # for search region
        H, W = search_size
        new_P_H, new_P_W = H // new_patch_size, W // new_patch_size
        search_patch_pos_embed = ops.interpolate(patch_pos_embed, size=(new_P_H, new_P_W), mode='bicubic',
                                                           align_corners=False)
        search_patch_pos_embed = ops.flatten(search_patch_pos_embed,start_dim=2)
        search_patch_pos_embed = search_patch_pos_embed.swapaxes(1, 2)

        # for template region
        H, W = template_size
        new_P_H, new_P_W = H // new_patch_size, W // new_patch_size
        template_patch_pos_embed = ops.interpolate(patch_pos_embed, size=(new_P_H, new_P_W), mode='bicubic',
                                                             align_corners=False)
        template_patch_pos_embed = ops.flatten(template_patch_pos_embed,start_dim=2).swapaxes(1, 2)

        self.pos_embed_z = ms.Parameter(template_patch_pos_embed)
        self.pos_embed_z0 = ms.Parameter(template_patch_pos_embed)
        self.pos_embed_z1 = ms.Parameter(template_patch_pos_embed)
        self.pos_embed_x = ms.Parameter(search_patch_pos_embed)

        # for cls token (keep it but not used)
        if self.add_cls_token and patch_start_index > 0:
            cls_pos_embed = self.pos_embed[:, 0:1, :]
            self.cls_pos_embed = ms.Parameter(cls_pos_embed)

        # separate token and segment token
        if self.add_sep_seg:
            self.template_segment_pos_embed = ms.Parameter(ops.zeros((1, 1, self.embed_dim)))
            self.template_segment_pos_embed = trunc_normal_(self.template_segment_pos_embed, std=.02)
            self.search_segment_pos_embed = ms.Parameter(ops.zeros((1, 1, self.embed_dim)))
            self.search_segment_pos_embed = trunc_normal_(self.search_segment_pos_embed, std=.02)

        # self.cls_token = None
        # self.pos_embed = None

        if self.return_inter:
            for i_layer in self.fpn_stage:
                if i_layer != 11:
                    norm_layer = partial(nn.LayerNorm, eps=1e-6)
                    layer = norm_layer(self.embed_dim)
                    layer_name = f'norm{i_layer}'
                    self.add_module(layer_name, layer)

    def forward_features(self, z_0, z_1_feat, x, identity, seqs_input):
        share_weight = self.word_embeddings.embedding_table.T
        out_list = []
        begin = self.bins * self.range
        begin_2 = self.bins * self.range + 1
        begin_3 = self.bins * self.range + 2
        begin_4 = self.bins * self.range + 3
        score = self.bins * self.range + 5
        B, H, W = x.shape[0], x.shape[2], x.shape[3]
        a = ops.cat((ops.ones((B, 1),dtype=x.dtype) * begin, ops.ones((B, 1),dtype=x.dtype) * begin_2,
                       ops.ones((B, 1),dtype=x.dtype) * begin_3,
                       ops.ones((B, 1),dtype=x.dtype) * begin_4,
                       ops.ones((B, 1),dtype=x.dtype) * score), axis=1)
        b = seqs_input
        # c = ops.cat([a], axis=1)
        c = ops.cat([b, a], axis=1)
        seqs_input_ = c.to(ms.int64)
        output_x_feat = x.copy()
        # original:output_x_feat = x.clone()

        tgt = self.word_embeddings(seqs_input_).permute(1, 0, 2)
        x = self.patch_embed(x)
        z_0 = self.patch_embed(z_0)
        z_1 = z_1_feat

        s_x = x.shape[1]
        s_z = z_0.shape[1] + z_1.shape[1]
        s_s = seqs_input.shape[1]

        z_0 += identity[:, 0, :].tile((B, self.pos_embed_z.shape[1], 1))
        z_1 += identity[:, 1, :].tile((B, self.pos_embed_z.shape[1], 1))

        x += identity[:, 2, :].tile((B, self.pos_embed_x.shape[1], 1))
        query_embed_ = self.position_embeddings.embedding_table.unsqueeze(1)
        prev_embed_ = self.prev_position_embeddings.embedding_table.unsqueeze(1)
        query_embed = ops.cat([prev_embed_, query_embed_], axis=0)
        #query_embed = ops.cat([query_embed_], axis=0)
        query_embed = query_embed.tile((1, B, 1))

        tgt = tgt.swapaxes(0, 1)
        query_embed = query_embed.swapaxes(0, 1)
        # print(self.pos_embed_z0.value())
        # print(self.pos_embed_z1.value())
        # print(self.pos_embed_x.value())
        z_0 += self.pos_embed_z0
        z_1 += self.pos_embed_z1
        x += self.pos_embed_x
        s_s = seqs_input_.shape[1]

        mask = generate_square_subsequent_mask(s_z, s_x, s_s)

        tgt += query_embed[:, :tgt.shape[1]]

        z = ops.cat((z_0, z_1), axis=1)
        zx = combine_tokens(z, x, mode=self.cat_mode)
        zxs = ops.cat((zx, tgt), axis=1)
        zxs = self.pos_drop(zxs)
        m1 = zxs[:, -5:-1]
        for j, blk in enumerate(self.blocks):
            zxs = blk(zxs, padding_mask=mask)
        zxs_numpy = zxs.numpy()
        m3 = zxs[:, -5:-1]
        for j, blk in enumerate(self.extension):
            zxs = blk(zxs, padding_mask=mask)

        lens_z = self.pos_embed_z.shape[1]
        lens_x = self.pos_embed_x.shape[1]
        z_0_feat = zxs[:, :lens_z]
        z_1_feat = zxs[:, lens_z:lens_z*2]
        x_feat = zxs[:, lens_z*2:lens_z*2+lens_x]
        m2 = zxs[:, -5:-1]
        x_out = self.norm(zxs[:, -5:-1])
        score_feat = zxs[:, -1]
        seq_feat = x_out
        at = ops.matmul(x_out, share_weight)
        out = at + self.output_bias
        temp = out.swapaxes(0, 1)

        out_list.append(out.unsqueeze(0))
        out = ops.softmax(out,-1)
        value, extra_seq = out.topk(dim=-1, k=1)[0], out.topk(dim=-1, k=1)[1]
        for i in range(4):
            value, extra_seq = out[:, i, :].topk(dim=-1, k=1)[0], out[:, i, :].topk(dim=-1, k=1)[1]
            if i == 0:
                seqs_output = extra_seq
                values = value
            else:
                seqs_output = ops.cat([seqs_output, extra_seq], axis=-1)
                values = ops.cat([values, value], axis=-1)

        output = {'seqs': seqs_output, 'class': values, 'feat': temp, "state": "val/test", "x_feat": ops.stop_gradient(output_x_feat), "seq_feat": seq_feat}
        return output, z_0_feat, z_1_feat, x_feat, score_feat

    def construct(self, z_0, z_1_feat, x, identity, seqs_input, **kwargs):
        """
        Joint feature extraction and relation modeling for the basic ViT backbone.
        Args:
            z (ops.Tensor): template feature, [B, C, H_z, W_z]
            x (ops.Tensor): search region feature, [B, C, H_x, W_x]

        Returns:
            x (ops.Tensor): merged template and search region feature, [B, L_z+L_x, C]
            attn : None
        """
        output = self.forward_features(z_0, z_1_feat, x, identity, seqs_input)

        return output
