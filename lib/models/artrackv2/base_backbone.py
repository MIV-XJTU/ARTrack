from functools import partial

import torch
import torch.nn as nn
import torch.nn.functional as F
from timm.models.vision_transformer import resize_pos_embed
from timm.models.layers import DropPath, to_2tuple, trunc_normal_

from lib.models.layers.patch_embed import PatchEmbed
from lib.models.artrackv2.utils import combine_tokens, recover_tokens

def generate_square_subsequent_mask(sz, sx, ss):
    r"""Generate a square mask for the sequence. The masked positions are filled with float('-inf').
        Unmasked positions are filled with float(0.0).
    """
    sum = sz + sx + ss
    mask = (torch.triu(torch.ones(sum, sum)) == 1).transpose(0, 1)
    mask[:, :] = 0
    mask[:int(sz/2), :int(sz/2)] = 1 #template self
    mask[int(sz/2):sz, int(sz/2):sz] = 1 # dt self
    mask[int(sz/2):sz, sz:sz+sx] = 1 # dt search
    mask[int(sz / 2):sz, -1] = 1  # dt search
    mask[sz:sz+sx, :sz+sx] = 1 # sr dt-t-sr
    mask[sz+sx:, :] = 1 # co dt-t-sr-co
    return ~mask

class BaseBackbone(nn.Module):
    def __init__(self):
        super().__init__()

        # for original ViT
        self.pos_embed = None
        self.img_size = [224, 224]
        self.patch_size = 16
        self.embed_dim = 384

        self.cat_mode = 'direct'

        self.pos_embed_z0 = None
        self.pos_embed_z1 = None
        self.pos_embed_x = None

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
                    param = nn.functional.interpolate(param, size=(new_patch_size, new_patch_size),
                                                      mode='bicubic', align_corners=False)
                    param = nn.Parameter(param)
                old_patch_embed[name] = param
            self.patch_embed = PatchEmbed(img_size=self.img_size, patch_size=new_patch_size, in_chans=3,
                                          embed_dim=self.embed_dim)
            self.patch_embed.proj.bias = old_patch_embed['proj.bias']
            self.patch_embed.proj.weight = old_patch_embed['proj.weight']

        # for patch embedding
        patch_pos_embed = self.pos_embed[:, patch_start_index:, :]
        patch_pos_embed = patch_pos_embed.transpose(1, 2)
        B, E, Q = patch_pos_embed.shape
        P_H, P_W = self.img_size[0] // self.patch_size, self.img_size[1] // self.patch_size
        patch_pos_embed = patch_pos_embed.view(B, E, P_H, P_W)

        # for search region
        H, W = search_size
        new_P_H, new_P_W = H // new_patch_size, W // new_patch_size
        search_patch_pos_embed = nn.functional.interpolate(patch_pos_embed, size=(new_P_H, new_P_W), mode='bicubic',
                                                           align_corners=False)
        search_patch_pos_embed = search_patch_pos_embed.flatten(2).transpose(1, 2)

        # for template region
        H, W = template_size
        new_P_H, new_P_W = H // new_patch_size, W // new_patch_size
        template_patch_pos_embed = nn.functional.interpolate(patch_pos_embed, size=(new_P_H, new_P_W), mode='bicubic',
                                                             align_corners=False)
        template_patch_pos_embed = template_patch_pos_embed.flatten(2).transpose(1, 2)

        self.pos_embed_z0 = nn.Parameter(template_patch_pos_embed)
        self.pos_embed_z1 = nn.Parameter(template_patch_pos_embed)
        self.pos_embed_x = nn.Parameter(search_patch_pos_embed)

        # for cls token (keep it but not used)
        if self.add_cls_token and patch_start_index > 0:
            cls_pos_embed = self.pos_embed[:, 0:1, :]
            self.cls_pos_embed = nn.Parameter(cls_pos_embed)

        # separate token and segment token
        if self.add_sep_seg:
            self.template_segment_pos_embed = nn.Parameter(torch.zeros(1, 1, self.embed_dim))
            self.template_segment_pos_embed = trunc_normal_(self.template_segment_pos_embed, std=.02)
            self.search_segment_pos_embed = nn.Parameter(torch.zeros(1, 1, self.embed_dim))
            self.search_segment_pos_embed = trunc_normal_(self.search_segment_pos_embed, std=.02)


        if self.return_inter:
            for i_layer in self.fpn_stage:
                if i_layer != 11:
                    norm_layer = partial(nn.LayerNorm, eps=1e-6)
                    layer = norm_layer(self.embed_dim)
                    layer_name = f'norm{i_layer}'
                    self.add_module(layer_name, layer)

    def forward_features(self, z_0, z_1, x, identity, seqs_input):
        share_weight = self.word_embeddings.weight.T

        B, H, W = x.shape[0], x.shape[2], x.shape[3]

        seqs_input = seqs_input.to(torch.int64).to(x.device)
        tgt = self.word_embeddings(seqs_input).permute(1, 0, 2)
        query_embed = self.position_embeddings.weight.unsqueeze(1)
        query_embed = query_embed.repeat(1, B, 1)

        tgt = tgt.transpose(0, 1)
        query_embed = query_embed.transpose(0, 1)

        x = self.patch_embed(x)
        z_0 = self.patch_embed(z_0)
        z_1 = self.patch_embed(z_1)

        len_x = x.shape[1]
        len_z = z_0.shape[1] + z_1.shape[1]
        len_seq = seqs_input.shape[1]

        mask = generate_square_subsequent_mask(len_z, len_x, len_seq).to(tgt.device)

        if self.add_cls_token:
            cls_tokens = self.cls_token.expand(B, -1, -1)
            cls_tokens = cls_tokens + self.cls_pos_embed

        z_0 += self.pos_embed_z0
        z_1 += self.pos_embed_z1
        x += self.pos_embed_x
        tgt += query_embed
        
        z_0 += identity[:, 0, :].repeat(B, self.pos_embed_z0.shape[1], 1)
        z_1 += identity[:, 1, :].repeat(B, self.pos_embed_z1.shape[1], 1)

        x += identity[:, 2, :].repeat(B, self.pos_embed_x.shape[1], 1)

        if self.add_sep_seg:
            x += self.search_segment_pos_embed
            z += self.template_segment_pos_embed

        z = torch.cat((z_0, z_1), dim=1)

        x = combine_tokens(z, x, mode=self.cat_mode)
        x = torch.cat((x, tgt), dim=1)
        if self.add_cls_token:
            x = torch.cat([cls_tokens, x], dim=1)

        x = self.pos_drop(x)

        for i, blk in enumerate(self.blocks):
            x = blk(x, padding_mask=mask)
        #
        for j, blk in enumerate(self.extension):
            x = blk(x, padding_mask=mask)
        x_out = self.norm(x[:, -5:-1])
        score_feat = x[:, -1]

        lens_z = self.pos_embed_z0.shape[1]
        lens_x = self.pos_embed_x.shape[1]

        z_0_feat = x[:, :lens_z]
        z_1_feat = x[:, lens_z:lens_z*2]
        x_feat = x[:, lens_z*2:lens_z*2+lens_x]

        #x = recover_tokens(x, lens_z, lens_x, mode=self.cat_mode)
        at = torch.matmul(x_out, share_weight)
        at = at + self.output_bias
        at = at[:, -4:]
        at = at.transpose(0, 1)
        output = {'feat': at, 'score_feat':score_feat, "state": "train"}

        return output, z_0_feat, z_1_feat, x_feat

    def forward_track(self, z_0, z_1, x, identity):
        share_weight = self.word_embeddings.weight.T
        out_list = []

        x0 = self.bins * self.range
        y0 = self.bins * self.range + 1
        x1 = self.bins * self.range + 2
        y1 = self.bins * self.range + 3
        score = self.bins * self.range + 5

        B, H, W = x.shape[0], x.shape[2], x.shape[3]

        seq = torch.cat([torch.ones((B, 1)).to(x) * x0, torch.ones((B, 1)).to(x) * y0,
                       torch.ones((B, 1)).to(x) * x1,
                       torch.ones((B, 1)).to(x) * y1,
                       torch.ones((B, 1)).to(x) * score], dim=1)

        seq_all = torch.cat([seq], dim=1)

        seqs_input = seq_all.to(torch.int64).to(x.device)
        output_x_feat = x.clone()
        tgt = self.word_embeddings(seqs_input).permute(1, 0, 2)

        x = self.patch_embed(x)
        z_0 = self.patch_embed(z_0)
        z_1 = self.patch_embed(z_1)

        len_x = x.shape[1]
        len_z = z_0.shape[1] + z_1.shape[1]
        len_seq = seqs_input.shape[1]

        z_0 += identity[:, 0, :].repeat(B, self.pos_embed_z0.shape[1], 1)
        z_1 += identity[:, 1, :].repeat(B, self.pos_embed_z0.shape[1], 1)

        x += identity[:, 2, :].repeat(B, self.pos_embed_x.shape[1], 1)

        query_pos_embed = self.position_embeddings.weight.unsqueeze(1)
        query_pos_embed = query_pos_embed.repeat(1, B, 1)

        tgt = tgt.transpose(0, 1)
        query_pos_embed = query_pos_embed.transpose(0, 1)

        z_0 += self.pos_embed_z0
        z_1 += self.pos_embed_z1
        x += self.pos_embed_x


        mask = generate_square_subsequent_mask(len_z, len_x, len_seq).to(tgt.device)

        tgt += query_pos_embed[:, :tgt.shape[1]]

        z = torch.cat((z_0, z_1), dim=1)

        zx = combine_tokens(z, x, mode=self.cat_mode)
        zxs = torch.cat((zx, tgt), dim=1)

        zxs = self.pos_drop(zxs)

        for j, blk in enumerate(self.blocks):
            zxs = blk(zxs, padding_mask=mask)

        for j, blk in enumerate(self.extension):
            zxs = blk(zxs, padding_mask=mask)

        lens_z_single = self.pos_embed_z0.shape[1]
        lens_x = self.pos_embed_x.shape[1]

        z_0_feat = zxs[:, :lens_z_single]
        z_1_feat = zxs[:, lens_z_single:lens_z_single * 2]
        x_feat = zxs[:, lens_z_single * 2:lens_z_single * 2 + lens_x]

        x_out = self.norm(zxs[:, -5:-1])
        score_feat = x[:, -1]

        possibility = torch.matmul(x_out, share_weight)
        out = possibility + self.output_bias
        temp = out.transpose(0, 1)

        out_list.append(out.unsqueeze(0))
        out = out.softmax(-1)

        value, extra_seq = out.topk(dim=-1, k=1)[0], out.topk(dim=-1, k=1)[1]
        for i in range(4):
            value, extra_seq = out[:, i, :].topk(dim=-1, k=1)[0], out[:, i, :].topk(dim=-1, k=1)[1]
            if i == 0:
                seqs_output = extra_seq
                values = value
            else:
                seqs_output = torch.cat([seqs_output, extra_seq], dim=-1)
                values = torch.cat([values, value], dim=-1)

        output = {'seqs': seqs_output, 'class': values, 'feat': temp, "state": "val/test",
                  "x_feat": output_x_feat.detach(), "score_feat": score_feat}

        return output, None, None, None

    def forward(self, z_0, z_1, x, identity, seqs_input, **kwargs):
        """
        Joint feature extraction and relation modeling for the basic ViT backbone.
        Args:
            z (torch.Tensor): template feature, [B, C, H_z, W_z]
            x (torch.Tensor): search region feature, [B, C, H_x, W_x]

        Returns:
            x (torch.Tensor): merged template and search region feature, [B, L_z+L_x, C]
            attn : None
        """
        if seqs_input == None:
            output = self.forward_track(z_0, z_1, x, identity)
        else:
            output = self.forward_features(z_0, z_1, x, identity, seqs_input)

        return output
