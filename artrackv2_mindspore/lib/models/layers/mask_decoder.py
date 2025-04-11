# -*- coding:utf-8 -*-
# author  : Skye Song
# file    : vit_decoder.py
# Copyright (c) Skye-Song. All Rights Reserved

import mindspore as ms
import mindspore.nn as nn
from mindspore import ops
from mindspore import Tensor
import sys

from lib.utils.box_ops import box_xywh_to_cxywh, box_cxcywh_to_xyxy
from lib.models.component.block import Block
from einops import rearrange

from lib.utils.image import *
from mindspore.common.initializer import initializer,Normal,XavierUniform,Constant

class MaskDecoder(nn.Cell):
	def __init__(self, mask_ratio=0.75, patch_size=16, num_patches=8 ** 2, embed_dim=1024, decoder_embed_dim=512,
	             decoder_depth=8, decoder_num_heads=16, pool_size=8,
	             mlp_ratio=4., norm_layer=nn.LayerNorm, norm_pix_loss=False):
		super().__init__()
		self.mask_ratio = mask_ratio

		self.num_patches = num_patches
		self.patch_size = patch_size

		self.decoder_embed = nn.Dense(embed_dim, decoder_embed_dim, has_bias=True)

		self.mask_token = ms.Parameter(ops.zeros((1, 1, decoder_embed_dim)))

		self.decoder_pos_embed = ms.Parameter(ops.zeros((1, num_patches, decoder_embed_dim)),
		                                      requires_grad=False)  # fixed sin-cos embedding

		self.decoder_blocks = nn.CellList([
			Block(decoder_embed_dim, decoder_num_heads, mlp_ratio, qkv_bias=True, norm_layer=norm_layer)
			for i in range(decoder_depth)])
		decoder_embed_dim_tuple=decoder_embed_dim
		if isinstance(decoder_embed_dim,int):
			decoder_embed_dim_tuple=tuple([decoder_embed_dim])
		self.decoder_norm = norm_layer(decoder_embed_dim_tuple)
		self.decoder_pred = nn.Dense(decoder_embed_dim, patch_size ** 2 * 3, has_bias=True)  # decoder to patch

		self.norm_pix_loss = norm_pix_loss

	def random_masking(self, x):
		"""
		Perform per-sample random masking by per-sample shuffling.
		Per-sample shuffling is done by argsort random noise.
		x: [N, L, D], sequence
		"""
		N, L, D = x.shape  # batch, length, dim
		len_keep = int(L * (1 - self.mask_ratio))

		noise = ops.rand(N, L, device=x.device)  # noise in [0, 1]

		# sort noise for each sample
		ids_shuffle = ops.argsort(noise, dim=1)  # ascend: small is keep, large is remove
		ids_restore = ops.argsort(ids_shuffle, dim=1)

		# keep the first subset
		ids_keep = ids_shuffle[:, :len_keep]
		x_keep = ops.gather_elements(x, dim=1, index=ids_keep.unsqueeze(-1).repeat(1, 1, D))

		# generate the binary mask: 0 is keep, 1 is remove
		mask = ops.ones([N, L], device=x.device)
		mask[:, :len_keep] = 0
		# unshuffle to get the binary mask
		mask = ops.gather_elements(mask, dim=1, index=ids_restore)

		# get the masked x
		mask_tokens = self.mask_token.repeat(x.shape[0], ids_restore.shape[1] - x_keep.shape[1], 1)
		x_ = ops.cat([x_keep, mask_tokens], axis=1)  # no cls token
		x_masked = ops.gather_elements(x_, dim=1, index=ids_restore.unsqueeze(-1).repeat(1, 1, x.shape[2]))  # unshuffle

		return x_masked, mask

	def forward_decoder(self, x, eval=False):
		# embed tokens

		x = self.decoder_embed(x)
		mask = None

		# append mask tokens to sequence
		if not eval:
			x, mask = self.random_masking(x)

		# add pos embed
		x = x + self.decoder_pos_embed

		# apply Transformer blocks
		for blk in self.decoder_blocks:
			x = blk(x)
		x = self.decoder_norm(x)

		# predictor projection
		x = self.decoder_pred(x)
		return x, mask

	def unpatchify(self, x):
		"""
        x: (N, L, patch_size**2 *3)
        imgs: (N, 3, H, W)
        """
		p = self.patch_size
		h = w = int(x.shape[1] ** .5)
		assert h * w == x.shape[1]

		x = x.reshape((x.shape[0], h, w, p, p, 3))
		x = ops.permute(x, (0,5,1,3,2,4))
		imgs = x.reshape((x.shape[0], 3, h * p, h * p))
		return imgs

	def patchify(self, imgs):
		"""
		imgs: (N, 3, H, W)
		x: (N, L, patch_size**2 *3)
		"""
		p = self.patch_size
		assert imgs.shape[2] == imgs.shape[3] and imgs.shape[2] % p == 0

		h = w = imgs.shape[2] // p
		x = imgs.reshape((imgs.shape[0], 3, h, p, w, p))
		x = ops.permute(x, (0,2,4,3,5,1))
		x = x.reshape((imgs.shape[0], h * w, p ** 2 * 3))

		return x

	def forward_loss(self, imgs, pred, mask=None):
		"""
		imgs: [N, 3, H, W]
		pred: [N, L, p*p*3]
		mask: [N, L], 0 is keep, 1 is remove,
		"""
		target = self.patchify(imgs)
		if self.norm_pix_loss:
			mean = target.mean(dim=-1, keepdims=True)
			var = target.var(dim=-1, keepdims=True)
			target = (target - mean) / (var + 1.e-6) ** .5

		loss = (pred - target) ** 2
		loss = loss.mean(dim=-1)  # [N, L], mean loss per patc
		if mask == None:
			loss = loss.sum() / pred.shape[1] / pred.shape[0]  # mean loss on removed patches
		else:
			loss = loss.sum() / pred.shape[1] / pred.shape[0]
		return loss

	def construct(self, x, images=None, gt_bboxes=None, eval=False,):
		x_numpy = x.asnumpy()
		x_numpy = rearrange(x_numpy, 'b c h w -> b (h w) c')
		x = Tensor(x_numpy)
		pred, mask = self.forward_decoder(x, eval)  # [N, L, p*p*3]
		if eval:
			return self.unpatchify(pred)
		if mask != None:
			loss = self.forward_loss(imgs=images, pred=pred, mask=mask)
		else:
			loss = self.forward_loss(imgs=images, pred=pred)
		pred = self.unpatchify(pred)
		return pred, loss


def mask_decoder():
	model = MaskDecoder(
		mask_ratio=0.75, patch_size=16, num_patches=8 ** 2, embed_dim=1024, decoder_embed_dim=512, decoder_depth=8,
		decoder_num_heads=16, mlp_ratio=4., norm_layer=nn.LayerNorm, norm_pix_loss=False)
	return model


def build_maskdecoder(cfg):
	pool_size = int(cfg.DATA.TEMPLATE.SIZE / cfg.MODEL.BACKBONE.PATCHSIZE)

	num_patches = (cfg.DATA.TEMPLATE.SIZE // cfg.MODEL.BACKBONE.PATCHSIZE) ** 2

	model = MaskDecoder(
		mask_ratio=cfg.MODEL.DECODER.MASK_RATIO,
		patch_size=cfg.MODEL.BACKBONE.PATCHSIZE,
		num_patches=num_patches,
		embed_dim=cfg.MODEL.BACKBONE.EMBEDDIM,
		decoder_embed_dim=cfg.MODEL.DECODER.EMBEDDIM,
		decoder_depth=cfg.MODEL.DECODER.DEPTH,
		decoder_num_heads=cfg.MODEL.DECODER.NUMHEADS,
		pool_size=pool_size,
		mlp_ratio=cfg.MODEL.DECODER.MLPRATIO,
		norm_layer=nn.LayerNorm,
		norm_pix_loss=False)
	return model
