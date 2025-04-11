# -*- coding:utf-8 -*-
# author  : Skye Song
# file    : attention.py
# Copyright (c) Skye-Song. All Rights Reserved
import sys
sys.path.append("/home/baiyifan/code/AR2_mindspore_cp/2stage")
import mindspore as ms
import mindspore.nn as nn
import mindspore.ops as ops
from einops import rearrange

from lib.utils.image import *


class Attention(nn.Cell):
	def __init__(self, dim, num_heads=8, qkv_bias=False, attn_drop=0., proj_drop=0.):
		super().__init__()
		self.num_heads = num_heads
		head_dim = dim // num_heads
		self.scale = head_dim ** -0.5

		self.qkv = nn.Dense(dim, dim * 3, has_bias=qkv_bias)
		self.attn_drop = nn.Dropout(p=attn_drop)
		self.proj = nn.Dense(dim, dim)
		self.proj_drop = nn.Dropout(p=proj_drop)

	def construct(self, x, padding_mask=None, **kwargs):
		B, N, C = x.shape
		qkv = self.qkv(x).reshape(B, N, 3, self.num_heads, C // self.num_heads).permute(2, 0, 3, 1, 4)
		q, k, v = qkv[0], qkv[1], qkv[2]  # (B, head, N, C//head)

		attn = (q @ k.swapaxes(-2, -1)) * self.scale  # (B, head, N, N)

		if padding_mask is not None:
			assert padding_mask.size()[0] == B
			assert padding_mask.size()[1] == N
			attn = attn.masked_fill(padding_mask.unsqueeze(1).unsqueeze(2), float("-inf"))

		attn = ops.softmax(attn,axis=-1)
		attn = self.attn_drop(attn)

		x = (attn @ v).swapaxes(1, 2).reshape(B, N, C)
		x = self.proj(x)
		x = self.proj_drop(x)
		return x


class ClsMixAttention(nn.Cell):
	def __init__(self,
	             dim,
	             num_heads,
	             qkv_bias=False,
	             attn_drop=0.,
	             proj_drop=0.,
	             ):
		super().__init__()
		self.num_heads = num_heads
		head_dim = dim // num_heads
		self.scale = head_dim ** -0.5

		self.qkv = nn.Dense(dim, dim * 3, has_bias=qkv_bias)
		self.attn_drop = nn.Dropout(p=attn_drop)
		self.proj = nn.Dense(dim, dim)
		self.proj_drop = nn.Dropout(p=proj_drop)

	def construct(self, x, t_h, t_w, s_h, s_w, online_size=1, padding_mask=None):
		B, N, C = x.shape
		qkv = self.qkv(x).reshape(B, N, 3, self.num_heads, C // self.num_heads).permute(2, 0, 3, 1, 4)
		q, k, v = qkv[0], qkv[1], qkv[2]  # (B, head, N, C)

		q_cls, q_t, q_s = ops.split(q, [1, t_h * t_w * (1 + online_size), s_h * s_w], axis=2)
		k_cls, k_t, k_s = ops.split(k, [1, t_h * t_w * (1 + online_size), s_h * s_w], axis=2)
		v_cls, v_t, v_s = ops.split(v, [1, t_h * t_w * (1 + online_size), s_h * s_w], axis=2)
		# cls token attention
		attn = (q_cls @ k.swapaxes(-2, -1)) * self.scale  # (B, head, N_q, N)
		if padding_mask is not None:
			assert padding_mask.size()[0] == B
			assert padding_mask.size()[1] == N
			attn = attn.masked_fill(padding_mask.unsqueeze(1).unsqueeze(2), float("-inf"))
		attn = ops.softmax(attn,axis=-1)
		attn = self.attn_drop(attn)
		x_cls = rearrange(attn @ v, 'b h t d -> b t (h d)')

		# template attention
		attn = (q_t @ k_t.swapaxes(-2, -1)) * self.scale  # (B, head, N_q, N)
		if padding_mask is not None:
			assert padding_mask.size()[0] == B
			assert padding_mask.size()[1] == N
			attn = attn.masked_fill(padding_mask.unsqueeze(1).unsqueeze(2), float("-inf"))
		attn = ops.softmax(attn,axis=-1)
		attn = self.attn_drop(attn)
		x_t = rearrange(attn @ v_t, 'b h t d -> b t (h d)')

		# search region attention
		attn = (q_s @ k.swapaxes(-2, -1)) * self.scale  # (B, head, N_s, N)
		if padding_mask is not None:
			assert padding_mask.size()[0] == B
			assert padding_mask.size()[1] == N
			attn = attn.masked_fill(padding_mask.unsqueeze(1).unsqueeze(2), float("-inf"))
		attn = ops.softmax(attn,axis=-1)
		attn = self.attn_drop(attn)
		x_s = rearrange(attn @ v, 'b h t d -> b t (h d)')

		x = ops.cat([x_cls, x_t, x_s], axis=1)

		x = self.proj(x)
		x = self.proj_drop(x)
		return x


class MixAttention(nn.Cell):
	def __init__(self,
	             dim,
	             num_heads,
	             qkv_bias=False,
	             attn_drop=0.,
	             proj_drop=0.,
	             ):
		super().__init__()
		self.num_heads = num_heads
		head_dim = dim // num_heads
		self.scale = head_dim ** -0.5

		self.qkv = nn.Dense(dim, dim * 3, has_bias=qkv_bias)
		self.attn_drop = nn.Dropout(p=attn_drop)
		self.proj = nn.Dense(dim, dim)
		self.proj_drop = nn.Dropout(p=proj_drop)

	def construct(self, x, t_h, t_w, s_h, s_w, padding_mask=None):
		B, N, C = x.shape
		qkv = self.qkv(x).reshape(B, N, 3, self.num_heads, C // self.num_heads).permute(2, 0, 3, 1, 4)
		q, k, v = qkv[0], qkv[1], qkv[2]  # (B, head, N, C)

		q_t, q_s = ops.split(q, [t_h * t_w * 2, s_h * s_w], axis=2)
		k_t, k_s = ops.split(k, [t_h * t_w * 2, s_h * s_w], axis=2)
		v_t, v_s = ops.split(v, [t_h * t_w * 2, s_h * s_w], axis=2)

		# template attention
		attn = (q_t @ k_t.swapaxes(-2, -1)) * self.scale  # (B, head, N_q, N)
		if padding_mask is not None:
			assert padding_mask.size()[0] == B
			assert padding_mask.size()[1] == N
			attn = attn.masked_fill(padding_mask.unsqueeze(1).unsqueeze(2), float("-inf"))
		attn = ops.softmax(attn,axis=-1)
		attn = self.attn_drop(attn)
		x_t = rearrange(attn @ v_t, 'b h t d -> b t (h d)')

		# search region attention
		attn = (q_s @ k.swapaxes(-2, -1)) * self.scale  # (B, head, N_s, N)
		if padding_mask is not None:
			assert padding_mask.size()[0] == B
			assert padding_mask.size()[1] == N
			attn = attn.masked_fill(padding_mask.unsqueeze(1).unsqueeze(2), float("-inf"))
		attn = ops.softmax(attn,axis=-1)
		attn = self.attn_drop(attn)
		x_s = rearrange(attn @ v, 'b h t d -> b t (h d)')

		x = ops.cat([x_t, x_s], axis=1)

		x = self.proj(x)
		x = self.proj_drop(x)
		return x


class NottAttention(nn.Cell):
	def __init__(self,
	             dim,
	             num_heads,
	             qkv_bias=False,
	             attn_drop=0.,
	             proj_drop=0.,
	             ):
		super().__init__()
		self.num_heads = num_heads
		head_dim = dim // num_heads
		self.scale = head_dim ** -0.5

		self.qkv = nn.Dense(dim, dim * 3, has_bias=qkv_bias)
		self.attn_drop = nn.Dropout(p=attn_drop)
		self.proj = nn.Dense(dim, dim)
		self.proj_drop = nn.Dropout(p=proj_drop)

	def construct(self, x, t_h, t_w, s_h, s_w, padding_mask=None):
		B, N, C = x.shape
		qkv = self.qkv(x).reshape(B, N, 3, self.num_heads, C // self.num_heads).permute(2, 0, 3, 1, 4)
		q, k, v = qkv[0], qkv[1], qkv[2]  # (B, head, N, C)

		q_t, q_s = ops.split(q, [t_h * t_w * 2, s_h * s_w], axis=2)
		k_t, k_s = ops.split(k, [t_h * t_w * 2, s_h * s_w], axis=2)
		v_t, v_s = ops.split(v, [t_h * t_w * 2, s_h * s_w], axis=2)

		# template attention
		attn = (q_t @ k_s.swapaxes(-2, -1)) * self.scale  # (B, head, N_q, N)
		if padding_mask is not None:
			assert padding_mask.size()[0] == B
			assert padding_mask.size()[1] == N
			attn = attn.masked_fill(padding_mask.unsqueeze(1).unsqueeze(2), float("-inf"))
		attn = ops.softmax(attn,axis=-1)
		attn = self.attn_drop(attn)
		x_t = rearrange(attn @ v_s, 'b h t d -> b t (h d)')

		# search region attention
		attn = (q_s @ k.swapaxes(-2, -1)) * self.scale  # (B, head, N_s, N)
		if padding_mask is not None:
			assert padding_mask.size()[0] == B
			assert padding_mask.size()[1] == N
			attn = attn.masked_fill(padding_mask.unsqueeze(1).unsqueeze(2), float("-inf"))
		attn = ops.softmax(attn,axis=-1)
		attn = self.attn_drop(attn)
		x_s = rearrange(attn @ v, 'b h t d -> b t (h d)')

		x = ops.cat([x_t, x_s], axis=1)

		x = self.proj(x)
		x = self.proj_drop(x)
		return x


class NossAttention(nn.Cell):
	def __init__(self,
	             dim,
	             num_heads,
	             qkv_bias=False,
	             attn_drop=0.,
	             proj_drop=0.,
	             ):
		super().__init__()
		self.num_heads = num_heads
		head_dim = dim // num_heads
		self.scale = head_dim ** -0.5

		self.qkv = nn.Dense(dim, dim * 3, has_bias=qkv_bias)
		self.attn_drop = nn.Dropout(p=attn_drop)
		self.proj = nn.Dense(dim, dim)
		self.proj_drop = nn.Dropout(p=proj_drop)

	def construct(self, x, t_h, t_w, s_h, s_w, padding_mask=None):
		B, N, C = x.shape
		qkv = self.qkv(x).reshape(B, N, 3, self.num_heads, C // self.num_heads).permute(2, 0, 3, 1, 4)
		q, k, v = qkv[0], qkv[1], qkv[2]  # (B, head, N, C)

		q_t, q_s = ops.split(q, [t_h * t_w * 2, s_h * s_w], axis=2)
		k_t, k_s = ops.split(k, [t_h * t_w * 2, s_h * s_w], axis=2)
		v_t, v_s = ops.split(v, [t_h * t_w * 2, s_h * s_w], axis=2)

		# template attention
		attn = (q_t @ k.swapaxes(-2, -1)) * self.scale  # (B, head, N_q, N)
		if padding_mask is not None:
			assert padding_mask.size()[0] == B
			assert padding_mask.size()[1] == N
			attn = attn.masked_fill(padding_mask.unsqueeze(1).unsqueeze(2), float("-inf"))
		attn = ops.softmax(attn,axis=-1)
		attn = self.attn_drop(attn)
		x_t = rearrange(attn @ v, 'b h t d -> b t (h d)')

		# search region attention
		attn = (q_s @ k_t.swapaxes(-2, -1)) * self.scale  # (B, head, N_s, N)
		if padding_mask is not None:
			assert padding_mask.size()[0] == B
			assert padding_mask.size()[1] == N
			attn = attn.masked_fill(padding_mask.unsqueeze(1).unsqueeze(2), float("-inf"))
		attn = ops.softmax(attn,axis=-1)
		attn = self.attn_drop(attn)
		x_s = rearrange(attn @ v_t, 'b h t d -> b t (h d)')

		x = ops.cat([x_t, x_s], axis=1)

		x = self.proj(x)
		x = self.proj_drop(x)
		return x


class CrossAttention(nn.Cell):
	def __init__(self,
	             dim,
	             num_heads,
	             qkv_bias=False,
	             attn_drop=0.,
	             proj_drop=0.,
	             ):
		super().__init__()
		self.num_heads = num_heads
		head_dim = dim // num_heads
		self.scale = head_dim ** -0.5

		self.qkv = nn.Dense(dim, dim * 3, has_bias=qkv_bias)
		self.attn_drop = nn.Dropout(p=attn_drop)
		self.proj = nn.Dense(dim, dim)
		self.proj_drop = nn.Dropout(p=proj_drop)

	def construct(self, x, t_h, t_w, s_h, s_w):
		B, N, C = x.shape
		qkv = self.qkv(x).reshape(B, N, 3, self.num_heads, C // self.num_heads).permute(2, 0, 3, 1, 4)
		q, k, v = qkv[0], qkv[1], qkv[2]  # (B, head, N, C)

		q_t, q_s = ops.split(q, [t_h * t_w * 2, s_h * s_w], axis=2)
		k_t, k_s = ops.split(k, [((t_h + 1) // 2) ** 2 * 2, s_h * s_w // 4], axis=4)
		v_t, v_s = ops.split(v, [((t_h + 1) // 2) ** 2 * 2, s_h * s_w // 4], axis=4)

		# template attention
		attn = (q_t @ k_s.swapaxes(-2, -1)) * self.scale
		attn = ops.softmax(attn,axis=-1)
		attn = self.attn_drop(attn)
		x_t = rearrange(attn @ v_s, 'b h t d -> b t (h d)')

		# search region attention
		attn = (q_s @ k_t.swapaxes(-2, -1)) * self.scale
		attn = ops.softmax(attn,axis=-1)
		attn = self.attn_drop(attn)
		x_s = rearrange(attn @ v_t, 'b h t d -> b t (h d)')

		x = ops.cat([x_t, x_s], axis=1)

		x = self.proj(x)
		x = self.proj_drop(x)
		return x
