# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.

# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.
# --------------------------------------------------------
# References:
# timm: https://github.com/rwightman/pytorch-image-models/tree/master/timm
# DeiT: https://github.com/facebookresearch/deit
# --------------------------------------------------------

from functools import partial

import torch
import torch.nn as nn

import timm.models.vision_transformer
from timm.models.vision_transformer import Mlp
from models.models_layers import Block, PatchEmbed
import math


class VisionTransformer(timm.models.vision_transformer.VisionTransformer):
    """ Vision Transformer with support for global average pooling
    """

    def __init__(self, global_pool=False, mlp_token=False, num_tokens=1, pred_att=False, last_layers=2, topkatt=15, **kwargs):
        super(VisionTransformer, self).__init__(**kwargs)

        self.global_pool = global_pool
        if self.global_pool:
            norm_layer = kwargs['norm_layer']
            embed_dim = kwargs['embed_dim']
            self.fc_norm = norm_layer(embed_dim)

            del self.norm  # remove the original norm

        num_patches = self.patch_embed.num_patches
        embed_dim = kwargs['embed_dim']
        img_size = 224
        patch_size = kwargs['patch_size']
        in_chans = 3
        self.cls_token = nn.Parameter(torch.zeros(1, num_tokens, embed_dim))
        self.pos_embed = nn.Parameter(torch.zeros(
            1, num_patches + num_tokens, embed_dim), requires_grad=False)  # fixed sin-cos embedding
        self.patch_embed = PatchEmbed(
            img_size, patch_size, in_chans, embed_dim)

        if mlp_token:
            mlp_ratio = kwargs['mlp_ratio']
            self.cls_token_mlp = Mlp(
                in_features=embed_dim, hidden_features=embed_dim*mlp_ratio, act_layer=nn.GELU)

        if pred_att:
            num_heads = kwargs['num_heads']
            qkv_bias = kwargs['qkv_bias']
            norm_layer = kwargs['norm_layer']
            depth = kwargs['depth']
            mlp_ratio = kwargs['mlp_ratio']
            qk_scale = None
            self.blocks = nn.ModuleList([
                Block(
                    dim=embed_dim, num_heads=num_heads, mlp_ratio=mlp_ratio, qkv_bias=qkv_bias, qk_scale=qk_scale,
                    drop=0.0, attn_drop=0.0, drop_path=0.0, norm_layer=norm_layer)
                for i in range(depth)])

        self.mlp_token = mlp_token
        self.num_tokens = num_tokens
        self.pred_att = pred_att
        self.last_layers = last_layers
        self.topkatt = topkatt
        print("mlp_token,num_tokens,pred_att,last_layers,topkatt")
        print(mlp_token,num_tokens,pred_att,last_layers,topkatt)

    def forward_features(self, x):
        B, _, w, h = x.shape
        x = self.patch_embed(x)
        if self.mlp_token:
            cls_token = self.cls_token_mlp(self.cls_token)
            # stole cls_tokens impl from Phil Wang, thanks
            cls_tokens = cls_token.expand(B, -1, -1)
        else:
            cls_tokens = self.cls_token.expand(B, -1, -1)

        x = torch.cat((cls_tokens, x), dim=1)
        x = x + self.pos_embed
        x = self.pos_drop(x)

        teacher_feature = 0
        teacher_att = []
        teacher_att_idx = []
        for i in range(len(self.blocks)):
            blk = self.blocks[i]
            if self.pred_att and i == len(self.blocks)-1:
                x, att = blk(x, return_att=True)
                att_cls_rank = att[:, :, 0, 1:]  # 196
                att_rank = att_cls_rank.mean(dim=1)
                _, topk_idx = torch.topk(att_rank, k=self.topkatt, dim=1, largest=True)

                att_tmp = att[:, :, 1:].transpose(1,2)

                att_pick = torch.gather(att_tmp, dim=1, index=topk_idx.unsqueeze(
                    -1).unsqueeze(-1).repeat(1, 1, att_tmp.shape[2], att_tmp.shape[3]))
                att_cls = att[:, :, 0, :]
                att_pick = torch.cat((att_cls.unsqueeze(dim=1), att_pick), dim=1)
                att_pick = att_pick[:, :, :, 1:]
                teacher_att.append(att_pick.unsqueeze(dim=1))
                teacher_att_idx.append(topk_idx.unsqueeze(dim=1))

            else:
                x = blk(x)
            if i >= len(self.blocks)-self.last_layers:
                teacher_feature+=x[:, self.num_tokens:, :]

        teacher_feature /= self.last_layers
        if self.pred_att:
            teacher_att = torch.cat(teacher_att, dim=1)
            teacher_att_idx = torch.cat(teacher_att_idx, dim=1)
            return [teacher_feature, teacher_att, teacher_att_idx]
        else:
            return [teacher_feature]


def vit_base_patch16(**kwargs):
    model = VisionTransformer(
        patch_size=16, embed_dim=768, depth=12, num_heads=12, mlp_ratio=4, qkv_bias=True,
        norm_layer=partial(nn.LayerNorm, eps=1e-6), **kwargs)
    return model


def vit_large_patch16(**kwargs):
    model = VisionTransformer(
        patch_size=16, embed_dim=1024, depth=24, num_heads=16, mlp_ratio=4, qkv_bias=True,
        norm_layer=partial(nn.LayerNorm, eps=1e-6), **kwargs)
    return model


def vit_huge_patch14(**kwargs):
    model = VisionTransformer(
        patch_size=14, embed_dim=1280, depth=32, num_heads=16, mlp_ratio=4, qkv_bias=True,
        norm_layer=partial(nn.LayerNorm, eps=1e-6), **kwargs)
    return model
