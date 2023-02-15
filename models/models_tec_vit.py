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

from timm.models.vision_transformer import Mlp

from util.pos_embed import get_2d_sincos_pos_embed
from models.models_layers import PatchEmbed, Block


class FeatureAdaptor(nn.Module):
    """ Encoder Adaptor for ViT
    """

    def __init__(self, embed_dim=1024, depth=None, num_adap=None, mid_embed_dim=None, norm_layer=nn.LayerNorm):
        super().__init__()

        self.adap_before = norm_layer(embed_dim*depth)

        self.fuse_fcs = nn.ModuleList([
            nn.Linear(embed_dim*(depth//num_adap), embed_dim, bias=True)
            for _ in range(num_adap)])

        self.adap_mlps = nn.ModuleList([
            Mlp(in_features=embed_dim, hidden_features=mid_embed_dim,
                out_features=embed_dim, act_layer=nn.GELU)
            for _ in range(num_adap)])

        self.adap_after = norm_layer(embed_dim)
        self.num_adap = num_adap

        self.initialize_weights()

    def initialize_weights(self):
        # initialization
        self.apply(self._init_weights)

    def _init_weights(self, m):
        if isinstance(m, nn.Linear):
            # we use xavier_uniform following official JAX ViT:
            torch.nn.init.xavier_uniform_(m.weight)
            if isinstance(m, nn.Linear) and m.bias is not None:
                nn.init.constant_(m.bias, 0)
        elif isinstance(m, nn.LayerNorm):
            nn.init.constant_(m.bias, 0)
            nn.init.constant_(m.weight, 1.0)

    def forward(self, x):
        out = 0
        x = self.adap_before(x)
        xset = x.chunk(self.num_adap, dim=-1)
        for i in range(len(xset)):
            x = self.fuse_fcs[i](xset[i])
            x = self.adap_mlps[i](x) + x
            out += x
        out = self.adap_after(out)
        return out


class ViTDecoderFeature(nn.Module):
    """ Feature Decoder for ViT
    """

    def __init__(self,
                 embed_dim=1024, pred_dim=None, decoder_embed_dim=512, norm_layer=nn.LayerNorm):
        super().__init__()

        # Feature decoder specifics
        if pred_dim is None:
            pred_dim = embed_dim

        self.decoder_embed = nn.Linear(embed_dim, decoder_embed_dim, bias=True)
        self.mask_token = nn.Parameter(torch.zeros(1, 1, decoder_embed_dim))

        self.decoder_norm = norm_layer(decoder_embed_dim)
        self.decoder_pred = nn.Linear(
            decoder_embed_dim, pred_dim, bias=True)  # decoder to feature

        self.initialize_weights()

    def initialize_weights(self):
        # initialization
        torch.nn.init.normal_(self.mask_token, std=.02)
        self.apply(self._init_weights)

    def _init_weights(self, m):
        if isinstance(m, nn.Linear):
            # we use xavier_uniform following official JAX ViT:
            torch.nn.init.xavier_uniform_(m.weight)
            if isinstance(m, nn.Linear) and m.bias is not None:
                nn.init.constant_(m.bias, 0)
        elif isinstance(m, nn.LayerNorm):
            nn.init.constant_(m.bias, 0)
            nn.init.constant_(m.weight, 1.0)

    def forward(self, x, ids_restore, decoder_pos_embed, decoder_blocks):
        # embed tokens
        x = self.decoder_embed(x)

        # append mask tokens to sequence
        mask_tokens = self.mask_token.repeat(
            x.shape[0], ids_restore.shape[1] + 1 - x.shape[1], 1)
        x_ = torch.cat([x[:, 1:, :], mask_tokens], dim=1)  # no cls token
        x_ = torch.gather(
            x_, dim=1, index=ids_restore.unsqueeze(-1).repeat(1, 1, x.shape[2]))  # unshuffle
        x = torch.cat([x[:, :1, :], x_], dim=1)  # append cls token

        # add pos embed
        x = x + decoder_pos_embed

        # apply Transformer blocks
        for blk in decoder_blocks:
            x = blk(x)
        x = self.decoder_norm(x)

        # predictor projection
        x = self.decoder_pred(x)

        # remove cls token
        x = x[:, 1:, :]

        return x


class ViTDecoderAtt(nn.Module):
    """ Attention Decoder for ViT
    """

    def __init__(self,
                 embed_dim=1024, pred_dim=None, pred_num_heads=16,
                 decoder_embed_dim=512, norm_layer=nn.LayerNorm):
        super().__init__()

        # Att decoder specifics
        if pred_dim is None:
            pred_dim = embed_dim

        self.decoder_embed = nn.Linear(
            embed_dim, decoder_embed_dim, bias=True)
        self.mask_token = nn.Parameter(
            torch.zeros(1, 1, decoder_embed_dim))

        self.decoder_norm = norm_layer(decoder_embed_dim)

        self.decoder_qk = nn.Linear(decoder_embed_dim, 1 *
                                    pred_dim * 2, bias=True)
        self.scale = pred_dim ** -0.5

        self.pred_num_heads = pred_num_heads

        self.initialize_weights()

    def initialize_weights(self):
        # initialization

        # timm's trunc_normal_(std=.02) is effectively normal_(std=0.02) as cutoff is too big (2.)
        torch.nn.init.normal_(self.mask_token, std=.02)
        self.apply(self._init_weights)

    def _init_weights(self, m):
        if isinstance(m, nn.Linear):
            # we use xavier_uniform following official JAX ViT:
            torch.nn.init.xavier_uniform_(m.weight)
            if isinstance(m, nn.Linear) and m.bias is not None:
                nn.init.constant_(m.bias, 0)
        elif isinstance(m, nn.LayerNorm):
            nn.init.constant_(m.bias, 0)
            nn.init.constant_(m.weight, 1.0)

    def forward(self, x, ids_restore, att_idx, decoder_pos_embed, decoder_blocks):
        # embed tokens
        x = self.decoder_embed(x)

        # append mask tokens to sequence
        mask_tokens = self.mask_token.repeat(
            x.shape[0], ids_restore.shape[1] + 1 - x.shape[1], 1)
        x_ = torch.cat([x[:, 1:, :], mask_tokens], dim=1)  # no cls token
        x_ = torch.gather(
            x_, dim=1, index=ids_restore.unsqueeze(-1).repeat(1, 1, x.shape[2]))  # unshuffle
        x = torch.cat([x[:, :1, :], x_], dim=1)  # append cls token

        # add pos embed
        x = x + decoder_pos_embed

        # apply Transformer blocks
        for blk in decoder_blocks:
            x = blk(x)

        x = self.decoder_norm(x)
        B, N, C = x.shape
        qk = self.decoder_qk(x).reshape(B, N, 2, 1, self.pred_num_heads, -
                                        1).permute(2, 0, 3, 4, 1, 5)  # 2 B LAYER H L C
        # make torchscript happy (cannot use tensor as tuple) # B LAYER H L C
        q, k = qk[0], qk[1]
        q = q.transpose(-2, -1)  # B, LAYER, H, L, C - > B LAYER H C L
        q_cls = q[:, :, :, :, 0:1]  # B LAYER H C 1
        q_p_all = q[:, :, :, :, 1:].permute(
            0, 1, 4, 2, 3)  # B LAYER H C L -> B LAYER L H C

        q_p = torch.gather(q_p_all, dim=2, index=att_idx.unsqueeze(
            -1).unsqueeze(-1).repeat(1, 1, 1, q_p_all.shape[3], q_p_all.shape[4]))

        q_p = q_p.permute(0, 1, 3, 4, 2)
        q = torch.cat((q_cls, q_p), dim=-1)
        q = q.transpose(-2, -1)

        attn = (q @ k.transpose(-2, -1)) * self.scale

        att = attn.permute(0, 1, 3, 4, 2)
        att = att.transpose(-1, -2)
        att = att[:, :, :, :, 1:]  # B LAYER L1 H L2

        return att


class TECViT(nn.Module):
    """ Target-Enhanced Conditional pretraining on ViT backbone
    """

    def __init__(self, img_size=224, patch_size=16, in_chans=3,
                 embed_dim=1024, depth=24, num_heads=16,
                 decoder_embed_dim=512, decoder_depth=8, decoder_num_heads=16,
                 mlp_ratio=4., norm_layer=nn.LayerNorm,
                 mlp_token=False, num_tokens=1, pred_att=False, att_tau=None,
                 num_adap=None, pred_num_heads=None, pred_dim=None):
        super().__init__()

        # --------------------------------------------------------------------------
        # TEC encoder specifics
        self.patch_embed = PatchEmbed(
            img_size, patch_size, in_chans, embed_dim)
        num_patches = self.patch_embed.num_patches

        self.cls_token = nn.Parameter(torch.zeros(1, num_tokens, embed_dim))

        self.pos_embed = nn.Parameter(torch.zeros(
            1, num_patches + num_tokens, embed_dim), requires_grad=False)  # fixed sin-cos embedding

        self.blocks = nn.ModuleList([
            Block(embed_dim, num_heads, mlp_ratio, qkv_bias=True,
                  qk_scale=None, norm_layer=norm_layer)
            for i in range(depth)])
        self.norm = norm_layer(embed_dim)
        # --------------------------------------------------------------------------
        # Input Adaptor
        if mlp_token:
            self.cls_token_mlp = Mlp(
                in_features=embed_dim, hidden_features=embed_dim*mlp_ratio, act_layer=nn.GELU)
        # Encoder Adaptor
        if num_adap is None:
            num_adap = depth//3
        self.encoder_adaptor = FeatureAdaptor(embed_dim=embed_dim, depth=depth,
                                              num_adap=num_adap, mid_embed_dim=decoder_embed_dim*4)
        # --------------------------------------------------------------------------
        # TEC decoder specifics
        if pred_num_heads is None:
            pred_num_heads = num_heads
        if pred_dim is None:
            pred_dim = embed_dim

        self.forward_fea_decoder = ViTDecoderFeature(
            embed_dim, pred_dim, decoder_embed_dim, norm_layer)

        if pred_att:
            self.forward_att_decoder = ViTDecoderAtt(
                embed_dim, pred_dim, pred_num_heads, decoder_embed_dim, norm_layer)

        self.decoder_blocks = nn.ModuleList([
            Block(decoder_embed_dim, decoder_num_heads, mlp_ratio,
                  qkv_bias=True, qk_scale=None, norm_layer=norm_layer)
            for i in range(decoder_depth)])
        self.decoder_pos_embed = nn.Parameter(torch.zeros(
            1, num_patches + num_tokens, decoder_embed_dim), requires_grad=False)  # fixed sin-cos embedding
        # --------------------------------------------------------------------------

        self.pred_att = pred_att
        self.mlp_token = mlp_token
        self.num_tokens = num_tokens
        self.decoder_num_heads = decoder_num_heads
        self.num_heads = num_heads
        self.num_adap = num_adap
        self.att_tau = att_tau
        print("pred_att,mlp_token,num_tokens,decoder_num_heads,num_adap,att_tau")
        print(pred_att,mlp_token,num_tokens,decoder_num_heads,num_adap,att_tau)

        self.initialize_weights()

    def initialize_weights(self):
        # initialization
        # initialize (and freeze) pos_embed by sin-cos embedding
        pos_embed = get_2d_sincos_pos_embed(self.pos_embed.shape[-1], int(
            self.patch_embed.num_patches**.5), cls_token=True, num_tokens=self.num_tokens)
        self.pos_embed.data.copy_(
            torch.from_numpy(pos_embed).float().unsqueeze(0))

        decoder_pos_embed = get_2d_sincos_pos_embed(self.decoder_pos_embed.shape[-1], int(
            self.patch_embed.num_patches**.5), cls_token=True, num_tokens=self.num_tokens)
        self.decoder_pos_embed.data.copy_(
            torch.from_numpy(decoder_pos_embed).float().unsqueeze(0))

        # initialize patch_embed like nn.Linear (instead of nn.Conv2d)
        w = self.patch_embed.proj.weight.data
        torch.nn.init.xavier_uniform_(w.view([w.shape[0], -1]))

        # timm's trunc_normal_(std=.02) is effectively normal_(std=0.02) as cutoff is too big (2.)
        torch.nn.init.normal_(self.cls_token, std=.02)
        # initialize nn.Linear and nn.LayerNorm
        self.apply(self._init_weights)

    def _init_weights(self, m):
        if isinstance(m, nn.Linear):
            # we use xavier_uniform following official JAX ViT:
            torch.nn.init.xavier_uniform_(m.weight)
            if isinstance(m, nn.Linear) and m.bias is not None:
                nn.init.constant_(m.bias, 0)
        elif isinstance(m, nn.LayerNorm):
            nn.init.constant_(m.bias, 0)
            nn.init.constant_(m.weight, 1.0)

    def patchify(self, imgs):
        """
        imgs: (N, 3, H, W)
        x: (N, L, patch_size**2 *3)
        """
        p = self.patch_embed.patch_size[0]
        assert imgs.shape[2] == imgs.shape[3] and imgs.shape[2] % p == 0

        h = w = imgs.shape[2] // p
        x = imgs.reshape(shape=(imgs.shape[0], 3, h, p, w, p))
        x = torch.einsum('nchpwq->nhwpqc', x)
        x = x.reshape(shape=(imgs.shape[0], h * w, p**2 * 3))
        return x

    def unpatchify(self, x):
        """
        x: (N, L, patch_size**2 *3)
        imgs: (N, 3, H, W)
        """
        p = self.patch_embed.patch_size[0]
        h = w = int(x.shape[1]**.5)
        assert h * w == x.shape[1]

        x = x.reshape(shape=(x.shape[0], h, w, p, p, 3))
        x = torch.einsum('nhwpqc->nchpwq', x)
        imgs = x.reshape(shape=(x.shape[0], 3, h * p, h * p))
        return imgs

    def random_masking(self, x, mask_ratio):
        """
        Perform per-sample random masking by per-sample shuffling.
        Per-sample shuffling is done by argsort random noise.
        x: [N, L, D], sequence
        """
        N, L, D = x.shape  # batch, length, dim
        len_keep = int(L * (1 - mask_ratio))

        noise = torch.rand(N, L, device=x.device)  # noise in [0, 1]

        # sort noise for each sample
        # ascend: small is keep, large is remove
        ids_shuffle = torch.argsort(noise, dim=1)
        ids_restore = torch.argsort(ids_shuffle, dim=1)

        # keep the first subset
        ids_keep = ids_shuffle[:, :len_keep]
        x_masked = torch.gather(
            x, dim=1, index=ids_keep.unsqueeze(-1).repeat(1, 1, D))

        # generate the binary mask: 0 is keep, 1 is remove
        mask = torch.ones([N, L], device=x.device)
        mask[:, :len_keep] = 0
        # unshuffle to get the binary mask
        mask = torch.gather(mask, dim=1, index=ids_restore)

        return x_masked, mask, ids_restore

    def prepare_token(self, B):
        # append cls token
        if self.mlp_token:
            cls_token = self.cls_token_mlp(self.cls_token)
            cls_token = cls_token + self.pos_embed[:, :self.num_tokens, :]
        else:
            cls_token = self.cls_token + self.pos_embed[:, :self.num_tokens, :]

        cls_token = cls_token.expand(B, -1, -1)

        return cls_token

    def forward_encoder(self, x, mask_ratio, cls_tokens):
        # embed patches
        x = self.patch_embed(x)

        # add pos embed w/o cls token
        x = x + self.pos_embed[:, 1:, :]

        # masking: length -> length * mask_ratio
        x, mask, ids_restore = self.random_masking(x, mask_ratio)

        x = torch.cat((cls_tokens, x), dim=1)

        # apply Transformer blocks
        x_set = []
        for i in range(len(self.blocks)):
            blx = self.blocks[i]
            x = blx(x)
            if i != len(self.blocks)-1:
                x_set.append(x)
        x = self.norm(x)
        x_set.append(x)

        return x_set, mask, ids_restore

    def forward_loss_feature(self, target, pred, mask):
        """
        target: [N, L, C]
        pred: [N, L, C]
        mask: [N, L], 0 is keep, 1 is remove, 
        """
        mean = target.mean(dim=1, keepdim=True)
        var = target.var(dim=1, keepdim=True)
        target = (target - mean) / (var + 1.e-6)**.5
        loss = (pred - target) ** 2
        loss = loss.mean(dim=-1)  # [N, L], mean loss per patch

        loss = (loss * mask).sum() / mask.sum()  # mean loss on removed patches
        return loss

    def forward_loss_att(self, target, pred, mask):
        """
        imgs: [N, 3, H, W]
        pred: [N, L, p*p*3]
        mask: [N, L], 0 is keep, 1 is remove, 
        """
        N, L = mask.shape
        target = torch.nn.functional.softmax(target/self.att_tau, dim=-1)
        # n l l c
        loss = torch.sum(-target *
                         torch.nn.functional.log_softmax(pred, dim=-1), dim=-1).mean()
        return loss

    def forward(self, imgs, mask_ratio=0.75, targets=None):
        loss_set = {}

        B = imgs.shape[0]
        cls_tokens_large = self.prepare_token(B)

        latent_set, mask, ids_restore = self.forward_encoder(
            imgs, mask_ratio, cls_tokens_large)
        latent = torch.cat(latent_set, dim=-1)
        adap_latent = self.encoder_adaptor(latent)

        pred = self.forward_fea_decoder(
            adap_latent, ids_restore,
            self.decoder_pos_embed, self.decoder_blocks
        )
        loss_large = self.forward_loss_feature(targets[0], pred, mask)
        loss_set['loss_fea'] = [loss_large, 1.0]

        if self.pred_att:
            att = self.forward_att_decoder(
                adap_latent, ids_restore, targets[2],
                self.decoder_pos_embed, self.decoder_blocks
            )
            loss_large_att = self.forward_loss_att(targets[1], att, mask)
            loss_set['loss_att'] = [loss_large_att, 1.0]

        return loss_set, None, None

def mae_vit_base_patch16(**kwargs):
    model = TECViT(
        patch_size=16, embed_dim=768, depth=12, num_heads=12,
        decoder_embed_dim=512, decoder_depth=2, decoder_num_heads=16,
        mlp_ratio=4, norm_layer=partial(nn.LayerNorm, eps=1e-6), **kwargs)
    return model

def mae_vit_large_patch16(**kwargs):
    model = TECViT(
        patch_size=16, embed_dim=1024, depth=24, num_heads=16,
        decoder_embed_dim=768, decoder_depth=2, decoder_num_heads=16,
        mlp_ratio=4, norm_layer=partial(nn.LayerNorm, eps=1e-6), **kwargs)
    return model
