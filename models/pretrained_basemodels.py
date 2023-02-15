# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.

# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.
# --------------------------------------------------------
# References:
# timm: https://github.com/rwightman/pytorch-image-models/tree/master/timm
# DeiT: https://github.com/facebookresearch/deit
# --------------------------------------------------------

import torch
import torch.nn as nn

from models.basemodel_vit import vit_base_patch16, vit_large_patch16


def get_pretrained_path(model_name):
    model_pretrain_dict = {}
    model_pretrain_dict['mae1klarge'] = "../../pro/mae_pretrain_vit_large.pth"
    model_pretrain_dict['mae1kbase'] = "../../pro/mae_pretrain_vit_base.pth"
    model_pretrain_dict['mae1k300ep'] = "../../pro/mae_pretrain_vit_base_300ep.pth"
    model_pretrain_dict['ibot1kbase'] = "../../pro/ibot-vit-base-teacher.pth"

    return model_pretrain_dict[model_name]


class PretrainedBaseModel(nn.Module):
    """ Load the pretrained SSL model as the base model for TEC pretraining.
    """

    def __init__(self, pred_att=True, basemodel='mae1k', last_layers=2, topkatt=None):
        super().__init__()
        print("loading", basemodel)
        if 'mae1klarge' == basemodel:
            this_basemodel = vit_large_patch16(
                mlp_token=False, num_tokens=1, pred_att=pred_att, last_layers=last_layers, topkatt=topkatt)
        elif 'mae1kbase' == basemodel or 'ibot1kbase' == basemodel or 'mae1k300ep' == basemodel:
            this_basemodel = vit_base_patch16(
                mlp_token=False, num_tokens=1, pred_att=pred_att, last_layers=last_layers, topkatt=topkatt)
        else:
            print("Not implemented now.")
            exit()
        pretrain_path = get_pretrained_path(basemodel)
        this_pretrain = self.get_pretrain(pretrain_path)
        msg = this_basemodel.load_state_dict(this_pretrain, strict=False)
        print(msg)
        self.basemodel = this_basemodel

    def get_pretrain(self, pretrain_path):
        checkpoint = torch.load(pretrain_path, map_location='cpu')
        print("Load pre-trained checkpoint from: %s" % pretrain_path)
        if 'model' in checkpoint.keys():
            checkpoint_model = checkpoint['model']
        elif 'state_dict' in checkpoint.keys():
            checkpoint_model = checkpoint['state_dict']
        elif 'student' in checkpoint.keys():
            checkpoint_model = checkpoint['student']
            from collections import OrderedDict
            new_state_dict = OrderedDict()
            for k, v in checkpoint_model.items():
                name = k[16:]
                new_state_dict[name] = v
            checkpoint_model = new_state_dict
        else:
            checkpoint_model = checkpoint
        return checkpoint_model

    def forward(self, imgs):
        targets = []
        targets = self.basemodel.forward_features(imgs)
        return targets
