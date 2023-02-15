# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.

# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.
# --------------------------------------------------------
# References:
# DeiT: https://github.com/facebookresearch/deit
# BEiT: https://github.com/microsoft/unilm/tree/master/beit
# --------------------------------------------------------
import models.models_vit as models_vit

import util.misc as misc
import argparse
import numpy as np
import os
from pathlib import Path

import torch
import torch.backends.cudnn as cudnn


def get_args_parser():
    parser = argparse.ArgumentParser(
        'Convert TEC pre-training model for downstream tasks', add_help=False)

    # Model parameters
    parser.add_argument('--model', default='vit_base_patch16', type=str, metavar='MODEL',
                        help='Name of model to train')

    parser.add_argument('--input_size', default=224, type=int,
                        help='images input size')

    parser.add_argument('--mask_ratio', default=0.75, type=float,
                        help='Masking ratio (percentage of removed patches).')

    # Dataset parameters
    parser.add_argument('--output_dir', default='./output_dir_convert',
                        help='path where to save, empty for no saving')
    parser.add_argument('--log_dir', default='./output_dir_convert',
                        help='path where to tensorboard log')
    parser.add_argument('--device', default='cuda',
                        help='device to use for training / testing')
    parser.add_argument('--seed', default=0, type=int)
    parser.add_argument('--resume', default='',
                        help='resume from checkpoint')
    parser.add_argument('--ckptname', default='tec_checkpoint.pth',
                        help='resume from checkpoint')

    # distributed training parameters
    parser.add_argument('--world_size', default=1, type=int,
                        help='number of distributed processes')
    parser.add_argument('--local_rank', default=-1, type=int)
    parser.add_argument('--dist_on_itp', action='store_true')
    parser.add_argument('--dist_url', default='env://',
                        help='url used to set up distributed training')
    parser.add_argument(
        "--rank", default=0, type=int, help="""rank for distrbuted training."""
    )

    # tec
    parser.add_argument('--mlp_token', action='store_true',
                        help='Enable the input adaptor by using mlp enhanced cls token')

    return parser


def main(args):
    misc.init_distributed_mode(args)
    # misc.init_distributed_ddpjob(args)

    print('job dir: {}'.format(os.path.dirname(os.path.realpath(__file__))))
    print("{}".format(args).replace(', ', ',\n'))

    device = torch.device(args.device)

    # fix the seed for reproducibility
    seed = args.seed + misc.get_rank()
    torch.manual_seed(seed)
    np.random.seed(seed)

    cudnn.benchmark = True

    # define the model
    model = models_vit.__dict__[args.model](
        num_classes=1000,
        drop_path_rate=0.0,
        global_pool=True,
        mlp_token=args.mlp_token,
    )
    model.to(device)

    if args.resume is not None:
        checkpoint = torch.load(args.resume, map_location='cpu')
        print("Load pre-trained checkpoint from: %s" % args.resume)
        checkpoint_model = checkpoint['model']
        msg = model.load_state_dict(checkpoint_model, strict=False)
        print(msg)
    else:
        print("no ckpt!")
        exit()

    rep_cls_token = model.rep_token()
    model.cls_token.data = rep_cls_token.data
    del model.cls_token_mlp
    print("merge cls token done.")
    checkpoint_model = model.state_dict()
    for k in ['head.weight', 'head.bias', 'fc_norm.weight', 'fc_norm.bias']:
        print(f"Removing key {k} from pretrained checkpoint")
        del checkpoint_model[k]
    print(dict(checkpoint_model).keys())
    torch.save(checkpoint_model, os.path.join(
        args.output_dir, args.ckptname))
    print("save to", os.path.join(args.output_dir, args.ckptname))


if __name__ == '__main__':
    args = get_args_parser()
    args = args.parse_args()
    if args.output_dir:
        Path(args.output_dir).mkdir(parents=True, exist_ok=True)
    main(args)
