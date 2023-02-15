# Towards Sustainable Self-supervised Learning

**Target-Enhanced Conditional (TEC) pretraining: a Faster and Stronger self-supervised pretraining method.**

The official implementation of paper: [Towards Sustainable Self-supervised Learning](https://arxiv.org/abs/2210.11016)

## Introduction

Although increasingly training-expensive, most self-supervised learning (SSL) models have repeatedly been trained from scratch but not fully utilized, since only a few SOTAs are employed for downstream tasks. In this work, we explore a sustainable SSL framework with  two major challenges: i) learning a stronger new SSL model based on the existing pretrained SSL model, also called as base model,  in a cost-friendly manner, ii) allowing the training of the new model to be compatible with various base models. We propose a Target-Enhanced Conditional (TEC) scheme which introduces two components to the existing mask-reconstruction based SSL. Firstly, we propose patch-relation enhanced targets which enhances the target given by base model and  encourages
the new model to learn semantic-relation knowledge from the base model by using  incomplete inputs.
This hardening and target-enhancing help the new model surpass the base model, since they enforce additional patch relation modeling to handle incomplete input. Secondly, we introduce a conditional adapter that adaptively adjusts new model prediction to align with the target of different base models.
Extensive experimental results show  that our TEC scheme can accelerate the learning speed, and also improve SOTA SSL base models, e.g. MAE and iBOT, taking an explorative step towards sustainable SSL.

# TEC Performance

| Method | Network |Base model| Pretrain data | Epoch | Top 1 acc. |
|------|------|--------|---------|-------|------------|
|TEC|ViT-B| MAE 300ep |ImageNet-1k|100|83.9|
|TEC|ViT-B| MAE 300ep |ImageNet-1k|100|84.3|
|TEC|ViT-B| MAE       |ImageNet-1k|300|84.7|
|TEC|ViT-B| MAE       |ImageNet-1k|800|84.8|
|TEC|ViT-B| iBoT      |ImageNet-1k|300|84.8|
|TEC|ViT-B| iBoT      |ImageNet-1k|800|85.1|
|TEC|ViT-L| MAE       |ImageNet-1k|300|86.5|

# Training

## Requirement

- timm==0.3.2 pytorch 1.8.1

- Download the pretrained SSL model of MAE/iBoT from the official repo. Change the pretrained model path in `models/pretrained_basemodels.py` file.

## Pretraining and Finetuning

<details>
  <summary>MAE-ViT-B basemodel and ViT-B newmodel. </summary>

pretraining:

```shell
python -m torch.distributed.launch --nproc_per_node=8 --use_env main_pretrain.py \
--mlp_token \
--pred_att \
--topkatt 15 \
--att_tau 1.8 \
--basemodel mae1kbase \
--model mae_vit_base_patch16 \
--last_layers 2 \
--batch_size 256 \
--mask_ratio 0.75 \
--epochs 800 \
--warmup_epochs 40 \
--blr 1.5e-4 --weight_decay 0.05 \
--accum_iter 2 \
--data_path /dataset/imagenet-raw \
--output_dir output_dir; \
```

finetuning:

```shell
python -m torch.distributed.launch --nproc_per_node=8 main_finetune.py \
--mlp_token \
--accum_iter 1 \
--batch_size 128 \
--model vit_base_patch16 \
--finetune  output_dir/checkpoint-799.pth \
--epochs 100 \
--blr 5e-4 --layer_decay 0.55 \
--warmup_epochs 20 \
--weight_decay 0.05 --drop_path 0.1 --mixup 0.8 --cutmix 1.0 --reprob 0.25 \
--dist_eval --data_path /dataset/imagenet-raw \
--output_dir output_dir_finetune; \
```

</details>

<details>
  <summary>iBoT-ViT-B basemodel and ViT-B newmodel. </summary>

pretraining:

```shell
python -m torch.distributed.launch --nproc_per_node=8 --use_env main_pretrain.py \
--mlp_token \
--pred_att \
--topkatt 9 \
--att_tau 1.0 \
--basemodel ibot1kbase \
--model mae_vit_base_patch16 \
--last_layers 2 \
--batch_size 256 \
--mask_ratio 0.75 \
--epochs 800 \
--warmup_epochs 40 \
--blr 1.5e-4 --weight_decay 0.05 \
--accum_iter 2 \
--data_path /dataset/imagenet-raw \
--output_dir output_dir; \
```

finetuning:

```shell
python -m torch.distributed.launch --nproc_per_node=8 main_finetune.py \
--mlp_token \
--accum_iter 1 \
--batch_size 128 \
--model vit_base_patch16 \
--finetune  output_dir/checkpoint-799.pth \
--epochs 100 \
--blr 5e-4 --layer_decay 0.55 \
--warmup_epochs 20 \
--weight_decay 0.05 --drop_path 0.1 --mixup 0.8 --cutmix 1.0 --reprob 0.25 \
--dist_eval --data_path /dataset/imagenet-raw \
--output_dir output_dir_finetune; \
```

</details>

<details>
  <summary>MAE-ViT-L basemodel and ViT-L newmodel. </summary>

pretraining:

```shell
python -m torch.distributed.launch --nproc_per_node=8 --use_env main_pretrain.py \
--mlp_token \
--pred_att \
--topkatt 15 \
--att_tau 1.4 \
--basemodel mae1klarge \
--model mae_vit_large_patch16 \
--last_layers 2 \
--batch_size 256 \
--mask_ratio 0.75 \
--epochs 300 \
--warmup_epochs 40 \
--blr 1.5e-4 --weight_decay 0.05 \
--accum_iter 2 \
--data_path /dataset/imagenet-raw \
--output_dir output_dir; \
```

finetuning:

```shell
python -m torch.distributed.launch --nproc_per_node=8 main_finetune.py \
--mlp_token \
--accum_iter 2 \
--batch_size 64 \
--model vit_large_patch16 \
--finetune  output_dir/checkpoint-299.pth \
--epochs 50 \
--blr 1e-3 --layer_decay 0.65 \
--min_lr 1e-5 \
--warmup_epochs 5 \
--weight_decay 0.05 --drop_path 0.1 --mixup 0.8 --cutmix 1.0 --reprob 0.25 \
--dist_eval --data_path /dataset/imagenet-raw \
--output_dir output_dir_finetune; \
```

</details>

## Convert checkpoint

To use the pretrained checkpoint for downstream tasks, you need to convert the checkpoint as follows:

<details>
  <summary>Convert checkpoint. </summary>

```shell
python weight_convert.py \
--mlp_token \
--model vit_base_patch16 \
--resume path_to_pretrained_model  \
--output_dir output_dir_convert \
--ckptname output_ckpt_name.pth
```

</details>

# Citation
```
@article{gao2022towards,
  title={Towards Sustainable Self-supervised Learning},
  author={Gao, Shanghua and Zhou, Pan and Cheng, Ming-Ming and Yan, Shuicheng},
  journal={arXiv preprint arXiv:2210.11016},
  year={2022}
}
```

# Acknowledgement

This codebase is build based on the [MAE codebase](https://github.com/facebookresearch/mae). Thanks!