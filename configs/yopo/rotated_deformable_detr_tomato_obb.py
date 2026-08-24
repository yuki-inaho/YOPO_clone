# Copyright (c) OpenMMLab. All rights reserved.
# Tomato 2D OBB (rotated box) on the shared DeformableDETR transformer with a
# LIGHTWEIGHT HGNetV2-B2 backbone (YOPO-native, pretrained PPHGNetV2_B2).
# RGB branch + 2D OBB output only; 3D pose output is NOT used.
#
# Loss: rotate IoU (RotatedIoULoss) by default; switch `loss_iou` to
# `GDLoss(loss_type='gwd')` for the GWD variant (see
# rotated_deformable_detr_tomato_obb_gwd.py).
#
# Run:
#   .venv/bin/python tools/train.py configs/yopo/rotated_deformable_detr_tomato_obb.py \
#       --work-dir work_dirs/rddetr_tomato_riou_smoke \
#       --cfg-options train_cfg.max_epochs=1

from mmengine.config import read_base

with read_base():
    from yopo.configs._base_.default_runtime import *  # noqa: F401,F403

from mmcv.transforms import LoadImageFromFile
from mmengine.optim.scheduler import MultiStepLR
from mmengine.runner.loops import EpochBasedTrainLoop

from yopo.datasets import DOTATomatoDataset
from yopo.datasets.transforms import (LoadAnnotations, PackDetInputs,
                                      RandomFlip)
from yopo.models.backbones import HGNetV2
from yopo.models.data_preprocessors import DetDataPreprocessor
from yopo.models.dense_heads import RotatedDeformableDETRHead
from yopo.models.detectors import DeformableDETR
from yopo.models.losses import FocalLoss, L1Loss, RotatedIoULoss
from yopo.models.necks import ChannelMapper
from yopo.models.task_modules import (FocalLossCost, GDCost, HungarianAssigner,
                                      RBoxL1Cost)

# ── dataset ──────────────────────────────────────────────────────────────────
data_root = ('/workspace/data/tomato_obb_detection/'
             'fruits_detection_data_Jun30-2025_dota/')
backend_args = None

model = dict(
    type=DeformableDETR,
    num_queries=100,
    num_feature_levels=4,
    with_box_refine=False,
    as_two_stage=False,
    data_preprocessor=dict(
        type=DetDataPreprocessor,
        mean=[123.675, 116.28, 103.53],
        std=[58.395, 57.12, 57.375],
        bgr_to_rgb=True,
        pad_size_divisor=1,
        boxtype2tensor=False),
    backbone=dict(
        type=HGNetV2,
        name='B2',
        in_channels=3,
        return_idx=[1, 2, 3],
        freeze_at=0,
        freeze_norm=False,
        init_cfg=dict(
            type='Pretrained',
            checkpoint='/home/kasm-user/.cache/torch/hub/checkpoints/'
            'PPHGNetV2_B2_stage1.pth')),
    neck=dict(
        type=ChannelMapper,
        in_channels=[384, 768, 1536],
        kernel_size=1,
        out_channels=256,
        act_cfg=None,
        norm_cfg=dict(type='GN', num_groups=32),
        num_outs=4),
    encoder=dict(
        num_layers=4,
        layer_cfg=dict(
            self_attn_cfg=dict(embed_dims=256, batch_first=True),
            ffn_cfg=dict(
                embed_dims=256, feedforward_channels=1024, ffn_drop=0.1))),
    decoder=dict(
        num_layers=4,
        return_intermediate=True,
        layer_cfg=dict(
            self_attn_cfg=dict(
                embed_dims=256, num_heads=8, dropout=0.1, batch_first=True),
            cross_attn_cfg=dict(embed_dims=256, batch_first=True),
            ffn_cfg=dict(
                embed_dims=256, feedforward_channels=1024, ffn_drop=0.1)),
        post_norm_cfg=None),
    positional_encoding=dict(num_feats=128, normalize=True, offset=-0.5),
    bbox_head=dict(
        type=RotatedDeformableDETRHead,
        num_classes=1,
        angle_cfg=dict(width_longer=True, start_angle=0),
        angle_factor=3.141592653589793,
        sync_cls_avg_factor=True,
        loss_cls=dict(
            type=FocalLoss,
            use_sigmoid=True,
            gamma=2.0,
            alpha=0.25,
            loss_weight=2.0),
        loss_bbox=dict(type=L1Loss, loss_weight=5.0),
        loss_iou=dict(type=RotatedIoULoss, loss_weight=2.0, mode='log')),
    # training and testing settings
    train_cfg=dict(
        assigner=dict(
            type=HungarianAssigner,
            match_costs=[
                dict(type=FocalLossCost, weight=2.0),
                dict(
                    type=RBoxL1Cost,
                    weight=5.0,
                    box_format='xywha',
                    angle_factor=3.141592653589793),
                dict(
                    type=GDCost,
                    loss_type='kld',
                    fun='log1p',
                    tau=1,
                    sqrt=False,
                    weight=2.0)
            ])),
    test_cfg=dict(max_per_img=100))

train_pipeline = [
    dict(type=LoadImageFromFile, backend_args=backend_args),
    dict(type=LoadAnnotations, with_bbox=True, box_type='rbox'),
    dict(
        type=RandomFlip,
        prob=0.5,
        direction=['horizontal', 'vertical', 'diagonal']),
    dict(type=PackDetInputs)
]
train_dataloader = dict(
    batch_size=8,
    num_workers=4,
    persistent_workers=True,
    sampler=dict(type='DefaultSampler', shuffle=True),
    dataset=dict(
        type=DOTATomatoDataset,
        data_root=data_root,
        ann_file=data_root + 'labels/train',
        data_prefix=dict(img_path=data_root + 'images/train'),
        filter_cfg=dict(filter_empty_gt=True),
        pipeline=train_pipeline))

# Muon for matrix parameters + ScheduleFree AdamW for vector parameters.
# Fully-qualified paths are required because this config intentionally has
# default_scope=None; short custom registry names would resolve to None in
# MMEngine's DefaultOptimWrapperConstructor.
optim_wrapper = dict(
    type='yopo.engine.optimizers.deim_optimizers.ScheduleFreeOptimWrapper',
    constructor='DefaultOptimWrapperConstructor',
    optimizer=dict(
        type=(
            'yopo.engine.optimizers.deim_optimizers.'
            'MuonScheduleFreeOptimizer'),
        muon_lr=0.005,
        muon_weight_decay=0.01,
        momentum=0.95,
        nesterov=True,
        ns_steps=5,
        sf_lr=0.00025,
        sf_betas=[0.9, 0.95],
        sf_eps=1e-8,
        sf_weight_decay=0.000125),
    clip_grad=dict(max_norm=0.1, norm_type=2))

# learning policy
max_epochs = 50
train_cfg = dict(
    type=EpochBasedTrainLoop, max_epochs=max_epochs, val_interval=50)
val_cfg = None
test_cfg = None
val_dataloader = None
val_evaluator = None
test_dataloader = None
test_evaluator = None

param_scheduler = [
    dict(
        type=MultiStepLR,
        begin=0,
        end=max_epochs,
        by_epoch=True,
        milestones=[40],
        gamma=0.1)
]

auto_scale_lr = dict(base_batch_size=32)

load_from = None
resume = False
