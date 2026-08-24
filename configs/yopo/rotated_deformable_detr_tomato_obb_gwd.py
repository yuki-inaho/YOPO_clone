# Copyright (c) OpenMMLab. All rights reserved.
# Tomato 2D OBB (rotated box) on the shared DeformableDETR transformer with a
# LIGHTWEIGHT HGNetV2-B2 backbone (YOPO-native, pretrained PPHGNetV2_B2).
# RGB branch + 2D OBB output only; 3D pose output is NOT used.
#
# Loss: GWD (Gaussian Wasserstein Distance) by default. For the rotate-IoU
# variant change `loss_iou` to RotatedIoULoss (see the *_riou.py config).
#
# Run:
#   .venv/bin/python tools/train.py configs/yopo/rotated_deformable_detr_tomato_obb_gwd.py \
#       --work-dir work_dirs/rddetr_tomato_gwd

from mmengine.config import read_base

with read_base():
    from yopo.configs._base_.default_runtime import *  # noqa: F401,F403

from mmcv.transforms import LoadImageFromFile
from mmengine.optim.scheduler import MultiStepLR
from mmengine.runner.loops import EpochBasedTrainLoop

from yopo.datasets import DOTATomatoDataset
from yopo.datasets.transforms import (LoadAnnotations, PackDetInputs,
                                      RandomFlip, Resize)
from yopo.evaluation.metrics import RotatedIoUMetric
from yopo.models.backbones import HGNetV2
from yopo.models.data_preprocessors import DetDataPreprocessor
from yopo.models.dense_heads import RotatedDeformableDETRHead
from yopo.models.detectors import DeformableDETR
from yopo.models.losses import FocalLoss, GDLoss, L1Loss
from yopo.models.necks import ChannelMapper
from yopo.models.task_modules import (FocalLossCost, GDCost, HungarianAssigner,
                                      RBoxL1Cost)

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
        loss_iou=dict(
            type=GDLoss,
            loss_type='gwd',
            fun='log1p',
            tau=1,
            loss_weight=2.0)),
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
    batch_size=24,
    num_workers=8,
    persistent_workers=True,
    sampler=dict(type='DefaultSampler', shuffle=True),
    dataset=dict(
        type=DOTATomatoDataset,
        data_root=data_root,
        ann_file=data_root + 'labels/train',
        data_prefix=dict(img_path=data_root + 'images/train'),
        filter_cfg=dict(filter_empty_gt=True),
        pipeline=train_pipeline))

val_pipeline = [
    dict(type=LoadImageFromFile, backend_args=backend_args),
    dict(type=LoadAnnotations, with_bbox=True, box_type='rbox'),
    # Preserve native 736x512 geometry while providing scale_factor required
    # by rotated-box prediction postprocessing.
    dict(type=Resize, scale=(736, 512), keep_ratio=False),
    dict(type=PackDetInputs)
]
val_dataloader = dict(
    batch_size=8,
    num_workers=4,
    persistent_workers=True,
    sampler=dict(type='DefaultSampler', shuffle=False),
    dataset=dict(
        type=DOTATomatoDataset,
        data_root=data_root,
        ann_file=data_root + 'labels/valid',
        data_prefix=dict(img_path=data_root + 'images/valid'),
        test_mode=True,
        pipeline=val_pipeline))

# optimization: Muon (matrix params) + ScheduleFree (vector params), the
# DEIM_sandbox-style optimizer stack. fp16 AMP lowers activation memory while
# the wrapper keeps the ScheduleFree half in train mode before scaled updates.
optim_wrapper = dict(
    type='yopo.engine.optimizers.deim_optimizers.AmpScheduleFreeOptimWrapper',
    constructor='DefaultOptimWrapperConstructor',
    dtype='float16',
    loss_scale=1.0,
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

max_epochs = 50
train_cfg = dict(
    type=EpochBasedTrainLoop, max_epochs=max_epochs, val_interval=5)
val_cfg = dict(type='ValLoop')
val_evaluator = dict(
    type='yopo.evaluation.metrics.rotated_iou_metric.RotatedIoUMetric',
    iou_thr=0.5,
    score_thr=0.05,
    num_classes=1)
test_cfg = dict(type='TestLoop')
test_dataloader = val_dataloader
test_evaluator = val_evaluator

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

# Keep the two best real-rIoU validation checkpoints plus the final periodic
# checkpoint.  All are weights-only, so storage stays below roughly 600 MB.
default_hooks = dict(
    checkpoint=dict(
        type='yopo.engine.hooks.topk_checkpoint_hook.TopKCheckpointHook',
        interval=5,
        max_keep_ckpts=1,
        save_last=True,
        save_optimizer=False,
        topk=2,
        key_indicator='rbbox_mAP_50',
        rule='greater'))

load_from = None
resume = False
