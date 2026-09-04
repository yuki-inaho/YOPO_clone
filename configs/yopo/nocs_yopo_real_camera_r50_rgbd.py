_base_ = [
    './_base_/datasets/nocs_dataset.py',
    '../_base_/default_runtime.py'
]

# RGB-D (4-channel) variant: depth is concatenated to RGB as a 4th channel.
# The ResNet conv1 is built with in_channels=4; the pretrained dino/resnet
# weights are loaded non-strictly (conv1 shape mismatch is skipped).

load_from = 'https://download.openmmlab.com/mmdetection/v3.0/dino/dino-4scale_r50_improved_8xb2-12e_coco/dino-4scale_r50_improved_8xb2-12e_coco_20230818_162607-6f47a913.pth'


model = dict(
    type='DINO9DCenter2DPose',
    num_queries=100,  # num_matching_queries
    with_box_refine=True,
    as_two_stage=True,
    data_preprocessor=dict(
        type='DetDataPreprocessor',
        mean=[123.675, 116.28, 103.53, 127.5],
        std=[58.395, 57.12, 57.375, 127.5],
        bgr_to_rgb=False,
        pad_size_divisor=1),
    backbone=dict(
        type='ResNet',
        depth=50,
        in_channels=4,
        num_stages=4,
        out_indices=(1, 2, 3),
        frozen_stages=1,
        norm_cfg=dict(type='BN', requires_grad=False),
        norm_eval=True,
        style='pytorch',
        init_cfg=dict(type='Pretrained', checkpoint='torchvision://resnet50')),
    neck=dict(
        type='ChannelMapper',
        in_channels=[512, 1024, 2048],
        kernel_size=1,
        out_channels=256,
        act_cfg=None,
        norm_cfg=dict(type='GN', num_groups=32),
        num_outs=4),
    encoder=dict(
        num_layers=6,
        layer_cfg=dict(
            self_attn_cfg=dict(embed_dims=256, num_levels=4,
                               dropout=0.0),  # 0.1 for DeformDETR
            ffn_cfg=dict(
                embed_dims=256,
                feedforward_channels=2048,  # 1024 for DeformDETR
                ffn_drop=0.0))),  # 0.1 for DeformDETR
    decoder=dict(
        num_layers=6,
        return_intermediate=True,
        layer_cfg=dict(
            self_attn_cfg=dict(embed_dims=256, num_heads=8,
                               dropout=0.0),  # 0.1 for DeformDETR
            cross_attn_cfg=dict(embed_dims=256, num_levels=4,
                                dropout=0.0),  # 0.1 for DeformDETR
            ffn_cfg=dict(
                embed_dims=256,
                feedforward_channels=2048,  # 1024 for DeformDETR
                ffn_drop=0.0)),  # 0.1 for DeformDETR
        post_norm_cfg=None),
    positional_encoding=dict(
        num_feats=128,
        normalize=True,
        offset=0.0,  # -0.5 for DeformDETR
        temperature=20),  # 10000 for DeformDETR
    bbox_head=dict(
        type='DINO9DCenter2DPoseHead',
        num_classes=6,
        use_bbox_for_z=True,
        classwise_rotation=True,
        classwise_sizes=True,
        sync_cls_avg_factor=True,
        loss_cls=dict(
            type='FocalLoss',
            use_sigmoid=True,
            gamma=2.0,
            alpha=0.25,
            loss_weight=1.0),  # 1.0 in DeformDETR
        loss_bbox=dict(type='L1Loss', loss_weight=5.0),
        loss_iou=dict(type='GIoULoss', loss_weight=2.0),
        loss_centers_2d = dict(type='L1PoseLoss', loss_weight=5.0),
        loss_z = dict(type='L2PoseLoss', loss_weight=50.0),
        loss_rotation = dict(
            type='Rotation3DLoss',
            loss_weight=5.0,
            symmetric_classes=[0, 1, 3],
            num_angles=16),
        loss_sizes = dict(type='L2PoseLoss', loss_weight=50.0),
        ),
    dn_cfg=dict(  # TODO: Move to model.train_cfg ?
        label_noise_scale=0.5,
        box_noise_scale=1.0,  # 0.4 fo DN-DETR
        group_cfg=dict(dynamic=True, num_groups=None,
                       num_dn_queries=20)),  # TODO: half num_dn_queries
    # training and testing settings
    train_cfg=dict(
        assigner=dict(
            type='HungarianAssigner',
            match_costs=[
                dict(type='FocalLossCost', weight=2.0),
                dict(type='BBoxL1Cost', weight=5.0, box_format='xywh'),
                dict(type='IoUCost', iou_mode='giou', weight=2.0),
                dict(type='TranslationCost', weight=5.0),
                dict(
                    type='RotationCost',
                    symmetric_classes=[0, 1, 3],
                    weight=2.0),
            ])),
    test_cfg=dict(max_per_img=300))  # 100 for DeformDETR


# optimizer: AMUSE (official kjeiun/amuse update rule; schedule-free)
optim_wrapper = dict(
    type='yopo.engine.optimizers.amuse.AmuseOptimWrapper',
    optimizer=dict(
        type='yopo.engine.optimizers.amuse.AmuseOptimizer',
        lr=0.0001,  # 0.0002 for DeformDETR
        aux_lr=0.0001,
        warmup_steps=100,
        weight_decay=0.0001),
    clip_grad=dict(max_norm=0.1, norm_type=2),
    paramwise_cfg=dict(custom_keys={'backbone': dict(lr_mult=0.1)})
)  # custom_keys contains sampling_offsets and reference_points in DeformDETR  # noqa

# learning policy
max_epochs = 12
train_cfg = dict(
    type='EpochBasedTrainLoop', max_epochs=max_epochs, val_interval=1)

val_cfg = dict(type='ValLoop')
test_cfg = dict(type='TestLoop')

# AMUSE is schedule-free; do not attach an external LR scheduler.
param_scheduler = []

custom_hooks = [dict(type='ScheduleFreeOptimizerModeHook')]

# NOTE: `auto_scale_lr` is for automatically scaling LR,
# USER SHOULD NOT CHANGE ITS VALUES.
# base_batch_size = (8 GPUs) x (2 samples per GPU)
auto_scale_lr = dict(base_batch_size=16, enable=False)

visualizer = dict(
    type='PoseLocalVisualizer')


backend_args = None
scale = (640, 480)
dataset_type = 'NOCSDataset'
data_root = 'data/nocs/'
train_pipeline = [
    dict(type='LoadImageFromFile', backend_args=backend_args),
    dict(type='LoadDepthImageFromFile', backend_args=backend_args,
         to_float32=True),
    dict(type='Load9DPoseAnnotations', with_bbox=True,
         with_centers_2d=True, with_z=True),
    dict(type='ConcatDepthToImage'),
    dict(type='Resize', scale=scale, keep_ratio=True),
    dict(type='YOLOXHSVRandomAug'),
    dict(type='RandomTranslatePixels', prob=0.5, max_translate_offset=50),
    dict(type='RandomFlipFor9DPose', prob=0.5),
    dict(type='FilterAnnotations', min_gt_bbox_wh=(1e-2, 1e-2)),
    dict(type='Pack9DPoseInputs')
]
test_pipeline = [
    dict(type='LoadImageFromFile', backend_args=backend_args),
    dict(type='LoadDepthImageFromFile', backend_args=backend_args,
         to_float32=True),
    # If you don't have a gt annotation, delete the pipeline
    dict(type='Load9DPoseAnnotations', with_bbox=True,
         with_centers_2d=True, with_z=True),
    dict(type='ConcatDepthToImage'),
    dict(type='Resize', scale=scale, keep_ratio=True),
    dict(
        type='Pack9DPoseInputs',
        meta_keys=('img_id', 'img_path', 'ori_shape', 'img_shape',
                   'scale_factor', 'intrinsic', 'models_info_path'))
]

train_dataloader = dict(
    batch_size=16,
    num_workers=8,
    persistent_workers=True,
    sampler=dict(type='DefaultSampler', shuffle=True),
    dataset=dict(
        type='ConcatDataset',
        datasets=[
            dict(
                type=dataset_type,
                data_root=data_root,
                split='camera_train',
                pipeline=train_pipeline,
                backend_args=backend_args),
            dict(
                type=dataset_type,
                data_root=data_root,
                split='real_train',
                pipeline=train_pipeline,
                backend_args=backend_args)
        ])
)
val_dataloader = dict(
    batch_size=1,
    num_workers=2,
    persistent_workers=True,
    drop_last=False,
    sampler=dict(type='DefaultSampler', shuffle=False),
    dataset=dict(
        type=dataset_type,
        data_root=data_root,
        split='real_test',
        test_mode=True,
        pipeline=test_pipeline))
test_dataloader = val_dataloader
