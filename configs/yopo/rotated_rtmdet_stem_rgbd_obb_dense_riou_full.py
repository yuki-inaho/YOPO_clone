"""Dense RGB-D rotated RTMDet with an exact pretrained RGB feature path."""

_base_ = ['./rotated_deformable_detr_stem_rgbd_obb_riou_stage1.py']

custom_imports = dict(
    imports=[
        'yopo.datasets.dota_tomato',
        'yopo.datasets.transforms.raw_depth',
        'yopo.engine.hooks.rgbd_obb_transfer',
        'yopo.engine.optimizers.deim_optimizers',
        'yopo.evaluation.metrics.rotated_iou_metric',
        'yopo.models.backbones.dual_rgbd',
        'yopo.models.dense_heads.rotated_rtmdet_head',
    ],
    allow_failed_imports=False,
)

model = dict(
    _delete_=True,
    type='RTMDet',
    use_syncbn=False,
    data_preprocessor=dict(
        type='DetDataPreprocessor',
        mean=[123.675, 116.28, 103.53, 153.0],
        std=[58.395, 57.12, 57.375, 76.5],
        bgr_to_rgb=False,
        pad_size_divisor=1,
        boxtype2tensor=False,
        non_blocking=True,
    ),
    backbone=dict(
        type='RGBDResidualBackbone',
        beta_init=0.0,
        rgb_backbone=dict(
            type='HGNetV2',
            name='B2',
            in_channels=3,
            return_idx=[1, 2, 3],
            freeze_at=-1,
            freeze_norm=False,
            init_cfg=None,
        ),
        depth_backbone=dict(
            type='HGNetV2',
            name='B0',
            in_channels=1,
            return_idx=[1, 2, 3],
            freeze_at=-1,
            freeze_norm=False,
            init_cfg=dict(
                type='Pretrained',
                checkpoint=(
                    'work_dirs/rgbd3d_validmask_mae_fp32_smoke/'
                    'enc_b0_validmask_mae.pth'
                ),
            ),
        ),
    ),
    # Shape-identical to the RGB OBB checkpoint: no random pre-neck mapper.
    neck=dict(
        type='ChannelMapper',
        in_channels=[384, 768, 1536],
        kernel_size=1,
        out_channels=256,
        act_cfg=None,
        norm_cfg=dict(type='GN', num_groups=32),
        num_outs=4,
    ),
    bbox_head=dict(
        type='RotatedRTMDetSepBNHead',
        num_classes=1,
        in_channels=256,
        feat_channels=256,
        stacked_convs=2,
        strides=[8, 16, 32, 64],
        norm_cfg=dict(type='GN', num_groups=32),
        loss_cls=dict(
            type='QualityFocalLoss',
            use_sigmoid=True,
            beta=2.0,
            loss_weight=1.0,
        ),
        loss_bbox=dict(
            type='SmoothL1Loss',
            beta=1.0 / 9.0,
            loss_weight=1.0,
        ),
        loss_iou=dict(
            type='RotatedIoULoss',
            mode='linear',
            loss_weight=2.0,
        ),
    ),
    train_cfg=dict(
        assigner=dict(
            type='DynamicSoftLabelAssigner',
            soft_center_radius=3.0,
            topk=13,
            iou_weight=3.0,
            iou_calculator=dict(type='RBboxOverlaps2D'),
        ),
    ),
    test_cfg=dict(
        score_thr=0.05,
        nms_pre=2000,
        nms=dict(iou_threshold=0.1),
        max_per_img=300,
    ),
)

custom_hooks = [
    dict(
        type='RGBDFeatureTransferHook',
        checkpoint=(
            'work_dirs/rddetr_tomato_obb_corrected_gwd_stage2/'
            'selected_best.pth'
        ),
        report_filename='rgbd_feature_transfer_report.json',
    ),
]

max_epochs = 5
train_cfg = dict(type='EpochBasedTrainLoop', max_epochs=max_epochs, val_interval=1)
param_scheduler = [
    dict(
        type='MultiStepLR',
        begin=0,
        end=max_epochs,
        by_epoch=True,
        milestones=[4],
        gamma=0.1,
    ),
]
load_from = None
resume = False

