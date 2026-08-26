"""RT-DETR-style RGB-D OBB: dense O2M proposals plus O2O decoder."""

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
        'yopo.models.detectors.rotated_rt_detr',
    ],
    allow_failed_imports=False,
)

model = dict(
    _delete_=True,
    type='RotatedRTDETR',
    num_queries=256,
    num_feature_levels=4,
    with_box_refine=True,
    as_two_stage=True,
    dense_loss_weight=1.0,
    prediction_mode='query',
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
            type='HGNetV2', name='B2', in_channels=3,
            return_idx=[1, 2, 3], freeze_at=-1,
            freeze_norm=False, init_cfg=None,
        ),
        depth_backbone=dict(
            type='HGNetV2', name='B0', in_channels=1,
            return_idx=[1, 2, 3], freeze_at=-1,
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
    neck=dict(
        type='ChannelMapper',
        in_channels=[384, 768, 1536],
        kernel_size=1,
        out_channels=256,
        act_cfg=None,
        norm_cfg=dict(type='GN', num_groups=32),
        num_outs=4,
    ),
    auxiliary_dense_head=dict(
        type='RotatedRTMDetSepBNHead',
        num_classes=1,
        in_channels=256,
        feat_channels=256,
        stacked_convs=2,
        strides=[8, 16, 32, 64],
        norm_cfg=dict(type='GN', num_groups=32),
        loss_cls=dict(
            type='QualityFocalLoss', use_sigmoid=True,
            beta=2.0, loss_weight=1.0,
        ),
        loss_bbox=dict(
            type='SmoothL1Loss', beta=1.0 / 9.0, loss_weight=1.0,
        ),
        loss_iou=dict(
            type='RotatedIoULoss', mode='linear', loss_weight=2.0,
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
    ),
    encoder=dict(
        num_layers=4,
        layer_cfg=dict(
            self_attn_cfg=dict(embed_dims=256, batch_first=True),
            ffn_cfg=dict(
                embed_dims=256,
                feedforward_channels=1024,
                ffn_drop=0.1,
            ),
        ),
    ),
    decoder=dict(
        num_layers=4,
        return_intermediate=True,
        layer_cfg=dict(
            self_attn_cfg=dict(
                embed_dims=256, num_heads=8, dropout=0.1,
                batch_first=True,
            ),
            cross_attn_cfg=dict(embed_dims=256, batch_first=True),
            ffn_cfg=dict(
                embed_dims=256,
                feedforward_channels=1024,
                ffn_drop=0.1,
            ),
        ),
        post_norm_cfg=None,
    ),
    positional_encoding=dict(num_feats=128, normalize=True, offset=-0.5),
    bbox_head=dict(
        type='RotatedDeformableDETRHead',
        num_classes=1,
        angle_cfg=dict(width_longer=True, start_angle=0),
        angle_factor=3.141592653589793,
        sync_cls_avg_factor=True,
        loss_cls=dict(
            type='FocalLoss', use_sigmoid=True,
            gamma=2.0, alpha=0.25, loss_weight=2.0,
        ),
        loss_bbox=dict(type='SmoothL1Loss', beta=1.0 / 9.0, loss_weight=5.0),
        loss_iou=dict(
            type='RotatedIoULoss', mode='linear', loss_weight=2.0,
        ),
    ),
    train_cfg=dict(
        assigner=dict(
            type='HungarianAssigner',
            match_costs=[
                dict(type='FocalLossCost', weight=2.0),
                dict(
                    type='RBoxL1Cost', weight=5.0, box_format='xywha',
                    angle_factor=3.141592653589793,
                ),
                dict(
                    type='GDCost', loss_type='kld', fun='log1p',
                    tau=1, sqrt=False, weight=2.0,
                ),
            ],
        ),
    ),
    test_cfg=dict(max_per_img=256),
)

custom_hooks = [
    dict(
        type='RGBDFeatureTransferHook',
        checkpoint=(
            'work_dirs/rddetr_tomato_obb_corrected_gwd_stage2/'
            'selected_best.pth'
        ),
        report_filename='rgbd_feature_transfer_report.json',
        compatible_detector_roots=[
            'encoder', 'decoder', 'level_embed',
        ],
    ),
]

max_epochs = 5
train_cfg = dict(type='EpochBasedTrainLoop', max_epochs=max_epochs, val_interval=1)
param_scheduler = [
    dict(
        type='MultiStepLR', begin=0, end=max_epochs,
        by_epoch=True, milestones=[4], gamma=0.1,
    ),
]
load_from = None
resume = False
