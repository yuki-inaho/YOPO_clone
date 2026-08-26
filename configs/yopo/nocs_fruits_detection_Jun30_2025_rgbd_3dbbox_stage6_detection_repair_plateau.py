"""Repair 2D query ranking to saturation while retaining learned 3D pose.

The 1e-5 gate preserved 95.8% of Stage-4 IoU50 but did not move AP50.  This
continuation uses the 5e-5 rate that previously trained the same DINO 2D head
successfully.  Pose branches stay at effective 1e-6 and the RGB-D backbone at
1e-7.  Validation applies score filtering and 2D NMS once, then retains the
same query indices for every 3D field.
"""

_base_ = [
    './nocs_fruits_detection_Jun30_2025_rgbd_3dbbox_'
    'stage5_detection_repair_probe.py'
]

load_from = (
    'work_dirs/nocs_fruits_736x512_rgbd_3dbbox_'
    'stage5_detection_repair_probe/best_3d_iou_0.50_epoch_5.pth'
)
resume = False

optim_wrapper = dict(
    optimizer=dict(lr=5e-5),
    paramwise_cfg=dict(
        custom_keys={
            'backbone': dict(lr_mult=0.002),
            'backbone.rgb_backbone': dict(lr_mult=0.002),
            'backbone.depth_backbone': dict(lr_mult=0.002),
            'backbone.depth_adapters': dict(lr_mult=0.02),
            'backbone.depth_beta': dict(lr_mult=0.02),
            'bbox_head.cls_branches': dict(lr_mult=1.0),
            'bbox_head.reg_branches': dict(lr_mult=1.0),
            'bbox_head.cop_': dict(lr_mult=0.02),
            'bbox_head.depth_query_sampler': dict(lr_mult=0.02),
            'bbox_head.reg_z_branch': dict(lr_mult=0.02),
            'bbox_head.reg_size_branch': dict(lr_mult=0.02),
            'bbox_head.reg_rotation_branch': dict(lr_mult=0.02),
            'encoder': dict(lr_mult=1.0),
        },
    ),
)

max_epochs = 100
train_cfg = dict(max_epochs=max_epochs, val_interval=5)
param_scheduler = []

val_evaluator = dict(
    type='NOCSMetric',
    score_thr=0.2,
    nms_cfg=dict(type='nms', iou_threshold=0.5),
)

custom_hooks = [
    dict(
        type='EarlyStoppingHook',
        monitor='AP50',
        rule='greater',
        min_delta=2e-3,
        patience=6,
        strict=True,
        check_finite=True,
    ),
]

default_hooks = dict(
    checkpoint=dict(
        _delete_=True,
        type='CheckpointHook',
        interval=5,
        save_last=True,
        max_keep_ckpts=2,
        save_best=['AP50', '3d_iou_0.50'],
        rule=['greater', 'greater'],
        save_optimizer=False,
    ),
    logger=dict(interval=5),
)
