"""Bounded continuation until corrected 3D geometry reaches a plateau."""

_base_ = [
    './nocs_fruits_detection_Jun30_2025_rgbd_3dbbox_'
    'stage8_geometry_repair_continue10.py'
]

load_from = (
    'work_dirs/nocs_fruits_736x512_rgbd_3dbbox_'
    'stage8_geometry_repair_continue10/best_3d_iou_0.50_epoch_10.pth'
)
resume = False

max_epochs = 10
train_cfg = dict(max_epochs=max_epochs, val_interval=2)

custom_hooks = [
    dict(
        type='EarlyStoppingHook',
        monitor='3d_iou_0.50',
        rule='greater',
        min_delta=2e-3,
        patience=3,
        strict=True,
        check_finite=True,
    ),
]

default_hooks = dict(
    checkpoint=dict(
        _delete_=True,
        type='CheckpointHook',
        interval=2,
        save_last=True,
        max_keep_ckpts=1,
        save_best=['AP50', '3d_iou_0.50'],
        rule=['greater', 'greater'],
        save_optimizer=False,
    ),
    logger=dict(interval=5),
)
