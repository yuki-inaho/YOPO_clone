"""Continue the full portable 3D objective until 3D-IoU saturation."""

_base_ = ['./nocs_fruits_detection_Jun30_2025_rgbd_3dbbox_stage3_full.py']

load_from = (
    'work_dirs/nocs_fruits_736x512_rgbd_3dbbox_stage3_full/'
    'best_3d_iou_0.50_epoch_5.pth'
)
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

max_epochs = 50
train_cfg = dict(max_epochs=max_epochs, val_interval=5)
param_scheduler = []
default_hooks = dict(
    checkpoint=dict(
        _delete_=True,
        type='CheckpointHook',
        interval=5,
        save_last=True,
        max_keep_ckpts=2,
        save_best='3d_iou_0.50',
        rule='greater',
        save_optimizer=False,
    ),
    logger=dict(interval=5),
)
