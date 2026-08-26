"""Initial 50-epoch native-736x512 FULL run with selected KFIoU.

Pass the gated Stage-11 model-only checkpoint through CLI ``load_from`` and
keep ``resume=False`` across the geometry change.  Periodic checkpoints retain
ScheduleFree state and can be exactly resumed for a justified 25-epoch tail.
"""

_base_ = [
    './nocs_fruits_detection_Jun30_2025_rgbd_3dbbox_'
    'stage12_native736x512_kfiou_base.py'
]

load_from = None
resume = False

max_epochs = 50
train_cfg = dict(max_epochs=max_epochs, val_interval=5)

custom_hooks = [
    dict(type='ScheduleFreeOptimizerModeHook'),
    dict(
        type='EarlyStoppingHook',
        monitor='AP50_95',
        rule='greater',
        min_delta=1e-3,
        patience=4,
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
        save_best=['AP50_95', 'AP75', '3d_iou_0.25'],
        rule=['greater', 'greater', 'greater'],
        save_optimizer=True,
    ),
    logger=dict(interval=5),
)
