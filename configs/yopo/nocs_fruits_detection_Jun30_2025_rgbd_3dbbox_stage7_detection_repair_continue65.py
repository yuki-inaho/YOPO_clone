"""Continue Stage-6 detection repair to 100 cumulative epochs.

Stage 6 reached epoch 35 while its AP monitor compared resized GT boxes with
original-coordinate predictions. The corrected metric shows AP50 still
improving through epoch 35. This stage therefore adds at most 65 epochs and
uses the deployment-selected conf=0.2/NMS=0.3 policy. Full-precision AP50
early stopping ends the run after six validations (30 epochs) without a
material 0.002 gain.
"""

_base_ = [
    './nocs_fruits_detection_Jun30_2025_rgbd_3dbbox_'
    'stage6_detection_repair_plateau.py'
]

load_from = (
    'work_dirs/nocs_fruits_736x512_rgbd_3dbbox_'
    'stage6_detection_repair_plateau/epoch_35.pth'
)
resume = False

# 35 completed Stage-6 epochs + at most 65 here = 100 cumulative epochs.
max_epochs = 65
train_cfg = dict(max_epochs=max_epochs, val_interval=5)

val_evaluator = dict(
    type='NOCSMetric',
    score_thr=0.2,
    nms_cfg=dict(type='nms', iou_threshold=0.3),
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
