"""Ten-epoch geometry-coordinate repair, completing 100 cumulative epochs.

Stage 6 (35 epochs) plus Stage 7 (55 epochs before AP early stopping) trained
for 90 epoch-equivalents. This final continuation fixes the training-only
camera coordinate contract: center targets, matching and projected-OBB loss
all use the resized/flipped image pixels, while teacher-free inference keeps
the original intrinsic and rescales centers back before 3D reconstruction.

Detection paths receive only 1e-6 to preserve the Stage-7 AP optimum. The
center/depth/size/rotation and CoP paths receive 1e-5 so the previously biased
3D geometry can adapt within the remaining ten epochs.
"""

_base_ = [
    './nocs_fruits_detection_Jun30_2025_rgbd_3dbbox_'
    'stage7_detection_repair_continue65.py'
]

load_from = (
    'work_dirs/nocs_fruits_736x512_rgbd_3dbbox_'
    'stage7_detection_repair_continue65/best_AP50_epoch_35.pth'
)
resume = False

optim_wrapper = dict(
    optimizer=dict(lr=1e-5),
    paramwise_cfg=dict(
        custom_keys={
            'backbone': dict(lr_mult=0.01),
            'neck': dict(lr_mult=0.1),
            'encoder': dict(lr_mult=0.1),
            'decoder': dict(lr_mult=0.1),
            'bbox_head.cls_branches': dict(lr_mult=0.1),
            'bbox_head.reg_branches': dict(lr_mult=0.1),
            'bbox_head.reg_centers_2d_branch': dict(lr_mult=1.0),
            'bbox_head.reg_z_branch': dict(lr_mult=1.0),
            'bbox_head.reg_size_branch': dict(lr_mult=1.0),
            'bbox_head.reg_rotation_branch': dict(lr_mult=1.0),
            'bbox_head.cop_': dict(lr_mult=1.0),
            'bbox_head.depth_query_sampler': dict(lr_mult=1.0),
        },
    ),
)

max_epochs = 10
train_cfg = dict(max_epochs=max_epochs, val_interval=2)
param_scheduler = []

val_evaluator = dict(
    type='NOCSMetric',
    score_thr=0.30,
    nms_cfg=dict(type='nms', iou_threshold=0.375),
)

custom_hooks = []
default_hooks = dict(
    checkpoint=dict(
        _delete_=True,
        type='CheckpointHook',
        interval=2,
        save_last=True,
        max_keep_ckpts=2,
        save_best=['AP50', '3d_iou_0.50'],
        rule=['greater', 'greater'],
        save_optimizer=False,
    ),
    logger=dict(interval=5),
)
