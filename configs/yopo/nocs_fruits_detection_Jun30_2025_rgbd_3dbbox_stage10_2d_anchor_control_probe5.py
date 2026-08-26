"""Five-epoch control with 2D-anchored query assignment.

This isolates the assignment-contract change after Stage 8.  Translation and
rotation remain supervised losses, but they cannot change the query-to-GT
identity selected by Hungarian matching.
"""

_base_ = [
    './nocs_fruits_detection_Jun30_2025_rgbd_3dbbox_'
    'stage8_geometry_repair_continue10.py'
]

load_from = (
    'work_dirs/nocs_fruits_736x512_rgbd_3dbbox_'
    'stage8_geometry_repair_continue10/best_3d_iou_0.50_epoch_10.pth'
)
resume = False

model = dict(
    train_cfg=dict(
        assigner=dict(
            _delete_=True,
            type='HungarianAssigner',
            match_costs=[
                dict(type='FocalLossCost', weight=2.0),
                dict(type='BBoxL1Cost', weight=5.0, box_format='xywh'),
                dict(type='IoUCost', iou_mode='giou', weight=2.0),
            ],
        ),
    ),
)

# Shared features and 2D branches adapt at 1e-5.  The pretrained backbones and
# converged pose path stay at or below 1e-6 to make this a detection repair
# probe without silently disabling any 3D objective.
optim_wrapper = dict(
    optimizer=dict(lr=1e-5),
    paramwise_cfg=dict(
        custom_keys={
            'backbone': dict(lr_mult=0.01),
            'neck': dict(lr_mult=1.0),
            'encoder': dict(lr_mult=1.0),
            'decoder': dict(lr_mult=1.0),
            'bbox_head.cls_branches': dict(lr_mult=1.0),
            'bbox_head.reg_branches': dict(lr_mult=1.0),
            'bbox_head.reg_centers_2d_branch': dict(lr_mult=0.1),
            'bbox_head.reg_z_branch': dict(lr_mult=0.1),
            'bbox_head.reg_size_branch': dict(lr_mult=0.1),
            'bbox_head.reg_rotation_branch': dict(lr_mult=0.1),
            'bbox_head.cop_': dict(lr_mult=0.1),
            'bbox_head.depth_query_sampler': dict(lr_mult=0.1),
        },
    ),
)

probe_epochs = 5
train_cfg = dict(max_epochs=probe_epochs, val_interval=1)
param_scheduler = []
custom_hooks = []

# HBB AP must measure query ranking without an operating confidence threshold
# or NMS.  The normal result view remains conf=0.2/NMS=0.3 so all 3D fields
# stay aligned to the deployment policy selected in earlier stages.
hbb_iou_thrs = [
    0.50, 0.55, 0.60, 0.65, 0.70,
    0.75, 0.80, 0.85, 0.90, 0.95,
]
val_evaluator = dict(
    type='NOCSMetric',
    score_thr=0.2,
    nms_cfg=dict(type='nms', iou_threshold=0.3),
    iou_thrs=hbb_iou_thrs,
    hbb_selection=dict(score_thr=0.0, nms_cfg=None),
)

default_hooks = dict(
    checkpoint=dict(
        _delete_=True,
        type='CheckpointHook',
        interval=1,
        save_last=True,
        max_keep_ckpts=2,
        save_best=['AP50_95', 'AP50', '3d_iou_0.50'],
        rule=['greater', 'greater', 'greater'],
        save_optimizer=False,
    ),
    logger=dict(interval=5),
)
