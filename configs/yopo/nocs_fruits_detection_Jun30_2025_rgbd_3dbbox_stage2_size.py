"""Portable Stage 2: retain z and add metric 3D size."""

_base_ = ['./nocs_fruits_detection_Jun30_2025_rgbd_3dbbox_curriculum_base.py']

load_from = (
    'work_dirs/nocs_fruits_736x512_rgbd_3dbbox_stage1_z/epoch_5.pth'
)
custom_hooks = []

model = dict(
    train_cfg=dict(
        assigner=dict(
            type='HungarianAssigner',
            match_costs=[
                dict(type='FocalLossCost', weight=2.0),
                dict(type='BBoxL1Cost', weight=5.0, box_format='xywh'),
                dict(type='IoUCost', iou_mode='giou', weight=2.0),
                dict(type='TranslationCost', weight=5.0),
            ],
        ),
    ),
    bbox_head=dict(
        loss_z=dict(loss_weight=50.0),
        loss_sizes=dict(loss_weight=50.0),
        loss_rotation=dict(loss_weight=0.0),
        loss_projection=None,
        loss_obb_aux=dict(loss_weight=1.0),
    ),
)
