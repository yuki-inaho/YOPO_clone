"""Portable Stage 3: full z/size/rotation plus projected-OBB consistency."""

_base_ = ['./nocs_fruits_detection_Jun30_2025_rgbd_3dbbox_curriculum_base.py']

load_from = (
    'work_dirs/nocs_fruits_736x512_rgbd_3dbbox_stage2_size/epoch_5.pth'
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
                dict(
                    type='RotationCost',
                    symmetric_classes=[],
                    weight=2.0,
                ),
            ],
        ),
    ),
    bbox_head=dict(
        # Matching/projection run in resized (and possibly flipped) pixels;
        # inference still consumes the preserved original-image intrinsic.
        train_intrinsic_to_image_space=True,
        loss_z=dict(loss_weight=50.0),
        loss_sizes=dict(loss_weight=50.0),
        loss_rotation=dict(loss_weight=5.0),
        projection_geometry_source='target',
        loss_projection=dict(
            _delete_=True,
            type='ProjectedEllipsoidGWDLoss',
            loss_weight=1.0,
            tau=1.0,
            normalize=True,
            detach_center=True,
            detach_depth=True,
            detach_size=True,
            fail_on_invalid=True,
        ),
        loss_obb_aux=dict(loss_weight=1.0),
    ),
)

default_hooks = dict(
    checkpoint=dict(
        _delete_=True,
        type='CheckpointHook',
        interval=5,
        save_last=True,
        max_keep_ckpts=1,
        save_best='3d_iou_0.50',
        rule='greater',
        save_optimizer=False,
    ),
    logger=dict(interval=5),
)
