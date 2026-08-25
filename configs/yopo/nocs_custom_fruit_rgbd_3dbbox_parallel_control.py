"""Matched five-epoch control using only the historical parallel pose path."""

_base_ = ["./nocs_custom_fruit_rgbd_3dbbox_cop_curriculum_base.py"]

load_from = (
    "work_dirs/rgbd3d_long_ft80_lr1e6/"
    "best_3d_iou_0.50_epoch_10.pth"
)

model = dict(
    train_cfg=dict(
        assigner=dict(
            type="HungarianAssigner",
            match_costs=[
                dict(type="FocalLossCost", weight=2.0),
                dict(type="BBoxL1Cost", weight=5.0, box_format="xywh"),
                dict(type="IoUCost", iou_mode="giou", weight=2.0),
                dict(type="TranslationCost", weight=5.0),
                dict(
                    type="RotationCost",
                    symmetric_classes=[],
                    weight=2.0,
                ),
            ],
        ),
    ),
    bbox_head=dict(
        cop_prediction_mode="parallel",
        cop_encoder_pose_supervision=True,
        cop_use_bbox_conditioning=False,
        cop_fusion_mode="residual",
        distill_attributes=(),
        loss_z=dict(loss_weight=50.0),
        loss_sizes=dict(loss_weight=50.0),
        loss_rotation=dict(loss_weight=5.0),
    ),
)

default_hooks = dict(
    checkpoint=dict(
        _delete_=True,
        type="CheckpointHook",
        interval=100,
        save_last=False,
        save_best="3d_iou_0.50",
        rule="greater",
        save_optimizer=False,
    ),
)
