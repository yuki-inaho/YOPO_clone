"""Stage 4: full OBB center -> depth -> size -> rotation CoP training."""

_base_ = ["./nocs_custom_fruit_rgbd_3dbbox_cop_curriculum_base.py"]

load_from = (
    "work_dirs/nocs_custom_fruit_rgbd_3dbbox_cop_stage3_obb_depth_size/"
    "epoch_5.pth"
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
        cop_fusion_mode="depth_dense",
        distill_attributes=("center", "z", "size", "rotation"),
        loss_z=dict(loss_weight=50.0),
        loss_sizes=dict(loss_weight=50.0),
        loss_rotation=dict(loss_weight=5.0),
    ),
)

# Full pose quality is meaningful again, so retain the best real evaluator
# checkpoint while keeping optimizer state out of the bounded artifact.
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
