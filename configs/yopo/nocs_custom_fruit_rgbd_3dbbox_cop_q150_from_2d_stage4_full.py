"""Q150 Stage 4: full CoP z/size/rotation from the 2D foundation."""

_base_ = ["./nocs_custom_fruit_rgbd_3dbbox_cop_q150_from_2d_base.py"]

load_from = (
    "work_dirs/nocs_custom_fruit_rgbd_3dbbox_cop_q150_from_2d_stage3_size/"
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

default_hooks = dict(
    checkpoint=dict(
        _delete_=True,
        type="CheckpointHook",
        interval=5,
        save_last=True,
        max_keep_ckpts=1,
        save_best="3d_iou_0.50",
        rule="greater",
        save_optimizer=False,
    ),
    logger=dict(interval=1),
)
