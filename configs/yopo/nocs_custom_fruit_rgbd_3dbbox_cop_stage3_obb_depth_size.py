"""Stage 3: OBB center -> depth -> 3D size; rotation remains disabled."""

_base_ = ["./nocs_custom_fruit_rgbd_3dbbox_cop_curriculum_base.py"]

load_from = (
    "work_dirs/nocs_custom_fruit_rgbd_3dbbox_cop_stage2_obb_depth/epoch_5.pth"
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
            ],
        ),
    ),
    bbox_head=dict(
        cop_fusion_mode="depth_dense",
        distill_attributes=("center", "z", "size"),
        loss_z=dict(loss_weight=50.0),
        loss_sizes=dict(loss_weight=50.0),
        loss_rotation=dict(loss_weight=0.0),
    ),
)
