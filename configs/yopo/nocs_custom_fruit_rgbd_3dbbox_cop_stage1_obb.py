"""Stage 1: frozen 2D OBB center teacher; no 3D pose supervision."""

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
            ],
        ),
    ),
    bbox_head=dict(
        distill_attributes=("center",),
        loss_z=dict(loss_weight=0.0),
        loss_sizes=dict(loss_weight=0.0),
        loss_rotation=dict(loss_weight=0.0),
    ),
)
