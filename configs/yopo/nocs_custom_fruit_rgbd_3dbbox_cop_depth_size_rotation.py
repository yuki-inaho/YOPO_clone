"""Short CoP experiment: 2D box -> depth -> 3D size -> rotation."""

_base_ = ["./nocs_custom_fruit_rgbd_3dbbox_long_finetune_lr1e6.py"]

# Single public capacity knob.  Keep decoder capacity and post-processing in
# lockstep so increasing one cannot silently truncate at the other.
max_objects = 100

model = dict(
    num_queries=max_objects,
    bbox_head=dict(
        cop_prediction_mode="chain",
        cop_chain_order=("z", "size", "rotation"),
        cop_use_bbox_conditioning=True,
        cop_fusion_mode="depth_dense",
        cop_depth_context=dict(
            num_levels=3, roi_size=3, vectorize_layers=True),
        cop_encoder_pose_supervision=False,
        test_cfg=dict(max_per_img=max_objects),
    ),
    train_cfg=dict(
        encoder_assigner=dict(
            type="HungarianAssigner",
            match_costs=[
                dict(type="FocalLossCost", weight=2.0),
                dict(type="BBoxL1Cost", weight=5.0, box_format="xywh"),
                dict(type="IoUCost", iou_mode="giou", weight=2.0),
            ],
        ),
    ),
)

# Isolated five-epoch quality gate from the previous best weights.  This is a
# fresh optimizer, not a resume, and changes only the CoP routing/order.
load_from = (
    "work_dirs/rgbd3d_long_ft80_lr1e6/"
    "best_3d_iou_0.50_epoch_10.pth"
)
resume = False
max_epochs = 5
train_cfg = dict(max_epochs=max_epochs, val_interval=5)
param_scheduler = [
    dict(
        type="CosineAnnealingLR",
        T_max=max_epochs,
        eta_min=1e-6,
        begin=0,
        end=max_epochs,
        by_epoch=True,
    ),
]
