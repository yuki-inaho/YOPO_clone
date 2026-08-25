"""Shared settings for cumulative OBB -> depth -> size -> rotation training."""

_base_ = ["./nocs_custom_fruit_rgbd_3dbbox_long_finetune_lr1e6.py"]

# One capacity knob controls decoder queries and both post-processing caps.
max_objects = 100
stage_epochs = 5

obb_teacher_checkpoint = (
    "work_dirs/rddetr_tomato_riou_linear_ft20/"
    "best_rbbox_mAP_50_epoch_20.pth"
)
pose_teacher_checkpoint = (
    "work_dirs/rgbd3d_long_ft80_lr1e6/"
    "best_3d_iou_0.50_epoch_10.pth"
)

model = dict(
    num_queries=max_objects,
    test_cfg=dict(max_per_img=max_objects),
    bbox_head=dict(
        cop_prediction_mode="chain",
        cop_chain_order=("z", "size", "rotation"),
        cop_use_bbox_conditioning=True,
        # Stage 1 has no 3D supervision yet.  The later stages switch this to
        # depth_dense and enable the explicit pre-fusion depth query.
        cop_fusion_mode="query_dense",
        cop_depth_context=dict(num_levels=3, roi_size=3),
        cop_encoder_pose_supervision=False,
        distill_attributes=(),
        # The head-only OBB adapter is retained as an explicit ablation but is
        # rejected for the main curriculum by the NOCS center-error gate.
        center_teacher_source="pose_center",
        obb_center_teacher_checkpoint=obb_teacher_checkpoint,
        pose_teacher_checkpoint=pose_teacher_checkpoint,
        distill_loss_weights=dict(
            center=2.0,
            z=5.0,
            size=5.0,
            rotation=1.0,
        ),
        # Keep every teacher query for the first isolated runs. The emitted
        # score tensor is logged and this can be raised without code changes.
        distill_score_threshold=0.2,
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

resume = False
train_cfg = dict(max_epochs=stage_epochs, val_interval=stage_epochs)
param_scheduler = [
    dict(
        type="CosineAnnealingLR",
        T_max=stage_epochs,
        eta_min=1e-6,
        begin=0,
        end=stage_epochs,
        by_epoch=True,
    ),
]

# Stages 1-3 are incomplete pose models, so retain only their final weights as
# the next stage's initialization rather than ranking them by 3D IoU.
default_hooks = dict(
    checkpoint=dict(
        _delete_=True,
        type="CheckpointHook",
        interval=stage_epochs,
        save_last=True,
        max_keep_ckpts=1,
        save_optimizer=False,
    ),
    logger=dict(interval=1),
)
