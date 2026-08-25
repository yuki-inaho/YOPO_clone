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
    data_preprocessor=dict(non_blocking=True),
    num_queries=max_objects,
    test_cfg=dict(max_per_img=max_objects),
    bbox_head=dict(
        cop_prediction_mode="chain",
        cop_chain_order=("z", "size", "rotation"),
        cop_use_bbox_conditioning=True,
        # Stage 1 has no 3D supervision yet.  The later stages switch this to
        # depth_dense and enable the explicit pre-fusion depth query.
        cop_fusion_mode="query_dense",
        cop_depth_context=dict(
            num_levels=3, roi_size=3, vectorize_layers=True),
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

# Pin host batches and let the data preprocessor enqueue asynchronous H2D
# copies. These two settings are intentionally paired; either can be disabled
# with a config override during the real-GPU quality gate.
train_dataloader = dict(pin_memory=True)
val_dataloader = dict(pin_memory=True)

# Fresh FP16 training makes the 9D transformer predictions non-finite before
# Hungarian matching.  BF16 avoids the overflow but MMCV 2.2's CUDA
# multi-scale deformable-attention kernel does not implement BF16.  Keep this
# short curriculum on the verified FP32 ScheduleFree path; batch 26 fits the
# 32 GiB RTX 5090 used for these runs.
optim_wrapper = dict(
    _delete_=True,
    type="ScheduleFreeOptimWrapper",
    optimizer=dict(
        type="AdamWScheduleFreeOptimizer",
        lr=1e-6,
        weight_decay=1e-4,
        warmup_steps=0,
    ),
    clip_grad=dict(max_norm=0.1, norm_type=2),
    paramwise_cfg=dict(
        custom_keys=dict(
            backbone=dict(lr_mult=0.1),
            encoder=dict(lr_mult=0.5),
            rgb_backbone=dict(lr_mult=0.1),
            depth_backbone=dict(lr_mult=0.1),
        ),
    ),
    constructor="DefaultOptimWrapperConstructor",
    accumulative_counts=1,
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
