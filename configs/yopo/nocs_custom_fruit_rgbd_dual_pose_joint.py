"""Jointly train parallel and CoP pose heads on shared 2D assignments."""

_base_ = ["./nocs_custom_fruit_rgbd_nmsfree_control_continue5.py"]

# The time-control checkpoint is the strongest verified CoP model and retains
# the historical parallel head parameters. Joint training updates both paths
# from the same query features and the same 2D-matched GT identities.
load_from = (
    "work_dirs/nocs_custom_fruit_rgbd_nmsfree_control_continue5/"
    "best_3d_iou_0.50_epoch_5.pth"
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
        # ``auxiliary`` keeps the parallel path as the primary inference path
        # and supervises CoP with separate, explicitly named auxiliary losses.
        cop_prediction_mode="auxiliary",
        cop_encoder_pose_supervision=True,
        cop_aux_loss_weights=dict(z=1.0, size=1.0, rotation=1.0),
        # Split the former single-path pose budget evenly across both heads.
        loss_z=dict(loss_weight=25.0),
        loss_sizes=dict(loss_weight=25.0),
        loss_rotation=dict(loss_weight=2.5),
        # Direct GT supervision replaces the frozen teacher in this joint run.
        distill_attributes=(),
        obb_center_teacher_checkpoint=None,
        pose_teacher_checkpoint=None,
    ),
)

# The parallel head is still near its historical initialization and uses the
# verified fresh-head LR. CoP starts from the strongest checkpoint, so it is
# updated at one fiftieth of that rate to avoid destructive forgetting while
# remaining jointly trainable. The shared detector stays conservative.
optim_wrapper = dict(
    paramwise_cfg=dict(
        custom_keys={
            "bbox_head.cop_": dict(lr_mult=1.0),
            "bbox_head.depth_query_sampler": dict(lr_mult=1.0),
            "bbox_head.reg_z_branch": dict(lr_mult=50.0),
            "bbox_head.reg_size_branch": dict(lr_mult=50.0),
            "bbox_head.reg_rotation_branch": dict(lr_mult=50.0),
        },
    ),
)

# The extra parallel loss path is small, but batch 20 leaves a safety margin
# below the 32 GiB device limit for encoder pose supervision and validation.
train_dataloader = dict(batch_size=20)
train_cfg = dict(max_epochs=10, val_interval=5)

param_scheduler = [
    dict(
        type="CosineAnnealingLR",
        T_max=10,
        eta_min=1e-6,
        begin=0,
        end=10,
        by_epoch=True,
    ),
]

default_hooks = dict(
    checkpoint=dict(
        _delete_=True,
        type="CheckpointHook",
        interval=5,
        save_last=True,
        max_keep_ckpts=2,
        save_best="3d_iou_0.50",
        rule="greater",
        save_optimizer=False,
    ),
    logger=dict(interval=1),
)
