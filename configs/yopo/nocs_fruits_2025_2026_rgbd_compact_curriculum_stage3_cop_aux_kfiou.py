"""Stage 3: jointly train parallel pose and an auxiliary CoP chain.

The former pose-loss budget is split evenly between the two paths.  A
covariance KFIoU objective introduces 2D OBB consistency before the final GWD
and projection stage.  Start from Stage 2's best 3d_iou_0.25 checkpoint.
"""

_base_ = [
    "./nocs_fruits_2025_2026_rgbd_compact_curriculum_stage2_parallel_pose.py"
]

model = dict(
    bbox_head=dict(
        cop_prediction_mode="auxiliary",
        cop_use_bbox_conditioning=True,
        cop_aux_loss_weights=dict(z=1.0, size=1.0, rotation=1.0),
        cop_obb_rotation_refinement=False,
        loss_z=dict(loss_weight=25.0),
        loss_sizes=dict(loss_weight=25.0),
        loss_rotation=dict(loss_weight=2.5),
        loss_obb_aux=dict(
            _delete_=True,
            type="GaussianKFIoULoss",
            loss_weight=1.0,
            fail_on_invalid=True,
            eps=1e-7,
        ),
    ),
)

optim_wrapper = dict(
    paramwise_cfg=dict(
        custom_keys={
            # The auxiliary CoP modules are absent from the Stage 2 topology
            # and therefore start fresh at the full topology-recovery LR.
            "bbox_head.cop_": dict(lr_mult=1.0),
            "bbox_head.depth_query_sampler": dict(lr_mult=1.0),
        },
    ),
)

max_epochs = 20
train_cfg = dict(max_epochs=max_epochs, val_interval=5)

custom_hooks = [
    dict(type="ScheduleFreeOptimizerModeHook"),
    dict(
        type="EarlyStoppingHook",
        monitor="3d_iou_0.25",
        rule="greater",
        min_delta=1e-3,
        patience=3,
        strict=True,
        check_finite=True,
    ),
]

load_from = None
resume = False
