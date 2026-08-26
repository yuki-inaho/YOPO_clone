"""Stage 2: add independent metric pose heads without CoP coupling.

Start from Stage 1's best AP50_95 checkpoint with ``load_from``.  Hungarian
assignment remains 2D-only, preserving the learned query/object identities.
"""

_base_ = [
    "./nocs_fruits_2025_2026_rgbd_compact_curriculum_stage1_2d_full.py"
]

model = dict(
    bbox_head=dict(
        cop_prediction_mode="parallel",
        loss_z=dict(loss_weight=50.0),
        loss_sizes=dict(loss_weight=50.0),
        loss_rotation=dict(loss_weight=5.0),
    ),
)

optim_wrapper = dict(
    paramwise_cfg=dict(
        custom_keys={
            "bbox_head.reg_z_branch": dict(lr_mult=0.25),
            "bbox_head.reg_size_branch": dict(lr_mult=0.25),
            "bbox_head.reg_rotation_branch": dict(lr_mult=0.25),
        },
    ),
)

val_evaluator = dict(compute_pose_metrics=True)

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

default_hooks = dict(
    checkpoint=dict(
        _delete_=True,
        type="CheckpointHook",
        interval=5,
        save_last=True,
        max_keep_ckpts=2,
        save_best=["3d_iou_0.25", "AP50_95", "AP75"],
        rule=["greater", "greater", "greater"],
        save_optimizer=True,
    ),
)

load_from = None
resume = False
