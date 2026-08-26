"""Stage 1: establish 2D detection before metric-pose training.

Train on the joint native 800x600 RGB-D set.  Only HBB classification,
regression, GIoU, and projected 2D center losses update the network.  The
parallel metric-pose branches are retained for checkpoint compatibility but
receive zero loss and zero learning rate in this stage.
"""

_base_ = [
    "./nocs_fruits_2025_2026_rgbd_3dbbox_"
    "b1b0_e4d4_ffn1024_amp_finetune.py"
]

model = dict(
    bbox_head=dict(
        cop_prediction_mode="parallel",
        cop_use_bbox_conditioning=False,
        cop_obb_rotation_refinement=False,
        expose_obb_aux_predictions=False,
        loss_z=dict(loss_weight=0.0),
        loss_sizes=dict(loss_weight=0.0),
        loss_rotation=dict(loss_weight=0.0),
        loss_projection=None,
        loss_obb_aux=None,
    ),
)

# Preserve the transplanted pose branches exactly until their curriculum
# stage.  Center-2D is part of the detection foundation and trains at base LR.
optim_wrapper = dict(
    paramwise_cfg=dict(
        custom_keys={
            "bbox_head.reg_centers_2d_branch": dict(lr_mult=1.0),
            "bbox_head.reg_z_branch": dict(lr_mult=0.0),
            "bbox_head.reg_size_branch": dict(lr_mult=0.0),
            "bbox_head.reg_rotation_branch": dict(lr_mult=0.0),
            "bbox_head.cop_": dict(lr_mult=0.0),
            "bbox_head.depth_query_sampler": dict(lr_mult=0.0),
        },
    ),
)

val_evaluator = dict(compute_pose_metrics=False)

max_epochs = 50
train_cfg = dict(max_epochs=max_epochs, val_interval=5)

custom_hooks = [
    dict(type="ScheduleFreeOptimizerModeHook"),
    dict(
        type="EarlyStoppingHook",
        monitor="AP50_95",
        rule="greater",
        min_delta=1e-3,
        patience=4,
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
        save_best=["AP50_95", "AP75"],
        rule=["greater", "greater"],
        save_optimizer=True,
    ),
    logger=dict(interval=5),
)

load_from = None
resume = False
