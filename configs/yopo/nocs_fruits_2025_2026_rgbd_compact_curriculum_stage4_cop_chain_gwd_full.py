"""Stage 4: make CoP primary and fine-tune full 2D/3D consistency.

Start from Stage 3's best 3d_iou_0.25 checkpoint.  The already trained CoP
chain becomes the inference path, followed by 2D OBB GWD and projected
ellipsoid GWD consistency.  The nominal 100 epochs are bounded by validation
plateau stopping.
"""

_base_ = [
    "./nocs_fruits_2025_2026_rgbd_compact_curriculum_stage3_cop_aux_kfiou.py"
]

model = dict(
    bbox_head=dict(
        cop_prediction_mode="chain",
        cop_encoder_pose_supervision=False,
        cop_obb_rotation_refinement=True,
        loss_z=dict(loss_weight=50.0),
        loss_sizes=dict(loss_weight=50.0),
        loss_rotation=dict(loss_weight=5.0),
        loss_obb_aux=dict(
            _delete_=True,
            type="GaussianGWDLoss",
            loss_weight=1.0,
            tau=1.0,
            normalize=True,
            include_center=False,
            fail_on_invalid=True,
        ),
        loss_projection=dict(
            _delete_=True,
            type="ProjectedEllipsoidGWDLoss",
            loss_weight=1.0,
            tau=1.0,
            normalize=True,
            detach_center=True,
            detach_depth=True,
            detach_size=True,
            fail_on_invalid=True,
        ),
    ),
)

optim_wrapper = dict(
    # The CoP-primary GWD path can overflow query activations under FP16 even
    # with static loss scaling and gradient clipping.  BF16 keeps AMP tensor
    # cores and the same memory class while providing FP32-like exponent range.
    dtype="bfloat16",
    loss_scale=1.0,
    optimizer=dict(lr=5e-5),
    paramwise_cfg=dict(
        custom_keys={
            "bbox_head.cop_": dict(lr_mult=0.25),
            "bbox_head.depth_query_sampler": dict(lr_mult=0.25),
        },
    ),
)

max_epochs = 100
train_cfg = dict(max_epochs=max_epochs, val_interval=5)

custom_hooks = [
    dict(type="ScheduleFreeOptimizerModeHook"),
    dict(
        type="EarlyStoppingHook",
        monitor="3d_iou_0.25",
        rule="greater",
        min_delta=1e-3,
        patience=6,
        strict=True,
        check_finite=True,
    ),
]

load_from = None
resume = False
