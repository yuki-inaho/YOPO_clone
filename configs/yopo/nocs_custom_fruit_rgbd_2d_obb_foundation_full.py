"""Twenty-epoch 2D OBB foundation before further 3D rotation training."""

_base_ = [
    "./nocs_custom_fruit_rgbd_3dbbox_cop_q150_stage4_projection_gwd_obb_aux.py"
]

# Start from the verified Q150 Stage4 model, but train only the converged 2D
# detection/center objectives and the NOCS OBB Gaussian objective.  The CoP
# feature chain remains the OBB feature extractor; z/size/rotation/projection
# objectives are disabled so this stage answers one question: can the query
# OBB representation itself be made accurate with a full schedule?
model = dict(
    bbox_head=dict(
        distill_attributes=(),
        obb_center_teacher_checkpoint=None,
        pose_teacher_checkpoint=None,
        loss_z=dict(loss_weight=0.0),
        loss_sizes=dict(loss_weight=0.0),
        loss_rotation=dict(loss_weight=0.0),
        loss_projection=None,
        loss_obb_aux=dict(loss_weight=5.0),
    ),
)

train_cfg = dict(max_epochs=20, val_interval=5)

param_scheduler = [
    dict(
        type="CosineAnnealingLR",
        T_max=20,
        eta_min=1e-6,
        begin=0,
        end=20,
        by_epoch=True,
    ),
]

default_hooks = dict(
    checkpoint=dict(
        _delete_=True,
        type="CheckpointHook",
        interval=5,
        save_last=True,
        max_keep_ckpts=4,
        save_optimizer=False,
    ),
    logger=dict(interval=1),
)
