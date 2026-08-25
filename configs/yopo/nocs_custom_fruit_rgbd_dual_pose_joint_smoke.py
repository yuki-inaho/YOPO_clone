"""One-epoch finite/VRAM gate for joint parallel and CoP training."""

_base_ = ["./nocs_custom_fruit_rgbd_dual_pose_joint.py"]

train_cfg = dict(max_epochs=1, val_interval=1)
param_scheduler = [
    dict(
        type="CosineAnnealingLR",
        T_max=1,
        eta_min=1e-6,
        begin=0,
        end=1,
        by_epoch=True,
    ),
]
default_hooks = dict(
    checkpoint=dict(
        _delete_=True,
        type="CheckpointHook",
        interval=1,
        save_last=True,
        max_keep_ckpts=1,
        save_optimizer=False,
    ),
    logger=dict(interval=1),
)
