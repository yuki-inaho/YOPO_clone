"""Continue the accepted OBB-refined Stage 4 for fifteen recovery epochs."""

_base_ = ["./nocs_custom_fruit_rgbd_2d_obb_foundation_stage4_refinement.py"]

load_from = (
    "work_dirs/nocs_custom_fruit_rgbd_2d_obb_foundation_stage4_refinement/"
    "best_3d_iou_0.50_epoch_5.pth"
)

train_cfg = dict(max_epochs=15, val_interval=5)

param_scheduler = [
    dict(
        type="CosineAnnealingLR",
        T_max=15,
        eta_min=1e-6,
        begin=0,
        end=15,
        by_epoch=True,
    ),
]

default_hooks = dict(
    checkpoint=dict(
        _delete_=True,
        type="CheckpointHook",
        interval=5,
        save_last=True,
        max_keep_ckpts=3,
        save_best="3d_iou_0.50",
        rule="greater",
        save_optimizer=False,
    ),
    logger=dict(interval=1),
)
