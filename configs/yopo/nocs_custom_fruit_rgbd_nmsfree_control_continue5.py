"""Five-epoch time control from the final OBB-refined RGB-D checkpoint."""

_base_ = [
    "./nocs_custom_fruit_rgbd_2d_obb_foundation_stage4_refinement_continue15.py"
]

load_from = (
    "work_dirs/nocs_custom_fruit_rgbd_2d_obb_foundation_stage4_refinement_continue15/"
    "best_3d_iou_0.50_epoch_15.pth"
)

train_cfg = dict(max_epochs=5, val_interval=5)

param_scheduler = [
    dict(
        type="CosineAnnealingLR",
        T_max=5,
        eta_min=1e-6,
        begin=0,
        end=5,
        by_epoch=True,
    ),
]

default_hooks = dict(
    checkpoint=dict(
        _delete_=True,
        type="CheckpointHook",
        interval=5,
        save_last=True,
        max_keep_ckpts=1,
        save_best="3d_iou_0.50",
        rule="greater",
        save_optimizer=False,
    ),
    logger=dict(interval=1),
)
