"""One-epoch finite/VRAM gate for weak-DA corrected tomato OBB training."""

_base_ = [
    "./rotated_deformable_detr_tomato_obb_corrected_da_weak_riou_stage1.py"
]

train_cfg = dict(max_epochs=1, val_interval=1)
param_scheduler = []
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
