"""One-update GPU smoke for the portable Q256 Stage-Z curriculum."""

_base_ = ['./nocs_fruits_detection_Jun30_2025_rgbd_3dbbox_stage1_z.py']

train_dataloader = dict(
    batch_size=12,
    num_workers=0,
    persistent_workers=False,
    dataset=dict(indices=12),
)
train_cfg = dict(max_epochs=1, val_interval=100)
default_hooks = dict(
    checkpoint=dict(
        _delete_=True,
        type='CheckpointHook',
        interval=1,
        save_last=True,
        max_keep_ckpts=1,
        save_optimizer=False,
    ),
    logger=dict(interval=1),
)
