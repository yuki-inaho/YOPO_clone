"""Two-iteration GPU smoke for Stage-10 MAL integration."""

_base_ = [
    './nocs_fruits_detection_Jun30_2025_rgbd_3dbbox_'
    'stage10_2d_anchor_mal_probe5.py'
]

train_dataloader = dict(
    batch_size=2,
    num_workers=0,
    persistent_workers=False,
    pin_memory=False,
)
val_dataloader = dict(
    batch_size=1,
    num_workers=0,
    persistent_workers=False,
    pin_memory=False,
)
train_cfg = dict(
    _delete_=True,
    type='IterBasedTrainLoop',
    max_iters=2,
    val_interval=100,
)
param_scheduler = []
custom_hooks = []

default_hooks = dict(
    checkpoint=dict(
        _delete_=True,
        type='CheckpointHook',
        by_epoch=False,
        interval=100,
        save_last=False,
        max_keep_ckpts=1,
    ),
    logger=dict(interval=1),
)
