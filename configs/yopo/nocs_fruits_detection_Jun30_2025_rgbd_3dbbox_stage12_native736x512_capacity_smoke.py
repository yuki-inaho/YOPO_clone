"""Two-iteration RTX 5090 capacity probe for (B,4,512,736) tensors.

Supply the adopted Stage-11 checkpoint with ``--cfg-options load_from=...``.
Start at b16 and adjust only ``train_dataloader.batch_size``.  Accept the
largest batch completing both iterations with finite losses and adequate VRAM
headroom.  This probe performs no validation and saves no checkpoint.
"""

_base_ = [
    './nocs_fruits_detection_Jun30_2025_rgbd_3dbbox_'
    'stage12_native736x512_base.py'
]

load_from = None
resume = False

train_dataloader = dict(
    batch_size=16,
    num_workers=2,
    persistent_workers=False,
    pin_memory=True,
)
train_cfg = dict(
    _delete_=True,
    type='IterBasedTrainLoop',
    max_iters=2,
    val_interval=100,
)
val_dataloader = None
val_cfg = None
val_evaluator = None

param_scheduler = []
custom_hooks = [dict(type='ScheduleFreeOptimizerModeHook')]
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

