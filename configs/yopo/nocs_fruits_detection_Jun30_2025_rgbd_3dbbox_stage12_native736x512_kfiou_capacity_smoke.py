"""Two-step capacity check for the exact native KFIoU objective.

Supply the selected Stage-11 checkpoint using ``--cfg-options load_from=...``.
The probe performs no validation and registers no checkpoint hook.
"""

_base_ = [
    './nocs_fruits_detection_Jun30_2025_rgbd_3dbbox_'
    'stage12_native736x512_kfiou_base.py'
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

default_hooks = dict(
    checkpoint=None,
    logger=dict(interval=1),
)
