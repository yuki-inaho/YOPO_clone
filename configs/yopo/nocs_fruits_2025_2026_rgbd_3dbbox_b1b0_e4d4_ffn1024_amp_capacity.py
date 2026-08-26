"""Three-update 800x600 AMP capacity probe for the compact YOPO model.

Override ``train_dataloader.batch_size`` and ``load_from`` on the command
line.  Validation and checkpoint writes are disabled so each probe measures
only the real joint-data training path.
"""

_base_ = [
    "./nocs_fruits_2025_2026_rgbd_3dbbox_"
    "b1b0_e4d4_ffn1024_amp_finetune.py"
]

train_dataloader = dict(
    batch_size=30,
    num_workers=4,
    persistent_workers=False,
)
train_cfg = dict(
    _delete_=True,
    type="IterBasedTrainLoop",
    max_iters=3,
    val_interval=4,
)
val_cfg = None
val_dataloader = None
val_evaluator = None
param_scheduler = []

# The inherited early-stopping hook requires validation metrics.  Capacity
# probing only needs ScheduleFree's train/eval mode contract.
custom_hooks = [dict(type="ScheduleFreeOptimizerModeHook")]
default_hooks = dict(
    checkpoint=dict(
        _delete_=True,
        type="CheckpointHook",
        interval=1000,
        save_last=False,
        save_optimizer=False,
    ),
    logger=dict(interval=1),
)

load_from = None
resume = False
