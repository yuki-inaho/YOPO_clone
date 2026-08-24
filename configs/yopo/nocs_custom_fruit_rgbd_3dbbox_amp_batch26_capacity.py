"""Four-iteration capacity verification for the 20-epoch RGB-D config."""

_base_ = ["./nocs_custom_fruit_rgbd_3dbbox_amp_20ep.py"]

train_dataloader = dict(
    batch_size=26,
    num_workers=2,
    persistent_workers=False,
)
train_cfg = dict(_delete_=True, type="IterBasedTrainLoop", max_iters=4, val_interval=5)
val_dataloader = None
val_cfg = None
val_evaluator = None
param_scheduler = []
default_hooks = dict(
    checkpoint=dict(
        _delete_=True,
        type="CheckpointHook",
        interval=100,
        save_last=False,
        save_optimizer=False,
    ),
    logger=dict(interval=1),
)
