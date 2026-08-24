"""One-iteration AMP smoke for the transferred frozen-RGB 3D model."""

_base_ = ["./nocs_custom_fruit_rgbd_3dbbox_transfer.py"]

train_dataloader = dict(
    batch_size=1,
    num_workers=0,
    persistent_workers=False,
)
train_cfg = dict(_delete_=True, type="IterBasedTrainLoop", max_iters=1, val_interval=2)
val_dataloader = None
val_cfg = None
val_evaluator = None
param_scheduler = []
default_hooks = dict(
    checkpoint=dict(
        _delete_=True,
        type="CheckpointHook",
        interval=1,
        save_last=True,
        save_optimizer=False,
    ),
    logger=dict(interval=1),
)

# Keep ScheduleFree's required train-mode transition while enabling fp16 AMP.
optim_wrapper = dict(
    type="AmpScheduleFreeOptimWrapper",
    dtype="float16",
    # Stable 3D rotation geometry keeps the fp16 backward finite, so use the
    # normal static scale used by the preceding 2D RIoU experiment.
    loss_scale=1.0,
    accumulative_counts=1,
)
