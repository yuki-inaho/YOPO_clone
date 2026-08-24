"""20-epoch RGB-D 3D BBOX transfer run with high-utilization fp16 AMP."""

_base_ = ["./nocs_custom_fruit_rgbd_3dbbox_transfer.py"]

# Batch 27 is finite for one iteration but OOMs on the next data batch; batch
# 28 OOMs immediately. Batch 26 is finite for four consecutive updates at a
# measured peak of 19,326 / 20,475 MiB, so it is the safe high-utilization
# setting without storing activations for microbatches.
train_dataloader = dict(
    batch_size=26,
    num_workers=4,
    persistent_workers=True,
)

max_epochs = 20
train_cfg = dict(
    type="EpochBasedTrainLoop",
    max_epochs=max_epochs,
    val_interval=1,
)

# With 12 optimizer updates per epoch, preserve an approximately two-epoch
# warm-up, then decay across this deliberately bounded 20-epoch transfer run.
param_scheduler = [
    dict(
        type="LinearLR",
        start_factor=1e-3,
        by_epoch=False,
        begin=0,
        end=24,
    ),
    dict(
        type="CosineAnnealingLR",
        T_max=max_epochs,
        eta_min=1e-6,
        begin=0,
        end=max_epochs,
        by_epoch=True,
    ),
]

# The real batch is close to the previous effective batch 24, so do not retain
# three microbatch graphs. Keep the AMP-aware ScheduleFree wrapper.  Unlike the
# inherited scratch run, this transfer still initializes its 3D decoder/head
# randomly; use the conventional DeformDETR-scale base LR and retain inherited
# backbone/encoder multipliers.
optim_wrapper = dict(
    type="AmpScheduleFreeOptimWrapper",
    accumulative_counts=1,
    dtype="float16",
    loss_scale=1.0,
    optimizer=dict(lr=2e-4),
)

# Run NOCSMetric on every real validation split and retain only its best model
# weights.  This intentionally avoids a Top-K checkpoint pool and optimizer
# state to keep storage bounded.
default_hooks = dict(
    checkpoint=dict(
        _delete_=True,
        type="CheckpointHook",
        interval=100,
        save_last=False,
        save_best="3d_iou_0.50",
        rule="greater",
        save_optimizer=False,
    ),
    logger=dict(interval=1),
)

# This detector produces exactly ``num_queries * num_classes`` scores per
# image (100 * 1 for the custom fruit task).  Asking predict() for the
# inherited 300 candidates makes torch.topk fail before NOCSMetric runs.
model = dict(test_cfg=dict(max_per_img=100))
