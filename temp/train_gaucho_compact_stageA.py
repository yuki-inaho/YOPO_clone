"""Stage A head warm-up on the pinned environment.

Everything except the GauCho 2D ellipse branch is warm-started from the
released compact checkpoint -- loading it leaves exactly the 30 new tensors
missing and nothing unexpected -- so the detector keeps the compact chain's low
base learning rate and its bfloat16 AMP, while the one randomly initialized
module gets a larger multiplier.  Nothing is frozen: Stage A of the design note
is "establish the 2D task", not "freeze the backbone".

The RGB backbone's ``init_cfg`` is disabled because ``load_from`` overwrites
those weights immediately; fetching them is pure waste and the download stalls
on this link.
"""

_base_ = ["../configs/yopo/nocs_fruits_Jun30_2025_rgbd_gaucho_compact_stageA.py"]

max_epochs = 20
train_cfg = dict(type="EpochBasedTrainLoop", max_epochs=max_epochs,
                 val_interval=2)

model = dict(backbone=dict(rgb_backbone=dict(init_cfg=None)))

train_dataloader = dict(batch_size=16, num_workers=8)

# Merges into the compact chain's custom_keys rather than replacing them.
optim_wrapper = dict(
    paramwise_cfg=dict(
        custom_keys={
            "bbox_head.reg_ellipse2d_branch": dict(lr_mult=10.0),
        }
    )
)

custom_hooks = [
    dict(type="ScheduleFreeOptimizerModeHook"),
    dict(
        type="EarlyStoppingHook",
        monitor="ellipse/rbbox_mAP_50",
        rule="greater",
        min_delta=1e-3,
        patience=4,
        strict=False,
        check_finite=True,
    ),
]

default_hooks = dict(
    logger=dict(type="LoggerHook", interval=20),
    checkpoint=dict(
        _delete_=True,
        type="CheckpointHook",
        interval=2,
        max_keep_ckpts=3,
        save_best="ellipse/rbbox_mAP_50",
        rule="greater",
    ),
)
