"""Stage A FULL: adapt the whole detector to this dataset, not just the head.

The warm-up stage plateaued with the ellipse AP pinned at ~95% of the
detector's own AP50 -- the ellipse head reproduces whatever the detector finds
(matched rotated IoU ~0.687), so the remaining gap is detector quality on this
dataset, not ellipse geometry.  The released checkpoint was trained on the
2025+2026 tomato set; this is stem data at a different resolution.

So this stage stops privileging the new head and lets everything adapt, which
is the same warm-up -> FULL shape the 2D reference implementation used (its
warm-up ended at 0.004 and FULL carried it to 0.851).

The learning rate is the repository's own "learn the task" value: the compact
curriculum uses 1e-4 for stages 1-3 and halves it to 5e-5 only for the final
refinement.  Nothing here is invented.
"""

_base_ = ["../configs/yopo/nocs_fruits_Jun30_2025_rgbd_gaucho_compact_stageA.py"]

max_epochs = 40
train_cfg = dict(type="EpochBasedTrainLoop", max_epochs=max_epochs,
                 val_interval=2)

model = dict(backbone=dict(rgb_backbone=dict(init_cfg=None)))

train_dataloader = dict(batch_size=16, num_workers=8)

optim_wrapper = dict(
    optimizer=dict(lr=1e-4),
    paramwise_cfg=dict(
        custom_keys={
            # The head is warmed up now; train it like every other head.
            "bbox_head.reg_ellipse2d_branch": dict(lr_mult=1.0),
        }
    ),
)

custom_hooks = [
    dict(type="ScheduleFreeOptimizerModeHook"),
    dict(
        type="EarlyStoppingHook",
        monitor="ellipse/rbbox_mAP_50",
        rule="greater",
        min_delta=1e-3,
        patience=6,
        strict=False,
        check_finite=True,
    ),
]

default_hooks = dict(
    logger=dict(type="LoggerHook", interval=20),
    checkpoint=dict(
        _delete_=True,
        type="CheckpointHook",
        interval=4,
        max_keep_ckpts=3,
        save_best="ellipse/rbbox_mAP_50",
        rule="greater",
    ),
)

# Set at launch to the warm-up stage's best checkpoint.
load_from = None
