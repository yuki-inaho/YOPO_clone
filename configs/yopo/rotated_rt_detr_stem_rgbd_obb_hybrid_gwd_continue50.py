"""Long GWD continuation from the selected 15-epoch hybrid checkpoint."""

_base_ = ['./rotated_rt_detr_stem_rgbd_obb_hybrid_gwd_full.py']

max_epochs = 50
train_cfg = dict(max_epochs=max_epochs, val_interval=1)

# Preserve the useful high-LR phase, then reduce only for the final quarter.
param_scheduler = [
    dict(
        type='MultiStepLR', begin=0, end=max_epochs,
        by_epoch=True, milestones=[38], gamma=0.1,
    ),
]

# Supplied explicitly from the preceding run's measured best checkpoint.
load_from = None
resume = False
