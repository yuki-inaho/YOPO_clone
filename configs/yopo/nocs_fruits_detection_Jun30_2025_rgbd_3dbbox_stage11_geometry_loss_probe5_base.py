"""Shared five-epoch contract for independent Stage-11 loss probes.

Launch each child config from the *same* Stage-10 epoch-85 best checkpoint
with ``--cfg-options load_from=/absolute/path/to/best_AP50_95_epoch_85.pth``.
Do not pass ``--resume``: these are model-only branches with fresh optimizer
state, not continuations of the stopped 100-epoch run.

The mature checkpoint used a base LR of 1e-5.  A 3e-6 probe LR limits drift
while still allowing the neck/decoder/head to react within five epochs.  The
existing param-wise multipliers are inherited unchanged.
"""

_base_ = [
    './nocs_fruits_detection_Jun30_2025_rgbd_3dbbox_'
    'stage10_2d_anchor_mal_full.py'
]

# Keep artifact paths out of the repository config.  ``resume=False`` makes
# any CLI ``load_from`` checkpoint a weights-only initialization even if the
# source file happens to contain optimizer metadata.
load_from = None
resume = False

optim_wrapper = dict(
    optimizer=dict(lr=3e-6),
)
param_scheduler = []

probe_epochs = 5
train_cfg = dict(max_epochs=probe_epochs, val_interval=1)

# ScheduleFree must be put into train/eval mode around loops.  Remove only the
# full run's AP early-stopping hook; a fixed five-epoch horizon is the fair
# comparison contract for both independent probes.
custom_hooks = [
    dict(type='ScheduleFreeOptimizerModeHook'),
]

# Evaluate every epoch with the inherited NOCS evaluator (including corrected
# full-SO(3) 3D OBB IoU).  Best files are deliberately weights-only so that a
# successful probe can seed another independent experiment without carrying
# optimizer history across variants.
default_hooks = dict(
    checkpoint=dict(
        _delete_=True,
        type='CheckpointHook',
        interval=1,
        save_last=True,
        max_keep_ckpts=2,
        save_best=['AP50_95', 'AP50', '3d_iou_0.50'],
        rule=['greater', 'greater', 'greater'],
        save_optimizer=False,
    ),
    logger=dict(interval=5),
)
