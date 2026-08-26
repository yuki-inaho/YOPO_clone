"""Long fresh run for the accepted Stage-10 MAL configuration.

Run only after the five-epoch MAL probe passes the documented 2D/3D gate.
Start from the Stage-8 foundation checkpoint supplied at launch; do not call
this a resume because the probe intentionally does not save optimizer state.
Early stopping bounds the nominal 100 epochs at the measured AP plateau.
"""

_base_ = [
    './nocs_fruits_detection_Jun30_2025_rgbd_3dbbox_'
    'stage10_2d_anchor_mal_probe5.py'
]

# The Stage-8 foundation checkpoint is intentionally supplied at launch via
# ``--cfg-options load_from=...``.  This keeps the config portable and avoids
# encoding an artifact-root-specific absolute path.
load_from = None
resume = False

max_epochs = 100
train_cfg = dict(max_epochs=max_epochs, val_interval=5)

custom_hooks = [
    dict(type='ScheduleFreeOptimizerModeHook'),
    dict(
        type='EarlyStoppingHook',
        monitor='AP50_95',
        rule='greater',
        min_delta=2e-3,
        patience=6,
        strict=True,
        check_finite=True,
    ),
]

default_hooks = dict(
    checkpoint=dict(
        _delete_=True,
        type='CheckpointHook',
        interval=5,
        save_last=True,
        max_keep_ckpts=2,
        save_best=['AP50_95', 'AP50', '3d_iou_0.50'],
        rule=['greater', 'greater', 'greater'],
        # Periodic epoch_N checkpoints must remain exactly resumable, including
        # ScheduleFree's optimizer state.  Metric best_*.pth files are
        # intentionally weights-only; ``max_keep_ckpts`` bounds periodic files.
        save_optimizer=True,
    ),
    logger=dict(interval=5),
)
