"""One-iteration GPU smoke for the GauCho 2D/3D ellipse path.

Proves the whole chain runs end to end on real RGB-D data: dataloader ->
Cholesky charts -> dual quadric -> losses -> backward -> checkpoint.  It makes
no claim about accuracy.

Usage from the repository root::

    .venv/bin/python tools/train.py temp/smoke_gaucho3d_1iter.py \
        --work-dir work_dirs/smoke_gaucho3d
"""

_base_ = ["../configs/yopo/nocs_fruits_Jun30_2025_rgbd_gaucho_stageC_projection.py"]

max_epochs = 1
train_cfg = dict(type="EpochBasedTrainLoop", max_epochs=max_epochs,
                 val_interval=max_epochs)

# Keep the smoke inside a single GPU and a handful of iterations.
train_dataloader = dict(batch_size=1, num_workers=0, persistent_workers=False)
val_dataloader = dict(batch_size=1, num_workers=0, persistent_workers=False)

default_hooks = dict(
    logger=dict(type="LoggerHook", interval=1),
    checkpoint=dict(type="CheckpointHook", interval=1, max_keep_ckpts=1),
)

# Any invalid geometry must surface as a validity count, never stop the smoke.
load_from = None
resume = False
