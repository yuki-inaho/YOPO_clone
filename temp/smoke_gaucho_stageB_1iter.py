"""Few-iteration GPU smoke for the GauCho 2D ellipse + 3D ellipsoid path.

Stage B rather than Stage C: the dual-quadric projection term routes through
``projected_ellipsoid_loss.gaussian_wasserstein_distance``, which uses the
multi-dimensional ``Tensor.all(dim=tuple)`` form introduced after torch 2.1.
Stage B's objectives are self-contained, so this smoke runs on the older
interim stack as well as on the pinned one.

Run from the repository root.

Note: written while the pinned environment was still installing, hence
the Stage B rather than Stage C base.  The pinned stack runs Stage C
fine; use temp/train_gaucho_stageB_native.py for real Stage B runs.
"""

_base_ = ["../configs/yopo/nocs_fruits_Jun30_2025_rgbd_gaucho_stageB_ellipsoid.py"]

max_epochs = 1
train_cfg = dict(type="EpochBasedTrainLoop", max_epochs=max_epochs,
                 val_interval=max_epochs + 1)

# A handful of real RGB-D frames is enough to exercise dataloader -> Cholesky
# chart -> loss -> backward -> checkpoint.
train_dataloader = dict(
    batch_size=1,
    num_workers=0,
    persistent_workers=False,
    dataset=dict(indices=4),
)

val_dataloader = None
val_cfg = None
val_evaluator = None
test_dataloader = None
test_cfg = None
test_evaluator = None

default_hooks = dict(
    logger=dict(type="LoggerHook", interval=1),
    # ``_delete_`` matters: the inherited hook is a TopKCheckpointHook and a
    # plain merge would leave its ``topk`` key behind for a hook that does not
    # accept it.
    checkpoint=dict(
        _delete_=True,
        type="CheckpointHook",
        interval=1,
        max_keep_ckpts=1,
    ),
)

# The base config's RGB transfer hook and the depth backbone's MAE
# initialization both need artifacts that a geometry smoke has no use for.
# Random initialization is fine: this run asserts that the chain executes and
# stays finite, never that it is accurate.
model = dict(
    backbone=dict(depth_backbone=dict(init_cfg=None)),
)
custom_hooks = []
load_from = None
resume = False
