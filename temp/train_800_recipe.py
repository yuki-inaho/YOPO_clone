"""Match the reference's training recipe, which is where the difference is.

The geometry oracle settles the direction.  Replacing every overlapping
prediction's box with its annotation's box -- perfect ellipse geometry, same
scores, same duplicates -- scores 0.8215 against a 0.8170 target.  So geometry
could reach the target only by removing 95% of its own error, and the detector
itself is the binding constraint.  Adding geometry losses was the wrong lever,
which the reference confirms from the other side: its best model regresses
geometry with the KLD *alone*, with the angle and axis terms all at zero.

What is different is the recipe.  The reference trains at physical batch 48 with
Schedule-Free Adam at 2.5e-4 (Muon 1e-3), and climbs 0.405 -> 0.782 -> 0.801 ->
0.838 -> 0.851 over 22 epochs.  This work has been running batch 20 at 1e-4 and
plateauing at 0.73.

This run takes the reference's learning rate and as much batch as 24 GB allows,
and runs long enough to see the trend rather than the first bounce.  Geometry
losses are exactly the baseline's.
"""

_base_ = ["./gaucho_800x600_base.py"]

model = dict(bbox_head=dict(loss_ellipsoid=dict(include_center=False)))

# 800x557 at batch 20 measured 18.9 GB allocated; 24 lands near 22 GB, which is
# the most that leaves room for allocator fragmentation over a long run.
train_dataloader = dict(batch_size=24)

# The reference's Schedule-Free Adam rate.  The inherited chain uses 1e-4, and
# the per-branch multipliers below it are kept, so this scales the whole
# schedule rather than re-tuning it branch by branch.
optim_wrapper = dict(optimizer=dict(lr=2.5e-4))

max_epochs = 32
train_cfg = dict(type="EpochBasedTrainLoop", max_epochs=max_epochs,
                 val_interval=2)

load_from = None
