"""Anchored depth with a robust z loss, because the anchor's error has a tail.

With the anchor in place and its residual initialised to zero, the model starts
at exactly "trust the sensor", which scores shared AP20 0.386.  Training moves
it *down*, to a plateau of 0.23-0.26 that two independent runs agree on and that
more epochs do not improve -- the 16-epoch run peaked at 0.258 on epoch 10 and
the 40-epoch run at 0.249 on epoch 8, both declining afterwards.

So the objective disagrees with the metric, and the shape of the anchor's error
says how.  Read at the annotated centre it is 5.2 mm at the median but 63 mm at
p90; read at the *predicted* centre, which is what training sees, the tail runs
to 109 mm.  ``loss_z`` is L2, so that tail dominates the gradient and the
cheapest way to reduce it is to pull the residual toward the mean -- which
spoils the many centres where the sensor was already right.

L1 charges the tail linearly instead of quadratically.  Nothing else changes,
including the weight, so the comparison against the L2 runs is single-factor.
"""

_base_ = ["./train_1024_anchor.py"]

model = dict(
    bbox_head=dict(
        loss_z=dict(_delete_=True, type="L1PoseLoss", loss_weight=50.0),
    ),
)

max_epochs = 16
train_cfg = dict(type="EpochBasedTrainLoop", max_epochs=max_epochs,
                 val_interval=2)
