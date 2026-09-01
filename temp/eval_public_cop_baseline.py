"""Evaluate the released compact CoP checkpoint on OUR split, for a DoD-B baseline.

The released model reports 3D IoU@0.25 = 0.0166 on its own 2025+2026 validation
split, which is not the split anything here is measured on.  Running it through
the same dataloader and the same evaluator as the GauCho runs produces the only
number that is actually comparable.

The GauCho branches stay off: this checkpoint has no such weights, and the
point is to measure the model as released.
"""

_base_ = ["./train_gaucho_stageB_native.py"]

model = dict(
    bbox_head=dict(
        gaucho_ellipse2d=False,
        gaucho_ellipsoid=False,
        loss_ellipse2d=None,
        loss_ellipsoid=None,
        expose_gaucho_predictions=False,
    ),
)
