"""Add the envelope corner term, which is the only thing that targets the angle.

The 2D residual was decomposed over matched pairs: centre within 8.6% of object
scale, extents within 3%, and an angle error of 17 degrees median / 67 degrees
at p90.  Reproducing that distribution numerically costs 0.20 of IoU from the
angle alone against 0.12 from centre and extent together, landing on the
measured 0.706 matched IoU.

Tripling the ellipse KLD weight moved that number by 0.0006, and there is a
structural reason: the KLD reduces to the Frobenius norm of ``L_p^{-1} L_g``,
which for a near-circular shape is nearly a rotation and nearly constant --
the objects here have a median aspect ratio of 1.20.  The gradient in the
rotation direction essentially does not exist, so no amount of weight can
recruit it.

``EllipseEnvelopeCornerLoss`` compares the corners of the oriented envelope,
which is exactly what the metric compares, and is strongly angle-sensitive at
the same aspect ratios: 0.008 at 5 degrees, 0.087 at 17, 0.470 at 45, and back
down to 0.033 at 90 -- tracking the metric's own shape, including the fact that
a near-square box rotated a quarter turn is almost itself again.

The KLD stays: it keeps scale and shape well-posed, and the corner term is
deliberately an auxiliary.  Weight 5.0 puts the two on a comparable scale at
the measured error; nothing else changes from the best-2D baseline.
"""

_base_ = ["./gaucho_800x600_base.py"]

model = dict(
    bbox_head=dict(
        loss_ellipsoid=dict(include_center=False),
        loss_ellipse2d_corner=dict(
            type="EllipseEnvelopeCornerLoss",
            loss_weight=5.0,
        ),
    ),
)

max_epochs = 16
train_cfg = dict(type="EpochBasedTrainLoop", max_epochs=max_epochs,
                 val_interval=2)

load_from = None
