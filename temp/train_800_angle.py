"""Explicit, eccentricity-weighted angle supervision -- the diagnosed gap.

Where the 2D metric is lost, measured over matched pairs::

    centre_error_rel   median 0.086     long_ratio  median 0.989
    angle_error_deg    median 17.2      short_ratio median 1.035
                       p90    67.1      aspect_gt   median 1.200

Centre and extent are within a few percent.  Reproducing the measured
distribution numerically costs 0.20 of IoU from the angle alone against 0.12
from centre and extent together, which lands on the measured 0.706 matched IoU.

Two things were ruled out first.  Tripling the ellipse KLD weight moved matched
IoU by 0.0006 -- the KLD reduces to the Frobenius norm of ``L_p^{-1} L_g``,
nearly constant under rotation once the shape is near-circular, so there is no
rotation gradient to recruit.  A corner-set Chamfer term made it worse (angle
p90 67 -> 75 degrees) because comparing corners as a set makes a quarter turn
nearly free.

The reference implementation supervises the angle explicitly instead, and
weights that term by the target's eccentricity, ``1 - (b/a)^2``: a circle has no
identifiable orientation, so charging for its angle is charging for annotation
noise, while an elongated target is charged in full.  The geodesic form is used
rather than ``1 - cos`` because the latter has zero gradient at exactly the 90
degree error where the observed tail sits.

Weight 2.0 puts it on the same scale as the KLD at the measured error.  The KLD
stays and keeps scale and shape well-posed; nothing else changes from the
best-2D baseline.
"""

_base_ = ["./gaucho_800x600_base.py"]

model = dict(
    bbox_head=dict(
        loss_ellipsoid=dict(include_center=False),
        loss_ellipse2d_angle=dict(
            type="EllipseAngleGeodesicLoss",
            loss_weight=2.0,
            eccentricity_floor=0.0,
        ),
    ),
)

max_epochs = 16
train_cfg = dict(type="EpochBasedTrainLoop", max_epochs=max_epochs,
                 val_interval=2)

load_from = None
