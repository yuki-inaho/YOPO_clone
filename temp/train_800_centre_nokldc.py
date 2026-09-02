"""Take the centre out of the 2D KLD, now that something else carries it.

The explicit centre term moved rotated-NMS mAP50 from 0.7261 to 0.7365, but the
centre error itself did not move (0.0851 -> 0.0855 of object scale), so the gain
came from elsewhere and the centre is still not being fixed.

The 2D KLD still has ``include_center=True``, and its centre term is a
Mahalanobis norm under the *predicted* shape.  That is the exact structure the
3D branch was diagnosed with and cured of: a displaced centre can be paid for by
inflating the ellipse, and the closed-form optimum at a fixed displacement is
``Sigma_gt + d d^T``.  The measurement agrees that it is happening here too --
the predicted extents run 1.035 short-axis and 1.025 in area against the
annotation.  It also means the term pulls only weakly on the centre, which is
why tripling the ellipse weight did nothing.

With ``EllipseCentreLoss`` charging the centre in units of object size, the
KLD's own centre term is now both redundant and counterproductive, so it is
switched off exactly as it was in 3D.  Single factor against the run before it.
"""

_base_ = ["./train_800_centre.py"]

model = dict(
    bbox_head=dict(
        loss_ellipse2d=dict(include_center=False),
    ),
)
