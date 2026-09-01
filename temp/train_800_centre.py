"""Charge the ellipse centre directly, which is where the 2D AP actually is.

Decomposing the 2D residual by AP -- replacing one part of each matched
prediction's geometry with the annotation's and re-scoring -- gives::

    replace nothing   mAP50 0.7312
    replace centre          0.7949   (+0.0637)
    replace extent          0.7503   (+0.0191)
    replace angle           0.7331   (+0.0019)
    replace everything      0.8215   (+0.0903)

The centre carries 70% of the headroom and the angle carries two thousandths,
which is the opposite of what the IoU-level decomposition suggested and why the
two orientation losses built on that reading did nothing.

The ellipse's centre is currently supervised only through the KLD, whose centre
term is a Mahalanobis norm under the predicted shape -- the same structure that
let the 3D branch absorb a centre error by inflating itself.  Normalised that
way it barely pulls, which is why tripling the ellipse weight moved the matched
overlap by 0.0006.  ``EllipseCentreLoss`` charges it in units of the target's
own size instead.

Built on the current best model, so the depth anchor and its window are
unchanged and the difference is attributable to this term.
"""

_base_ = ["./train_800_anchor_w17.py"]

model = dict(
    bbox_head=dict(
        loss_ellipse2d_centre=dict(
            type="EllipseCentreLoss",
            loss_weight=5.0,
            beta=0.05,
        ),
    ),
)
