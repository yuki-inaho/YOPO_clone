"""Is the ellipse branch stuck on optimisation, or on capacity?

Over eight epochs from a warm start, neither ellipse loss moved at all::

    ep   loss_ellipse2d   loss_ellipse2d_angle
     1           0.1665                 0.1399
     8           0.1669                 0.1408

That is the *training* loss, so the model is not fitting the data it is shown
-- this is not a generalisation gap and not a weighting problem.  Two
explanations remain: the gradient reaching the ellipse branch is too small to
move it against the rest of the objective, or the branch cannot express a
better ellipse from the features it is given.

This run removes every excuse on the optimisation side -- the angle term is
weighted an order of magnitude up and the ellipse branches get a twenty-fold
learning rate.  If the training angle loss still does not move, the limit is
representational, and the next step is the architecture rather than the
objective.
"""

_base_ = ["./train_800_angle.py"]

model = dict(
    bbox_head=dict(
        loss_ellipse2d_angle=dict(loss_weight=20.0),
    ),
)

optim_wrapper = dict(
    paramwise_cfg=dict(
        custom_keys={
            "bbox_head.reg_ellipsoid_branch": dict(lr_mult=10.0),
            "bbox_head.reg_ellipse2d_branch": dict(lr_mult=20.0),
        }
    ),
)

max_epochs = 6
train_cfg = dict(type="EpochBasedTrainLoop", max_epochs=max_epochs,
                 val_interval=2)
