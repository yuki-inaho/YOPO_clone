"""Stage C at native resolution: one-way dual-quadric projection consistency.

``Q* -> C* = P Q* P^T`` is the only exact 3D -> 2D path in the design note.
Perspective Jacobian covariance transport and corner projection are
approximations and are not used anywhere in this implementation.

The 2D side is detached.  Optimizing ``2D head <-> projected 3D head`` alone
lets both collapse onto a common wrong ellipse, so Stage B's direct 3D
objective stays on as the second, independent ground of supervision -- the head
refuses to build without it.

The projection target is the OBB Gaussian *after* the resize pipeline, so the
dual quadric has to be projected with the image-space intrinsic.  The head now
fails closed if that flag is off; the compact chain sets it.
"""

_base_ = ["./train_gaucho_stageB_native.py"]

model = dict(
    bbox_head=dict(
        loss_ellipsoid_projection=dict(
            type="DualQuadricProjectionGWDLoss",
            loss_weight=1.0,
            tau=1.0,
            normalize=True,
            include_center=True,
            detach_target=True,
            fail_on_invalid=False,
        ),
    ),
)

max_epochs = 20
train_cfg = dict(type="EpochBasedTrainLoop", max_epochs=max_epochs,
                 val_interval=2)

optim_wrapper = dict(
    paramwise_cfg=dict(
        custom_keys={
            # Both GauCho heads are trained by now; no branch is privileged.
            "bbox_head.reg_ellipsoid_branch": dict(lr_mult=1.0),
            "bbox_head.reg_ellipse2d_branch": dict(lr_mult=1.0),
        }
    ),
)

load_from = None
