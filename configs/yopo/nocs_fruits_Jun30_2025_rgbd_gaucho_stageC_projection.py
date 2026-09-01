"""Stage C: one-way dual-quadric projection consistency.

``Q* -> C* = P Q* P^T`` is the only exact 3D -> 2D path; perspective Jacobian
covariance transport and corner projection are approximations and are not used.

The 2D side is detached.  Optimizing ``2D head <-> projected 3D head`` alone
would let both collapse onto a common wrong ellipse, so the direct 3D objective
from Stage B stays on as the second, independent ground of supervision -- the
head refuses to build otherwise.

HYBRID, like Stage B: the inherited chain keeps its 6D rotation head and
``loss_rotation``, and also the older approximate ``loss_projection`` /
``loss_obb_aux``.  So this stage does not introduce projection -- it adds the
*exact* dual-quadric projection alongside the approximate one already running.
Both deviations from the design note are deliberate and recorded in
temp/workdoc_Aug31-2026_gaucho3d_rgbd_ellipsoid.md.
"""

_base_ = ["./nocs_fruits_Jun30_2025_rgbd_gaucho_stageB_ellipsoid.py"]

model = dict(
    bbox_head=dict(
        # The pipeline resizes, so the annotation this stage is
        # supervised against lives in resized image pixels and the dual quadric is projected with this K.
        # Without this the stored original-image K is used and the
        # error hides entirely in the shape term.
        train_intrinsic_to_image_space=True,
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

load_from = None
resume = False
