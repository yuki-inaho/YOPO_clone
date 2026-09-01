"""Stage C on the compact architecture: one-way dual-quadric consistency.

``Q* -> C* = P Q* P^T`` is the only exact 3D -> 2D path.  Perspective Jacobian
covariance transport and corner projection are approximations and are not used.

The 2D side is detached: optimizing ``2D head <-> projected 3D head`` alone
lets both collapse onto a common wrong ellipse.  Stage B's direct 3D objective
therefore stays on as the second, independent ground of supervision -- the head
refuses to build without it.

HYBRID, like Stage B: the inherited compact chain keeps its 6D rotation head
and ``loss_rotation`` (weight 5.0), and it also keeps the older approximate
``loss_projection``/``loss_obb_aux`` terms.  So this stage does not introduce
projection -- it adds the *exact* dual-quadric projection alongside the
approximate one already running.  Both deviations from the design note are
deliberate and are recorded in temp/workdoc_Aug31-2026_gaucho3d_rgbd_ellipsoid.md.
"""

_base_ = ["./nocs_fruits_Jun30_2025_rgbd_gaucho_compact_stageB.py"]

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

# Supplied at launch as Stage B's best checkpoint.  Pointing back at the
# released model here would silently discard the previous stage.
load_from = None
