"""Stage B: add the metric 3D GauCho ellipsoid on top of Stage A.

The canonical 3D state becomes ``Ellipsoid3D(t, Sigma)``.  ``Sigma`` is built
from a scale--shape Cholesky chart, so the GauCho branch itself regresses no
rotation.

This is a HYBRID, not the pure GauCho-3D of the design note: the inherited
compact chain keeps its 6D rotation head and ``loss_rotation`` (weight 5.0), so
a full SO(3) pose objective is still trained alongside the SPD shape.  That is
a deliberate deviation from section 8.4, kept because the released checkpoint
this warm-starts from was optimized under that loss.  The 3D
centre reuses the already-supervised 2D centre and depth branches rather than
adding a second, competing translation estimator.

Direct 3D supervision only; the dual-quadric projection term arrives in Stage C
so the two heads cannot start by agreeing with each other.
"""

_base_ = ["./nocs_fruits_Jun30_2025_rgbd_gaucho_stageA_ellipse2d.py"]

model = dict(
    bbox_head=dict(
        # The pipeline resizes, so the annotation this stage is
        # supervised against lives in resized image pixels and the 3D centres are back-projected with this K.
        # Without this the stored original-image K is used and the
        # error hides entirely in the shape term.
        train_intrinsic_to_image_space=True,
        gaucho_ellipsoid=True,
        gaucho_chart="scale_shape",
        # Tomato fruit and stem radii are centimetre-scale.  The prior fixes
        # where a zero-initialized head starts; it is not a constraint.
        gaucho_size_prior=0.03,
        gaucho_use_bbox_conditioning=True,
        loss_ellipsoid=dict(
            type="Ellipsoid3DKLDLoss",
            loss_weight=2.0,
            tau=1.0,
            include_center=True,
            fail_on_invalid=False,
        ),
    ),
)

load_from = None
resume = False
