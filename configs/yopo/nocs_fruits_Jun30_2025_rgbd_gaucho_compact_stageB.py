"""Stage B on the compact architecture: add the metric 3D GauCho ellipsoid.

The canonical 3D state becomes ``Ellipsoid3D(t, Sigma)`` with ``Sigma`` built
from a scale--shape Cholesky chart, so the GauCho branch itself regresses no
rotation.

This is a HYBRID, not the pure GauCho-3D of the design note: the inherited
compact chain keeps its 6D rotation head and ``loss_rotation`` (weight 5.0), so
at the model level a full SO(3) pose objective is still trained alongside the
SPD shape.  That is a deliberate deviation from the note's section 8.4 -- the
released checkpoint this warm-starts from was optimized under that loss, and
removing it is a separate, measured experiment.  The 3D centre
reuses the already-supervised 2D centre and depth branches rather than adding a
second, competing translation estimator.

Direct 3D supervision only.  The dual-quadric projection term waits for Stage C
so the 2D and 3D heads cannot begin by simply agreeing with each other.
"""

_base_ = ["./nocs_fruits_Jun30_2025_rgbd_gaucho_compact_stageA.py"]

model = dict(
    bbox_head=dict(
        gaucho_ellipsoid=True,
        gaucho_chart="scale_shape",
        # Tomato stem radii are centimetre-scale.  The prior only fixes where a
        # zero-initialized head starts; it is not a constraint on the output.
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

# Supplied at launch as Stage A's best checkpoint.  Pointing back at the
# released model here would silently discard the previous stage.
load_from = None
