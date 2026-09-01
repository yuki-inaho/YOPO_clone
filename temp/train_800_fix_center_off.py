"""800x600 Stage B with the KLD centre term removed -- the fix under test.

The centre term of ``D_KL(target || prediction)`` is a Mahalanobis norm under
the *predicted* shape, so a displaced centre can be paid for by growing Sigma
along the displacement.  That is not a training artefact: the closed-form
minimiser over Sigma at a fixed centre error ``d`` is ``Sigma_gt + d d^T``, and
for this data's measured 23 mm median range error and 18 mm ground-truth extent
it predicts a 2.74x optical-axis bloat.  The trained model measured 2.87x.

The centre is already supervised, twice, by ``loss_centers_2d`` and ``loss_z``.
Dropping it here leaves the KLD as a pure shape objective and removes the
incentive entirely -- ``tests/test_gaucho3d_shared_match_metric.py`` pins that
the gradient on the error axis goes from strongly negative to zero.
"""

_base_ = ["./gaucho_800x600_base.py"]

model = dict(
    bbox_head=dict(
        loss_ellipsoid=dict(include_center=False),
    ),
)
