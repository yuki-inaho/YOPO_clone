"""Rotation-only projection GWD weighted by observed OBB anisotropy."""

_base_ = [
    "./nocs_custom_fruit_rgbd_3dbbox_cop_q150_stage4_projection_gwd.py"
]

# Continuous observability weighting:
#   a = (lambda_max - lambda_min) / (lambda_max + lambda_min)
#   w = a / mean_positive(a)
# The loss implementation normalizes the batch weights to mean one, so this
# changes which instances supply rotation gradient without changing its scale.
model = dict(
    bbox_head=dict(
        loss_projection=dict(
            target_anisotropy_power=1.0,
        ),
    ),
)
