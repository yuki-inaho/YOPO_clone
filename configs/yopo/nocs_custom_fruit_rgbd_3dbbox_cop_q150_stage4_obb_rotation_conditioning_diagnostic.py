"""Teacher-free conditioned inference exposing query OBB Gaussians."""

_base_ = [
    "./nocs_custom_fruit_rgbd_3dbbox_cop_q150_stage4_obb_rotation_conditioning_inference.py"
]

model = dict(
    bbox_head=dict(
        expose_obb_aux_predictions=True,
    ),
)
