"""Teacher-free legacy OBB-aux inference exposing query Gaussians."""

_base_ = [
    "./nocs_custom_fruit_rgbd_3dbbox_cop_q150_stage4_projection_gwd_obb_aux_w1_inference.py"
]

model = dict(
    bbox_head=dict(
        expose_obb_aux_predictions=True,
    ),
)
