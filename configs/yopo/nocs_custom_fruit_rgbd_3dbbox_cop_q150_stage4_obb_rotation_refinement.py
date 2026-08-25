"""Refine final rotation with a supervised post-rotation OBB descriptor."""

_base_ = [
    "./nocs_custom_fruit_rgbd_3dbbox_cop_q150_stage4_projection_gwd_obb_aux_w1.py"
]

model = dict(
    bbox_head=dict(
        cop_obb_rotation_refinement=True,
    ),
)
