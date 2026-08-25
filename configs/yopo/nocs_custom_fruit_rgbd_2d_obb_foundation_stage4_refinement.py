"""Test OBB-to-rotation refinement after the 2D OBB foundation."""

_base_ = ["./nocs_custom_fruit_rgbd_2d_obb_foundation_stage4_control.py"]

# This is the only model change relative to the matched recovery control.  The
# descriptor is predicted at the already-supervised post-rotation OBB feature
# and is fed back only to recompute the final rotation output.
model = dict(
    bbox_head=dict(
        cop_obb_rotation_refinement=True,
    ),
)
