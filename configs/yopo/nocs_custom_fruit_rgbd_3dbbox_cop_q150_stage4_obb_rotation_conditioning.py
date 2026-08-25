"""Feed a supervised 2D OBB spin-2 descriptor into CoP rotation."""

_base_ = [
    "./nocs_custom_fruit_rgbd_3dbbox_cop_q150_stage4_projection_gwd_obb_aux_w1.py"
]

# Keep the successful low-weight OBB supervision and the direct projection
# loss, but make the OBB covariance an explicit input to the final rotation
# stage.  The head uses the scale-free spin-2 descriptor
# ((sigma_xx-sigma_yy)/trace, 2*sigma_xy/trace).
model = dict(
    bbox_head=dict(
        cop_obb_rotation_conditioning=True,
    ),
)
