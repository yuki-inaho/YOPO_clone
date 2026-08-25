"""Projection GWD plus a lower-weight query-level OBB auxiliary head."""

_base_ = [
    "./nocs_custom_fruit_rgbd_3dbbox_cop_q150_stage4_projection_gwd_obb_aux.py"
]

# Weight 5 improved 3D IoU but slightly worsened matched SO(3) error. Keep
# every other condition fixed and reduce only the auxiliary supervision scale.
model = dict(
    bbox_head=dict(
        loss_obb_aux=dict(
            loss_weight=1.0,
        ),
    ),
)
