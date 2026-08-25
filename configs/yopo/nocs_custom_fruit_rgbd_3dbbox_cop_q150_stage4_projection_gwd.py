"""Rotation-only 3D ellipsoid -> 2D OBB projection GWD ablation."""

_base_ = ["./nocs_custom_fruit_rgbd_3dbbox_cop_q150_stage4_continue_control.py"]

model = dict(
    bbox_head=dict(
        projection_geometry_source="target",
        loss_projection=dict(
            _delete_=True,
            type="ProjectedEllipsoidGWDLoss",
            loss_weight=1.0,
            tau=1.0,
            normalize=True,
            detach_center=True,
            detach_depth=True,
            detach_size=True,
            fail_on_invalid=True,
        ),
    ),
)
