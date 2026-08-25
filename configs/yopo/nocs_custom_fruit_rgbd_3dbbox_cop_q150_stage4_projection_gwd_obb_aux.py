"""Projection GWD plus query-level 2D OBB Gaussian auxiliary head."""

_base_ = ["./nocs_custom_fruit_rgbd_3dbbox_cop_q150_stage4_projection_gwd.py"]

model = dict(
    bbox_head=dict(
        loss_obb_aux=dict(
            type="GaussianGWDLoss",
            loss_weight=5.0,
            tau=1.0,
            normalize=True,
            include_center=False,
            fail_on_invalid=True,
        ),
    ),
)
