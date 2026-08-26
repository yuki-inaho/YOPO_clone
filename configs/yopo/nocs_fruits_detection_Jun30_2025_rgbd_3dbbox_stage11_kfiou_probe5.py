"""Independent five-epoch covariance KFIoU probe from Stage-10 epoch 85.

This changes only the compact 2D OBB auxiliary objective.  A 20-iteration
calibration from the epoch-85 checkpoint measured about 0.010 per decoder
layer at weight 0.25, versus about 0.095 for the mature GWD term.  Weight 1.0
therefore gives a conservative expected contribution of about 0.040 per
layer: large enough to test the geometry while remaining below the replaced
objective.  Stop the full probe if any ``loss_obb_aux`` exceeds 0.15.
"""

_base_ = [
    './nocs_fruits_detection_Jun30_2025_rgbd_3dbbox_'
    'stage11_geometry_loss_probe5_base.py'
]

model = dict(
    bbox_head=dict(
        loss_obb_aux=dict(
            _delete_=True,
            type='GaussianKFIoULoss',
            loss_weight=1.0,
            fail_on_invalid=True,
            eps=1e-7,
        ),
    ),
)
