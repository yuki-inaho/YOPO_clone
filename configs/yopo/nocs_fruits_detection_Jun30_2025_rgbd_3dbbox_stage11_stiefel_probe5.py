"""Independent five-epoch raw-rotation Stiefel probe from Stage-10 epoch 85.

The semantic SO(3), projected-GWD and compact-GWD losses remain unchanged.
The new target-free frame term is intentionally weak: epoch-85 CoP rotation
outputs are initialized near two 0.5 columns, whose raw beta=1 penalty is about
0.16.  A 0.05 weight therefore contributes about 0.008 per decoder layer,
well below the approximately 1.6 semantic rotation loss.  Confirm that scale
in a one-batch/one-step smoke log before starting the independent probe; stop
and lower the weight if any ``loss_rotation_frame`` exceeds 0.03.
"""

_base_ = [
    './nocs_fruits_detection_Jun30_2025_rgbd_3dbbox_'
    'stage11_geometry_loss_probe5_base.py'
]

model = dict(
    bbox_head=dict(
        loss_rotation_frame=dict(
            type='Rotation6DStiefelLoss',
            beta=1.0,
            loss_weight=0.05,
        ),
    ),
)
