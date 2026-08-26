"""DEIM MAL probe on the 2D-anchored Stage-10 control."""

_base_ = [
    './nocs_fruits_detection_Jun30_2025_rgbd_3dbbox_'
    'stage10_2d_anchor_control_probe5.py'
]

# The only change from the control is classification supervision.  HBB IoU is
# the first quality source because it is available for decoder, encoder and DN
# queries without adding parameters or changing checkpoint keys.
model = dict(
    bbox_head=dict(
        loss_cls=dict(
            _delete_=True,
            type='MatchabilityAwareLoss',
            use_sigmoid=True,
            gamma=1.5,
            loss_weight=1.0,
            force_float32=True,
        ),
        quality_target=dict(
            type='MatchabilityQualityPolicy',
            source='hbb_iou',
        ),
    ),
)
