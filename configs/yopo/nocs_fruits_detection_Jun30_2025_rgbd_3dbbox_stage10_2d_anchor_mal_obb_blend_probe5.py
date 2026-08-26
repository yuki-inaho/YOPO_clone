"""Optional OBB-GWD quality blend after the HBB-only MAL probe passes."""

_base_ = [
    './nocs_fruits_detection_Jun30_2025_rgbd_3dbbox_'
    'stage10_2d_anchor_mal_probe5.py'
]

model = dict(
    bbox_head=dict(
        quality_target=dict(
            _delete_=True,
            type='MatchabilityQualityPolicy',
            source='blend',
            obb_weight=0.25,
            tau=1.0,
            normalize=True,
            include_center=False,
            missing_obb='hbb_iou',
            fail_on_invalid=True,
        ),
    ),
)
