"""Five-epoch isolated probe of the 5 cm physical ellipsoid constraint.

The train-time term is a soft one-sided penalty; the inference projection is
the hard guarantee.  Oversized GT annotations are excluded only from the 3D
shape KLD, while still supervising detection, centre, depth and the 2D ellipse.
No ellipse-aware matching/ranking changes are mixed into this ablation.
"""

_base_ = ["./train_800_fix_center_off.py"]

max_epochs = 5
train_cfg = dict(
    _delete_=True,
    type="EpochBasedTrainLoop",
    max_epochs=max_epochs,
    val_interval=1,
)

model = dict(
    bbox_head=dict(
        loss_ellipsoid_max_axis=dict(
            type="EllipsoidMaxAxisLoss",
            max_diameter=0.05,
            loss_weight=0.25,
        ),
        ellipsoid_gt_max_diameter=0.05,
        ellipsoid_max_diameter=0.05,
    ),
)

val_evaluator = [
    dict(type="NOCSMetric", prefix="legacy_nocs"),
    dict(type="EllipseEnvelopeRotatedIoUMetric", iou_thr=0.5,
         score_thr=0.05, num_classes=1, prefix="ellipse2d_raw"),
    dict(type="EllipseEnvelopeRotatedIoUMetric", iou_thr=0.5,
         score_thr=0.05, nms_iou_threshold=0.20, num_classes=1,
         prefix="ellipse2d_nms02"),
    dict(type="ProjectedEllipsoidRotatedIoUMetric", iou_thr=0.5,
         score_thr=0.05, nms_iou_threshold=0.20, num_classes=1,
         prefix="projected_ellipsoid"),
    dict(type="GauCho3DSharedMatchMetric", score_thr=0.20,
         nms_iou_threshold=0.20, match_iou_threshold=0.50,
         iou_3d_thresholds=(0.10, 0.20, 0.25, 0.50), num_classes=1,
         prefix="gaucho3d"),
]

custom_hooks = [dict(type="ScheduleFreeOptimizerModeHook")]
default_hooks = dict(
    checkpoint=dict(
        _delete_=True,
        type="CheckpointHook",
        interval=1,
        max_keep_ckpts=2,
        save_best="gaucho3d/shared_AP_20",
        rule="greater",
    ),
)


