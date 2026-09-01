"""Five-epoch ellipse-aware probe; this file does not start training itself.

This is the smallest controlled change suggested by rotated RTMDet/O2-DEIM:

1. Hungarian assignment sees the same ellipse geometry that validation scores.
2. Matchability confidence is trained on an HBB/ellipse-quality blend.
3. The existing 20 denoising queries receive ellipse supervision; their count
   is deliberately not increased until this missing objective is measured.

The encoder keeps its HBB-only assigner because it has no ellipse branch.  Its
quality target explicitly falls back to HBB IoU for the same reason.
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
    train_cfg=dict(
        assigner=dict(
            _delete_=True,
            type="HungarianAssigner",
            match_costs=[
                dict(type="FocalLossCost", weight=2.0),
                dict(type="BBoxL1Cost", weight=5.0, box_format="xywh"),
                dict(type="IoUCost", iou_mode="giou", weight=2.0),
                dict(
                    type="Ellipse2DKLDCost",
                    weight=1.0,
                    tau=1.0,
                    symmetric=True,
                ),
            ],
        ),
    ),
    bbox_head=dict(
        gaucho_ellipse2d_dn=True,
        quality_target=dict(
            _delete_=True,
            type="MatchabilityQualityPolicy",
            source="ellipse_kld_blend",
            obb_weight=0.5,
            include_center=True,
            ellipse_symmetric=True,
            tau=1.0,
            # Encoder proposals intentionally have no ellipse output.
            missing_obb="hbb_iou",
            fail_on_invalid=False,
        ),
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
        save_best="ellipse2d_nms02/rbbox_mAP_50",
        rule="greater",
    ),
)


