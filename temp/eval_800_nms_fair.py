"""Re-score an 800x600 checkpoint under the reference implementation's protocol.

``EllipseEnvelopeRotatedIoUMetric`` defaults to no suppression, which is the
raw number every run in this work has logged.  The 2D reference this is measured
against evaluates at rotated NMS IoU 0.2, and an unsuppressed 256-query emission
scored against that is not a like-for-like comparison: on the 736x512
checkpoint the same weights read 0.6533 raw and 0.7113 suppressed.

Both are reported here so neither can be quoted without the other.
"""

_base_ = ["./train_800_fix_center_off.py"]

_evaluator = [
    dict(type="NOCSMetric", prefix="legacy_nocs"),
    # The raw number, unchanged, so the run's own logs stay comparable.
    dict(type="EllipseEnvelopeRotatedIoUMetric", iou_thr=0.5, score_thr=0.05,
         num_classes=1, prefix="ellipse2d_raw"),
    # The reference protocol.
    dict(type="EllipseEnvelopeRotatedIoUMetric", iou_thr=0.5, score_thr=0.05,
         nms_iou_threshold=0.20, num_classes=1, prefix="ellipse2d_nms02"),
    # Threshold sweep at the reference protocol.  mAP can never exceed recall,
    # and recall after NMS is already below the DoD target, so the question is
    # whether the objects are missed outright or detected with an ellipse too
    # loose to clear IoU 0.5.  The gap between these two answers it.
    dict(type="EllipseEnvelopeRotatedIoUMetric", iou_thr=0.25, score_thr=0.05,
         nms_iou_threshold=0.20, num_classes=1, prefix="ellipse2d_nms02_iou25"),
    dict(type="EllipseEnvelopeRotatedIoUMetric", iou_thr=0.75, score_thr=0.05,
         nms_iou_threshold=0.20, num_classes=1, prefix="ellipse2d_nms02_iou75"),
    # The ceiling: this detector with perfect ellipse geometry.  The
    # distance from here to the 0.8170 target is what geometry cannot
    # reach, and therefore whether the objective or the detector is next.
    dict(type="EllipseEnvelopeRotatedIoUMetric", iou_thr=0.5,
         score_thr=0.05, nms_iou_threshold=0.20, num_classes=1,
         geometry_oracle=True, prefix="ellipse2d_oracle"),
    # Which part of the geometry is worth what, in AP.  The IoU-level
    # decomposition says the residual is orientation; these say whether
    # fixing orientation alone would actually move the reported number.
    dict(type="EllipseEnvelopeRotatedIoUMetric", iou_thr=0.5,
         score_thr=0.05, nms_iou_threshold=0.20, num_classes=1,
         geometry_oracle=True, oracle_fields="angle",
         prefix="oracle_angle"),
    dict(type="EllipseEnvelopeRotatedIoUMetric", iou_thr=0.5,
         score_thr=0.05, nms_iou_threshold=0.20, num_classes=1,
         geometry_oracle=True, oracle_fields="size",
         prefix="oracle_size"),
    dict(type="EllipseEnvelopeRotatedIoUMetric", iou_thr=0.5,
         score_thr=0.05, nms_iou_threshold=0.20, num_classes=1,
         geometry_oracle=True, oracle_fields="centre",
         prefix="oracle_centre"),
    dict(type="EllipseEnvelopeRotatedIoUMetric", iou_thr=0.5,
         score_thr=0.05, nms_iou_threshold=0.20, num_classes=1,
         geometry_oracle=True, oracle_fields="size_angle",
         prefix="oracle_size_angle"),
    dict(type="ProjectedEllipsoidRotatedIoUMetric", iou_thr=0.5,
         score_thr=0.05, nms_iou_threshold=0.20, num_classes=1,
         prefix="projected_ellipsoid"),
    # What perfect range would be worth, with the bearing and the shape
    # left exactly as predicted.
    dict(type="GauCho3DSharedMatchMetric", score_thr=0.20,
         nms_iou_threshold=0.20, match_iou_threshold=0.50,
         iou_3d_thresholds=(0.10, 0.20, 0.25, 0.50), num_classes=1,
         depth_oracle=True, prefix="gaucho3d_depth_oracle"),
    dict(type="GauCho3DSharedMatchMetric", score_thr=0.20,
         nms_iou_threshold=0.20, match_iou_threshold=0.50,
         iou_3d_thresholds=(0.10, 0.20, 0.25, 0.50), num_classes=1,
         prefix="gaucho3d"),
]

val_evaluator = _evaluator
test_evaluator = _evaluator
test_dataloader = {"_delete_": True, **_base_.val_dataloader}
test_cfg = dict(_delete_=True, type="TestLoop")
