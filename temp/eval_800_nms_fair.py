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
    dict(type="ProjectedEllipsoidRotatedIoUMetric", iou_thr=0.5,
         score_thr=0.05, nms_iou_threshold=0.20, num_classes=1,
         prefix="projected_ellipsoid"),
    dict(type="GauCho3DSharedMatchMetric", score_thr=0.20,
         nms_iou_threshold=0.20, match_iou_threshold=0.50,
         iou_3d_thresholds=(0.10, 0.20, 0.25, 0.50), num_classes=1,
         prefix="gaucho3d"),
]

val_evaluator = _evaluator
test_evaluator = _evaluator
test_dataloader = {"_delete_": True, **_base_.val_dataloader}
test_cfg = dict(_delete_=True, type="TestLoop")
