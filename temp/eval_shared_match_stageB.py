"""Re-score the existing Stage B checkpoint under the shared correspondence.

This is the "before" column.  It changes nothing about the model; it only
replaces the evaluator list so the same weights are measured by metrics that
actually read the 3D ellipsoid, under one correspondence decided in the image.
"""

_base_ = ["./train_gaucho_stageB_native.py"]

_evaluator = [
    # Renamed: this is the legacy YOPO cuboid, not the GauCho ellipsoid.
    dict(type="NOCSMetric", prefix="legacy_nocs"),
    # The independent 2D GauCho branch (the DoD-A number).
    dict(type="EllipseEnvelopeRotatedIoUMetric", iou_thr=0.5, score_thr=0.05,
         num_classes=1, prefix="ellipse2d_head"),
    # The 3D ellipsoid seen through the camera.
    dict(type="ProjectedEllipsoidRotatedIoUMetric", iou_thr=0.5,
         score_thr=0.05, nms_iou_threshold=0.20, num_classes=1,
         prefix="projected_ellipsoid"),
    # The 3D ellipsoid itself, on a correspondence fixed in the image.
    dict(type="GauCho3DSharedMatchMetric", score_thr=0.20,
         nms_iou_threshold=0.20, match_iou_threshold=0.50,
         iou_3d_thresholds=(0.10, 0.20, 0.25, 0.50), num_classes=1,
         prefix="gaucho3d"),
]

val_evaluator = _evaluator
test_evaluator = _evaluator
# The inherited chain sets ``test_dataloader = None``; a dict cannot merge
# onto None, so the val loader is installed wholesale.
test_dataloader = {"_delete_": True, **_base_.val_dataloader}
test_cfg = dict(_delete_=True, type="TestLoop")
