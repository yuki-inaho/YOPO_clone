"""Shared 800x600 Stage-B base: correct evaluation, aggressive batch.

Two changes from the 736x512 lineage, both requested and both deliberate.

Resolution.  ``ResizeforPose`` with ``keep_ratio`` maps the native 736x512
frame into 800x557 -- a 1.087x upscale, 1.18x the pixels.  18.1% of the
annotated stems have a short side under 16 px at native, so the small-object
end of this dataset is where the detector is losing recall.

Evaluation.  The metric list is replaced wholesale, because the previous one
did not measure the thing this work builds.  ``NOCSMetric`` is renamed to
``legacy_nocs`` -- it scores the pre-existing YOPO cuboid
(``translations/rotations/sizes``) and reads none of the ellipsoid fields.
``ellipse2d_head`` scores the independent 2D branch (the DoD-A number).
``projected_ellipsoid`` scores the 3D ellipsoid seen through the camera.
``gaucho3d`` scores the ellipsoid itself, on a correspondence decided once in
the image and never re-decided in 3D.

Checkpoint selection moves to ``gaucho3d/shared_AP_20``: a coupled criterion
that requires the prediction to find the right object *and* to have the right
3D shape, so a model cannot win by improving one at the other's expense.  The
0.20 threshold rather than 0.50 is a property of the data -- these are ~20 mm
objects, where a few millimetres of annotation jitter alone crosses 0.50.
"""

_base_ = ["../configs/yopo/nocs_fruits_Jun30_2025_rgbd_gaucho_compact_stageB.py"]

scale_800 = (800, 600)

train_pipeline = [
    dict(type="LoadImageFromFile", backend_args=None),
    dict(type="LoadRawDepthImageWithValidMask"),
    dict(type="Load9DPoseAnnotations", with_bbox=True, with_centers_2d=True,
         with_z=True, with_obb_gaussian=True),
    dict(type="ConcatRawDepthToImage", depth_scale=255.0),
    dict(type="ResizeforPose", scale=scale_800, keep_ratio=True),
    dict(type="ResizeOBBGaussians"),
    dict(type="RandomFlipFor9DPose", prob=0.5),
    dict(type="FilterAnnotations", min_gt_bbox_wh=(0.01, 0.01)),
    dict(type="Pack9DPoseInputs"),
]

val_pipeline = [
    dict(type="LoadImageFromFile", backend_args=None),
    dict(type="LoadRawDepthImageWithValidMask"),
    dict(type="Load9DPoseAnnotations", with_bbox=True, with_centers_2d=True,
         with_z=True, with_obb_gaussian=True),
    dict(type="ConcatRawDepthToImage", depth_scale=255.0),
    dict(type="ResizeforPose", scale=scale_800, keep_ratio=True),
    dict(type="ResizeOBBGaussians"),
    dict(type="Pack9DPoseInputs",
         meta_keys=("img_id", "img_path", "ori_shape", "img_shape",
                    "scale_factor", "intrinsic", "models_info_path")),
]

max_epochs = 24
train_cfg = dict(type="EpochBasedTrainLoop", max_epochs=max_epochs,
                 val_interval=2)

model = dict(backbone=dict(rgb_backbone=dict(init_cfg=None)))

# Measured rather than guessed.  736x512 at batch 12 peaked at 10.2 GB
# allocated on a 24 GB card, leaving half the board idle.  The 800x557 smoke at
# batch 16 measured 15.2 GB allocated / 16.1 GB reserved, which puts the
# marginal cost at 0.8 GB per sample and 2.4 GB fixed.  Batch 20 therefore lands
# near 18.4 GB allocated / 19.5 GB reserved -- the working point, with enough
# margin left for allocator fragmentation over a 24-epoch run.  Batch 24 would
# reach ~22.4 GB and is not worth the OOM risk mid-run.
train_dataloader = dict(batch_size=20, num_workers=8,
                        dataset=dict(pipeline=train_pipeline))
val_dataloader = dict(batch_size=4, num_workers=4,
                      dataset=dict(pipeline=val_pipeline))

optim_wrapper = dict(
    optimizer=dict(lr=1e-4),
    paramwise_cfg=dict(
        custom_keys={
            "bbox_head.reg_ellipsoid_branch": dict(lr_mult=10.0),
            "bbox_head.reg_ellipse2d_branch": dict(lr_mult=1.0),
        }
    ),
)

val_evaluator = [
    dict(type="NOCSMetric", prefix="legacy_nocs"),
    dict(type="EllipseEnvelopeRotatedIoUMetric", iou_thr=0.5, score_thr=0.05,
         num_classes=1, prefix="ellipse2d_head"),
    dict(type="ProjectedEllipsoidRotatedIoUMetric", iou_thr=0.5,
         score_thr=0.05, nms_iou_threshold=0.20, num_classes=1,
         prefix="projected_ellipsoid"),
    dict(type="GauCho3DSharedMatchMetric", score_thr=0.20,
         nms_iou_threshold=0.20, match_iou_threshold=0.50,
         iou_3d_thresholds=(0.10, 0.20, 0.25, 0.50), num_classes=1,
         prefix="gaucho3d"),
]

custom_hooks = [
    dict(type="ScheduleFreeOptimizerModeHook"),
    dict(
        type="EarlyStoppingHook",
        monitor="gaucho3d/shared_AP_20",
        rule="greater",
        min_delta=1e-4,
        patience=6,
        strict=False,
        check_finite=True,
    ),
]

default_hooks = dict(
    logger=dict(type="LoggerHook", interval=20),
    checkpoint=dict(
        _delete_=True,
        type="CheckpointHook",
        interval=4,
        max_keep_ckpts=3,
        save_best="gaucho3d/shared_AP_20",
        rule="greater",
    ),
)

load_from = None
