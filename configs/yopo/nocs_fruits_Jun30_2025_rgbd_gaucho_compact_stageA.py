"""Stage A on the compact architecture, warm-started from the released model.

Why this exists alongside ``nocs_fruits_Jun30_2025_rgbd_gaucho_stageA_ellipse2d``:
that config inherits the ``nocs_custom_fruit_rgbd_3dbbox_transfer`` chain, whose
two initialization artifacts (the RGB transfer checkpoint and the depth MAE
encoder) are not present in a fresh clone.  Training it means training from
random weights on 1,196 frames, which cannot reach a number worth comparing
against the 2D reference implementation.

The released ``large_2025_2026_compact_cop_best3d_epoch5`` checkpoint is a
compact RGB-D model with the same single class and the same 256 queries, and it
already reaches AP50 0.6929 on its own validation split.  Building on it makes
this a head warm-up on a competent detector -- the same shape of curriculum the
reference implementation used -- instead of a cold start.

The model comes from the compact chain; the data, pipelines, and per-frame
intrinsics come from the Jun30-2025 conversion, whose labels carry their own
intrinsics and OBB coordinate scale.
"""

_base_ = [
    "./nocs_fruits_2025_2026_rgbd_compact_curriculum_stage4_cop_chain_gwd_full.py"
]

data_root = "data/fruits_detection_Jun30-2025_stem_rgbd_736x512/"
dataset_type = "NOCSCustomFruitDataset"
backend_args = None

# Identical to the validated Jun30 pipelines.  ``ResizeforPose`` with
# ``keep_ratio`` turns the native 736x512 frame into the 640x445 input the
# conversion was checked against; ``ResizeOBBGaussians`` carries the OBB
# annotation through the same transform so the ellipse target stays aligned.
train_pipeline = [
    dict(type="LoadImageFromFile", backend_args=backend_args),
    dict(type="LoadRawDepthImageWithValidMask"),
    dict(
        type="Load9DPoseAnnotations",
        with_bbox=True,
        with_centers_2d=True,
        with_z=True,
        with_obb_gaussian=True,
    ),
    dict(type="ConcatRawDepthToImage", depth_scale=255.0),
    dict(type="ResizeforPose", scale=(640, 480), keep_ratio=True),
    dict(type="ResizeOBBGaussians"),
    dict(type="RandomFlipFor9DPose", prob=0.5),
    dict(type="FilterAnnotations", min_gt_bbox_wh=(0.01, 0.01)),
    dict(type="Pack9DPoseInputs"),
]

val_pipeline = [
    dict(type="LoadImageFromFile", backend_args=backend_args),
    dict(type="LoadRawDepthImageWithValidMask"),
    dict(
        type="Load9DPoseAnnotations",
        with_bbox=True,
        with_centers_2d=True,
        with_z=True,
        with_obb_gaussian=True,
    ),
    dict(type="ConcatRawDepthToImage", depth_scale=255.0),
    dict(type="ResizeforPose", scale=(640, 480), keep_ratio=True),
    dict(type="ResizeOBBGaussians"),
    dict(
        type="Pack9DPoseInputs",
        meta_keys=("img_id", "img_path", "ori_shape", "img_shape",
                   "scale_factor", "intrinsic", "models_info_path"),
    ),
]

# One dataset, not the 2025+2026 concatenation the compact chain trains on.
train_dataloader = dict(
    batch_size=12,
    num_workers=8,
    persistent_workers=True,
    pin_memory=True,
    dataset=dict(
        _delete_=True,
        type=dataset_type,
        data_root=data_root,
        split="real_train",
        obb_coordinate_scale=0.8,
        intrinsic=[443.9066, 449.1953, 321.3503, 230.8687],
        pipeline=train_pipeline,
        backend_args=backend_args,
    ),
)

val_dataloader = dict(
    batch_size=4,
    num_workers=4,
    persistent_workers=True,
    dataset=dict(
        _delete_=True,
        type=dataset_type,
        data_root=data_root,
        split="custom_val",
        obb_coordinate_scale=0.8,
        intrinsic=[443.9066, 449.1953, 321.3503, 230.8687],
        pipeline=val_pipeline,
        backend_args=backend_args,
    ),
)
model = dict(
    bbox_head=dict(
        expose_gaucho_predictions=True,
        gaucho_ellipse2d=True,
        gaucho_classwise=True,
        loss_ellipse2d=dict(
            type="Ellipse2DKLDLoss",
            loss_weight=2.0,
            tau=1.0,
            include_center=True,
            fail_on_invalid=False,
        ),
    ),
)

# The 2D ellipse envelope against the annotated OBB, with the reference
# implementation's definition, next to the existing 3D pose metric.
val_evaluator = [
    dict(type="NOCSMetric"),
    dict(
        type="EllipseEnvelopeRotatedIoUMetric",
        iou_thr=0.5,
        score_thr=0.05,
        num_classes=1,
        prefix="ellipse",
    ),
]

# The base config runs no test loop; leave it that way.
# The released compact checkpoint is supplied at launch via
#   --cfg-options load_from=<path>
# rather than hard-coded here, matching the convention the Stage-10 configs in
# this directory already use: it keeps the config portable and avoids encoding
# an artifact-root-specific absolute path.
load_from = None
resume = False
