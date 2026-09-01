"""FULL fine-tune at the dataset's native resolution.

The inherited pipeline resized 736x512 down to 640x445.  Measured over 679
annotated stems, that leaves a median OBB of 29.6 x 23.4 px with **18.1% of
objects under 16 px on their short side** -- the regime where detection is most
resolution-sensitive.  The reference number this is measured against was
produced at 512x736, i.e. no downscaling at all, and the released compact
checkpoint was trained at 800x600, so native input is also closer to its
pretraining distribution.

``ResizeforPose`` is kept but set to the native size, which yields
``scale_factor == (1.0, 1.0)``.  Removing the transform outright would drop
``scale_factor`` from the metainfo, which both ``rescale=True`` inference and
the ellipse metric's frame mapping rely on.
"""

_base_ = ["../configs/yopo/nocs_fruits_Jun30_2025_rgbd_gaucho_compact_stageA.py"]

native_scale = (736, 512)

train_pipeline = [
    dict(type="LoadImageFromFile", backend_args=None),
    dict(type="LoadRawDepthImageWithValidMask"),
    dict(
        type="Load9DPoseAnnotations",
        with_bbox=True,
        with_centers_2d=True,
        with_z=True,
        with_obb_gaussian=True,
    ),
    dict(type="ConcatRawDepthToImage", depth_scale=255.0),
    dict(type="ResizeforPose", scale=native_scale, keep_ratio=True),
    dict(type="ResizeOBBGaussians"),
    dict(type="RandomFlipFor9DPose", prob=0.5),
    dict(type="FilterAnnotations", min_gt_bbox_wh=(0.01, 0.01)),
    dict(type="Pack9DPoseInputs"),
]

val_pipeline = [
    dict(type="LoadImageFromFile", backend_args=None),
    dict(type="LoadRawDepthImageWithValidMask"),
    dict(
        type="Load9DPoseAnnotations",
        with_bbox=True,
        with_centers_2d=True,
        with_z=True,
        with_obb_gaussian=True,
    ),
    dict(type="ConcatRawDepthToImage", depth_scale=255.0),
    dict(type="ResizeforPose", scale=native_scale, keep_ratio=True),
    dict(type="ResizeOBBGaussians"),
    dict(
        type="Pack9DPoseInputs",
        meta_keys=("img_id", "img_path", "ori_shape", "img_shape",
                   "scale_factor", "intrinsic", "models_info_path"),
    ),
]

max_epochs = 40
train_cfg = dict(type="EpochBasedTrainLoop", max_epochs=max_epochs,
                 val_interval=2)

model = dict(backbone=dict(rgb_backbone=dict(init_cfg=None)))

# 1.32x the pixels of the resized run, which peaked at 10.9 GB for batch 16.
train_dataloader = dict(batch_size=12, num_workers=8,
                        dataset=dict(pipeline=train_pipeline))
val_dataloader = dict(batch_size=4, num_workers=4,
                      dataset=dict(pipeline=val_pipeline))

optim_wrapper = dict(
    optimizer=dict(lr=1e-4),
    paramwise_cfg=dict(
        custom_keys={"bbox_head.reg_ellipse2d_branch": dict(lr_mult=1.0)}
    ),
)

custom_hooks = [
    dict(type="ScheduleFreeOptimizerModeHook"),
    dict(
        type="EarlyStoppingHook",
        monitor="ellipse/rbbox_mAP_50",
        rule="greater",
        min_delta=1e-3,
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
        save_best="ellipse/rbbox_mAP_50",
        rule="greater",
    ),
)

load_from = None
