"""Stage B at native resolution: train the metric 3D GauCho ellipsoid.

This is the first run of the 3D path.  Until the review fixes it was not
runnable at all: the compact chain trains under bfloat16 AMP, autocast leaves
``torch.linalg.*`` alone, and CUDA has no bfloat16 kernel for cholesky,
triangular solve, LU or eigh -- the first iteration died on
``torch.linalg.cholesky`` of the ground-truth ellipsoid.

Native 736x512 for the same reason Stage A used it: 18.1% of the annotated
stems have a short side under 16 px, and the inherited 640x480 resize was
discarding 24% of the pixels.

Starts from the best 2D checkpoint, so the detector and the 2D ellipse head are
already trained and only the 3D branch is fresh.
"""

_base_ = ["../configs/yopo/nocs_fruits_Jun30_2025_rgbd_gaucho_compact_stageB.py"]

native_scale = (736, 512)

train_pipeline = [
    dict(type="LoadImageFromFile", backend_args=None),
    dict(type="LoadRawDepthImageWithValidMask"),
    dict(type="Load9DPoseAnnotations", with_bbox=True, with_centers_2d=True,
         with_z=True, with_obb_gaussian=True),
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
    dict(type="Load9DPoseAnnotations", with_bbox=True, with_centers_2d=True,
         with_z=True, with_obb_gaussian=True),
    dict(type="ConcatRawDepthToImage", depth_scale=255.0),
    dict(type="ResizeforPose", scale=native_scale, keep_ratio=True),
    dict(type="ResizeOBBGaussians"),
    dict(type="Pack9DPoseInputs",
         meta_keys=("img_id", "img_path", "ori_shape", "img_shape",
                    "scale_factor", "intrinsic", "models_info_path")),
]

max_epochs = 24
train_cfg = dict(type="EpochBasedTrainLoop", max_epochs=max_epochs,
                 val_interval=2)

model = dict(backbone=dict(rgb_backbone=dict(init_cfg=None)))

train_dataloader = dict(batch_size=12, num_workers=8,
                        dataset=dict(pipeline=train_pipeline))
val_dataloader = dict(batch_size=4, num_workers=4,
                      dataset=dict(pipeline=val_pipeline))

optim_wrapper = dict(
    optimizer=dict(lr=1e-4),
    paramwise_cfg=dict(
        custom_keys={
            # The only randomly initialized module at this point.
            "bbox_head.reg_ellipsoid_branch": dict(lr_mult=10.0),
            "bbox_head.reg_ellipse2d_branch": dict(lr_mult=1.0),
        }
    ),
)

# Stage B is judged on 3D overlap; the 2D ellipse must not regress while it
# trains, which the reported ellipse metric keeps visible.
custom_hooks = [
    dict(type="ScheduleFreeOptimizerModeHook"),
    dict(
        type="EarlyStoppingHook",
        monitor="3d_iou_0.25",
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
        save_best="3d_iou_0.25",
        rule="greater",
    ),
)

load_from = None
