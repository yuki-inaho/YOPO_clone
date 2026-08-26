"""Shared native-736x512 RGB-D geometry contract.

This layer changes no head, matcher, or loss.  RGB and raw depth remain at the
dataset's native 736x512 resolution: no resize and no pad are performed.
``AssertIdentityImageGeometry`` validates the contract and records only the
identity scale metadata required by rescaled prediction paths.
"""

_base_ = [
    './nocs_fruits_detection_Jun30_2025_rgbd_3dbbox_'
    'stage10_2d_anchor_mal_full.py'
]

custom_imports = dict(
    imports=[
        'yopo.datasets.pose_estimation.nocs_custom_fruit_dataset',
        'yopo.datasets.transforms.raw_depth',
        'yopo.datasets.transforms.identity_geometry',
        'yopo.engine.hooks.rgbd_pose_transfer',
        'yopo.engine.optimizers.deim_optimizers',
        'yopo.models.backbones.dual_rgbd',
        'yopo.models.losses.projected_ellipsoid_loss',
        'yopo.models.losses.stable_rotation',
    ],
    allow_failed_imports=False,
)

backend_args = None
native_image_size_wh = (736, 512)

train_pipeline = [
    dict(type='LoadImageFromFile', backend_args=backend_args),
    dict(type='LoadRawDepthImageWithValidMask'),
    dict(
        type='Load9DPoseAnnotations',
        with_bbox=True,
        with_centers_2d=True,
        with_z=True,
        with_obb_gaussian=True,
    ),
    dict(type='ConcatRawDepthToImage', depth_scale=255.0),
    dict(
        type='AssertIdentityImageGeometry',
        image_size=native_image_size_wh,
        channels=4,
    ),
    dict(type='RandomFlipFor9DPose', prob=0.5),
    dict(type='FilterAnnotations', min_gt_bbox_wh=(1e-2, 1e-2)),
    dict(type='Pack9DPoseInputs'),
]

val_pipeline = [
    dict(type='LoadImageFromFile', backend_args=backend_args),
    dict(type='LoadRawDepthImageWithValidMask'),
    dict(
        type='Load9DPoseAnnotations',
        with_bbox=True,
        with_centers_2d=True,
        with_z=True,
        with_obb_gaussian=True,
    ),
    dict(type='ConcatRawDepthToImage', depth_scale=255.0),
    dict(
        type='AssertIdentityImageGeometry',
        image_size=native_image_size_wh,
        channels=4,
    ),
    dict(
        type='Pack9DPoseInputs',
        meta_keys=(
            'img_id', 'img_path', 'ori_shape', 'img_shape', 'scale_factor',
            'intrinsic', 'models_info_path',
        ),
    ),
]

# Keep standalone test/inference on exactly the same native geometry.  The
# inherited Stage-10 test pipeline still resized to 640x480; it is unused by
# the training loop but would otherwise silently change coordinates when this
# config is reused for final evaluation or visualization.
test_pipeline = val_pipeline

train_dataloader = dict(
    # Native pixels are 1.324x the former 640x445 tensor.  Start the RTX 5090
    # capacity search at b16, then change this single knob via CLI according
    # to measured peak VRAM and a practical safety margin.
    batch_size=16,
    dataset=dict(pipeline=train_pipeline),
)
val_dataloader = dict(
    batch_size=1,
    dataset=dict(pipeline=val_pipeline),
)

# Make the proof-safe broad phase explicit in resolved Stage-12 configs.
# ``False`` remains available for diagnostic all-pairs exact A/B runs.
val_evaluator = dict(two_phase_3d_iou=True)

model = dict(
    data_preprocessor=dict(
        # Native dimensions are already batch-uniform; divisor 1 guarantees
        # that preprocessing adds no hidden spatial padding.
        pad_size_divisor=1,
    ),
)
