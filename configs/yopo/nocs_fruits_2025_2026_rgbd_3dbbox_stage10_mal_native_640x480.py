"""Joint 2025+2026 RGB-D training on preprocessed 640x480 canvases.

The input roots already contain resized RGB, nearest-neighbor mapped depth,
updated camera intrinsics and updated 2D annotations.  Runtime geometric
resize transforms are intentionally absent.  The 2025 content is a top-left
640x445 letterbox inside 640x480; the 2026 4:3 content fills the canvas.
"""

_base_ = [
    "./nocs_fruits_detection_Jun30_2025_rgbd_3dbbox_"
    "stage10_2d_anchor_mal_full.py"
]

dataset_type = "NOCSCustomFruitDataset"
backend_args = None
joint_root = "data/fruits_rgbd_2025_2026_640x480_preprocessed/"
data_roots = [joint_root + "2025/", joint_root + "2026/"]

native_train_pipeline = [
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
    dict(type="RandomFlipFor9DPose", prob=0.5),
    dict(type="FilterAnnotations", min_gt_bbox_wh=(1e-2, 1e-2)),
    dict(type="Pack9DPoseInputs"),
]

native_val_pipeline = [
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
    dict(
        type="Pack9DPoseInputs",
        meta_keys=(
            "img_id",
            "img_path",
            "ori_shape",
            "img_shape",
            "scale_factor",
            "intrinsic",
            "models_info_path",
        ),
    ),
]

train_datasets = [
    dict(
        type=dataset_type,
        data_root=data_root,
        split="real_train",
        obb_coordinate_scale=1.0,
        pipeline=native_train_pipeline,
        backend_args=backend_args,
    )
    for data_root in data_roots
]

val_datasets = [
    dict(
        type=dataset_type,
        data_root=data_root,
        split="custom_val",
        obb_coordinate_scale=1.0,
        pipeline=native_val_pipeline,
        backend_args=backend_args,
    )
    for data_root in data_roots
]

train_dataloader = dict(
    _delete_=True,
    batch_size=20,
    num_workers=6,
    persistent_workers=True,
    pin_memory=True,
    sampler=dict(type="DefaultSampler", shuffle=True),
    dataset=dict(type="ConcatDataset", datasets=train_datasets),
)

val_dataloader = dict(
    _delete_=True,
    batch_size=1,
    num_workers=4,
    persistent_workers=True,
    pin_memory=True,
    drop_last=False,
    sampler=dict(type="DefaultSampler", shuffle=False),
    dataset=dict(type="ConcatDataset", datasets=val_datasets),
)

# Both stored canvases are already identical and fixed-size.  Divisor 1 means
# the preprocessor performs neither resize nor additional spatial padding.
model = dict(data_preprocessor=dict(pad_size_divisor=1))
