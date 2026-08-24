"""Tiny valid-mask MAE pretraining run for the custom raw depth branch."""

_base_ = ["../_base_/default_runtime.py"]

custom_imports = dict(
    imports=[
        "yopo.datasets.pose_estimation.nocs_custom_fruit_dataset",
        "yopo.datasets.transforms.raw_depth",
        "yopo.models.data_preprocessors.rgbd_valid_mask",
        "yopo.models.detectors.valid_mask_mae_depth",
    ],
    allow_failed_imports=False,
)

dataset_type = "NOCSCustomFruitDataset"
data_root = "data/nocs_custom/"
custom_intrinsic = [443.9066, 449.1953, 321.3503, 230.8687]

mae_pipeline = [
    dict(type="LoadImageFromFile"),
    dict(type="LoadRawDepthImageWithValidMask"),
    dict(type="ConcatRawDepthToImage", depth_scale=255.0, append_valid_mask=True),
    dict(type="Resize", scale=(640, 480), keep_ratio=True),
    dict(type="PackDetInputs"),
]

train_dataloader = dict(
    batch_size=16,
    num_workers=4,
    persistent_workers=True,
    sampler=dict(type="DefaultSampler", shuffle=True),
    dataset=dict(
        type=dataset_type,
        data_root=data_root,
        split="real_train",
        intrinsic=custom_intrinsic,
        pipeline=mae_pipeline,
    ),
)
val_dataloader = None
val_cfg = None
val_evaluator = None
test_dataloader = None
test_cfg = None
test_evaluator = None

model = dict(
    type="ValidMaskMAEDepth",
    data_preprocessor=dict(
        type="RGBDValidMaskDataPreprocessor",
        mean=[123.675, 116.28, 103.53, 153.0, 0.5],
        std=[58.395, 57.12, 57.375, 76.5, 0.5],
        bgr_to_rgb=False,
        pad_size_divisor=1,
    ),
    encoder=dict(
        type="HGNetV2",
        name="B0",
        in_channels=1,
        return_idx=[1, 2, 3],
        freeze_at=-1,
        freeze_norm=True,
        init_cfg=None,
    ),
    hidden_channels=128,
    mask_ratio=0.75,
    loss_weight=1.0,
)

optim_wrapper = dict(
    type="OptimWrapper",
    optimizer=dict(type="AdamW", lr=1e-4, weight_decay=1e-4),
    clip_grad=dict(max_norm=1.0, norm_type=2),
)
train_cfg = dict(type="EpochBasedTrainLoop", max_epochs=1, val_interval=1)
default_hooks = dict(checkpoint=dict(type="CheckpointHook", interval=1, save_last=True))
load_from = None
resume = False
