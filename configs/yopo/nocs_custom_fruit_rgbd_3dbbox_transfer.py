"""Custom-fruit RGB-D 3D BBOX training with frozen transferred RGB features."""

_base_ = ["./nocs_custom_real_hgnetv2_rgbd_deim_cop_dual.py"]

custom_imports = dict(
    imports=[
        "yopo.datasets.pose_estimation.nocs_custom_fruit_dataset",
        "yopo.datasets.transforms.raw_depth",
        "yopo.models.backbones.frozen_rgbd",
        "yopo.engine.hooks.rgb_backbone_transfer",
        "yopo.models.losses.stable_rotation",
        "yopo.models.losses.projected_ellipsoid_loss",
    ],
    allow_failed_imports=False,
)

dataset_type = "NOCSCustomFruitDataset"
data_root = "data/nocs_custom/"
custom_intrinsic = [443.9066, 449.1953, 321.3503, 230.8687]
backend_args = None
scale = (640, 480)

# Do not apply HSV perturbations to depth.  RandomFlipFor9DPose updates RGB-D
# pixels, intrinsics, 2D boxes and all pose targets in one consistent step.
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
    dict(type="Resize", scale=scale, keep_ratio=True),
    dict(type="ResizeOBBGaussians"),
    dict(type="RandomFlipFor9DPose", prob=0.5),
    dict(type="FilterAnnotations", min_gt_bbox_wh=(1e-2, 1e-2)),
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
    dict(type="Resize", scale=scale, keep_ratio=True),
    dict(type="ResizeOBBGaussians"),
    dict(
        type="Pack9DPoseInputs",
        meta_keys=(
            "img_id", "img_path", "ori_shape", "img_shape", "scale_factor",
            "intrinsic", "models_info_path",
        ),
    ),
]

train_dataloader = dict(
    _delete_=True,
    batch_size=8,
    num_workers=4,
    persistent_workers=True,
    sampler=dict(type="DefaultSampler", shuffle=True),
    dataset=dict(
        type=dataset_type,
        data_root=data_root,
        split="real_train",
        obb_coordinate_scale=0.8,
        intrinsic=custom_intrinsic,
        pipeline=train_pipeline,
        backend_args=backend_args,
    ),
)

val_dataloader = dict(
    _delete_=True,
    batch_size=1,
    num_workers=2,
    persistent_workers=True,
    drop_last=False,
    sampler=dict(type="DefaultSampler", shuffle=False),
    dataset=dict(
        type=dataset_type,
        data_root=data_root,
        split="custom_val",
        obb_coordinate_scale=0.8,
        intrinsic=custom_intrinsic,
        pipeline=val_pipeline,
        backend_args=backend_args,
    ),
)

model = dict(
    backbone=dict(
        type="FrozenRGBDDualBackbone",
        freeze_rgb=True,
        rgb_backbone=dict(init_cfg=None),
        depth_backbone=dict(
            freeze_at=-1,
            init_cfg=dict(
                type="Pretrained",
                checkpoint=(
                    "work_dirs/rgbd3d_validmask_mae_fp32_smoke/"
                    "enc_b0_validmask_mae.pth"
                ),
            ),
        ),
    ),
    train_cfg=dict(
        assigner=dict(
            type="HungarianAssigner",
            match_costs=[
                dict(type="FocalLossCost", weight=2.0),
                dict(type="BBoxL1Cost", weight=5.0, box_format="xywh"),
                dict(type="IoUCost", iou_mode="giou", weight=2.0),
                dict(type="TranslationCost", weight=5.0),
                dict(type="RotationCost", symmetric_classes=[], weight=2.0),
            ],
        )
    ),
    bbox_head=dict(
        loss_rotation=dict(
            type="AMPStableRotation3DLoss",
            acos_clamp=0.9999,
            normalize_eps=1e-2,
            loss_weight=5.0,
            # None, rather than an empty list: the base loss only enters its
            # symmetry branch when this is non-None.
            symmetric_classes=None,
        ),
    ),
)

# The old depth-MAE checkpoint was trained without a valid mask.  It is not a
# transfer source; a finite valid-mask checkpoint is produced in step 7.
load_from = None
resume = False

rgb_transfer_checkpoint = (
    "work_dirs/rddetr_tomato_riou_linear_ft20/"
    "best_rbbox_mAP_50_epoch_20.pth"
)
custom_hooks = [
    dict(
        type="RGBBackboneTransferHook",
        checkpoint=rgb_transfer_checkpoint,
        report_filename="partial_transfer_report.json",
    )
]
