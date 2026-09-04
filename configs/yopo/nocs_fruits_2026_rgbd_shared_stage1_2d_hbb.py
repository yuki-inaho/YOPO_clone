"""Stage 1 HBB training with the shared B2/B0 RGB-D feature contract v1."""

_base_ = ['./nocs_fruits_2025_2026_rgbd_compact_curriculum_stage1_2d_full.py']

custom_imports = dict(
    imports=[
        'yopo.datasets.pose_estimation.nocs_custom_fruit_dataset',
        'yopo.datasets.transforms.identity_geometry',
        'yopo.datasets.transforms.raw_depth',
        'yopo.engine.hooks.schedulefree_optimizer_mode',
        'yopo.engine.optimizers.amuse',
        'yopo.evaluation.metrics.ellipse_rotated_iou_metric',
        'yopo.evaluation.metrics.gaucho3d_shared_match_metric',
        'yopo.models.backbones.dual_rgbd',
        'yopo.models.losses.gaucho3d_loss',
        'yopo.models.losses.projected_ellipsoid_loss',
        'yopo.models.losses.stable_rotation',
        'yopo.models.necks.portable_hybrid_encoder',
    ],
    allow_failed_imports=False,
)

data_root = 'data/fruit_obb_rgbd_2026_yopo_3dobb_800x600_preprocessed/'
backend_args = None

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
    dict(type='ConcatRawDepthToImage', depth_scale=1.0, bgr_to_rgb=True),
    dict(type='AssertIdentityImageGeometry', image_size=(800, 600), channels=4),
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
    dict(type='ConcatRawDepthToImage', depth_scale=1.0, bgr_to_rgb=True),
    dict(type='AssertIdentityImageGeometry', image_size=(800, 600), channels=4),
    dict(
        type='Pack9DPoseInputs',
        meta_keys=(
            'img_id',
            'img_path',
            'ori_shape',
            'img_shape',
            'scale_factor',
            'intrinsic',
            'models_info_path',
        ),
    ),
]

train_dataloader = dict(
    _delete_=True,
    batch_size=1,
    num_workers=6,
    persistent_workers=True,
    pin_memory=True,
    sampler=dict(type='DefaultSampler', shuffle=True),
    dataset=dict(
        type='NOCSCustomFruitDataset',
        data_root=data_root,
        split='real_train',
        obb_coordinate_scale=1.0,
        pipeline=train_pipeline,
        backend_args=backend_args,
    ),
)

val_dataloader = dict(
    _delete_=True,
    batch_size=1,
    num_workers=4,
    persistent_workers=True,
    pin_memory=True,
    drop_last=False,
    sampler=dict(type='DefaultSampler', shuffle=False),
    dataset=dict(
        type='NOCSCustomFruitDataset',
        data_root=data_root,
        split='custom_val',
        obb_coordinate_scale=1.0,
        pipeline=val_pipeline,
        backend_args=backend_args,
    ),
)

model = dict(
    data_preprocessor=dict(
        _delete_=True,
        type='DetDataPreprocessor',
        mean=[0.0, 0.0, 0.0, 0.0],
        std=[255.0, 255.0, 255.0, 1.0],
        bgr_to_rgb=False,
        pad_size_divisor=1,
        non_blocking=True,
    ),
    backbone=dict(
        _delete_=True,
        type='RGBDResidualBackbone',
        beta_init=0.0,
        rgb_backbone=dict(
            type='HGNetV2',
            name='B2',
            use_lab=True,
            in_channels=3,
            return_idx=[1, 2, 3],
            freeze_at=-1,
            freeze_norm=True,
            init_cfg=None,
        ),
        depth_backbone=dict(
            type='HGNetV2',
            name='B0',
            use_lab=True,
            in_channels=1,
            return_idx=[1, 2, 3],
            freeze_at=-1,
            freeze_norm=True,
            init_cfg=None,
        ),
    ),
    neck=dict(
        _delete_=True,
        type='PortableHybridEncoderNeck',
        in_channels=[384, 768, 1536],
        hidden_dim=256,
        num_heads=8,
        ffn_dim=1024,
        num_aifi_layers=1,
        num_outs=4,
    ),
    bbox_head=dict(
        cop_depth_context=dict(in_channels=[256, 512, 1024]),
    ),
)

optim_wrapper = dict(
    _delete_=True,
    type='yopo.engine.optimizers.amuse.AmpAmuseOptimWrapper',
    dtype='bfloat16',
    loss_scale=1.0,
    optimizer=dict(
        type='yopo.engine.optimizers.amuse.AmuseOptimizer',
        lr=1e-4,
        aux_lr=1e-4,
        warmup_steps=100,
        weight_decay=1e-4,
    ),
    clip_grad=dict(max_norm=0.1, norm_type=2),
    paramwise_cfg=dict(
        custom_keys={
            'backbone.rgb_backbone': dict(lr_mult=0.1),
            'backbone.depth_backbone': dict(lr_mult=0.1),
            'backbone.depth_adapters': dict(lr_mult=1.0),
            'backbone.depth_beta': dict(lr_mult=1.0),
            'neck': dict(lr_mult=1.0),
            'encoder': dict(lr_mult=0.5),
            'decoder': dict(lr_mult=0.5),
            'bbox_head.reg_centers_2d_branch': dict(lr_mult=1.0),
            'bbox_head.reg_z_branch': dict(lr_mult=0.0),
            'bbox_head.reg_size_branch': dict(lr_mult=0.0),
            'bbox_head.reg_rotation_branch': dict(lr_mult=0.0),
            'bbox_head.cop_': dict(lr_mult=0.0),
            'bbox_head.depth_query_sampler': dict(lr_mult=0.0),
        },
    ),
)

# AMUSE owns its averaging schedule.
param_scheduler = []
auto_scale_lr = dict(enable=False, base_batch_size=1)
randomness = dict(seed=20260903, deterministic=False)
load_from = None
resume = False
