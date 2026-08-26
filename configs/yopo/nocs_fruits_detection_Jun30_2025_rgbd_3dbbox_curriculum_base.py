"""Portable Q256 RGB-D CoP curriculum with composite 2D/3D transfer."""

_base_ = ['./nocs_custom_fruit_rgbd_nmsfree_control_continue5.py']

custom_imports = dict(
    imports=[
        'yopo.datasets.pose_estimation.nocs_custom_fruit_dataset',
        'yopo.datasets.transforms.raw_depth',
        'yopo.engine.hooks.rgbd_pose_transfer',
        'yopo.engine.optimizers.deim_optimizers',
        'yopo.models.backbones.dual_rgbd',
        'yopo.models.losses.projected_ellipsoid_loss',
        'yopo.models.losses.stable_rotation',
    ],
    allow_failed_imports=False,
)

data_root = 'data/fruits_detection_Jun30-2025_stem_rgbd_736x512/'
max_objects = 256
stage_epochs = 5

# Certified endpoint of the second 50-epoch GWD continuation.  It improved
# rbbox mAP50 from 0.195542872 to 0.201180086 on the full validation split.
feature_checkpoint = (
    'work_dirs/rtdetr_stem_rgbd_obb_hybrid_angle_gwd_continue50_plateau/'
    'best_rbbox_mAP_50_epoch_48.pth'
)
pose_checkpoint = (
    'work_dirs/nocs_custom_fruit_rgbd_nmsfree_control_continue5/'
    'best_3d_iou_0.50_epoch_5.pth'
)

train_dataloader = dict(
    # Capacity probe on RTX 5090: Q256/b20 peaks at 26,317 MiB and is
    # ~20% faster per epoch than b12 while retaining >6 GiB headroom.
    batch_size=20,
    num_workers=6,
    persistent_workers=True,
    pin_memory=True,
    dataset=dict(data_root=data_root),
)
val_dataloader = dict(
    batch_size=1,
    num_workers=4,
    persistent_workers=True,
    pin_memory=True,
    dataset=dict(data_root=data_root),
)

model = dict(
    num_queries=max_objects,
    test_cfg=dict(max_per_img=max_objects),
    data_preprocessor=dict(non_blocking=True),
    backbone=dict(
        _delete_=True,
        type='RGBDResidualBackbone',
        beta_init=0.0,
        rgb_backbone=dict(
            type='HGNetV2', name='B2', in_channels=3,
            return_idx=[1, 2, 3], freeze_at=-1,
            freeze_norm=False, init_cfg=None,
        ),
        depth_backbone=dict(
            type='HGNetV2', name='B0', in_channels=1,
            return_idx=[1, 2, 3], freeze_at=-1,
            freeze_norm=False,
            init_cfg=dict(
                type='Pretrained',
                checkpoint=(
                    'work_dirs/rgbd3d_validmask_mae_fp32_smoke/'
                    'enc_b0_validmask_mae.pth'
                ),
            ),
        ),
    ),
    neck=dict(
        in_channels=[384, 768, 1536],
    ),
    bbox_head=dict(
        test_cfg=dict(max_per_img=max_objects),
        cop_depth_context=dict(
            num_levels=3,
            roi_size=3,
            vectorize_layers=True,
            in_channels=[256, 512, 1024],
        ),
        # Portable labels are the direct curriculum teacher. Frozen teachers
        # from the old 50-frame split are deliberately disabled.
        distill_attributes=(),
        obb_center_teacher_checkpoint=None,
        pose_teacher_checkpoint=None,
    ),
)

custom_hooks = [
    dict(
        type='RGBDPoseCurriculumTransferHook',
        pose_checkpoint=pose_checkpoint,
        feature_checkpoint=feature_checkpoint,
        report_filename='rgbd_pose_curriculum_transfer_report.json',
        query_seed=736512,
    ),
]

optim_wrapper = dict(
    paramwise_cfg=dict(
        custom_keys={
            'backbone.rgb_backbone': dict(lr_mult=0.1),
            'backbone.depth_backbone': dict(lr_mult=0.1),
            'backbone.depth_adapters': dict(lr_mult=1.0),
            'backbone.depth_beta': dict(lr_mult=1.0),
            'encoder': dict(lr_mult=0.5),
            'bbox_head.cop_': dict(lr_mult=50.0),
            'bbox_head.depth_query_sampler': dict(lr_mult=50.0),
        },
    ),
)

train_cfg = dict(max_epochs=stage_epochs, val_interval=stage_epochs)
param_scheduler = []
default_hooks = dict(
    checkpoint=dict(
        _delete_=True,
        type='CheckpointHook',
        interval=stage_epochs,
        save_last=True,
        max_keep_ckpts=1,
        save_optimizer=False,
    ),
    logger=dict(interval=5),
)

load_from = None
resume = False
