_base_ = './housecat6d_yopo_r50.py'

load_from = 'https://github.com/RistoranteRist/mmlab-weights/releases/download/dino-swinl/dino-5scale_swin-l_8xb2-36e_coco-5486e051.pth'  # noqa

num_levels = 5
model = dict(
    num_feature_levels=num_levels,
    backbone=dict(
        _delete_=True,
        type='SwinTransformer',
        pretrain_img_size=384,
        embed_dims=192,
        depths=[2, 2, 18, 2],
        num_heads=[6, 12, 24, 48],
        window_size=12,
        mlp_ratio=4,
        qkv_bias=True,
        qk_scale=None,
        drop_rate=0.,
        attn_drop_rate=0.,
        drop_path_rate=0.2,
        patch_norm=True,
        out_indices=(0, 1, 2, 3),
        # Please only add indices that would be used
        # in FPN, otherwise some parameter will not be used
        with_cp=True),
    neck=dict(in_channels=[192, 384, 768, 1536], num_outs=num_levels),
    encoder=dict(layer_cfg=dict(self_attn_cfg=dict(num_levels=num_levels))),
    decoder=dict(layer_cfg=dict(cross_attn_cfg=dict(num_levels=num_levels))),
    bbox_head=dict(
        num_classes=10,
    )
)

optim_wrapper = dict(
    type='yopo.engine.optimizers.amuse.AmuseOptimWrapper',
    optimizer=dict(
        type='yopo.engine.optimizers.amuse.AmuseOptimizer',
        lr=0.0002,
        aux_lr=0.0002,
        warmup_steps=100,
        weight_decay=0.0001),
    clip_grad=dict(max_norm=0.1, norm_type=2),
    paramwise_cfg=dict(custom_keys=dict(backbone=dict(lr_mult=0.1))))

# AMUSE is schedule-free.
param_scheduler = []
custom_hooks = [dict(type='ScheduleFreeOptimizerModeHook')]


auto_scale_lr = dict(base_batch_size=16, enable=False)

# Override train_pipeline: use RandomFlipFor9DPose instead of RandomRotationFor9DPose
backend_args = None
scale = (640, 480)
train_pipeline = [
    dict(type='LoadImageFromFile', backend_args=backend_args),
    dict(type='Load9DPoseAnnotations', with_bbox=True,
         with_centers_2d=True, with_z=True),
    dict(type='ResizeforPose', scale=scale, keep_ratio=True),
    dict(type='YOLOXHSVRandomAug'),
    dict(type='RandomTranslatePixels', prob=0.5, max_translate_offset=50),
    dict(type='RandomFlipFor9DPose', prob=0.5),
    dict(type='FilterAnnotations', min_gt_bbox_wh=(1e-2, 1e-2)),
    dict(type='Pack9DPoseInputs')
]

train_dataloader = dict(
    batch_size=4,
    num_workers=2,
    dataset=dict(pipeline=train_pipeline))
