_base_ = ['./nocs_custom_real_r50_rgbd.py']

# ─────────────────────────────────────────────────────────────────────────────
# Depth MAE (masked autoencoder) self-supervised pretraining config.
#
# Pretrains the depth branch (HGNetV2-B0, in_channels=1, return_idx=[1,2,3])
# with masked depth reconstruction (FCMAE-style, arXiv:2301.00808). The model
# uses only the depth channel of the 4-channel (RGB+depth) input and randomly
# masks 75% of the depth pixels; the light decoder reconstructs the masked
# depth pixels from the backbone features (L1 loss on masked pixels only).
# ─────────────────────────────────────────────────────────────────────────────

model = dict(
    _delete_=True,
    type='MAEDepth',
    mask_ratio=0.75,
    loss_weight=1.0,
    data_preprocessor=dict(
        type='DetDataPreprocessor',
        mean=[123.675, 116.28, 103.53, 153.0],
        std=[58.395, 57.12, 57.375, 76.5],
        bgr_to_rgb=False,
        pad_size_divisor=1),
    encoder=dict(
        type='HGNetV2',
        name='B0',
        in_channels=1,
        return_idx=[1, 2, 3],
        freeze_at=-1,
        freeze_norm=False,
        init_cfg=dict(
            type='Pretrained',
            checkpoint='https://github.com/Peterande/storage/releases/'
            'download/dfinev1.0/PPHGNetV2_B0_stage1.pth')))

# Train on the same custom RGB-D data (only the depth channel is used).
train_dataloader = dict(
    batch_size=8,
    num_workers=4,
    persistent_workers=True)

max_epochs = 20
train_cfg = dict(
    type='EpochBasedTrainLoop', max_epochs=max_epochs, val_interval=20)

val_dataloader = None
val_cfg = None
val_evaluator = None
test_dataloader = None
test_cfg = None
test_evaluator = None

optim_wrapper = dict(
    type='OptimWrapper',
    optimizer=dict(type='AdamW', lr=1e-4, weight_decay=0.05),
    clip_grad=dict(max_norm=0.1, norm_type=2),
    paramwise_cfg=dict(custom_keys={'encoder': dict(lr_mult=0.1)}))

param_scheduler = [
    dict(
        type='LinearLR',
        start_factor=1e-3,
        by_epoch=False,
        begin=0,
        end=200),
    dict(
        type='CosineAnnealingLR',
        T_max=max_epochs,
        eta_min=1e-6,
        begin=0,
        end=max_epochs,
        by_epoch=True),
]

default_hooks = dict(checkpoint=dict(type='CheckpointHook', interval=5))
resume = False
load_from = None