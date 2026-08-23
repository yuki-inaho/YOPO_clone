_base_ = ['./nocs_custom_real_r50_rgbd.py']

# Custom RGB-D (single-class fruit) fine-tune config with a DEIM-style
# HGNetV2 backbone (ported from the mmrotate sandbox, BaseModule version),
# trained on the 4-channel RGB-D custom data (data/nocs_custom).
#
# Architecture:
#   backbone = HGNetV2 (B2)  -> stage outputs (return_idx=[1,2,3]) with
#                               out_channels [384, 768, 1536], in_channels=4
#   neck     = ChannelMapper -> 4 levels at 256ch (num_outs=4)
# Pretrained weights are the DEIM/DFINE PPHGNetV2_B2_stage1.pth (COCO
# backbone), loaded non-strictly (the 3-channel stem is skipped for the
# RGB-D 4-channel stem, which is randomly initialized).

custom_intrinsic = [443.9066, 449.1953, 321.3503, 230.8687]

model = dict(
    _delete_=True,
    type='DINO9DCenter2DPose',
    num_queries=100,
    with_box_refine=True,
    as_two_stage=True,
    data_preprocessor=dict(
        type='DetDataPreprocessor',
        mean=[123.675, 116.28, 103.53, 127.5],
        std=[58.395, 57.12, 57.375, 127.5],
        bgr_to_rgb=False,
        pad_size_divisor=1),
    backbone=dict(
        type='HGNetV2',
        name='B2',
        in_channels=4,
        return_idx=[1, 2, 3],
        freeze_at=0,
        freeze_norm=True,
        init_cfg=dict(
            type='Pretrained',
            checkpoint='https://github.com/Peterande/storage/releases/download/'
            'dfinev1.0/PPHGNetV2_B2_stage1.pth')),
    neck=dict(
        type='ChannelMapper',
        in_channels=[384, 768, 1536],
        kernel_size=1,
        out_channels=256,
        act_cfg=None,
        norm_cfg=dict(type='GN', num_groups=32),
        num_outs=4),
    encoder=dict(
        num_layers=6,
        layer_cfg=dict(
            self_attn_cfg=dict(embed_dims=256, num_levels=4, dropout=0.0),
            ffn_cfg=dict(embed_dims=256, feedforward_channels=2048,
                         ffn_drop=0.0))),
    decoder=dict(
        num_layers=6,
        return_intermediate=True,
        layer_cfg=dict(
            self_attn_cfg=dict(embed_dims=256, num_heads=8, dropout=0.0),
            cross_attn_cfg=dict(embed_dims=256, num_levels=4, dropout=0.0),
            ffn_cfg=dict(embed_dims=256, feedforward_channels=2048,
                         ffn_drop=0.0)),
        post_norm_cfg=None),
    positional_encoding=dict(num_feats=128, normalize=True, offset=0.0,
                             temperature=20),
    bbox_head=dict(
        type='DINO9DCenter2DPoseHead',
        num_classes=1,
        use_bbox_for_z=True,
        classwise_rotation=True,
        classwise_sizes=True,
        sync_cls_avg_factor=True,
        loss_cls=dict(type='FocalLoss', use_sigmoid=True, gamma=2.0,
                      alpha=0.25, loss_weight=1.0),
        loss_bbox=dict(type='L1Loss', loss_weight=5.0),
        loss_iou=dict(type='GIoULoss', loss_weight=2.0),
        loss_centers_2d=dict(type='L1PoseLoss', loss_weight=5.0),
        loss_z=dict(type='L2PoseLoss', loss_weight=50.0),
        loss_rotation=dict(type='Rotation3DLoss', loss_weight=5.0),
        loss_sizes=dict(type='L2PoseLoss', loss_weight=50.0)),
    dn_cfg=dict(label_noise_scale=0.5, box_noise_scale=1.0,
                group_cfg=dict(dynamic=True, num_groups=None,
                               num_dn_queries=20)),
    train_cfg=dict(
        assigner=dict(
            type='HungarianAssigner',
            match_costs=[
                dict(type='FocalLossCost', weight=2.0),
                dict(type='BBoxL1Cost', weight=5.0, box_format='xywh'),
                dict(type='IoUCost', iou_mode='giou', weight=2.0),
                dict(type='TranslationCost', weight=5.0),
                dict(type='RotationCost', symmetric_classes=[0, 1, 3],
                     weight=2.0),
            ])),
    test_cfg=dict(max_per_img=300))

# Pretraining checkpoint to load (full DINO9D head is built from the DEIM
# backbone only; _base_ sets load_from=None for smoke runs).
load_from = None

optim_wrapper = dict(
    type='OptimWrapper',
    optimizer=dict(type='AdamW', lr=0.0001, weight_decay=0.0001),
    clip_grad=dict(max_norm=0.1, norm_type=2),
    paramwise_cfg=dict(custom_keys={'backbone': dict(lr_mult=0.1)}))
