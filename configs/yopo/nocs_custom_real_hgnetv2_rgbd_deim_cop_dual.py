_base_ = ['./nocs_custom_real_hgnetv2_rgbd_dual.py']

# ─────────────────────────────────────────────────────────────────────────────
# Integrated dual-path RGB-D config: RGBDDualBackbone + CoP head + depth MAE.
#
# Inherits the dual-path backbone (RGB=HGNetV2-B2 pretrained, depth=HGNetV2-B0)
# and the DEIM training stack (ScheduleFree, batch 24, cosine 100 epochs, TopK
# checkpoint, TensorBoard). Differences from the plain dual config:
#   - depth_backbone is initialized from the depth-MAE-pretrained HGNetV2-B0
#     checkpoint (work_dirs/mae_depth/enc_b0_mae.pth) instead of the raw
#     PPHGNetV2_B0_stage1.pth, so the 1-channel depth backbone starts from
#     masked-depth-reconstruction features.
#   - CoP (Chain-of-Prediction) auxiliary head is enabled (inherited from the
#     dual config which sets bbox_head.use_cop_chain=True).
# ─────────────────────────────────────────────────────────────────────────────

model = dict(
    backbone=dict(
        depth_backbone=dict(
            init_cfg=dict(
                type='Pretrained',
                checkpoint='work_dirs/mae_depth/enc_b0_mae.pth'))))

load_from = None
resume = False

# ── Validation for TopK checkpoint ranking ──────────────────────────────────
# The inherited TopKCheckpointHook (key_indicator='3d_iou_0.50', topk=3) only
# prunes the checkpoint pool in after_val_epoch. Without a val loop no metric
# is produced and every epoch checkpoint is kept. Enable validation here so the
# K-best pool is maintained.
dataset_type = 'NOCSDataset'
data_root = 'data/nocs_custom/'
custom_intrinsic = [443.9066, 449.1953, 321.3503, 230.8687]
val_backend_args = None
val_scale = (640, 480)
val_pipeline = [
    dict(type='LoadImageFromFile', backend_args=val_backend_args),
    dict(type='LoadDepthImageFromFile', backend_args=val_backend_args,
         to_float32=True),
    dict(type='Load9DPoseAnnotations', with_bbox=True,
         with_centers_2d=True, with_z=True),
    dict(type='ConcatDepthToImage'),
    dict(type='Resize', scale=val_scale, keep_ratio=True),
    dict(
        type='Pack9DPoseInputs',
        meta_keys=('img_id', 'img_path', 'ori_shape', 'img_shape',
                   'scale_factor', 'intrinsic', 'models_info_path'),
    ),
]
val_dataloader = dict(
    _delete_=True,
    batch_size=1,
    num_workers=2,
    persistent_workers=True,
    drop_last=False,
    sampler=dict(type='DefaultSampler', shuffle=False),
    dataset=dict(
        type=dataset_type,
        data_root=data_root,
        split='custom_val',
        intrinsic=custom_intrinsic,
        pipeline=val_pipeline,
        backend_args=val_backend_args))
val_cfg = dict(_delete_=True, type='ValLoop')
val_evaluator = dict(_delete_=True, type='NOCSMetric')

# Validate every 10 epochs (val on 50 frames is cheap but not free).
train_cfg = dict(
    type='EpochBasedTrainLoop', max_epochs=100, val_interval=10)

# K-best only: disable the periodic (every-epoch) checkpoint saving inherited
# from the deim base (interval=1 used to dump all 100 checkpoints, ~140MB each)
# and keep only the best + top-K (3) checkpoints ranked by 3d_iou_0.50.
default_hooks = dict(
    checkpoint=dict(
        type='TopKCheckpointHook',
        interval=100,
        topk=3,
        key_indicator='3d_iou_0.50',
        rule='greater',
        save_best='3d_iou_0.50'))
