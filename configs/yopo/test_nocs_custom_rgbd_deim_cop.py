_base_ = ['./nocs_custom_real_hgnetv2_rgbd_deim_cop.py']

# Test-only config: evaluate the trained CoP model on the custom val split
# (scene_1_val from test_list.txt) and optionally dump visualizations.

# ── Val / test data: custom val split ─────────────────────────────────────
dataset_type = 'NOCSDataset'
data_root = 'data/nocs_custom/'
custom_intrinsic = [443.9066, 449.1953, 321.3503, 230.8687]

# num_queries=100, so cap max_per_img at the query budget (deim base used 300).
model = dict(test_cfg=dict(max_per_img=100))

backend_args = None
scale = (640, 480)
test_pipeline = [
    dict(type='LoadImageFromFile', backend_args=backend_args),
    dict(type='LoadDepthImageFromFile', backend_args=backend_args,
         to_float32=True),
    dict(type='Load9DPoseAnnotations', with_bbox=True,
         with_centers_2d=True, with_z=True),
    dict(type='ConcatDepthToImage'),
    dict(type='Resize', scale=scale, keep_ratio=True),
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
        test_mode=True,
        intrinsic=custom_intrinsic,
        pipeline=test_pipeline,
        backend_args=backend_args))
val_cfg = dict(_delete_=True, type='ValLoop')
val_evaluator = dict(
    _delete_=True,
    type='NOCSMetric',
    format_only=False,
    score_thr=0.2)
test_cfg = dict(_delete_=True, type='TestLoop')

test_dataloader = val_dataloader
test_evaluator = val_evaluator

# ── Visualization ───────────────────────────────────────────────────────
vis_backends = [
    dict(type='LocalVisBackend'),
]
visualizer = dict(
    type='PoseLocalVisualizer', vis_backends=vis_backends, name='visualizer')
default_scope = 'yopo'

# Training hooks disabled for a clean test run.
default_hooks = dict(
    checkpoint=dict(type='CheckpointHook', interval=1),
    logger=dict(type='LoggerHook', interval=50),
    param_scheduler=dict(type='ParamSchedulerHook'),
    sampler_seed=dict(type='DistSamplerSeedHook'),
)