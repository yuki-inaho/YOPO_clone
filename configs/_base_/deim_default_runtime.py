default_scope = 'yopo'

# ── Default hooks: replace periodic CheckpointHook with TopKCheckpointHook.
#    Keeps the best-K checkpoints (by the val metric) in addition to the
#    periodic best/last checkpoints managed internally.
default_hooks = dict(
    timer=dict(type='IterTimerHook'),
    logger=dict(type='LoggerHook', interval=50),
    param_scheduler=dict(type='ParamSchedulerHook'),
    checkpoint=dict(
        type='TopKCheckpointHook',
        interval=1,
        topk=3,
        key_indicator='3d_iou_0.50',
        rule='greater',
        save_best='3d_iou_0.50'),
    sampler_seed=dict(type='DistSamplerSeedHook'),
    visualization=dict(type='DetVisualizationHook'))

env_cfg = dict(
    cudnn_benchmark=False,
    mp_cfg=dict(mp_start_method='fork', opencv_num_threads=0),
    dist_cfg=dict(backend='nccl'),
)

# ── TensorBoard: keep Local (JSON) + add TensorBoard event files.
vis_backends = [
    dict(type='LocalVisBackend'),
    dict(type='TensorboardVisBackend'),
]
visualizer = dict(
    type='DetLocalVisualizer', vis_backends=vis_backends, name='visualizer')
log_processor = dict(type='LogProcessor', window_size=50, by_epoch=True)

log_level = 'INFO'
load_from = None
resume = False
