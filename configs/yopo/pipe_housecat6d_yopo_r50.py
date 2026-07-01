_base_ = './housecat6d_yopo_r50.py'

data_root = 'data/housecat6d_pipe_nyx650_250'

model = dict(
    train_cfg=dict(
        assigner=dict(
            match_costs=[
                dict(type='FocalLossCost', weight=2.0),
                dict(type='BBoxL1Cost', weight=5.0, box_format='xywh'),
                dict(type='IoUCost', iou_mode='giou', weight=2.0),
                dict(type='TranslationCost', weight=5.0),
                dict(
                    type='RotationCost',
                    weight=2.0,
                    symmetric_classes=[1, 2, 7, 9]),
            ])))

train_dataloader = dict(dataset=dict(data_root=data_root, split='train'))
val_dataloader = dict(dataset=dict(data_root=data_root, split='val'))
test_dataloader = dict(dataset=dict(data_root=data_root, split='val'))

vis_backends = [
    dict(type='LocalVisBackend'),
    dict(type='TensorboardVisBackend'),
]
visualizer = dict(
    type='PoseLocalVisualizer',
    vis_backends=vis_backends,
    name='visualizer')

default_hooks = dict(
    logger=dict(type='LoggerHook', interval=10),
    checkpoint=dict(
        type='CheckpointHook',
        interval=1,
        save_best='AP50',
        rule='greater',
        max_keep_ckpts=3),
    early_stopping=dict(
        type='EarlyStoppingHook',
        monitor='AP50',
        rule='greater',
        patience=10,
        min_delta=0.001))

custom_hooks = [
    dict(
        type='EMAHook',
        ema_type='ExpMomentumEMA',
        momentum=0.0002,
        update_buffers=True,
        priority=49),
]
