"""Stage 2: add compact-Gaussian OBB supervision while retaining HBB assignment."""

_base_ = ['./nocs_fruits_2026_rgbd_shared_stage1_2d_hbb.py']

model = dict(
    bbox_head=dict(
        cop_prediction_mode='auxiliary',
        cop_use_bbox_conditioning=True,
        expose_obb_aux_predictions=True,
        loss_obb_aux=dict(
            _delete_=True,
            type='GaussianKFIoULoss',
            loss_weight=1.0,
            fail_on_invalid=True,
            eps=1e-7,
        ),
    ),
)

optim_wrapper = dict(
    paramwise_cfg=dict(
        custom_keys={
            'bbox_head.cop_': dict(lr_mult=1.0),
            'bbox_head.depth_query_sampler': dict(lr_mult=1.0),
        },
    ),
)

val_evaluator = [
    dict(
        type='NOCSMetric',
        score_thr=0.2,
        nms_cfg=dict(type='nms', iou_threshold=0.3),
        iou_thrs=[value / 100 for value in range(50, 100, 5)],
        hbb_selection=dict(score_thr=0.0, nms_cfg=None),
        compute_pose_metrics=False,
    ),
    dict(
        type='EllipseEnvelopeRotatedIoUMetric',
        pred_field='obb_aux_obb',
        iou_thr=0.5,
        score_thr=0.05,
        num_classes=1,
        prefix='obb',
    ),
]

max_epochs = 30
train_cfg = dict(max_epochs=max_epochs, val_interval=5)
custom_hooks = [
    dict(type='ScheduleFreeOptimizerModeHook'),
    dict(
        type='EarlyStoppingHook',
        monitor='obb/rbbox_mAP_50',
        rule='greater',
        min_delta=1e-3,
        patience=4,
        strict=True,
        check_finite=True,
    ),
]
default_hooks = dict(
    checkpoint=dict(
        _delete_=True,
        type='CheckpointHook',
        interval=5,
        save_last=True,
        max_keep_ckpts=2,
        save_best=['AP50_95', 'AP75', 'obb/rbbox_mAP_50'],
        rule=['greater', 'greater', 'greater'],
        save_optimizer=True,
    ),
)
load_from = None
resume = False
