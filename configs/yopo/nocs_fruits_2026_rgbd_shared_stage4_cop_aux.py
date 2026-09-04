"""Stage 4: train metric pose through parallel and auxiliary CoP paths."""

_base_ = ['./nocs_fruits_2026_rgbd_shared_stage3_ellipse2d.py']

model = dict(
    bbox_head=dict(
        cop_prediction_mode='auxiliary',
        cop_aux_loss_weights=dict(z=1.0, size=1.0, rotation=1.0),
        loss_z=dict(loss_weight=25.0),
        loss_sizes=dict(loss_weight=25.0),
        loss_rotation=dict(loss_weight=2.5),
    ),
)

optim_wrapper = dict(
    paramwise_cfg=dict(
        custom_keys={
            'bbox_head.reg_z_branch': dict(lr_mult=0.25),
            'bbox_head.reg_size_branch': dict(lr_mult=0.25),
            'bbox_head.reg_rotation_branch': dict(lr_mult=0.25),
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
        compute_pose_metrics=True,
    ),
    dict(
        type='EllipseEnvelopeRotatedIoUMetric', pred_field='obb_aux_obb',
        iou_thr=0.5, score_thr=0.05, num_classes=1, prefix='obb'),
    dict(
        type='EllipseEnvelopeRotatedIoUMetric', pred_field='ellipse_obb',
        iou_thr=0.5, score_thr=0.05, num_classes=1, prefix='ellipse'),
]

max_epochs = 30
train_cfg = dict(max_epochs=max_epochs, val_interval=5)
custom_hooks = [
    dict(type='ScheduleFreeOptimizerModeHook'),
    dict(
        type='EarlyStoppingHook',
        monitor='3d_iou_0.25',
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
        save_best=[
            '3d_iou_0.25', 'AP50_95', 'AP75',
            'obb/rbbox_mAP_50', 'ellipse/rbbox_mAP_50'],
        rule=['greater'] * 5,
        save_optimizer=True,
    ),
)
load_from = None
resume = False
