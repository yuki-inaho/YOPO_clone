"""Stage 7: add one-way dual-quadric projection to the direct 3D objective."""

_base_ = ['./nocs_fruits_2026_rgbd_shared_stage6_ellipsoid_direct.py']

model = dict(
    bbox_head=dict(
        loss_ellipsoid_projection=dict(
            _delete_=True,
            type='DualQuadricProjectionGWDLoss',
            loss_weight=1.0,
            tau=1.0,
            normalize=True,
            include_center=True,
            detach_target=True,
            fail_on_invalid=True,
        ),
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
    dict(type='GauCho3DSharedMatchMetric', prefix='ellipsoid'),
    dict(
        type='ProjectedEllipsoidRotatedIoUMetric',
        iou_thr=0.5,
        score_thr=0.05,
        num_classes=1,
        prefix='projection',
    ),
    dict(type='GauCho3DSharedMatchMetric', prefix='projection'),
]

max_epochs = 50
train_cfg = dict(max_epochs=max_epochs, val_interval=5)
custom_hooks = [
    dict(type='ScheduleFreeOptimizerModeHook'),
    dict(
        type='EarlyStoppingHook',
        monitor='projection/shared_AP_25',
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
            'projection/shared_AP_25', 'projection/AP_50',
            'ellipsoid/shared_AP_25', '3d_iou_0.25', 'AP50_95',
            'obb/rbbox_mAP_50', 'ellipse/rbbox_mAP_50'],
        rule=['greater'] * 7,
        save_optimizer=True,
    ),
)
load_from = None
resume = False
