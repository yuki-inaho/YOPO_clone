"""Stage 6: add direct GauCho 3D ellipsoid supervision without exact projection."""

_base_ = ['./nocs_fruits_2026_rgbd_shared_stage5_cop_chain.py']

model = dict(
    bbox_head=dict(
        gaucho_ellipsoid=True,
        gaucho_chart='scale_shape',
        gaucho_size_prior=0.03,
        gaucho_use_bbox_conditioning=True,
        loss_ellipsoid=dict(
            type='Ellipsoid3DKLDLoss',
            loss_weight=2.0,
            tau=1.0,
            include_center=True,
            fail_on_invalid=True,
        ),
        loss_ellipsoid_projection=None,
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
]

max_epochs = 30
train_cfg = dict(max_epochs=max_epochs, val_interval=5)
custom_hooks = [
    dict(type='ScheduleFreeOptimizerModeHook'),
    dict(
        type='EarlyStoppingHook',
        monitor='ellipsoid/shared_AP_25',
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
            'ellipsoid/shared_AP_25', '3d_iou_0.25', 'AP50_95',
            'obb/rbbox_mAP_50', 'ellipse/rbbox_mAP_50'],
        rule=['greater'] * 5,
        save_optimizer=True,
    ),
)
load_from = None
resume = False
