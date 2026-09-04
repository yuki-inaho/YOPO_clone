"""Contracts for the shared B2/B0 RGB-D YOPO curriculum."""

from copy import deepcopy
from pathlib import Path

from mmengine.config import Config


CONFIG_DIR = Path('configs/yopo')
NAMES = (
    'nocs_fruits_2026_rgbd_shared_stage1_2d_hbb.py',
    'nocs_fruits_2026_rgbd_shared_stage2_obb.py',
    'nocs_fruits_2026_rgbd_shared_stage3_ellipse2d.py',
    'nocs_fruits_2026_rgbd_shared_stage4_cop_aux.py',
    'nocs_fruits_2026_rgbd_shared_stage5_cop_chain.py',
    'nocs_fruits_2026_rgbd_shared_stage6_ellipsoid_direct.py',
    'nocs_fruits_2026_rgbd_shared_stage7_ellipsoid_projection.py',
)


def _configs() -> list[Config]:
    return [Config.fromfile(CONFIG_DIR / name) for name in NAMES]


def test_every_stage_preserves_shared_architecture_data_and_bf16_amuse() -> None:
    for config in _configs():
        assert config.model.backbone.rgb_backbone.name == 'B2'
        assert config.model.backbone.depth_backbone.name == 'B0'
        assert config.model.neck.type == 'PortableHybridEncoderNeck'
        assert config.train_dataloader.dataset.split == 'real_train'
        assert config.optim_wrapper.type.endswith('AmpAmuseOptimWrapper')
        assert config.optim_wrapper.dtype == 'bfloat16'
        assert config.param_scheduler == []
        nocs = (
            config.val_evaluator[0]
            if isinstance(config.val_evaluator, list)
            else config.val_evaluator
        )
        assert nocs.score_thr == 0.2
        assert nocs.nms_cfg == {'type': 'nms', 'iou_threshold': 0.3}
        assert nocs.hbb_selection == {'score_thr': 0.0, 'nms_cfg': None}
        assert nocs.iou_thrs == [value / 100 for value in range(50, 100, 5)]


def test_curriculum_adds_one_geometry_family_at_a_time() -> None:
    hbb, obb, ellipse, cop_aux, cop_chain, direct, projection = _configs()

    assert hbb.model.bbox_head.loss_obb_aux is None
    assert obb.model.bbox_head.loss_obb_aux.type == 'GaussianKFIoULoss'
    assert obb.model.bbox_head.loss_z.loss_weight == 0.0
    assert obb.model.bbox_head.cop_prediction_mode == 'auxiliary'

    assert ellipse.model.bbox_head.gaucho_ellipse2d is True
    assert ellipse.model.bbox_head.loss_ellipse2d.type == 'Ellipse2DKLDLoss'
    assert cop_aux.model.bbox_head.loss_z.loss_weight == 25.0
    assert cop_aux.model.bbox_head.cop_prediction_mode == 'auxiliary'

    assert cop_chain.model.bbox_head.cop_prediction_mode == 'chain'
    assert cop_chain.model.bbox_head.loss_obb_aux.type == 'GaussianGWDLoss'
    assert direct.model.bbox_head.gaucho_ellipsoid is True
    assert direct.model.bbox_head.loss_ellipsoid.type == 'Ellipsoid3DKLDLoss'
    assert direct.model.bbox_head.loss_ellipsoid_projection is None
    assert projection.model.bbox_head.loss_ellipsoid_projection.type == (
        'DualQuadricProjectionGWDLoss'
    )


def test_obb_and_ellipse_metrics_use_explicit_prediction_fields() -> None:
    obb, ellipse = _configs()[1:3]
    assert obb.val_evaluator[1].pred_field == 'obb_aux_obb'
    assert [metric.pred_field for metric in ellipse.val_evaluator[1:]] == [
        'obb_aux_obb',
        'ellipse_obb',
    ]


def test_each_stage_selects_a_best_checkpoint_on_its_new_primary_metric() -> None:
    configs = _configs()
    expected = [
        'AP50_95',
        'obb/rbbox_mAP_50',
        'ellipse/rbbox_mAP_50',
        '3d_iou_0.25',
        '3d_iou_0.25',
        'ellipsoid/shared_AP_25',
        'projection/shared_AP_25',
    ]
    for config, metric in zip(configs, expected, strict=True):
        assert config.custom_hooks[1].monitor == metric
        saved = config.default_hooks.checkpoint.save_best
        saved = [saved] if isinstance(saved, str) else saved
        assert metric in saved


def test_projection_stage_reports_2d_and_shared_3d_projection_metrics() -> None:
    projection = _configs()[-1]
    assert projection.val_evaluator[-2].type == (
        'ProjectedEllipsoidRotatedIoUMetric'
    )
    assert projection.val_evaluator[-2].prefix == 'projection'
    assert projection.val_evaluator[-1].type == 'GauCho3DSharedMatchMetric'
    assert projection.val_evaluator[-1].prefix == 'projection'


def test_stage10_reverse_kld_and_two_update_smoke_contracts() -> None:
    stage10 = Config.fromfile(
        CONFIG_DIR / 'nocs_fruits_2026_rgbd_shared_stage10_reverse_kld.py'
    )
    smoke = Config.fromfile(
        CONFIG_DIR /
        'nocs_fruits_2026_rgbd_shared_stage10_reverse_kld_capacity_smoke.py'
    )

    assert stage10.model.bbox_head.loss_ellipsoid.direction == (
        'prediction_to_target'
    )
    assert stage10.max_epochs == 15
    assert stage10.train_cfg.max_epochs == 15
    assert stage10.train_cfg.val_interval == 5
    assert stage10.train_dataloader.batch_size == 24
    assert stage10.optim_wrapper.type.endswith('AmpAmuseOptimWrapper')

    assert smoke.train_cfg.type == 'IterBasedTrainLoop'
    assert smoke.train_cfg.max_iters == 2
    assert smoke.train_dataloader.batch_size == 24
    assert smoke.val_dataloader is None
    assert smoke.val_cfg is None
    assert smoke.val_evaluator is None
    assert smoke.load_from.endswith(
        'best_projection_shared_AP_25_epoch_5.pth'
    )


def test_stage10_changes_only_kld_direction_and_epoch_cap_from_stage8() -> None:
    stage8 = Config.fromfile(
        CONFIG_DIR / 'nocs_fruits_2026_rgbd_shared_stage8_sensor_depth_anchor.py'
    ).to_dict()
    stage10 = Config.fromfile(
        CONFIG_DIR / 'nocs_fruits_2026_rgbd_shared_stage10_reverse_kld.py'
    ).to_dict()

    normalized = deepcopy(stage10)
    direction = normalized['model']['bbox_head']['loss_ellipsoid'].pop(
        'direction'
    )
    assert direction == 'prediction_to_target'
    assert normalized['max_epochs'] == 15
    assert normalized['train_cfg']['max_epochs'] == 15
    normalized['max_epochs'] = stage8['max_epochs']
    normalized['train_cfg']['max_epochs'] = stage8['train_cfg']['max_epochs']
    assert normalized == stage8
