from __future__ import annotations

from mmengine.config import Config

from tools.test import ensure_test_components


CONFIG_ROOT = 'configs/yopo/'


def _load(name: str) -> Config:
    return Config.fromfile(CONFIG_ROOT + name)


def test_mal_probe_changes_only_quality_classification_from_2d_control():
    control = _load(
        'nocs_fruits_detection_Jun30_2025_rgbd_3dbbox_'
        'stage10_2d_anchor_control_probe5.py')
    mal = _load(
        'nocs_fruits_detection_Jun30_2025_rgbd_3dbbox_'
        'stage10_2d_anchor_mal_probe5.py')

    assert control.load_from == mal.load_from
    assert control.model.num_queries == mal.model.num_queries == 256
    assert control.model.train_cfg == mal.model.train_cfg
    assert control.optim_wrapper == mal.optim_wrapper
    assert control.model.bbox_head.loss_cls.type == 'FocalLoss'
    assert mal.model.bbox_head.loss_cls.type == 'MatchabilityAwareLoss'
    assert mal.model.bbox_head.quality_target.source == 'hbb_iou'


def test_stage10_assignment_is_2d_only_and_pose_losses_remain_enabled():
    config = _load(
        'nocs_fruits_detection_Jun30_2025_rgbd_3dbbox_'
        'stage10_2d_anchor_control_probe5.py')

    costs = config.model.train_cfg.assigner.match_costs
    assert [cost.type for cost in costs] == [
        'FocalLossCost', 'BBoxL1Cost', 'IoUCost']
    head = config.model.bbox_head
    assert head.loss_z.loss_weight > 0
    assert head.loss_sizes.loss_weight > 0
    assert head.loss_rotation.loss_weight > 0


def test_obb_blend_and_full_schedule_are_config_only_extensions():
    mal = _load(
        'nocs_fruits_detection_Jun30_2025_rgbd_3dbbox_'
        'stage10_2d_anchor_mal_probe5.py')
    blend = _load(
        'nocs_fruits_detection_Jun30_2025_rgbd_3dbbox_'
        'stage10_2d_anchor_mal_obb_blend_probe5.py')
    full = _load(
        'nocs_fruits_detection_Jun30_2025_rgbd_3dbbox_'
        'stage10_2d_anchor_mal_full.py')

    assert mal.model.bbox_head.loss_cls == blend.model.bbox_head.loss_cls
    assert blend.model.bbox_head.quality_target.source == 'blend'
    assert blend.model.bbox_head.quality_target.obb_weight == 0.25
    assert full.train_cfg.max_epochs == 100
    assert full.load_from is None
    hooks_by_type = {hook.type: hook for hook in full.custom_hooks}
    assert hooks_by_type['ScheduleFreeOptimizerModeHook']
    assert hooks_by_type['EarlyStoppingHook'].monitor == 'AP50_95'
    assert full.default_hooks.checkpoint.save_optimizer is True


def test_stage10_separates_raw_hbb_ap_from_operating_3d_selection():
    config = _load(
        'nocs_fruits_detection_Jun30_2025_rgbd_3dbbox_'
        'stage10_2d_anchor_mal_probe5.py')

    evaluator = config.val_evaluator
    assert evaluator.score_thr == 0.2
    assert evaluator.nms_cfg.iou_threshold == 0.3
    assert evaluator.hbb_selection.score_thr == 0.0
    assert evaluator.hbb_selection.nms_cfg is None
    assert len(evaluator.iou_thrs) == 10
    assert evaluator.iou_thrs[0] == 0.5
    assert evaluator.iou_thrs[-1] == 0.95


def test_offline_test_reuses_validation_components_without_config_duplication():
    config = _load(
        'nocs_fruits_detection_Jun30_2025_rgbd_3dbbox_'
        'stage10_2d_anchor_mal_probe5.py')

    assert config.test_dataloader is None
    assert config.test_cfg is None
    assert config.test_evaluator is None
    ensure_test_components(config)
    assert config.test_dataloader == config.val_dataloader
    assert config.test_cfg.type == 'TestLoop'
    assert config.test_evaluator == config.val_evaluator


def test_smoke_uses_bounded_iterations_and_small_loader():
    smoke = _load(
        'nocs_fruits_detection_Jun30_2025_rgbd_3dbbox_'
        'stage10_2d_anchor_mal_smoke.py')

    assert smoke.train_cfg.type == 'IterBasedTrainLoop'
    assert smoke.train_cfg.max_iters == 2
    assert smoke.train_dataloader.batch_size == 2
    assert smoke.train_dataloader.num_workers == 0
    assert smoke.model.bbox_head.loss_cls.type == 'MatchabilityAwareLoss'


def test_ocd_probe_changes_only_the_denoising_box_strategy():
    mal = _load(
        'nocs_fruits_detection_Jun30_2025_rgbd_3dbbox_'
        'stage10_2d_anchor_mal_probe5.py')
    ocd = _load(
        'nocs_fruits_detection_Jun30_2025_rgbd_3dbbox_'
        'stage10_2d_anchor_mal_ocd_probe5.py')

    assert mal.model.bbox_head == ocd.model.bbox_head
    assert mal.model.train_cfg == ocd.model.train_cfg
    assert mal.optim_wrapper == ocd.optim_wrapper
    assert 'box_noise' not in mal.model.dn_cfg
    assert ocd.model.dn_cfg.box_noise.type == 'BoxOnlyOCDNoise'
    assert ocd.model.dn_cfg.box_noise.positive_noise_scale == 0.5
    assert tuple(ocd.model.dn_cfg.box_noise.negative_noise_scale) == (0.5, 1.0)
