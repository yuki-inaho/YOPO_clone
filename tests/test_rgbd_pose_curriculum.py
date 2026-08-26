from pathlib import Path

import pytest
import torch
from mmengine.config import Config

from yopo.engine.hooks.rgbd_pose_transfer import (
    build_detection_repair_transfer_state,
    build_rgbd_pose_curriculum_transfer_state,
)


def _transfer_states():
    pose = {
        'backbone.old.weight': torch.randn(2, 2),
        'neck.old.weight': torch.randn(2, 2),
        'query_embedding.weight': torch.arange(6.0).reshape(3, 2),
        'encoder.weight': torch.randn(2, 2),
        'bbox_head.weight': torch.randn(2, 2),
        'bbox_head.distillation_teacher.old.weight': torch.randn(2, 2),
    }
    feature = {
        'backbone.rgb.weight': torch.randn(2, 2),
        'backbone.depth.weight': torch.randn(2, 2),
        'neck.mapper.weight': torch.randn(2, 2),
        'bbox_head.incompatible.weight': torch.randn(2, 2),
    }
    target = {
        'backbone.rgb.weight': torch.zeros(2, 2),
        'backbone.depth.weight': torch.zeros(2, 2),
        'neck.mapper.weight': torch.zeros(2, 2),
        'query_embedding.weight': torch.zeros(5, 2),
        'encoder.weight': torch.zeros(2, 2),
        'bbox_head.weight': torch.zeros(2, 2),
        'bbox_head.depth_query_sampler.input_projections.0.weight': (
            torch.zeros(2, 2, 1, 1)
        ),
    }
    return pose, feature, target


def test_composite_transfer_uses_feature_pyramid_and_expands_pose_queries():
    pose, feature, target = _transfer_states()

    selected, report = build_rgbd_pose_curriculum_transfer_state(
        pose, feature, target, query_seed=17)

    assert set(selected) == {
        'backbone.rgb.weight',
        'backbone.depth.weight',
        'neck.mapper.weight',
        'query_embedding.weight',
        'encoder.weight',
        'bbox_head.weight',
    }
    torch.testing.assert_close(
        selected['query_embedding.weight'][:3],
        pose['query_embedding.weight'],
    )
    assert selected['query_embedding.weight'].shape == (5, 2)
    assert report['source_queries'] == 3
    assert report['target_queries'] == 5
    assert report['loaded_groups']['backbone'] == 2
    assert report['loaded_groups']['neck'] == 1
    assert report['target_only_keys'] == [
        'bbox_head.depth_query_sampler.input_projections.0.weight'
    ]
    assert report['rebuilt_pose_key_count'] == 1


def test_composite_transfer_rejects_missing_feature_and_pose_mismatch():
    pose, feature, target = _transfer_states()
    del feature['neck.mapper.weight']
    with pytest.raises(KeyError, match='feature checkpoint misses target keys'):
        build_rgbd_pose_curriculum_transfer_state(pose, feature, target)

    pose, feature, target = _transfer_states()
    pose['encoder.weight'] = torch.randn(3, 2)
    with pytest.raises(ValueError, match='pose tensor shape mismatch'):
        build_rgbd_pose_curriculum_transfer_state(pose, feature, target)


def test_portable_curriculum_uses_consistent_training_image_geometry():
    config = Config.fromfile(
        'configs/yopo/'
        'nocs_fruits_detection_Jun30_2025_rgbd_3dbbox_stage3_full.py')

    assert config.model.bbox_head.train_intrinsic_to_image_space is True
    assert [step.type for step in config.train_dataloader.dataset.pipeline][4:6] == [
        'ResizeforPose', 'ResizeOBBGaussians']
    assert [step.type for step in config.val_dataloader.dataset.pipeline][4:6] == [
        'ResizeforPose', 'ResizeOBBGaussians']


def test_stage8_repairs_geometry_without_reopening_detection_learning_rate():
    config = Config.fromfile(
        'configs/yopo/'
        'nocs_fruits_detection_Jun30_2025_rgbd_3dbbox_'
        'stage8_geometry_repair_continue10.py')

    assert config.train_cfg.max_epochs == 10
    assert config.train_cfg.val_interval == 2
    assert config.val_evaluator.score_thr == pytest.approx(0.30)
    assert config.val_evaluator.nms_cfg.iou_threshold == pytest.approx(0.375)
    keys = config.optim_wrapper.paramwise_cfg.custom_keys
    assert keys['bbox_head.cls_branches'].lr_mult == pytest.approx(0.1)
    assert keys['bbox_head.reg_centers_2d_branch'].lr_mult == pytest.approx(1.0)


def test_stage9_is_a_bounded_geometry_plateau_continuation():
    config = Config.fromfile(
        'configs/yopo/'
        'nocs_fruits_detection_Jun30_2025_rgbd_3dbbox_'
        'stage9_geometry_plateau_continue10.py')

    assert config.train_cfg.max_epochs == 10
    assert config.train_cfg.val_interval == 2
    assert config.custom_hooks[0].monitor == '3d_iou_0.50'
    assert config.custom_hooks[0].min_delta == pytest.approx(2e-3)
    assert config.custom_hooks[0].patience == 3
    assert config.default_hooks.checkpoint.max_keep_ckpts == 1


def test_portable_3dobb_curriculum_config_covers_dataset_maximum():
    config_path = Path(
        'configs/yopo/'
        'nocs_fruits_detection_Jun30_2025_rgbd_3dbbox_curriculum_base.py'
    )
    cfg = Config.fromfile(config_path)

    assert cfg.model.num_queries == 256
    assert cfg.model.test_cfg.max_per_img == 256
    assert cfg.model.bbox_head.test_cfg.max_per_img == 256
    assert cfg.train_dataloader.dataset.data_root.endswith(
        'fruits_detection_Jun30-2025_stem_rgbd_736x512/')
    assert cfg.val_dataloader.dataset.data_root == (
        cfg.train_dataloader.dataset.data_root)
    assert cfg.model.backbone.type == 'RGBDResidualBackbone'
    assert cfg.model.neck.in_channels == [384, 768, 1536]
    assert cfg.model.bbox_head.cop_depth_context.in_channels == [256, 512, 1024]


def test_detection_repair_probe_updates_2d_path_and_protects_pose_path():
    config_path = Path(
        'configs/yopo/'
        'nocs_fruits_detection_Jun30_2025_rgbd_3dbbox_'
        'stage5_detection_repair_probe.py'
    )
    cfg = Config.fromfile(config_path)

    assert cfg.load_from.endswith(
        'nocs_fruits_736x512_rgbd_3dbbox_stage4_plateau/'
        'best_3d_iou_0.50_epoch_50.pth'
    )
    assert cfg.optim_wrapper.optimizer.lr == pytest.approx(1e-5)
    custom_keys = cfg.optim_wrapper.paramwise_cfg.custom_keys
    assert custom_keys['backbone.rgb_backbone'].lr_mult == pytest.approx(0.01)
    assert custom_keys['backbone.depth_backbone'].lr_mult == pytest.approx(0.01)
    assert custom_keys['bbox_head.cls_branches'].lr_mult == pytest.approx(1.0)
    assert custom_keys['bbox_head.reg_branches'].lr_mult == pytest.approx(1.0)
    assert custom_keys['bbox_head.cop_'].lr_mult == pytest.approx(0.1)
    assert custom_keys['bbox_head.reg_z_branch'].lr_mult == pytest.approx(0.1)
    assert custom_keys['bbox_head.reg_size_branch'].lr_mult == pytest.approx(0.1)
    assert custom_keys['bbox_head.reg_rotation_branch'].lr_mult == pytest.approx(0.1)
    assert cfg.train_cfg.max_epochs == 5
    assert cfg.train_cfg.val_interval == 5
    assert cfg.default_hooks.checkpoint.save_best == [
        'AP50', '3d_iou_0.50'
    ]


def test_detection_repair_transfer_replaces_only_2d_query_path():
    source = {
        'backbone.weight': torch.full((2, 2), 1.0),
        'encoder.weight': torch.full((2, 2), 2.0),
        'decoder.weight': torch.full((2, 2), 3.0),
        'query_embedding.weight': torch.arange(6.0).reshape(3, 2),
        'level_embed': torch.full((2, 2), 4.0),
        'memory_trans_fc.weight': torch.full((2, 2), 5.0),
        'bbox_head.cls_branches.0.weight': torch.full((2, 2), 6.0),
        'bbox_head.reg_branches.0.weight': torch.full((2, 2), 7.0),
        'bbox_head.cop_z_out.0.weight': torch.full((2, 2), 8.0),
    }
    target = {
        key: torch.zeros_like(value) for key, value in source.items()
    }
    target['query_embedding.weight'] = torch.zeros(5, 2)

    selected, report = build_detection_repair_transfer_state(
        source, target, query_seed=23)

    assert set(selected) == {
        'encoder.weight',
        'decoder.weight',
        'query_embedding.weight',
        'level_embed',
        'memory_trans_fc.weight',
        'bbox_head.cls_branches.0.weight',
        'bbox_head.reg_branches.0.weight',
    }
    torch.testing.assert_close(
        selected['query_embedding.weight'][:3],
        source['query_embedding.weight'],
    )
    assert selected['query_embedding.weight'].shape == (5, 2)
    assert report['source_queries'] == 3
    assert report['target_queries'] == 5
    assert report['loaded_key_count'] == 7


def test_detection_repair_transfer_rejects_missing_or_mismatched_2d_keys():
    source = {'encoder.weight': torch.zeros(2, 2)}
    target = {
        'encoder.weight': torch.zeros(2, 2),
        'decoder.weight': torch.zeros(2, 2),
    }
    with pytest.raises(KeyError, match='misses target 2D keys'):
        build_detection_repair_transfer_state(source, target)

    source['decoder.weight'] = torch.zeros(3, 2)
    with pytest.raises(ValueError, match='2D tensor shape mismatch'):
        build_detection_repair_transfer_state(source, target)


def test_detection_transfer_probe_uses_proven_foundation_head():
    config_path = Path(
        'configs/yopo/'
        'nocs_fruits_detection_Jun30_2025_rgbd_3dbbox_'
        'stage5_2d_transfer_probe.py'
    )
    cfg = Config.fromfile(config_path)

    assert len(cfg.custom_hooks) == 1
    hook = cfg.custom_hooks[0]
    assert hook.type == 'DetectionRepairTransferHook'
    assert hook.checkpoint.endswith(
        'nocs_custom_fruit_rgbd_2d_foundation_full/'
        'best_AP50_epoch_50.pth'
    )
    assert cfg.train_cfg.max_epochs == 5
    assert cfg.default_hooks.checkpoint.save_best == [
        'AP50', '3d_iou_0.50'
    ]


def test_detection_repair_plateau_runs_to_100_with_pose_protection():
    config_path = Path(
        'configs/yopo/'
        'nocs_fruits_detection_Jun30_2025_rgbd_3dbbox_'
        'stage6_detection_repair_plateau.py'
    )
    cfg = Config.fromfile(config_path)

    assert cfg.load_from.endswith(
        'nocs_fruits_736x512_rgbd_3dbbox_stage5_detection_repair_probe/'
        'best_3d_iou_0.50_epoch_5.pth'
    )
    assert cfg.optim_wrapper.optimizer.lr == pytest.approx(5e-5)
    custom_keys = cfg.optim_wrapper.paramwise_cfg.custom_keys
    assert custom_keys['backbone.rgb_backbone'].lr_mult == pytest.approx(0.002)
    assert custom_keys['bbox_head.cop_'].lr_mult == pytest.approx(0.02)
    assert custom_keys['bbox_head.cls_branches'].lr_mult == pytest.approx(1.0)
    assert custom_keys['bbox_head.reg_branches'].lr_mult == pytest.approx(1.0)
    assert cfg.train_cfg.max_epochs == 100
    assert cfg.train_cfg.val_interval == 5
    assert cfg.val_evaluator.score_thr == pytest.approx(0.2)
    assert cfg.val_evaluator.nms_cfg.iou_threshold == pytest.approx(0.5)
    assert cfg.custom_hooks[0].monitor == 'AP50'
    assert cfg.custom_hooks[0].patience == 6


def test_stage7_continues_to_100_cumulative_epochs_with_corrected_metric():
    config_path = Path(
        'configs/yopo/'
        'nocs_fruits_detection_Jun30_2025_rgbd_3dbbox_'
        'stage7_detection_repair_continue65.py'
    )
    cfg = Config.fromfile(config_path)

    assert cfg.max_epochs == 65
    assert cfg.train_cfg.max_epochs == 65
    assert cfg.train_cfg.val_interval == 5
    assert cfg.load_from.endswith(
        'stage6_detection_repair_plateau/epoch_35.pth')
    assert cfg.val_evaluator.score_thr == pytest.approx(0.2)
    assert cfg.val_evaluator.nms_cfg.iou_threshold == pytest.approx(0.3)
    assert cfg.custom_hooks[0].monitor == 'AP50'
