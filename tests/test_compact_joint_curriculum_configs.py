from pathlib import Path

import pytest
from mmengine.config import Config


CONFIG_DIR = Path("configs/yopo")
STAGES = [
    CONFIG_DIR
    / "nocs_fruits_2025_2026_rgbd_compact_curriculum_stage1_2d_full.py",
    CONFIG_DIR
    / "nocs_fruits_2025_2026_rgbd_compact_curriculum_stage2_parallel_pose.py",
    CONFIG_DIR
    / "nocs_fruits_2025_2026_rgbd_compact_curriculum_stage3_cop_aux_kfiou.py",
    CONFIG_DIR
    / "nocs_fruits_2025_2026_rgbd_compact_curriculum_stage4_cop_chain_gwd_full.py",
]


@pytest.fixture(scope="module")
def configs():
    return [Config.fromfile(path) for path in STAGES]


def test_all_stages_keep_joint_native_data_and_amp_batch_30(configs):
    for stage_index, config in enumerate(configs):
        assert (
            "yopo.datasets.transforms.identity_geometry"
            in config.custom_imports.imports
        )
        assert config.train_dataloader.batch_size == 30
        assert config.model.data_preprocessor.pad_size_divisor == 1
        assert config.optim_wrapper.type == "AmpScheduleFreeOptimWrapper"
        if stage_index < 3:
            assert config.optim_wrapper.dtype == "float16"
            assert config.optim_wrapper.loss_scale == pytest.approx(0.25)
            assert config.optim_wrapper.optimizer.lr == pytest.approx(1e-4)
        else:
            assert config.optim_wrapper.dtype == "bfloat16"
            assert config.optim_wrapper.loss_scale == pytest.approx(1.0)
            assert config.optim_wrapper.optimizer.lr == pytest.approx(5e-5)
        roots = [
            dataset.data_root
            for dataset in config.train_dataloader.dataset.datasets
        ]
        assert roots == [
            "data/fruits_rgbd_2025_2026_800x600_preprocessed/2025/",
            "data/fruits_rgbd_2025_2026_800x600_preprocessed/2026/",
        ]
        pipeline_types = {
            transform.type
            for transform in config.train_dataloader.dataset.datasets[0].pipeline
        }
        assert "Resize" not in pipeline_types
        assert "AssertIdentityImageGeometry" in pipeline_types
        identity = next(
            transform
            for transform in config.train_dataloader.dataset.datasets[0].pipeline
            if transform.type == "AssertIdentityImageGeometry"
        )
        assert tuple(identity.image_size) == (800, 600)
        assert identity.channels == 4
        val_pipeline_types = {
            transform.type
            for transform in config.val_dataloader.dataset.datasets[0].pipeline
        }
        assert "AssertIdentityImageGeometry" in val_pipeline_types


def test_stage1_is_pose_independent_detection(configs):
    config = configs[0]
    head = config.model.bbox_head

    assert head.cop_prediction_mode == "parallel"
    assert head.loss_z.loss_weight == 0.0
    assert head.loss_sizes.loss_weight == 0.0
    assert head.loss_rotation.loss_weight == 0.0
    assert head.loss_projection is None
    assert head.loss_obb_aux is None
    assert config.val_evaluator.compute_pose_metrics is False
    assert config.default_hooks.checkpoint.save_best == ["AP50_95", "AP75"]

    custom_keys = config.optim_wrapper.paramwise_cfg.custom_keys
    assert custom_keys["bbox_head.reg_z_branch"].lr_mult == 0.0
    assert custom_keys["bbox_head.reg_size_branch"].lr_mult == 0.0
    assert custom_keys["bbox_head.reg_rotation_branch"].lr_mult == 0.0
    assert custom_keys["bbox_head.cop_"].lr_mult == 0.0


def test_stage2_adds_parallel_pose_without_changing_2d_assignment(configs):
    config = configs[1]
    head = config.model.bbox_head

    assert head.cop_prediction_mode == "parallel"
    assert head.loss_z.loss_weight == 50.0
    assert head.loss_sizes.loss_weight == 50.0
    assert head.loss_rotation.loss_weight == 5.0
    assert head.loss_obb_aux is None
    assert config.val_evaluator.compute_pose_metrics is True
    assert [cost.type for cost in config.model.train_cfg.assigner.match_costs] == [
        "FocalLossCost",
        "BBoxL1Cost",
        "IoUCost",
    ]


def test_stage3_splits_pose_budget_and_introduces_kfiou_cop(configs):
    head = configs[2].model.bbox_head

    assert head.cop_prediction_mode == "auxiliary"
    assert head.cop_aux_loss_weights == {
        "z": 1.0,
        "size": 1.0,
        "rotation": 1.0,
    }
    assert head.loss_z.loss_weight == 25.0
    assert head.loss_sizes.loss_weight == 25.0
    assert head.loss_rotation.loss_weight == 2.5
    assert head.loss_obb_aux.type == "GaussianKFIoULoss"
    assert head.cop_obb_rotation_refinement is False


def test_stage4_makes_cop_primary_and_switches_to_gwd(configs):
    config = configs[3]
    head = config.model.bbox_head

    assert head.cop_prediction_mode == "chain"
    assert head.cop_encoder_pose_supervision is False
    assert head.cop_obb_rotation_refinement is True
    assert head.loss_obb_aux.type == "GaussianGWDLoss"
    assert head.loss_projection.type == "ProjectedEllipsoidGWDLoss"
    assert config.train_cfg.max_epochs == 100
    assert config.custom_hooks[1].monitor == "3d_iou_0.25"
