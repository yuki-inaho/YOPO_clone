from pathlib import Path

import pytest
from mmengine.config import Config


CONFIG = Path(
    "configs/yopo/"
    "nocs_tmt4_02_20260804_rgbd_compact_stage4_finetune_800x600.py"
)
COP_INFERENCE_CONFIG = Path(
    "configs/yopo/"
    "nocs_tmt4_02_20260804_rgbd_compact_stage4_"
    "finetune_cop_inference_800x600.py"
)


@pytest.fixture(scope="module")
def config():
    return Config.fromfile(CONFIG)


def test_finetune_keeps_compact_stage4_objective_and_stable_amp(config):
    head = config.model.bbox_head

    assert head.cop_prediction_mode == "auxiliary"
    assert head.cop_encoder_pose_supervision is True
    assert head.cop_aux_loss_weights == {
        "z": 1.0,
        "size": 1.0,
        "rotation": 1.0,
    }
    assert head.loss_z.loss_weight == pytest.approx(25.0)
    assert head.loss_sizes.loss_weight == pytest.approx(25.0)
    assert head.loss_rotation.loss_weight == pytest.approx(2.5)
    assert head.loss_obb_aux.type == "GaussianGWDLoss"
    assert head.loss_projection.type == "ProjectedEllipsoidGWDLoss"
    assert head.cop_obb_rotation_refinement is True
    assert config.optim_wrapper.type == "AmpScheduleFreeOptimWrapper"
    assert config.optim_wrapper.dtype == "bfloat16"
    assert config.optim_wrapper.optimizer.lr == pytest.approx(5e-5)
    assert config.resume is False
    assert config.load_from.endswith("best_3d_iou_0.25_epoch_5.pth")

    custom_keys = config.optim_wrapper.paramwise_cfg.custom_keys
    for name in (
        "bbox_head.reg_z_branch",
        "bbox_head.reg_size_branch",
        "bbox_head.reg_rotation_branch",
        "bbox_head.cop_",
        "bbox_head.depth_query_sampler",
    ):
        assert custom_keys[name].lr_mult == pytest.approx(0.5)


def test_finetune_uses_materialized_450_50_native_split(config):
    train = config.train_dataloader
    valid = config.val_dataloader

    assert train.batch_size == 29
    assert train.dataset.type == "NOCSCustomFruitDataset"
    assert train.dataset.split == "real_train"
    assert valid.dataset.type == "NOCSCustomFruitDataset"
    assert valid.dataset.split == "custom_val"
    assert train.dataset.data_root == valid.dataset.data_root

    for dataset in (train.dataset, valid.dataset):
        pipeline_types = [step.type for step in dataset.pipeline]
        assert "Resize" not in pipeline_types
        assert "ResizeforPose" not in pipeline_types
        identity = next(
            step
            for step in dataset.pipeline
            if step.type == "AssertIdentityImageGeometry"
        )
        assert tuple(identity.image_size) == (800, 600)
        assert identity.channels == 4


def test_finetune_has_bounded_validation_and_best_checkpoint_contract(config):
    assert config.train_cfg.max_epochs == 30
    assert config.train_cfg.val_interval == 5
    assert config.default_hooks.checkpoint.interval == 5
    assert config.default_hooks.checkpoint.save_best == [
        "3d_iou_0.25",
        "AP50_95",
        "AP75",
    ]
    assert config.custom_hooks[1].monitor == "3d_iou_0.25"
    assert config.custom_hooks[1].check_finite is True


def test_cop_inference_view_uses_same_dual_path_checkpoint_contract(config):
    cop = Config.fromfile(COP_INFERENCE_CONFIG)

    assert config.model.bbox_head.cop_prediction_mode == "auxiliary"
    assert cop.model.bbox_head.cop_prediction_mode == "chain"
    assert cop.model.bbox_head.cop_encoder_pose_supervision is False
    assert cop.load_from is None
    assert cop.train_dataloader == config.train_dataloader
    assert cop.val_dataloader == config.val_dataloader
