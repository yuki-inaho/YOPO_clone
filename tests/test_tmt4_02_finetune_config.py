from pathlib import Path

import pytest
from mmengine.config import Config


CONFIG = Path(
    "configs/yopo/"
    "nocs_tmt4_02_20260804_rgbd_compact_stage4_finetune_800x600.py"
)


@pytest.fixture(scope="module")
def config():
    return Config.fromfile(CONFIG)


def test_finetune_keeps_compact_stage4_objective_and_stable_amp(config):
    head = config.model.bbox_head

    assert head.cop_prediction_mode == "chain"
    assert head.loss_obb_aux.type == "GaussianGWDLoss"
    assert head.loss_projection.type == "ProjectedEllipsoidGWDLoss"
    assert config.optim_wrapper.type == "AmpScheduleFreeOptimWrapper"
    assert config.optim_wrapper.dtype == "bfloat16"
    assert config.optim_wrapper.optimizer.lr == pytest.approx(5e-5)
    assert config.resume is False
    assert config.load_from.endswith("best_3d_iou_0.25_epoch_5.pth")


def test_finetune_uses_materialized_450_50_native_split(config):
    train = config.train_dataloader
    valid = config.val_dataloader

    assert train.batch_size == 30
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
