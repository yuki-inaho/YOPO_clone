"""Production shared RGB-D YOPO configuration contracts."""

from __future__ import annotations

from mmengine.config import Config
from mmengine.optim import AmpOptimWrapper

from yopo.engine.optimizers.amuse import AmpAmuseOptimWrapper


def test_amp_amuse_wrapper_keeps_amp_and_schedule_free_mode_contracts() -> None:
    assert issubclass(AmpAmuseOptimWrapper, AmpOptimWrapper)
    assert 'step' in AmpAmuseOptimWrapper.__dict__


def test_shared_rgbd_stage1_uses_contract_architecture_and_amuse() -> None:
    config = Config.fromfile(
        'configs/yopo/nocs_fruits_2026_rgbd_shared_stage1_2d_hbb.py'
    )

    assert config.model.backbone.type == 'RGBDResidualBackbone'
    assert config.model.backbone.rgb_backbone.name == 'B2'
    assert config.model.backbone.rgb_backbone.use_lab is True
    assert config.model.backbone.rgb_backbone.freeze_norm is True
    assert config.model.backbone.depth_backbone.name == 'B0'
    assert config.model.backbone.depth_backbone.use_lab is True
    assert config.model.backbone.depth_backbone.freeze_norm is True
    assert config.model.neck.type == 'PortableHybridEncoderNeck'
    assert config.model.neck.in_channels == [384, 768, 1536]
    assert config.model.neck.num_outs == 4
    assert config.model.data_preprocessor.mean == [0.0, 0.0, 0.0, 0.0]
    assert config.model.data_preprocessor.std == [255.0, 255.0, 255.0, 1.0]
    assert config.optim_wrapper.type.endswith('AmpAmuseOptimWrapper')
    assert config.optim_wrapper.dtype == 'bfloat16'
    assert config.optim_wrapper.loss_scale == 1.0
    assert config.optim_wrapper.optimizer.type.endswith('AmuseOptimizer')
    assert config.param_scheduler == []

    train_dataset = config.train_dataloader.dataset
    assert train_dataset.type == 'NOCSCustomFruitDataset'
    assert train_dataset.split == 'real_train'
    concat = next(item for item in train_dataset.pipeline if item.type == 'ConcatRawDepthToImage')
    assert concat.depth_scale == 1.0
    assert concat.bgr_to_rgb is True
