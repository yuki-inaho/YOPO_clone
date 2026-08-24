"""Unit contract for freezing only the transferred RGB branch."""

from __future__ import annotations

import torch
from mmengine.config import Config

from yopo.registry import MODELS
from yopo.utils import register_all_modules


CONFIG_PATH = "configs/yopo/nocs_custom_real_hgnetv2_rgbd_deim_cop_dual.py"


def _build_frozen_model():
    from yopo.models.backbones.frozen_rgbd import FrozenRGBDDualBackbone

    register_all_modules()
    config = Config.fromfile(CONFIG_PATH)
    config.model.backbone.type = "FrozenRGBDDualBackbone"
    # The inherited exploratory config froze the depth stem.  This transfer
    # run freezes RGB only, so depth must be fully trainable from construction.
    config.model.backbone.depth_backbone.freeze_at = -1
    model = MODELS.build(config.model)
    assert isinstance(model.backbone, FrozenRGBDDualBackbone)
    return model


def test_freeze_parameter_contract():
    """RGB has no gradient; depth/fusion/3D head retain a finite path."""
    model = _build_frozen_model()
    model.train()

    assert not model.backbone.rgb_backbone.training
    assert all(not parameter.requires_grad for parameter in model.backbone.rgb_backbone.parameters())
    assert all(parameter.requires_grad for parameter in model.backbone.depth_backbone.parameters())
    assert all(parameter.requires_grad for parameter in model.backbone.fuse.parameters())
    assert all(parameter.requires_grad for parameter in model.bbox_head.parameters())

    features = model.backbone(torch.randn(1, 4, 128, 128))
    feature_loss = sum(feature.square().mean() for feature in features)
    head_loss = 1e-8 * sum(
        parameter.square().mean() for parameter in model.bbox_head.parameters()
    )
    (feature_loss + head_loss).backward()

    assert all(parameter.grad is None for parameter in model.backbone.rgb_backbone.parameters())
    assert any(parameter.grad is not None for parameter in model.backbone.depth_backbone.parameters())
    assert any(parameter.grad is not None for parameter in model.backbone.fuse.parameters())
    assert any(parameter.grad is not None for parameter in model.bbox_head.parameters())
