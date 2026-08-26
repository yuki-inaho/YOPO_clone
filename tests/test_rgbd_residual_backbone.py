import pytest
import torch
import torch.nn as nn

from yopo.models.backbones.dual_rgbd import RGBDResidualBackbone
from yopo.registry import MODELS


@MODELS.register_module(force=True)
class _TinyModalityBackbone(nn.Module):
    def __init__(self, in_channels, channels=(8, 16, 32), return_idx=(0, 1, 2)):
        super().__init__()
        self.return_idx = list(return_idx)
        self._out_channels = list(channels)
        layers = []
        current = in_channels
        for output in channels:
            layers.append(nn.Conv2d(current, output, 3, stride=2, padding=1))
            current = output
        self.layers = nn.ModuleList(layers)

    def forward(self, x):
        outputs = []
        for layer in self.layers:
            x = layer(x)
            outputs.append(x)
        return tuple(outputs[index] for index in self.return_idx)


def _make_model(beta_init=0.0):
    return RGBDResidualBackbone(
        rgb_backbone=dict(type='_TinyModalityBackbone', in_channels=3),
        depth_backbone=dict(type='_TinyModalityBackbone', in_channels=1),
        beta_init=beta_init,
    )


def test_zero_depth_residual_preserves_rgb_features_exactly():
    torch.manual_seed(7)
    model = _make_model(beta_init=0.0).eval()
    inputs = torch.randn(2, 4, 33, 47)

    expected = model.rgb_backbone(inputs[:, :3])
    actual = model(inputs)

    assert [feature.shape[1] for feature in actual] == [8, 16, 32]
    for rgb, fused in zip(expected, actual):
        torch.testing.assert_close(fused, rgb, rtol=0, atol=0)


def test_depth_gate_receives_gradient_at_identity_start():
    model = _make_model(beta_init=0.0).train()
    outputs = model(torch.randn(2, 4, 32, 48))
    sum(feature.mean() for feature in outputs).backward()

    assert model.depth_beta.grad is not None
    assert torch.isfinite(model.depth_beta.grad).all()
    assert model.depth_beta.grad.abs().sum() > 0


def test_residual_backbone_exposes_same_fused_and_raw_depth_features():
    torch.manual_seed(13)
    model = _make_model(beta_init=0.25).eval()
    inputs = torch.randn(2, 4, 32, 48)

    expected_fused = model(inputs)
    fused, raw_depth = model.forward_with_depth_features(inputs)
    expected_depth = model.depth_backbone(inputs[:, 3:4])

    assert len(fused) == len(raw_depth) == 3
    for expected, actual in zip(expected_fused, fused):
        torch.testing.assert_close(actual, expected, rtol=0, atol=0)
    for expected, actual in zip(expected_depth, raw_depth):
        torch.testing.assert_close(actual, expected, rtol=0, atol=0)


def test_residual_backbone_rejects_non_rgbd_input():
    model = _make_model()
    with pytest.raises(ValueError, match='4 channels'):
        model(torch.randn(1, 3, 16, 16))
