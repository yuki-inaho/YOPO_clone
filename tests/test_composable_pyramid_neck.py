"""Focused contracts for the configurable RGB-D feature-pyramid neck."""

import pytest
import torch

from yopo.models.necks.channel_mapper import ChannelMapper


def _inputs() -> tuple[torch.Tensor, ...]:
    return (
        torch.randn(2, 8, 16, 23),
        torch.randn(2, 8, 8, 12),
        torch.randn(2, 8, 4, 6),
    )


def _mapper_config() -> dict:
    return dict(
        type="ChannelMapper",
        in_channels=[8, 8, 8],
        out_channels=8,
        kernel_size=1,
        num_outs=4,
        norm_cfg=None,
        act_cfg=None,
    )


def test_composable_neck_without_refiner_matches_mapper() -> None:
    from yopo.models.necks.composable_pyramid_neck import ComposablePyramidNeck

    direct = ChannelMapper(**{k: v for k, v in _mapper_config().items() if k != "type"})
    composed = ComposablePyramidNeck(mapper=_mapper_config(), refiner=None)
    composed.mapper.load_state_dict(direct.state_dict())
    inputs = _inputs()

    expected = direct(inputs)
    actual = composed(inputs)

    assert len(actual) == 4
    for expected_level, actual_level in zip(expected, actual):
        torch.testing.assert_close(actual_level, expected_level)


def test_zero_initialized_refiner_is_exact_identity_on_odd_shapes() -> None:
    from yopo.models.necks.composable_pyramid_neck import ResidualPyramidRefiner

    features = (
        torch.randn(2, 8, 16, 23),
        torch.randn(2, 8, 8, 12),
        torch.randn(2, 8, 4, 6),
        torch.randn(2, 8, 2, 3),
    )
    refiner = ResidualPyramidRefiner(
        channels=8,
        num_levels=4,
        beta_init=0.0,
    )

    outputs = refiner(features)

    assert [tuple(x.shape) for x in outputs] == [tuple(x.shape) for x in features]
    for source, output in zip(features, outputs):
        assert torch.equal(source, output)


def test_refiner_rejects_invalid_feature_contract() -> None:
    from yopo.models.necks.composable_pyramid_neck import ResidualPyramidRefiner

    refiner = ResidualPyramidRefiner(channels=8, num_levels=4)
    wrong_channels = list(_inputs()) + [torch.randn(2, 7, 2, 3)]

    with pytest.raises(ValueError, match="channels"):
        refiner(tuple(wrong_channels))
