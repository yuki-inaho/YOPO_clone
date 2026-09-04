"""Strict JAX RGB-D feature checkpoint to YOPO mapping tests."""

from __future__ import annotations

import numpy as np
import pytest
import torch

from yopo.utils.jax_feature_transfer import convert_jax_feature_arrays


def _source() -> dict[str, np.ndarray]:
    prefix = 'ema::'
    return {
        prefix + 'backbone/fusion_beta': np.asarray([0.1, 0.2, 0.3], np.float32),
        prefix + 'backbone/fusion_params/adapters/0/kernel': np.arange(
            24, dtype=np.float32
        ).reshape(1, 1, 3, 8),
        prefix + 'encoder/projections/0/kernel': np.arange(48, dtype=np.float32).reshape(
            1, 1, 6, 8
        ),
        prefix + 'encoder/aifi/0/attention/query/kernel': np.eye(8, dtype=np.float32),
        prefix + 'encoder/aifi/0/attention/key/kernel': np.eye(8, dtype=np.float32) * 2,
        prefix + 'encoder/aifi/0/attention/value/kernel': np.eye(8, dtype=np.float32) * 3,
        prefix + 'encoder/aifi/0/attention/query/bias': np.ones(8, np.float32),
        prefix + 'encoder/aifi/0/attention/key/bias': np.ones(8, np.float32) * 2,
        prefix + 'encoder/aifi/0/attention/value/bias': np.ones(8, np.float32) * 3,
    }


def _target() -> dict[str, torch.Tensor]:
    return {
        'backbone.depth_beta': torch.empty(3),
        'backbone.depth_adapters.0.weight': torch.empty(8, 3, 1, 1),
        'neck.projections.0.weight': torch.empty(8, 6, 1, 1),
        'neck.aifi.0.attention.in_proj_weight': torch.empty(24, 8),
        'neck.aifi.0.attention.in_proj_bias': torch.empty(24),
        'neck.derived_p6.weight': torch.empty(8, 8, 3, 3),
    }


def test_converter_transposes_convs_and_packs_qkv() -> None:
    converted, report = convert_jax_feature_arrays(_source(), _target(), weights='ema')

    assert report.ok
    assert report.missing == ()
    assert report.excluded_target == ('neck.derived_p6.weight',)
    np.testing.assert_array_equal(
        converted['neck.projections.0.weight'].numpy(),
        _source()['ema::encoder/projections/0/kernel'].transpose(3, 2, 0, 1),
    )
    assert torch.equal(
        converted['neck.aifi.0.attention.in_proj_weight'][:8], torch.eye(8)
    )
    assert torch.equal(
        converted['neck.aifi.0.attention.in_proj_weight'][16:], torch.eye(8) * 3
    )
    assert torch.equal(
        converted['neck.aifi.0.attention.in_proj_bias'][8:16], torch.full((8,), 2.0)
    )


def test_converter_fails_closed_on_shape_mismatch() -> None:
    target = _target()
    target['neck.projections.0.weight'] = torch.empty(7, 6, 1, 1)

    with pytest.raises(ValueError, match='shape'):
        convert_jax_feature_arrays(_source(), target, weights='ema', strict=True)


def test_converter_rejects_source_kernel_that_is_not_canonical_hwio() -> None:
    source = _source()
    key = 'ema::encoder/projections/0/kernel'
    source[key] = source[key].transpose(3, 2, 0, 1)

    with pytest.raises(ValueError, match='shape'):
        convert_jax_feature_arrays(source, _target(), weights='ema', strict=True)


def test_converter_fails_closed_on_missing_shared_key() -> None:
    source = _source()
    source.pop('ema::backbone/fusion_beta')

    with pytest.raises(ValueError, match='missing'):
        convert_jax_feature_arrays(source, _target(), weights='ema', strict=True)
