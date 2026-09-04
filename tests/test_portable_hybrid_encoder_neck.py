"""Contract tests for the portable JAX-equivalent hybrid feature stack."""

from __future__ import annotations

from pathlib import Path

import pytest
import torch
import torch.nn.functional as F

from yopo.models.necks.portable_hybrid_encoder import PortableHybridEncoderNeck
from yopo.utils.feature_contract import (
    feature_contract_fingerprint,
    load_feature_contract,
    validate_neck_feature_contract,
)


CONTRACT = Path(__file__).parents[1] / "contracts" / "rgbd_feature_stack_v1.json"


def _neck() -> PortableHybridEncoderNeck:
    return PortableHybridEncoderNeck(
        in_channels=(6, 10, 14),
        hidden_dim=8,
        num_heads=2,
        ffn_dim=16,
        num_aifi_layers=1,
        num_outs=4,
    )


def test_portable_neck_preserves_odd_shared_shapes_and_derives_p6() -> None:
    neck = _neck().eval()
    inputs = (
        torch.randn(1, 6, 9, 11),
        torch.randn(1, 10, 5, 6),
        torch.randn(1, 14, 3, 3),
    )

    shared = neck.forward_shared(inputs)
    outputs = neck(inputs)

    assert [tuple(value.shape[-2:]) for value in shared] == [(9, 11), (5, 6), (3, 3)]
    assert [tuple(value.shape[-2:]) for value in outputs] == [
        (9, 11),
        (5, 6),
        (3, 3),
        (2, 2),
    ]
    for shared_value, output in zip(shared, outputs[:3], strict=True):
        torch.testing.assert_close(shared_value, output, rtol=0, atol=0)


def test_bottom_up_pool_matches_explicit_same_padding_divide_four() -> None:
    source = torch.arange(15, dtype=torch.float32).reshape(1, 1, 3, 5)
    expected = F.avg_pool2d(F.pad(source, (0, 1, 0, 1)), 2, stride=2)

    actual = PortableHybridEncoderNeck.same_divide4_pool(source)

    torch.testing.assert_close(actual, expected, rtol=0, atol=0)


def test_portable_neck_rejects_invalid_channels() -> None:
    neck = _neck()
    with pytest.raises(ValueError, match="level 1 channels"):
        neck((torch.randn(1, 6, 8, 8), torch.randn(1, 9, 4, 4), torch.randn(1, 14, 2, 2)))


def test_yopo_reads_the_same_v1_contract_fingerprint() -> None:
    payload = load_feature_contract(CONTRACT)
    assert feature_contract_fingerprint(payload) == payload["fingerprint_sha256"]

    production_neck = PortableHybridEncoderNeck(
        in_channels=(384, 768, 1536),
        hidden_dim=256,
        num_heads=8,
        ffn_dim=1024,
        num_aifi_layers=1,
        num_outs=4,
    )
    validate_neck_feature_contract(production_neck, payload)
