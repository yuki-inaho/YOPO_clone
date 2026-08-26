from __future__ import annotations

import math

import pytest
import torch
import torch.nn.functional as F

from yopo.models.losses.matchability_aware_loss import (
    MatchabilityAwareLoss,
    matchability_aware_loss,
)
from yopo.registry import MODELS


def _manual_mal(
        pred: torch.Tensor,
        labels: torch.Tensor,
        quality: torch.Tensor,
        gamma: float) -> torch.Tensor:
    """Small, explicit reference implementation for test expectations."""
    target = torch.zeros_like(pred)
    positive = torch.zeros_like(pred, dtype=torch.bool)
    rows = torch.arange(pred.shape[0])
    foreground = (labels >= 0) & (labels < pred.shape[1])
    rows = rows[foreground]
    columns = labels[foreground]
    positive[rows, columns] = True
    target[rows, columns] = quality[foreground].pow(gamma)
    negative_weight = pred.sigmoid().detach().pow(gamma)
    modulation = torch.where(
        positive, torch.ones_like(pred), negative_weight)
    return (
        F.binary_cross_entropy_with_logits(pred, target, reduction='none')
        * modulation).sum(dim=-1)


def test_matches_explicit_multiclass_formula():
    gamma = 1.5
    pred = torch.tensor([[0.4, -0.3], [-0.2, 0.7]], dtype=torch.float64)
    labels = torch.tensor([0, 2])
    quality = torch.tensor([0.5, 0.0], dtype=torch.float64)

    actual = matchability_aware_loss(
        pred, (labels, quality), gamma=gamma, reduction='none')
    expected = _manual_mal(pred, labels, quality, gamma)

    torch.testing.assert_close(actual, expected, rtol=1e-12, atol=1e-12)


def test_zero_quality_matched_positive_is_not_a_negative():
    pred = torch.zeros((2, 1), dtype=torch.float64, requires_grad=True)
    labels = torch.tensor([0, 1])
    quality = torch.zeros(2, dtype=torch.float64)

    loss = matchability_aware_loss(
        pred, (labels, quality), gamma=1.5, reduction='none')

    expected_positive = torch.tensor(math.log(2.0), dtype=torch.float64)
    expected_negative = expected_positive * 0.5**1.5
    torch.testing.assert_close(loss[0], expected_positive)
    torch.testing.assert_close(loss[1], expected_negative)
    assert loss[0] > loss[1]


def test_quality_and_negative_modulation_are_detached():
    gamma = 1.5
    pred = torch.tensor([[0.2], [0.4]], dtype=torch.float64,
                        requires_grad=True)
    quality = torch.tensor([0.6, 0.0], dtype=torch.float64,
                           requires_grad=True)
    labels = torch.tensor([0, 1])

    loss = matchability_aware_loss(
        pred, (labels, quality), gamma=gamma, reduction='sum')
    loss.backward()

    positive_target = quality.detach()[0].pow(gamma)
    expected_positive_grad = (
        pred.detach()[0, 0].sigmoid() - positive_target)
    detached_negative_weight = pred.detach()[1, 0].sigmoid().pow(gamma)
    expected_negative_grad = (
        detached_negative_weight * pred.detach()[1, 0].sigmoid())
    expected = torch.stack((expected_positive_grad, expected_negative_grad))
    torch.testing.assert_close(pred.grad[:, 0], expected)
    assert quality.grad is None


def test_weight_reduction_avg_factor_and_override_follow_loss_api():
    pred = torch.tensor([[0.1], [-0.4], [0.7]], dtype=torch.float64)
    labels = torch.tensor([0, 1, 0])
    quality = torch.tensor([0.8, 0.0, 0.3], dtype=torch.float64)
    sample_weight = torch.tensor([1.0, 0.0, 2.0], dtype=torch.float64)
    base = matchability_aware_loss(
        pred, (labels, quality), reduction='none')

    module = MatchabilityAwareLoss(reduction='mean', loss_weight=2.5)
    actual = module(
        pred, (labels, quality), sample_weight, avg_factor=4.0)
    expected = 2.5 * (base * sample_weight).sum() / 4.0
    torch.testing.assert_close(actual, expected)

    overridden = module(
        pred, (labels, quality), sample_weight,
        reduction_override='none')
    torch.testing.assert_close(overridden, 2.5 * base * sample_weight)


def test_bfloat16_can_compute_bce_in_float32_and_backpropagate():
    pred = torch.tensor([[0.2], [-0.7]], dtype=torch.bfloat16,
                        requires_grad=True)
    labels = torch.tensor([0, 1])
    quality = torch.tensor([0.75, 0.0], dtype=torch.bfloat16)

    loss = MatchabilityAwareLoss(force_float32=True)(
        pred, (labels, quality))
    loss.backward()

    assert loss.dtype == torch.float32
    assert torch.isfinite(loss)
    assert pred.grad is not None
    assert torch.isfinite(pred.grad).all()


def test_module_is_registry_buildable_after_explicit_import():
    module = MODELS.build(
        dict(
            type='MatchabilityAwareLoss',
            gamma=1.3,
            reduction='sum',
            loss_weight=0.5,
            force_float32=False,
        ))

    assert isinstance(module, MatchabilityAwareLoss)
    assert module.use_sigmoid is True
    assert module.gamma == 1.3
    assert module.reduction == 'sum'
    assert module.loss_weight == 0.5
    assert module.force_float32 is False


@pytest.mark.parametrize('gamma', [0.0, -1.0, float('nan'), float('inf')])
def test_invalid_gamma_is_rejected(gamma):
    with pytest.raises(ValueError, match='gamma'):
        MatchabilityAwareLoss(gamma=gamma)


def test_target_shape_and_reduction_errors_are_explicit():
    module = MatchabilityAwareLoss()
    pred = torch.zeros((2, 1))
    labels = torch.zeros(2, dtype=torch.long)
    quality = torch.ones(1)

    with pytest.raises(ValueError, match='quality shape'):
        module(pred, (labels, quality))
    with pytest.raises(ValueError, match='reduction_override'):
        module(
            pred, (labels, torch.ones(2)),
            reduction_override='unsupported')
