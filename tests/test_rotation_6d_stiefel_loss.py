"""Focused contracts for raw-6D Stiefel frame regularization."""

import pytest
import torch

from yopo.models.losses.rotation_6d_stiefel_loss import (
    Rotation6DStiefelLoss,
    rotation_6d_stiefel_loss,
)
from yopo.registry import MODELS


def test_orthonormal_columns_have_exactly_zero_loss():
    prediction = torch.tensor(
        [
            [1.0, 0.0, 0.0, 0.0, 1.0, 0.0],
            [0.0, 0.0, 1.0, 0.0, 1.0, 0.0],
        ]
    )

    loss = rotation_6d_stiefel_loss(prediction)

    torch.testing.assert_close(loss, torch.zeros_like(loss), atol=0, rtol=0)


def test_scaled_and_parallel_columns_are_penalized():
    # beta=1 gives 2.5 / 4 for the first Gram residual and
    # (0.5 + 0.5) / 4 for the second.
    prediction = torch.tensor(
        [
            [2.0, 0.0, 0.0, 0.0, 1.0, 0.0],
            [1.0, 0.0, 0.0, 1.0, 0.0, 0.0],
        ],
        dtype=torch.float64,
    )

    loss = rotation_6d_stiefel_loss(prediction, beta=1.0)

    expected = torch.tensor([0.625, 0.25], dtype=torch.float32)
    torch.testing.assert_close(loss, expected)
    assert torch.all(loss > 0)


@pytest.mark.parametrize(
    'prediction',
    [
        [[0.0, 0.0, 0.0, 0.0, 0.0, 0.0]],
        [[1.0, 0.0, 0.0, 1.0, 0.0, 0.0]],
    ],
)
def test_degenerate_frames_have_finite_loss_and_gradient(prediction):
    prediction = torch.tensor(prediction, requires_grad=True)

    loss = Rotation6DStiefelLoss()(prediction)
    loss.backward()

    assert torch.isfinite(loss)
    assert prediction.grad is not None
    assert torch.isfinite(prediction.grad).all()


def test_weight_reduction_avg_factor_and_override_follow_loss_api():
    prediction = torch.tensor(
        [
            [2.0, 0.0, 0.0, 0.0, 1.0, 0.0],
            [1.0, 0.0, 0.0, 1.0, 0.0, 0.0],
        ]
    )
    component_weight = torch.tensor(
        [
            [2.0, 2.0, 2.0, 2.0, 2.0, 2.0],
            [0.0, 0.0, 0.0, 0.0, 0.0, 0.0],
        ]
    )
    base = rotation_6d_stiefel_loss(prediction)
    module = Rotation6DStiefelLoss(reduction='mean', loss_weight=3.0)

    actual = module(
        prediction,
        target=torch.empty(0),
        weight=component_weight,
        avg_factor=4.0,
    )
    expected = 3.0 * (base * torch.tensor([2.0, 0.0])).sum() / 4.0
    torch.testing.assert_close(actual, expected)

    overridden = module(
        prediction,
        weight=torch.tensor([[1.0], [2.0]]),
        reduction_override='none',
    )
    torch.testing.assert_close(
        overridden, 3.0 * base * torch.tensor([1.0, 2.0]))


def test_bfloat16_autocast_computes_geometry_in_float32():
    prediction = torch.tensor(
        [[0.5, 0.5, 0.5, 0.5, 0.5, 0.5]],
        dtype=torch.bfloat16,
        requires_grad=True,
    )

    with torch.autocast(device_type='cpu', dtype=torch.bfloat16):
        loss = Rotation6DStiefelLoss()(prediction)
    loss.backward()

    assert loss.dtype == torch.float32
    assert torch.isfinite(loss)
    assert prediction.grad is not None
    assert torch.isfinite(prediction.grad).all()


def test_empty_prediction_has_differentiable_zero_mean():
    prediction = torch.empty((0, 6), requires_grad=True)

    loss = Rotation6DStiefelLoss()(prediction)
    loss.backward()

    torch.testing.assert_close(loss, torch.tensor(0.0))
    assert prediction.grad is not None


def test_module_is_registry_buildable_after_import():
    module = MODELS.build(
        dict(
            type='Rotation6DStiefelLoss',
            beta=0.5,
            reduction='sum',
            loss_weight=0.04,
        )
    )

    assert isinstance(module, Rotation6DStiefelLoss)
    assert module.beta == 0.5
    assert module.reduction == 'sum'
    assert module.loss_weight == 0.04


@pytest.mark.parametrize('beta', [0.0, -1.0, float('nan'), float('inf')])
def test_invalid_beta_is_rejected(beta):
    with pytest.raises(ValueError, match='beta'):
        Rotation6DStiefelLoss(beta=beta)


def test_shape_weight_and_reduction_errors_are_explicit():
    module = Rotation6DStiefelLoss()
    with pytest.raises(ValueError, match=r'\(\.\.\., 6\)'):
        module(torch.zeros((2, 3)))
    with pytest.raises(ValueError, match='weight must be'):
        module(torch.zeros((2, 6)), weight=torch.ones(3))
    with pytest.raises(ValueError, match='reduction_override'):
        module(
            torch.zeros((2, 6)), reduction_override='unsupported')


@pytest.mark.parametrize('loss_weight', [-1.0, float('nan'), float('inf')])
def test_invalid_loss_weight_is_rejected(loss_weight):
    with pytest.raises(ValueError, match='loss_weight'):
        Rotation6DStiefelLoss(loss_weight=loss_weight)
