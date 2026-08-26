"""Numerical and API contracts for compact Gaussian KFIoU loss."""

from __future__ import annotations

import pytest
import torch

from yopo.models.losses.gaussian_kfiou_loss import (
    GaussianKFIoULoss,
    gaussian_kfiou_similarity,
)
from yopo.registry import MODELS


def _compact(center: torch.Tensor, sigma: torch.Tensor) -> torch.Tensor:
    return torch.stack(
        (
            center[..., 0],
            center[..., 1],
            sigma[..., 0, 0],
            sigma[..., 0, 1],
            sigma[..., 1, 1],
        ),
        dim=-1,
    )


def test_known_isotropic_case_matches_normalized_closed_form():
    # For Sigma_p=aI and Sigma_t=bI, normalized similarity is
    # 3ab / (a^2 + ab + b^2).  With a=4, b=1 the loss is 3/7.
    predicted = torch.tensor(
        [[0.0, 0.0, 4.0, 0.0, 4.0]], dtype=torch.float64)
    target = torch.tensor(
        [[0.0, 0.0, 1.0, 0.0, 1.0]], dtype=torch.float64)

    actual = GaussianKFIoULoss(reduction="none")(predicted, target)
    expected = torch.tensor([3.0 / 7.0], dtype=torch.float64)

    torch.testing.assert_close(actual, expected, rtol=1e-12, atol=1e-12)


def test_identical_covariances_have_numerically_zero_loss():
    compact = torch.tensor(
        [[3.0, -2.0, 3.0, 1.0, 2.0]], dtype=torch.float64)

    loss = GaussianKFIoULoss(reduction="none")(compact, compact.clone())

    torch.testing.assert_close(loss, torch.zeros_like(loss), atol=1e-15, rtol=0)


def test_similarity_is_center_rotation_and_common_scale_invariant():
    angle = torch.tensor(0.73, dtype=torch.float64)
    cosine, sine = torch.cos(angle), torch.sin(angle)
    rotation = torch.stack(
        (torch.stack((cosine, -sine)), torch.stack((sine, cosine))))
    predicted_sigma = torch.tensor(
        [[3.0, 0.4], [0.4, 1.2]], dtype=torch.float64)
    target_sigma = torch.tensor(
        [[1.1, -0.2], [-0.2, 2.4]], dtype=torch.float64)
    predicted = _compact(torch.zeros(2, dtype=torch.float64), predicted_sigma)
    target = _compact(torch.ones(2, dtype=torch.float64), target_sigma)
    base, valid = gaussian_kfiou_similarity(predicted, target)

    transformed_predicted = _compact(
        torch.tensor([100.0, -50.0], dtype=torch.float64),
        7.0 * rotation @ predicted_sigma @ rotation.T,
    )
    transformed_target = _compact(
        torch.tensor([-20.0, 200.0], dtype=torch.float64),
        7.0 * rotation @ target_sigma @ rotation.T,
    )
    transformed, transformed_valid = gaussian_kfiou_similarity(
        transformed_predicted, transformed_target)

    assert bool(valid.item()) and bool(transformed_valid.item())
    torch.testing.assert_close(transformed, base, rtol=1e-12, atol=1e-12)


def test_covariance_gradient_is_finite_and_center_gradient_is_zero():
    predicted = torch.tensor(
        [[9.0, -4.0, 3.0, 0.3, 1.1]],
        dtype=torch.float64,
        requires_grad=True,
    )
    target = torch.tensor(
        [[0.0, 0.0, 1.2, -0.1, 2.2]], dtype=torch.float64)

    loss = GaussianKFIoULoss()(predicted, target)
    loss.backward()

    assert torch.isfinite(loss)
    assert predicted.grad is not None
    assert torch.isfinite(predicted.grad).all()
    torch.testing.assert_close(
        predicted.grad[0, :2], torch.zeros(2, dtype=torch.float64))
    assert predicted.grad[0, 2:].norm() > 0


def test_bfloat16_inputs_are_computed_in_float32_and_backpropagate():
    predicted = torch.tensor(
        [[0.0, 0.0, 3.0, 0.25, 1.0]],
        dtype=torch.bfloat16,
        requires_grad=True,
    )
    target = torch.tensor(
        [[0.0, 0.0, 1.0, 0.0, 2.0]], dtype=torch.bfloat16)

    with torch.autocast(device_type="cpu", dtype=torch.bfloat16):
        loss = GaussianKFIoULoss()(predicted, target)
    loss.backward()

    assert loss.dtype == torch.float32
    assert torch.isfinite(loss)
    assert predicted.grad is not None
    assert torch.isfinite(predicted.grad).all()


def test_invalid_positive_pair_fails_and_zero_weight_pair_is_ignored():
    invalid = torch.tensor(
        [[0.0, 0.0, 1.0, 2.0, 1.0]], dtype=torch.float64)
    target = torch.tensor(
        [[0.0, 0.0, 1.0, 0.0, 1.0]], dtype=torch.float64)
    module = GaussianKFIoULoss(fail_on_invalid=True)

    with pytest.raises(RuntimeError, match="1/1 invalid positive pairs"):
        module(invalid, target, weight=torch.ones(1, dtype=torch.float64))

    ignored = module(
        invalid,
        target,
        weight=torch.zeros(1, dtype=torch.float64),
    )
    torch.testing.assert_close(ignored, torch.zeros_like(ignored))


def test_invalid_pair_can_be_safely_masked_when_failure_is_disabled():
    predicted = torch.tensor(
        [
            [0.0, 0.0, 4.0, 0.0, 4.0],
            [0.0, 0.0, 1.0, 2.0, 1.0],
        ],
        dtype=torch.float64,
        requires_grad=True,
    )
    target = torch.tensor(
        [[0.0, 0.0, 1.0, 0.0, 1.0]] * 2, dtype=torch.float64)

    loss = GaussianKFIoULoss(fail_on_invalid=False)(
        predicted, target, avg_factor=1)
    loss.backward()

    torch.testing.assert_close(
        loss.detach(), torch.tensor(3.0 / 7.0, dtype=torch.float64))
    assert torch.isfinite(predicted.grad).all()
    torch.testing.assert_close(
        predicted.grad[1], torch.zeros(5, dtype=torch.float64))


def test_weight_reduction_avg_factor_and_override_follow_loss_api():
    predicted = torch.tensor(
        [
            [0.0, 0.0, 4.0, 0.0, 4.0],
            [0.0, 0.0, 1.0, 0.0, 1.0],
        ],
        dtype=torch.float64,
    )
    target = torch.tensor(
        [[0.0, 0.0, 1.0, 0.0, 1.0]] * 2, dtype=torch.float64)
    weight = torch.tensor([2.0, 3.0], dtype=torch.float64)
    module = GaussianKFIoULoss(loss_weight=2.5)

    actual = module(predicted, target, weight=weight, avg_factor=4.0)
    expected = torch.tensor(
        2.5 * (2.0 * 3.0 / 7.0) / 4.0, dtype=torch.float64)
    torch.testing.assert_close(actual, expected)

    per_pair = module(
        predicted,
        target,
        weight=weight,
        reduction_override="none",
    )
    torch.testing.assert_close(
        per_pair,
        torch.tensor([2.5 * 2.0 * 3.0 / 7.0, 0.0], dtype=torch.float64),
    )

    with pytest.raises(ValueError, match="avg_factor"):
        GaussianKFIoULoss(reduction="sum")(
            predicted, target, avg_factor=2.0)


def test_zero_weight_pairs_preserve_unreduced_shape_and_mean_denominator():
    predicted = torch.tensor(
        [
            [0.0, 0.0, 1.0, 2.0, 1.0],  # invalid but must not be evaluated
            [0.0, 0.0, 4.0, 0.0, 4.0],
        ],
        dtype=torch.float64,
    )
    target = torch.tensor(
        [[0.0, 0.0, 1.0, 0.0, 1.0]] * 2, dtype=torch.float64)
    weight = torch.tensor([0.0, 1.0], dtype=torch.float64)

    per_pair = GaussianKFIoULoss(reduction="none")(
        predicted, target, weight=weight)
    mean = GaussianKFIoULoss(reduction="mean")(
        predicted, target, weight=weight)

    torch.testing.assert_close(
        per_pair, torch.tensor([0.0, 3.0 / 7.0], dtype=torch.float64))
    torch.testing.assert_close(mean, per_pair.mean())


def test_invalid_shapes_values_weights_and_reductions_are_explicit():
    module = GaussianKFIoULoss()
    compact = torch.tensor(
        [[0.0, 0.0, 1.0, 0.0, 1.0]], dtype=torch.float32)

    with pytest.raises(ValueError, match=r"equal \(\.\.\., 5\)"):
        module(compact, torch.zeros(1, 4))
    with pytest.raises(RuntimeError, match="invalid positive pairs"):
        module(compact.clone().fill_(float("nan")), compact)
    with pytest.raises(ValueError, match="weight must have shape"):
        module(compact, compact, weight=torch.ones(2))
    with pytest.raises(ValueError, match="finite values"):
        module(compact, compact, weight=torch.tensor([float("nan")]))
    with pytest.raises(ValueError, match="non-negative"):
        module(compact, compact, weight=torch.tensor([-1.0]))
    with pytest.raises(ValueError, match="unsupported reduction"):
        module(compact, compact, reduction_override="unsupported")


def test_module_is_registry_buildable_after_explicit_import():
    module = MODELS.build(
        dict(
            type="GaussianKFIoULoss",
            loss_weight=0.25,
            reduction="sum",
            fail_on_invalid=False,
            eps=1e-6,
        ))

    assert isinstance(module, GaussianKFIoULoss)
    assert module.loss_weight == 0.25
    assert module.reduction == "sum"
    assert module.fail_on_invalid is False
    assert module.eps == 1e-6
