"""Contracts for the envelope corner loss.

It exists because the KLD cannot supply an orientation gradient for a
near-circular shape while the metric it is judged by still charges 0.20 of IoU
for the measured 17 degree error.  So the properties that matter are: it is
angle-sensitive where the KLD is not, it tracks the metric rather than the
covariance, and it stays differentiable on the circle where these objects live.
"""

import math

import pytest
import torch

from yopo.models.losses.ellipse_envelope_corner_loss import (
    EllipseEnvelopeCornerLoss, ellipse_envelope_corners)
from yopo.models.losses.gaucho3d_loss import ellipse_kld_from_cholesky


def _sigma(a, b, angle):
    rotation = torch.tensor([[math.cos(angle), -math.sin(angle)],
                             [math.sin(angle), math.cos(angle)]])
    return rotation @ torch.diag(torch.tensor([a * a, b * b])) @ rotation.T


def _compact(mean, sigma):
    return torch.tensor([[mean[0], mean[1], sigma[0, 0], sigma[0, 1],
                          sigma[1, 1]]])


# -- geometry ------------------------------------------------------------


def test_envelope_of_a_circle_is_a_square():
    corners = ellipse_envelope_corners(torch.zeros(1, 2),
                                       (torch.eye(2) * 100.0).unsqueeze(0))
    extent = corners[0].abs()
    # 1% trace softening moves the axes by well under a percent.
    assert torch.allclose(extent, torch.full_like(extent, 10.0), rtol=0.01)


def test_envelope_recovers_the_semi_axes():
    sigma = _sigma(11.0, 5.0, 0.4)
    corners = ellipse_envelope_corners(torch.zeros(1, 2), sigma.unsqueeze(0))[0]
    # Half-diagonal length is sqrt(a^2 + b^2) regardless of orientation.
    assert corners.norm(dim=-1).max().item() == pytest.approx(
        math.hypot(11.0, 5.0), rel=0.01)


# -- the property the KLD lacks ------------------------------------------


def test_angle_sensitivity_where_the_kld_has_none():
    """At aspect 1.2 the metric loses 0.20 of IoU by 17 degrees; the KLD does not see it."""
    a, b = 11.0, 9.17                      # aspect 1.2, as measured on this data
    target_sigma = _sigma(a, b, 0.0)
    target = _compact((10.0, 10.0), target_sigma)
    mean = torch.tensor([[10.0, 10.0]])
    loss = EllipseEnvelopeCornerLoss(loss_weight=1.0)

    corner_values, kld_values = [], []
    target_cholesky = torch.linalg.cholesky(target_sigma).unsqueeze(0)
    for degrees in (0.0, 5.0, 10.0, 17.0, 30.0, 45.0):
        sigma = _sigma(a, b, math.radians(degrees))
        corner_values.append(float(loss(mean, sigma.unsqueeze(0), target,
                                        reduction_override="none")[0]))
        kld_values.append(float(ellipse_kld_from_cholesky(
            mean, torch.linalg.cholesky(sigma).unsqueeze(0), mean,
            target_cholesky)[0]))

    assert corner_values[0] == pytest.approx(0.0, abs=1e-9)
    assert corner_values == sorted(corner_values), corner_values
    # The corner term separates 17 degrees from 0 by two orders of magnitude.
    assert corner_values[3] > 50 * max(corner_values[0], 1e-6)
    # The KLD barely moves over the same range -- this is the whole point.
    assert max(kld_values) < 0.25, kld_values
    assert corner_values[3] / max(corner_values[1], 1e-9) > 5.0


def test_ninety_degrees_costs_little_for_a_near_circular_shape():
    """The metric only drops to 0.714 at 90 degrees for aspect 1.2, and so must this."""
    a, b = 11.0, 9.17
    target = _compact((10.0, 10.0), _sigma(a, b, 0.0))
    mean = torch.tensor([[10.0, 10.0]])
    loss = EllipseEnvelopeCornerLoss(loss_weight=1.0)
    at_ninety = float(loss(mean, _sigma(a, b, math.pi / 2).unsqueeze(0), target,
                           reduction_override="none")[0])
    at_forty_five = float(loss(mean, _sigma(a, b, math.pi / 4).unsqueeze(0),
                               target, reduction_override="none")[0])
    assert at_ninety < at_forty_five, (at_ninety, at_forty_five)


# -- differentiability where these objects actually live -----------------


@pytest.mark.parametrize("state,name", [
    ([0.0, 0.0, 0.0], "circle"),
    ([0.4, -0.4, 0.0], "axis-aligned elongated"),
    ([0.1, -0.1, 0.0], "aspect 1.2"),
    ([0.2, -0.2, 0.3], "oblique"),
])
def test_gradient_is_finite_and_bounded(state, name):
    """``eigh`` would give NaN on the circle; these fruit are near-circular."""
    raw = torch.tensor([state], requires_grad=True)
    major = 10.0 * torch.exp(raw[:, 0])
    minor = 10.0 * torch.exp(raw[:, 1])
    off = raw[:, 2] * 10.0
    sigma = torch.stack(
        (torch.stack((major * major, off), dim=-1),
         torch.stack((off, minor * minor), dim=-1)), dim=-2)
    target = _compact((10.0, 10.0), _sigma(11.0, 9.17, 0.0))
    EllipseEnvelopeCornerLoss(loss_weight=1.0)(
        torch.tensor([[10.0, 10.0]]), sigma, target).backward()
    assert torch.isfinite(raw.grad).all(), name
    assert float(raw.grad.norm()) < 100.0, (name, float(raw.grad.norm()))


def test_exact_match_is_zero_and_weights_apply():
    target_sigma = _sigma(11.0, 9.17, 0.3)
    target = _compact((10.0, 10.0), target_sigma).repeat(2, 1)
    mean = torch.tensor([[10.0, 10.0], [10.0, 10.0]])
    sigma = torch.stack((target_sigma, _sigma(11.0, 9.17, 0.3 + 0.5)))
    loss = EllipseEnvelopeCornerLoss(loss_weight=1.0)

    per_sample = loss(mean, sigma, target, reduction_override="none")
    assert per_sample[0].item() == pytest.approx(0.0, abs=1e-9)
    assert per_sample[1].item() > 0.01
    dropped = loss(mean, sigma, target, weight=torch.tensor([1.0, 0.0]),
                   avg_factor=2)
    assert dropped.item() == pytest.approx(0.0, abs=1e-9)


def test_scale_invariance():
    """A 2 cm and a 5 cm object must contribute comparably."""
    loss = EllipseEnvelopeCornerLoss(loss_weight=1.0)
    values = []
    for scale in (1.0, 3.0):
        target = _compact((0.0, 0.0), _sigma(11.0 * scale, 9.17 * scale, 0.0))
        sigma = _sigma(11.0 * scale, 9.17 * scale, math.radians(17.0))
        values.append(float(loss(torch.zeros(1, 2), sigma.unsqueeze(0), target,
                                 reduction_override="none")[0]))
    assert values[0] == pytest.approx(values[1], rel=1e-3)


def test_degenerate_annotation_is_excluded():
    target = torch.tensor([[0.0, 0.0, 0.0, 0.0, 0.0]])
    loss = EllipseEnvelopeCornerLoss(loss_weight=1.0)
    value = loss(torch.zeros(1, 2), (torch.eye(2) * 100.0).unsqueeze(0), target,
                 avg_factor=1)
    assert torch.isfinite(value) and value.item() == 0.0


def test_gradient_survives_degenerate_targets():
    """Every negative query carries an all-zero target, so this is the common case.

    ``torch.where(valid, f(x), 0)`` does not stop a NaN -- the backward
    multiplies NaN by zero and gets NaN.  A run whose loss value looked finite
    produced ``grad_norm: nan`` for its whole training, the AMP scaler skipped
    every step, and the model never updated: the weights were bit-identical
    across eight epochs.  Checking the value is not enough; this checks the
    gradient.
    """
    raw = torch.tensor([[0.2, -0.2, 0.1], [0.1, -0.1, 0.0]], requires_grad=True)
    major = 10.0 * torch.exp(raw[:, 0])
    minor = 10.0 * torch.exp(raw[:, 1])
    off = raw[:, 2] * 10.0
    sigma = torch.stack(
        (torch.stack((major * major, off), dim=-1),
         torch.stack((off, minor * minor), dim=-1)), dim=-2)
    degenerate = torch.tensor([[0.0, 0.0, 0.0, 0.0, 0.0],
                               [0.0, 0.0, 121.0, 10.0, 84.0]])
    EllipseEnvelopeCornerLoss(loss_weight=1.0)(torch.zeros(2, 2), sigma, degenerate, avg_factor=2).backward()
    assert torch.isfinite(raw.grad).all()

    raw2 = raw.detach().clone().requires_grad_(True)
    major = 10.0 * torch.exp(raw2[:, 0])
    minor = 10.0 * torch.exp(raw2[:, 1])
    off = raw2[:, 2] * 10.0
    sigma = torch.stack(
        (torch.stack((major * major, off), dim=-1),
         torch.stack((off, minor * minor), dim=-1)), dim=-2)
    degenerate = torch.zeros(2, 5)
    EllipseEnvelopeCornerLoss(loss_weight=1.0)(torch.zeros(2, 2), sigma, degenerate, avg_factor=2).backward()
    assert torch.isfinite(raw2.grad).all()
