"""Contracts for the eccentricity-weighted angle term.

Two failed attempts define what this has to get right.  The KLD supplies no
rotation gradient for a near-circular shape, so tripling its weight moved the
matched IoU by 0.0006.  A corner-set Chamfer term made the angle *worse*
(p90 67 -> 75 degrees) because comparing corners as a set makes a quarter turn
nearly free.  So: sensitive at small angles, still sensitive at 90, and silent
where the annotation itself carries no orientation.
"""

import math

import pytest
import torch

from yopo.models.losses.ellipse_angle_loss import (EllipseAngleGeodesicLoss,
                                                   double_angle_direction,
                                                   shape_eccentricity)


def _sigma(a, b, angle):
    a, b = float(a), float(b)
    rotation = torch.tensor([[math.cos(angle), -math.sin(angle)],
                             [math.sin(angle), math.cos(angle)]])
    return rotation @ torch.diag(torch.tensor([a * a, b * b])) @ rotation.T


def _compact(sigma):
    return torch.tensor([[0.0, 0.0, sigma[0, 0], sigma[0, 1], sigma[1, 1]]])


# -- the pieces ----------------------------------------------------------


@pytest.mark.parametrize("ratio,expected", [(1.0, 0.0), (1.2, 0.3051),
                                            (1.5, 0.5556), (4.0, 0.9375)])
def test_eccentricity_matches_one_minus_axis_ratio_squared(ratio, expected):
    sigma = _sigma(10.0 * ratio, 10.0, 0.3).unsqueeze(0)
    assert float(shape_eccentricity(sigma)[0]) == pytest.approx(expected,
                                                                abs=1e-3)


def test_direction_is_pi_periodic():
    """A half turn is the same ellipse and must give the same direction."""
    first, _ = double_angle_direction(_sigma(11.0, 5.0, 0.3).unsqueeze(0))
    second, _ = double_angle_direction(
        _sigma(11.0, 5.0, 0.3 + math.pi).unsqueeze(0))
    assert torch.allclose(first, second, atol=1e-5)


def test_direction_tracks_the_double_angle():
    for degrees in (0.0, 20.0, 55.0):
        angle = math.radians(degrees)
        direction, _ = double_angle_direction(
            _sigma(12.0, 4.0, angle).unsqueeze(0))
        assert float(direction[0, 0]) == pytest.approx(math.cos(2 * angle),
                                                       abs=1e-4)
        assert float(direction[0, 1]) == pytest.approx(math.sin(2 * angle),
                                                       abs=1e-4)


# -- the properties the two failed attempts lacked -----------------------


def test_grows_linearly_and_does_not_saturate_at_ninety_degrees():
    """``1 - cos`` has zero gradient at 90 degrees; the observed tail is there."""
    target = _sigma(11.0, 9.17, 0.0)
    loss = EllipseAngleGeodesicLoss(loss_weight=1.0)
    values = [
        float(loss(_sigma(11.0, 9.17, math.radians(d)).unsqueeze(0),
                   _compact(target), reduction_override="none")[0])
        for d in (0, 17, 45, 90)
    ]
    assert values[0] == pytest.approx(0.0, abs=1e-6)
    assert values == sorted(values)
    # Linear in the angle: 90 degrees costs about five times what 17 does.
    assert values[3] / values[1] == pytest.approx(90 / 17, rel=0.05)


def test_circular_target_contributes_nothing():
    """A circle has no orientation; charging for it charges for annotation noise."""
    loss = EllipseAngleGeodesicLoss(loss_weight=1.0)
    value = loss(_sigma(11.0, 9.17, math.radians(45)).unsqueeze(0),
                 _compact(_sigma(10.0, 10.0, 0.0)), reduction_override="none")
    assert float(value[0]) == pytest.approx(0.0, abs=1e-6)


def test_elongated_target_is_charged_in_full():
    loss = EllipseAngleGeodesicLoss(loss_weight=1.0)
    target = _sigma(20.0, 5.0, 0.0)
    quarter = float(loss(_sigma(20.0, 5.0, math.pi / 2).unsqueeze(0),
                         _compact(target), reduction_override="none")[0])
    # eccentricity 0.9375 times a full quarter-turn error of 1.0.
    assert quarter == pytest.approx(0.9375, abs=1e-3)


def test_floor_restores_supervision_on_circles():
    loss = EllipseAngleGeodesicLoss(loss_weight=1.0, eccentricity_floor=0.5)
    value = float(loss(_sigma(11.0, 9.17, math.radians(90)).unsqueeze(0),
                       _compact(_sigma(10.0, 10.0, 0.0)),
                       reduction_override="none")[0])
    assert value == pytest.approx(0.5, abs=1e-3)


# -- differentiability where these objects live --------------------------


@pytest.mark.parametrize("state,name", [
    ([0.0, 0.0, 0.0], "circular prediction"),
    ([1e-5, -1e-5, 0.0], "near-circular prediction"),
    ([0.2, -0.2, 0.3], "oblique"),
    ([-0.1, 0.1, 0.0], "quarter-turn off"),
])
def test_gradient_is_finite(state, name):
    raw = torch.tensor([state], requires_grad=True)
    major = 10.0 * torch.exp(raw[:, 0])
    minor = 10.0 * torch.exp(raw[:, 1])
    off = raw[:, 2] * 10.0
    sigma = torch.stack(
        (torch.stack((major * major, off), dim=-1),
         torch.stack((off, minor * minor), dim=-1)), dim=-2)
    EllipseAngleGeodesicLoss(loss_weight=1.0)(
        sigma, _compact(_sigma(11.0, 9.17, 0.4))).backward()
    assert torch.isfinite(raw.grad).all(), name


def test_weights_and_degenerate_targets():
    target = _compact(_sigma(11.0, 9.17, 0.0)).repeat(2, 1)
    sigma = torch.stack((_sigma(11.0, 9.17, 0.0),
                         _sigma(11.0, 9.17, math.radians(60))))
    loss = EllipseAngleGeodesicLoss(loss_weight=1.0)
    per_sample = loss(sigma, target, reduction_override="none")
    assert per_sample[0].item() == pytest.approx(0.0, abs=1e-6)
    assert per_sample[1].item() > 0.05
    dropped = loss(sigma, target, weight=torch.tensor([1.0, 0.0]),
                   avg_factor=2)
    assert dropped.item() == pytest.approx(0.0, abs=1e-9)

    degenerate = torch.tensor([[0.0, 0.0, 0.0, 0.0, 0.0]])
    value = loss(_sigma(11.0, 9.17, 0.3).unsqueeze(0), degenerate, avg_factor=1)
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
    EllipseAngleGeodesicLoss(loss_weight=1.0)(sigma, degenerate, avg_factor=2).backward()
    assert torch.isfinite(raw.grad).all()

    raw2 = raw.detach().clone().requires_grad_(True)
    major = 10.0 * torch.exp(raw2[:, 0])
    minor = 10.0 * torch.exp(raw2[:, 1])
    off = raw2[:, 2] * 10.0
    sigma = torch.stack(
        (torch.stack((major * major, off), dim=-1),
         torch.stack((off, minor * minor), dim=-1)), dim=-2)
    degenerate = torch.zeros(2, 5)
    EllipseAngleGeodesicLoss(loss_weight=1.0)(sigma, degenerate, avg_factor=2).backward()
    assert torch.isfinite(raw2.grad).all()
