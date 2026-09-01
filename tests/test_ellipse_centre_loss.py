"""Contracts for the explicit ellipse-centre term.

It exists because of an AP-level decomposition, not an IoU-level one: replacing
the matched predictions' centre with the annotation's is worth +0.064 mAP50,
against +0.019 for the extent and +0.002 for the angle.  Two orientation losses
were built before that measurement and neither helped, so the properties that
matter here are that it charges the centre, that it does so in units of object
size, and that it survives the degenerate targets every background query brings.
"""

import pytest
import torch

from yopo.models.losses.ellipse_centre_loss import EllipseCentreLoss


def _target(cx=10.0, cy=10.0, xx=100.0, xy=0.0, yy=100.0):
    return torch.tensor([[cx, cy, xx, xy, yy]])


def test_exact_centre_is_free():
    loss = EllipseCentreLoss(loss_weight=1.0)
    value = loss(torch.tensor([[10.0, 10.0]]), _target(),
                 reduction_override="none")
    assert float(value[0]) == pytest.approx(0.0, abs=1e-9)


def test_grows_with_displacement():
    loss = EllipseCentreLoss(loss_weight=1.0)
    values = [
        float(loss(torch.tensor([[10.0 + d, 10.0]]), _target(),
                   reduction_override="none")[0])
        for d in (0.0, 0.5, 1.5, 3.0, 6.0)
    ]
    assert values == sorted(values)
    assert values[0] == 0.0
    # Beyond the quadratic region it is linear in the displacement.
    assert (values[4] - values[3]) == pytest.approx(
        (values[3] - values[2]) * 3.0 / 1.5, rel=0.05)


def test_normalised_by_object_size():
    """A displacement that is the same fraction of the object costs the same."""
    loss = EllipseCentreLoss(loss_weight=1.0)
    small = loss(torch.tensor([[11.0, 10.0]]), _target(xx=100.0, yy=100.0),
                 reduction_override="none")
    # Four times the variance is twice the scale, so twice the displacement.
    large = loss(torch.tensor([[12.0, 10.0]]), _target(xx=400.0, yy=400.0),
                 reduction_override="none")
    assert float(small[0]) == pytest.approx(float(large[0]), rel=1e-5)


def test_direction_does_not_matter():
    loss = EllipseCentreLoss(loss_weight=1.0)
    along = loss(torch.tensor([[13.0, 10.0]]), _target(),
                 reduction_override="none")
    across = loss(torch.tensor([[10.0, 13.0]]), _target(),
                  reduction_override="none")
    assert float(along[0]) == pytest.approx(float(across[0]), rel=1e-6)


def test_weights_apply():
    loss = EllipseCentreLoss(loss_weight=1.0)
    target = _target().repeat(2, 1)
    predicted = torch.tensor([[10.0, 10.0], [16.0, 10.0]])
    dropped = loss(predicted, target, weight=torch.tensor([1.0, 0.0]),
                   avg_factor=2)
    assert float(dropped) == pytest.approx(0.0, abs=1e-9)


def test_gradient_survives_degenerate_targets():
    """Background queries carry all-zero targets; masking a NaN does not remove it."""
    raw = torch.zeros(2, 2, requires_grad=True)
    target = torch.cat((_target(), torch.zeros(1, 5)))
    EllipseCentreLoss(loss_weight=1.0)(raw + 12.0, target,
                                       avg_factor=2).backward()
    assert torch.isfinite(raw.grad).all()

    raw2 = torch.zeros(2, 2, requires_grad=True)
    EllipseCentreLoss(loss_weight=1.0)(raw2 + 12.0, torch.zeros(2, 5),
                                       avg_factor=2).backward()
    assert torch.isfinite(raw2.grad).all()


def test_rejects_a_nonpositive_beta():
    with pytest.raises(ValueError, match="beta must be positive"):
        EllipseCentreLoss(beta=0.0)
