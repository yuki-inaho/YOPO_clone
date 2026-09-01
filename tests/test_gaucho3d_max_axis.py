"""The physical size bound: soft penalty in training, hard cap at inference.

These fruit do not exceed roughly 5 cm across.  That is knowledge about the
world, and it closes the escape route the KLD centre term used to pay for --
growing an ellipsoid along the line of sight until it covers a centre it could
not place.  A finite loss cannot guarantee a bound, so the two mechanisms are
tested as a pair: the penalty teaches it, the clamp enforces it.
"""

import numpy as np
import pytest
import torch

from yopo.models.losses.gaucho3d_geometry import (clamp_sigma_max_axis,
                                                  ellipsoid_max_diameter)
from yopo.models.losses.gaucho3d_loss import (Ellipsoid3DKLDLoss,
                                              EllipsoidMaxAxisLoss)


def _cholesky_for_diameters(diameters):
    """Cholesky factor of an axis-aligned ellipsoid with the given extents."""
    radii = torch.as_tensor(diameters, dtype=torch.float32) * 0.5
    return torch.diag_embed(radii)


# -- the penalty is exactly inert inside the bound -----------------------


@pytest.mark.parametrize("diameter", [0.005, 0.02, 0.0249, 0.049, 0.05])
def test_no_penalty_at_or_below_the_bound(diameter):
    """99% of this dataset is 2-4 cm; the term must not touch any of it."""
    loss = EllipsoidMaxAxisLoss(max_diameter=0.05, loss_weight=1.0)
    value = loss(_cholesky_for_diameters([[diameter] * 3]))
    assert value.item() == 0.0


def test_penalty_grows_beyond_the_bound():
    loss = EllipsoidMaxAxisLoss(max_diameter=0.05, loss_weight=1.0)
    values = [
        loss(_cholesky_for_diameters([[d, 0.02, 0.02]])).item()
        for d in (0.05, 0.06, 0.10, 0.15)
    ]
    assert values[0] == 0.0
    assert values == sorted(values)
    # ReLU(d/dmax - 1)^2: at 10 cm the excess is exactly 1.0.
    assert values[2] == pytest.approx(1.0, rel=1e-5)


def test_only_the_longest_axis_matters():
    """A pancake within the bound on its long axis is not penalized."""
    loss = EllipsoidMaxAxisLoss(max_diameter=0.05, loss_weight=1.0)
    assert loss(_cholesky_for_diameters([[0.049, 0.001, 0.001]])).item() == 0.0


# -- the sphere case, where eigen-decomposition backward usually breaks ---


def test_gradient_is_finite_for_a_sphere():
    """``eigh`` backward divides by eigenvalue gaps; ``eigvalsh`` does not.

    Tomatoes are near-spherical, so a NaN here would appear on real data rather
    than in a corner case.
    """
    raw = torch.full((4, 3), 0.04, requires_grad=True)
    cholesky = torch.diag_embed(raw * 0.5)
    EllipsoidMaxAxisLoss(max_diameter=0.05)(cholesky).backward()
    assert torch.isfinite(raw.grad).all()

    over = torch.full((4, 3), 0.08, requires_grad=True)
    EllipsoidMaxAxisLoss(max_diameter=0.05)(
        torch.diag_embed(over * 0.5)).backward()
    assert torch.isfinite(over.grad).all()
    # Positive gradient shrinks under descent, and no axis is ever grown.
    assert (over.grad >= 0).all()
    assert over.grad.sum() > 0.0
    # For an exact sphere the eigenvalues are degenerate, so the subgradient
    # lands on one arbitrary principal axis rather than spreading over three.
    # That is finite and correct for a max-axis constraint -- only a violating
    # axis needs to move -- but it means the term shrinks a sphere one axis at
    # a time.  The shape KLD is what keeps it spherical while it does.
    assert (over.grad > 0).sum(dim=-1).tolist() == [1, 1, 1, 1]


def test_max_diameter_matches_the_analytic_value():
    rotation = torch.tensor([[0.6, -0.8, 0.0], [0.8, 0.6, 0.0],
                             [0.0, 0.0, 1.0]])
    radii = torch.tensor([0.03, 0.01, 0.005])
    sigma = rotation @ torch.diag(radii.square()) @ rotation.T
    assert ellipsoid_max_diameter(sigma.unsqueeze(0)).item() == pytest.approx(
        0.06, rel=1e-5)


# -- the inference clamp -------------------------------------------------


def test_clamp_is_the_identity_below_the_bound():
    rotation = torch.linalg.qr(torch.randn(3, 3, generator=torch.Generator(
    ).manual_seed(0)))[0]
    sigma = (rotation @ torch.diag(torch.tensor([0.02, 0.015, 0.01]).square())
             @ rotation.T).unsqueeze(0)
    clamped, was_clamped = clamp_sigma_max_axis(sigma, 0.05)
    assert not bool(was_clamped.any())
    assert torch.allclose(clamped, sigma, atol=1e-12)


def test_clamp_caps_the_long_axis_and_keeps_the_rest():
    rotation = torch.linalg.qr(torch.randn(3, 3, generator=torch.Generator(
    ).manual_seed(1)))[0]
    radii = torch.tensor([0.05, 0.012, 0.008])          # 10 cm long axis
    sigma = (rotation @ torch.diag(radii.square()) @ rotation.T).unsqueeze(0)
    clamped, was_clamped = clamp_sigma_max_axis(sigma, 0.05)

    assert bool(was_clamped.all())
    assert ellipsoid_max_diameter(clamped).item() == pytest.approx(0.05,
                                                                   rel=1e-5)
    # The two shorter axes and the orientation survive untouched.
    before = torch.linalg.eigvalsh(sigma)[0].sqrt() * 2.0
    after = torch.linalg.eigvalsh(clamped)[0].sqrt() * 2.0
    assert after[0].item() == pytest.approx(before[0].item(), rel=1e-5)
    assert after[1].item() == pytest.approx(before[1].item(), rel=1e-5)
    _, vectors_before = torch.linalg.eigh(sigma[0])
    _, vectors_after = torch.linalg.eigh(clamped[0])
    assert torch.allclose(vectors_before.abs(), vectors_after.abs(), atol=1e-5)


def test_clamp_reports_per_instance():
    sigma = torch.stack([
        torch.diag(torch.tensor([0.01, 0.01, 0.01]).square()),   # 2 cm, fine
        torch.diag(torch.tensor([0.04, 0.01, 0.01]).square()),   # 8 cm, over
    ])
    _, was_clamped = clamp_sigma_max_axis(sigma, 0.05)
    assert was_clamped.tolist() == [False, True]


def test_clamp_rejects_a_nonpositive_bound():
    sigma = torch.eye(3).unsqueeze(0) * 1e-4
    with pytest.raises(ValueError, match="max_diameter must be positive"):
        clamp_sigma_max_axis(sigma, 0.0)


# -- dropping oversized annotations from shape supervision ---------------


def test_oversized_annotation_can_be_excluded_from_the_shape_loss():
    """0.91% of the training annotations exceed 5 cm, up to 10.98 cm.

    Asking the KLD to reproduce those while the max-axis term forbids them
    produces directly opposing gradients, so they are dropped from the shape
    term by weight -- the mechanism this pins.
    """
    predicted = _cholesky_for_diameters([[0.02, 0.02, 0.02],
                                         [0.02, 0.02, 0.02]])
    target = _cholesky_for_diameters([[0.02, 0.02, 0.02],
                                      [0.11, 0.02, 0.02]])
    centers = torch.zeros(2, 3)
    loss = Ellipsoid3DKLDLoss(loss_weight=1.0, include_center=False,
                              fail_on_invalid=False)

    unfiltered = loss(centers, predicted, centers, target, avg_factor=2)
    filtered = loss(centers, predicted, centers, target,
                    weight=torch.tensor([1.0, 0.0]), avg_factor=2)
    assert unfiltered.item() > 0.0
    # Row 0 is exact, so with row 1 dropped nothing is left to pay for.
    assert filtered.item() == pytest.approx(0.0, abs=1e-9)
