"""Contracts for the forward oriented-box KLD used by the parallel 9D head.

The closed form is checked against the definition it claims to equal, and
against the property it exists for: unlike ``KL(target || prediction)``, this
direction cannot be cheated by inflating the prediction.
"""

import numpy as np
import pytest
import torch

from yopo.models.losses.parallel_box3d_kld_loss import (
    DEFAULT_ALPHA, ParallelOrientedBox3DKLDLoss, oriented_box3d_kld)
from yopo.models.losses.gaucho3d_loss import ellipsoid_kld_from_cholesky


def _rotation(axis, angle):
    axis = np.asarray(axis, dtype=np.float64)
    axis = axis / np.linalg.norm(axis)
    cross = np.array([[0.0, -axis[2], axis[1]],
                      [axis[2], 0.0, -axis[0]],
                      [-axis[1], axis[0], 0.0]])
    return (np.eye(3) + np.sin(angle) * cross
            + (1 - np.cos(angle)) * cross @ cross)


def _brute_force_kld(mu_p, R_p, d_p, mu_t, R_t, d_t, alpha=DEFAULT_ALPHA):
    """``KL(N_p || N_t)`` straight from the definition, with explicit inverses."""
    sigma_p = R_p @ np.diag(alpha * np.square(d_p)) @ R_p.T
    sigma_t = R_t @ np.diag(alpha * np.square(d_t)) @ R_t.T
    inverse_t = np.linalg.inv(sigma_t)
    delta = (mu_p - mu_t).reshape(3, 1)
    return 0.5 * (
        float((delta.T @ inverse_t @ delta).item())
        + float(np.trace(inverse_t @ sigma_p))
        + float(np.log(np.linalg.det(sigma_t) / np.linalg.det(sigma_p)))
        - 3.0)


def _as_tensors(*arrays):
    return [torch.as_tensor(a, dtype=torch.float64).unsqueeze(0)
            for a in arrays]


# -- the closed form is the KLD it claims to be --------------------------


@pytest.mark.parametrize("seed", range(6))
def test_closed_form_matches_the_definition(seed):
    rng = np.random.default_rng(seed)
    mu_p = rng.normal(scale=0.05, size=3)
    mu_t = rng.normal(scale=0.05, size=3)
    R_p = _rotation(rng.normal(size=3), rng.uniform(0, np.pi))
    R_t = _rotation(rng.normal(size=3), rng.uniform(0, np.pi))
    d_p = rng.uniform(0.01, 0.06, size=3)
    d_t = rng.uniform(0.01, 0.06, size=3)

    closed = oriented_box3d_kld(*_as_tensors(mu_p, R_p, d_p, mu_t, R_t, d_t))
    assert closed.item() == pytest.approx(
        _brute_force_kld(mu_p, R_p, d_p, mu_t, R_t, d_t), rel=1e-9, abs=1e-12)


def test_exact_match_is_zero():
    R = _rotation([1.0, 2.0, 3.0], 0.7)
    d = np.array([0.03, 0.02, 0.025])
    mu = np.array([0.01, -0.02, 0.4])
    value = oriented_box3d_kld(*_as_tensors(mu, R, d, mu, R, d))
    assert value.item() == pytest.approx(0.0, abs=1e-12)


def test_invariant_to_a_common_rigid_transform():
    """Both boxes moved together is the same configuration, so the same KLD."""
    rng = np.random.default_rng(11)
    mu_p, mu_t = rng.normal(size=3) * 0.05, rng.normal(size=3) * 0.05
    R_p = _rotation(rng.normal(size=3), 0.4)
    R_t = _rotation(rng.normal(size=3), 1.1)
    d_p, d_t = np.array([0.03, 0.02, 0.04]), np.array([0.025, 0.03, 0.02])
    before = oriented_box3d_kld(*_as_tensors(mu_p, R_p, d_p, mu_t, R_t, d_t))

    world = _rotation([0.2, -1.0, 0.3], 0.9)
    shift = np.array([1.0, -2.0, 3.0])
    after = oriented_box3d_kld(*_as_tensors(
        world @ mu_p + shift, world @ R_p, d_p,
        world @ mu_t + shift, world @ R_t, d_t))
    assert after.item() == pytest.approx(before.item(), rel=1e-9)


def test_axis_sign_does_not_matter():
    """``q_ij^2`` is sign-blind: an extent has no front."""
    R = _rotation([0.0, 0.0, 1.0], 0.3)
    flipped = R.copy()
    flipped[:, 0] *= -1.0
    flipped[:, 1] *= -1.0          # keep it a proper rotation
    d = np.array([0.03, 0.02, 0.025])
    mu_t = np.zeros(3)
    a = oriented_box3d_kld(*_as_tensors(mu_t, R, d, mu_t, R, d))
    b = oriented_box3d_kld(*_as_tensors(mu_t, flipped, d, mu_t, R, d))
    assert b.item() == pytest.approx(a.item(), abs=1e-12)


# -- the property this direction exists for ------------------------------


def test_inflating_the_prediction_cannot_hide_a_centre_error():
    """The reason for using this direction rather than the GauCho one.

    With a fixed 40 mm displacement, growing the prediction along it strictly
    increases this loss, while it strictly decreases ``KL(target || pred)`` --
    the mechanism that produced the measured 2.87x optical-axis bloat.
    """
    R = np.eye(3)
    d_t = np.array([0.018, 0.018, 0.018])
    mu_t = np.array([0.0, 0.0, 0.40])
    mu_p = np.array([0.0, 0.0, 0.44])

    forward, reverse = [], []
    for stretch in (1.0, 2.0, 3.0, 4.0):
        d_p = np.array([0.018, 0.018, 0.018 * stretch])
        forward.append(oriented_box3d_kld(
            *_as_tensors(mu_p, R, d_p, mu_t, R, d_t)).item())

        chol_p = torch.diag_embed(torch.tensor(
            [np.sqrt(DEFAULT_ALPHA) * x for x in d_p],
            dtype=torch.float64)).unsqueeze(0)
        chol_t = torch.diag_embed(torch.tensor(
            [np.sqrt(DEFAULT_ALPHA) * x for x in d_t],
            dtype=torch.float64)).unsqueeze(0)
        reverse.append(ellipsoid_kld_from_cholesky(
            torch.tensor(mu_p, dtype=torch.float64).unsqueeze(0), chol_p,
            torch.tensor(mu_t, dtype=torch.float64).unsqueeze(0), chol_t
        ).item())

    assert forward == sorted(forward), (
        f"forward KLD must penalise inflation, got {forward}")
    assert reverse == sorted(reverse, reverse=True), (
        f"reverse KLD is expected to reward inflation, got {reverse}")


def test_gradient_pushes_size_toward_the_target():
    log_size = torch.zeros(1, 3, dtype=torch.float64, requires_grad=True)
    d_t = torch.tensor([[0.02, 0.02, 0.02]], dtype=torch.float64)
    d_p = 0.04 * torch.exp(log_size)          # twice too large
    rotation = torch.eye(3, dtype=torch.float64).unsqueeze(0)
    center = torch.zeros(1, 3, dtype=torch.float64)
    oriented_box3d_kld(center, rotation, d_p, center, rotation,
                       d_t).sum().backward()
    # Positive gradient shrinks under descent, on every axis.
    assert (log_size.grad > 0).all()


def test_gradient_is_finite_for_a_sphere():
    """These fruit are near-spherical; a NaN here would be on real data."""
    log_size = torch.zeros(4, 3, requires_grad=True)
    rotation = torch.eye(3).expand(4, 3, 3)
    d_p = 0.02 * torch.exp(log_size)
    d_t = torch.full((4, 3), 0.02)
    center_p = torch.randn(4, 3, generator=torch.Generator().manual_seed(3)) * 0.01
    center_t = torch.zeros(4, 3)
    oriented_box3d_kld(center_p, rotation, d_p, center_t, rotation,
                       d_t).sum().backward()
    assert torch.isfinite(log_size.grad).all()


# -- the module wrapper --------------------------------------------------


def test_module_bounds_and_weights():
    R = torch.eye(3).expand(2, 3, 3)
    d_t = torch.full((2, 3), 0.02)
    d_p = torch.stack((torch.full((3,), 0.02), torch.full((3,), 0.20)))
    center = torch.zeros(2, 3)
    loss = ParallelOrientedBox3DKLDLoss(loss_weight=1.0, tau=1.0)

    per_sample = loss(center, R, d_p, center, R, d_t,
                      reduction_override="none")
    assert per_sample[0].item() == pytest.approx(0.0, abs=1e-9)
    assert 0.0 < per_sample[1].item() < 1.0, "tau must bound the distance"

    dropped = loss(center, R, d_p, center, R, d_t,
                   weight=torch.tensor([1.0, 0.0]), avg_factor=2)
    assert dropped.item() == pytest.approx(0.0, abs=1e-9)


def test_nonpositive_size_is_excluded_not_propagated():
    R = torch.eye(3).expand(2, 3, 3)
    d_t = torch.full((2, 3), 0.02)
    d_p = torch.stack((torch.full((3,), 0.02), torch.tensor([0.02, -1.0, 0.02])))
    center = torch.zeros(2, 3)
    loss = ParallelOrientedBox3DKLDLoss(loss_weight=1.0)
    value = loss(center, R, d_p, center, R, d_t, avg_factor=2)
    assert torch.isfinite(value)
    assert int(loss.last_invalid_count.item()) == 1


@pytest.mark.parametrize("tau", [0.5, 1.0, 2.0, 5.0])
def test_bounding_is_zero_at_zero_for_every_tau(tau):
    """``1 - 1/(tau + d)`` happens to be right at tau=1 and wrong everywhere else.

    At tau=2 it charges 0.5 for a perfect prediction; at tau=0.5 it returns a
    negative loss.  The bounded form has to be ``d / (d + tau)``.
    """
    rotation = torch.eye(3).unsqueeze(0)
    size = torch.full((1, 3), 0.02)
    center = torch.zeros(1, 3)
    loss = ParallelOrientedBox3DKLDLoss(loss_weight=1.0, tau=tau)
    exact = loss(center, rotation, size, center, rotation, size,
                 reduction_override="none")
    assert float(exact[0]) == pytest.approx(0.0, abs=1e-9)

    wrong_size = torch.full((1, 3), 0.06)
    worse = loss(center, rotation, wrong_size, center, rotation, size,
                 reduction_override="none")
    assert 0.0 < float(worse[0]) < 1.0
