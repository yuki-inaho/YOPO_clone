from __future__ import annotations

import numpy as np
import pytest
import torch

from yopo.models.losses.projected_ellipsoid_loss import (
    GaussianGWDLoss,
    ProjectedEllipsoidGWDLoss,
    compact_gaussian,
    normalized_gaussian_anisotropy_weights,
    project_ellipsoid_to_gaussian,
)


def _intrinsic() -> torch.Tensor:
    return torch.tensor(
        [[[443.9066, 0.0, 321.3503],
          [0.0, 449.1953, 230.8687],
          [0.0, 0.0, 1.0]]],
        dtype=torch.float64,
    )


def _identity_6d(*, requires_grad: bool = False) -> torch.Tensor:
    return torch.tensor(
        [[1.0, 0.0, 0.0, 0.0, 1.0, 0.0]],
        dtype=torch.float64,
        requires_grad=requires_grad,
    )


def test_exact_sphere_projection_matches_perspective_solution():
    translation = torch.tensor([[0.0, 0.0, 0.5]], dtype=torch.float64)
    radius = 0.02
    size = torch.full((1, 3), 2 * radius, dtype=torch.float64)

    xy, sigma, valid = project_ellipsoid_to_gaussian(
        translation, _identity_6d(), size, _intrinsic())

    expected_radii = _intrinsic()[0, [0, 1], [0, 1]] * radius / np.sqrt(
        translation[0, 2].item() ** 2 - radius**2)
    torch.testing.assert_close(xy, _intrinsic()[:, :2, 2])
    torch.testing.assert_close(
        sigma[0].diagonal(), expected_radii.square(), rtol=1e-10, atol=1e-10)
    torch.testing.assert_close(
        sigma[0, 0, 1], torch.tensor(0.0, dtype=torch.float64))
    assert valid.tolist() == [True]


def test_anisotropic_projection_gwd_reaches_rotation_gradient():
    translation = torch.tensor([[0.03, -0.02, 0.5]], dtype=torch.float64)
    size = torch.tensor([[0.08, 0.03, 0.02]], dtype=torch.float64)
    predicted_rotation = _identity_6d(requires_grad=True)
    target_rotation = torch.tensor(
        [[0.0, 1.0, 0.0, -1.0, 0.0, 0.0]], dtype=torch.float64)
    target_xy, target_sigma, _ = project_ellipsoid_to_gaussian(
        translation, target_rotation, size, _intrinsic())
    target = compact_gaussian(target_xy, target_sigma)

    loss = ProjectedEllipsoidGWDLoss(
        detach_center=True, detach_depth=True, detach_size=True,
    )(
        centers_2d_px=torch.tensor([[348.0, 213.0]], dtype=torch.float64),
        depth=translation[:, 2:],
        rotations=predicted_rotation,
        sizes=size,
        target_gaussians=target,
        intrinsics=_intrinsic(),
        weight=torch.ones(1, dtype=torch.float64),
        avg_factor=1,
    )
    loss.backward()

    assert torch.isfinite(loss)
    assert predicted_rotation.grad is not None
    assert torch.isfinite(predicted_rotation.grad).all()
    assert predicted_rotation.grad.norm() > 0


def test_rotation_only_mode_detaches_center_depth_and_size():
    center = torch.tensor([[350.0, 210.0]], dtype=torch.float64, requires_grad=True)
    depth = torch.tensor([[0.5]], dtype=torch.float64, requires_grad=True)
    size = torch.tensor(
        [[0.08, 0.03, 0.02]], dtype=torch.float64, requires_grad=True)
    rotation = _identity_6d(requires_grad=True)
    target_rotation = torch.tensor(
        [[0.0, 1.0, 0.0, -1.0, 0.0, 0.0]], dtype=torch.float64)
    translation = torch.tensor([[0.03, -0.02, 0.5]], dtype=torch.float64)
    target_xy, target_sigma, _ = project_ellipsoid_to_gaussian(
        translation, target_rotation, size.detach(), _intrinsic())

    loss = ProjectedEllipsoidGWDLoss(
        detach_center=True, detach_depth=True, detach_size=True,
    )(
        center, depth, rotation, size,
        compact_gaussian(target_xy, target_sigma), _intrinsic(),
        weight=torch.ones(1, dtype=torch.float64), avg_factor=1,
    )
    loss.backward()

    assert center.grad is None
    assert depth.grad is None
    assert size.grad is None
    assert rotation.grad is not None and rotation.grad.norm() > 0


def test_sphere_projection_is_rotation_invariant():
    translation = torch.tensor([[0.02, 0.01, 0.5]], dtype=torch.float64)
    size = torch.full((1, 3), 0.04, dtype=torch.float64)
    first_rotation = _identity_6d()
    second_rotation = torch.tensor(
        [[0.0, 1.0, 0.0, -1.0, 0.0, 0.0]], dtype=torch.float64)

    first_xy, first_sigma, _ = project_ellipsoid_to_gaussian(
        translation, first_rotation, size, _intrinsic())
    second_xy, second_sigma, _ = project_ellipsoid_to_gaussian(
        translation, second_rotation, size, _intrinsic())

    torch.testing.assert_close(first_xy, second_xy, rtol=1e-10, atol=1e-10)
    torch.testing.assert_close(first_sigma, second_sigma, rtol=1e-10, atol=1e-10)


def test_invalid_positive_pair_fails_loudly():
    target = torch.tensor(
        [[320.0, 240.0, 100.0, 0.0, 100.0]], dtype=torch.float64)
    with pytest.raises(RuntimeError, match="invalid positive"):
        ProjectedEllipsoidGWDLoss(fail_on_invalid=True)(
            centers_2d_px=torch.tensor(
                [[320.0, 240.0]], dtype=torch.float64),
            depth=torch.tensor([[0.01]], dtype=torch.float64),
            rotations=_identity_6d(),
            sizes=torch.tensor([[0.08, 0.03, 0.02]], dtype=torch.float64),
            target_gaussians=target,
            intrinsics=_intrinsic(),
            weight=torch.ones(1, dtype=torch.float64),
            avg_factor=1,
        )


def test_compact_gaussian_gwd_covariance_only_has_finite_gradient():
    predicted = torch.tensor(
        [[0.0, 0.0, 0.04, 0.0, 0.01]],
        dtype=torch.float64,
        requires_grad=True,
    )
    target = torch.tensor(
        [[0.5, 0.5, 0.01, 0.0, 0.04]], dtype=torch.float64)
    loss = GaussianGWDLoss(
        include_center=False, loss_weight=5.0)(
            predicted, target, weight=torch.ones(1, dtype=torch.float64),
            avg_factor=1)
    loss.backward()

    assert torch.isfinite(loss)
    assert predicted.grad is not None
    assert torch.isfinite(predicted.grad).all()
    assert predicted.grad[0, :2].abs().sum() == 0
    assert predicted.grad[0, 2:].norm() > 0


def test_anisotropy_weights_are_continuous_and_unit_mean():
    targets = torch.tensor(
        [
            [0.0, 0.0, 1.0, 0.0, 1.0],
            [0.0, 0.0, 4.0, 0.0, 1.0],
            [0.0, 0.0, 9.0, 0.0, 1.0],
        ],
        dtype=torch.float64,
    )

    weights, anisotropy, valid = normalized_gaussian_anisotropy_weights(
        targets, power=1.0)

    torch.testing.assert_close(
        anisotropy, torch.tensor([0.0, 0.6, 0.8], dtype=torch.float64))
    torch.testing.assert_close(weights.mean(), torch.tensor(1.0, dtype=torch.float64))
    assert weights[0] == 0
    assert weights[0] < weights[1] < weights[2]
    assert valid.tolist() == [True, True, True]


def test_zero_anisotropy_power_preserves_uniform_weighting():
    targets = torch.tensor(
        [[0.0, 0.0, 4.0, 0.0, 1.0]], dtype=torch.float64)
    weights, _, valid = normalized_gaussian_anisotropy_weights(
        targets, power=0.0)

    torch.testing.assert_close(weights, torch.ones_like(weights))
    assert valid.tolist() == [True]


def test_projection_loss_rejects_negative_anisotropy_power():
    with pytest.raises(ValueError, match="target_anisotropy_power"):
        ProjectedEllipsoidGWDLoss(target_anisotropy_power=-1.0)


def test_projection_loss_applies_normalized_target_anisotropy_weights():
    translation = torch.tensor(
        [[0.0, 0.0, 0.5], [0.0, 0.0, 0.5]], dtype=torch.float64)
    sizes = torch.tensor(
        [[0.05, 0.045, 0.04], [0.08, 0.03, 0.02]], dtype=torch.float64)
    target_rotation = torch.tensor(
        [[0.0, 1.0, 0.0, -1.0, 0.0, 0.0]] * 2,
        dtype=torch.float64,
    )
    intrinsic = _intrinsic().repeat(2, 1, 1)
    target_xy, target_sigma, _ = project_ellipsoid_to_gaussian(
        translation, target_rotation, sizes, intrinsic)
    targets = compact_gaussian(target_xy, target_sigma)
    expected_weights, _, _ = normalized_gaussian_anisotropy_weights(
        targets, power=1.0)
    common = dict(
        centers_2d_px=intrinsic[:, :2, 2],
        depth=translation[:, 2:],
        rotations=_identity_6d().repeat(2, 1),
        sizes=sizes,
        target_gaussians=targets,
        intrinsics=intrinsic,
        weight=torch.ones(2, dtype=torch.float64),
        avg_factor=2,
    )

    uniform = ProjectedEllipsoidGWDLoss(
        reduction="none", target_anisotropy_power=0.0)(**common)
    weighted_loss = ProjectedEllipsoidGWDLoss(
        reduction="none", target_anisotropy_power=1.0)
    weighted = weighted_loss(**common)

    torch.testing.assert_close(weighted, uniform * expected_weights)
    assert weighted_loss.last_target_anisotropy_mean > 0
    torch.testing.assert_close(
        weighted_loss.last_anisotropy_weight_max,
        expected_weights.max(),
    )
