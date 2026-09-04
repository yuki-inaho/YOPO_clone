"""Acceptance contract tests for the GauCho-3D ellipsoid path.

These mirror the "implementation acceptance test" table of the RGB-D GauCho-3D
design note.  They check algebraic and geometric contracts only; nothing here
claims anything about learned accuracy.
"""

import math

import pytest
import torch

from yopo.models.losses.gaucho3d_geometry import (
    SPHERE,
    SPHEROID,
    TRIAXIAL,
    block_cholesky3d,
    cholesky3d_from_raw,
    cholesky3d_to_dual_plane,
    cholesky3d_to_raw,
    cholesky3d_to_scale_shape,
    decode_center_from_anchor,
    dual_conic_to_gaussian,
    dual_plane_cholesky3d,
    ellipsoid_from_rotation_size,
    ellipsoid_to_dual_quadric,
    front_margin,
    project_dual_quadric,
    project_ellipsoid_dual_quadric,
    ray_ellipsoid_roots,
    ray_frame,
    scale_shape_cholesky3d,
    sigma_from_cholesky,
    symmetry_class,
)
from yopo.models.losses.gaucho3d_loss import (
    DualConicResidualLoss,
    DualQuadricProjectionGWDLoss,
    Ellipsoid3DGWDLoss,
    Ellipsoid3DKLDLoss,
    EllipsoidFreeSpaceLoss,
    RayEllipsoidSurfaceLoss,
    ellipsoid_gwd,
    ellipsoid_kld_from_cholesky,
)

DTYPE = torch.float64


def _random_rotations(count: int, generator: torch.Generator) -> torch.Tensor:
    matrix = torch.randn(count, 3, 3, dtype=DTYPE, generator=generator)
    orthogonal, _ = torch.linalg.qr(matrix)
    determinant = torch.linalg.det(orthogonal)
    orthogonal[:, :, 0] = orthogonal[:, :, 0] * determinant.unsqueeze(-1)
    return orthogonal


def _front_ellipsoids(count: int, generator: torch.Generator):
    """Small camera-front ellipsoids with a realistic fruit scale."""
    center = torch.stack(
        (
            0.1 * torch.randn(count, dtype=DTYPE, generator=generator),
            0.1 * torch.randn(count, dtype=DTYPE, generator=generator),
            1.0 + torch.rand(count, dtype=DTYPE, generator=generator),
        ),
        dim=-1,
    )
    raw = 0.3 * torch.randn(count, 6, dtype=DTYPE, generator=generator)
    cholesky = scale_shape_cholesky3d(raw, 0.03)
    return center, cholesky


def _pinhole(count: int) -> torch.Tensor:
    intrinsic = torch.zeros(count, 3, 3, dtype=DTYPE)
    intrinsic[:, 0, 0] = 600.0
    intrinsic[:, 1, 1] = 600.0
    intrinsic[:, 0, 2] = 400.0
    intrinsic[:, 1, 2] = 300.0
    intrinsic[:, 2, 2] = 1.0
    return intrinsic


@pytest.fixture
def generator() -> torch.Generator:
    gen = torch.Generator()
    gen.manual_seed(20260831)
    return gen


def test_direct_cholesky_chart_is_always_spd(generator):
    raw = 3.0 * torch.randn(4096, 6, dtype=DTYPE, generator=generator)
    cholesky = cholesky3d_from_raw(raw)
    sigma = sigma_from_cholesky(cholesky)
    assert torch.isfinite(cholesky).all()
    assert (cholesky.diagonal(dim1=-2, dim2=-1) > 0).all()
    assert torch.linalg.eigvalsh(sigma).min() > 0
    # Cholesky of the reconstructed Sigma must succeed for every sample.
    assert torch.isfinite(torch.linalg.cholesky(sigma)).all()


def test_direct_cholesky_round_trip(generator):
    raw = torch.randn(2048, 6, dtype=DTYPE, generator=generator)
    recovered = cholesky3d_to_raw(cholesky3d_from_raw(raw))
    assert torch.allclose(recovered, raw, atol=1e-12)


def test_dual_plane_is_bijective_with_standard_cholesky(generator):
    raw = torch.randn(4096, 6, dtype=DTYPE, generator=generator)
    standard = cholesky3d_from_raw(raw)
    round_trip = dual_plane_cholesky3d(cholesky3d_to_dual_plane(standard))
    assert (standard - round_trip).abs().max() < 1e-10
    assert (
        sigma_from_cholesky(standard) - sigma_from_cholesky(round_trip)
    ).abs().max() < 1e-10


def test_dual_plane_zero_correlation_cannot_represent_general_spd(generator):
    """Two orthogonal 2D GauCho factors alone are one degree of freedom short."""
    raw = torch.randn(4096, 6, dtype=DTYPE, generator=generator)
    full = sigma_from_cholesky(dual_plane_cholesky3d(raw))
    reduced_raw = raw.clone()
    reduced_raw[:, 5] = 0.0
    reduced = sigma_from_cholesky(dual_plane_cholesky3d(reduced_raw))
    relative = (
        (full - reduced).norm(dim=(-2, -1)) / full.norm(dim=(-2, -1))
    )
    assert relative.median() > 1e-2


def test_scale_shape_chart_round_trip_and_determinant(generator):
    raw = torch.randn(4096, 6, dtype=DTYPE, generator=generator)
    rho0 = 0.04
    cholesky = scale_shape_cholesky3d(raw, rho0)
    recovered = cholesky3d_to_scale_shape(cholesky, rho0)
    assert (recovered - raw).abs().max() < 1e-10
    rho = torch.linalg.det(sigma_from_cholesky(cholesky)).pow(1.0 / 6.0)
    assert torch.allclose(rho, rho0 * raw[:, 0].exp(), rtol=1e-9)


def test_block_cholesky_has_six_degrees_of_freedom(generator):
    transverse = torch.zeros(256, 2, 2, dtype=DTYPE)
    transverse[:, 0, 0] = torch.rand(256, dtype=DTYPE, generator=generator) + 0.5
    transverse[:, 1, 0] = torch.randn(256, dtype=DTYPE, generator=generator)
    transverse[:, 1, 1] = torch.rand(256, dtype=DTYPE, generator=generator) + 0.5
    coupling = torch.randn(256, 2, dtype=DTYPE, generator=generator)
    thickness = torch.randn(256, dtype=DTYPE, generator=generator)
    cholesky = block_cholesky3d(transverse, coupling, thickness)
    assert torch.linalg.eigvalsh(sigma_from_cholesky(cholesky)).min() > 0
    assert torch.allclose(cholesky[:, :2, :2], transverse)


def test_dual_quadric_projection_matches_explicit_camera_matrix(generator):
    center, cholesky = _front_ellipsoids(512, generator)
    sigma = sigma_from_cholesky(cholesky)
    intrinsic = _pinhole(512)
    projection = torch.cat(
        (intrinsic, torch.zeros(512, 3, 1, dtype=DTYPE)), dim=-1)
    explicit = project_dual_quadric(
        ellipsoid_to_dual_quadric(center, sigma), projection)
    fast = intrinsic @ (
        sigma - center.unsqueeze(-1) * center.unsqueeze(-2)
    ) @ intrinsic.transpose(-1, -2)
    assert (explicit - fast).abs().max() < 1e-9


def test_projected_conic_is_front_facing_and_spd(generator):
    center, cholesky = _front_ellipsoids(512, generator)
    sigma = sigma_from_cholesky(cholesky)
    intrinsic = _pinhole(512)
    conic = intrinsic @ (
        sigma - center.unsqueeze(-1) * center.unsqueeze(-2)
    ) @ intrinsic.transpose(-1, -2)
    assert (conic[:, 2, 2] < 0).all()
    mean, shape, valid = project_ellipsoid_dual_quadric(
        center, sigma, intrinsic)
    assert valid.all()
    assert torch.linalg.eigvalsh(shape).min() > 0
    assert (front_margin(center, sigma) > 0).all()


def test_dual_conic_round_trip_is_canonical(generator):
    center, cholesky = _front_ellipsoids(512, generator)
    sigma = sigma_from_cholesky(cholesky)
    intrinsic = _pinhole(512)
    conic = intrinsic @ (
        sigma - center.unsqueeze(-1) * center.unsqueeze(-2)
    ) @ intrinsic.transpose(-1, -2)
    mean, shape, valid = dual_conic_to_gaussian(conic)
    top = torch.cat(
        (shape - mean.unsqueeze(-1) * mean.unsqueeze(-2), -mean.unsqueeze(-1)),
        dim=-1)
    bottom = torch.cat(
        (-mean.unsqueeze(-2), -torch.ones(512, 1, 1, dtype=DTYPE)), dim=-1)
    rebuilt = torch.cat((top, bottom), dim=-2)

    def canonical(matrix):
        return matrix / matrix[..., 2, 2].abs().unsqueeze(-1).unsqueeze(-1)

    assert valid.all()
    assert (canonical(conic) - canonical(rebuilt)).abs().max() < 1e-9


def test_projected_center_differs_from_projected_3d_center(generator):
    """The silhouette centre is not the projection of the 3D centre.

    Supervising ``||mu_2 - pi(t)||`` would therefore fit an identity that does
    not hold for a finite anisotropic ellipsoid.
    """
    center, cholesky = _front_ellipsoids(512, generator)
    sigma = sigma_from_cholesky(cholesky)
    intrinsic = _pinhole(512)
    mean, _, valid = project_ellipsoid_dual_quadric(center, sigma, intrinsic)
    naive = (intrinsic @ center.unsqueeze(-1)).squeeze(-1)
    naive = naive[:, :2] / naive[:, 2:3]
    separation = (mean - naive).norm(dim=-1)[valid]
    assert separation.max() > 1e-3


def test_ray_roots_satisfy_the_surface_equation(generator):
    center, cholesky = _front_ellipsoids(256, generator)
    # Aim rays straight at each centre so every sample intersects.
    directions = center / center.norm(dim=-1, keepdim=True)
    z_near, z_far, valid = ray_ellipsoid_roots(directions, center, cholesky)
    assert valid.all()
    assert (z_near <= z_far).all()
    surface = z_near.unsqueeze(-1) * directions - center
    solved = torch.linalg.solve_triangular(
        cholesky, surface.unsqueeze(-1), upper=False).squeeze(-1)
    assert (solved.square().sum(dim=-1) - 1.0).abs().max() < 1e-9


def test_ray_miss_is_reported_not_clamped(generator):
    center, cholesky = _front_ellipsoids(64, generator)
    # Point the rays away from the objects: every ray must miss.
    directions = torch.zeros(64, 3, dtype=DTYPE)
    directions[:, 0] = 1.0
    _, _, valid = ray_ellipsoid_roots(directions, center, cholesky)
    assert not valid.any()


def test_ray_frame_is_right_handed_and_aligned(generator):
    center, _ = _front_ellipsoids(512, generator)
    frame = ray_frame(center)
    identity = torch.eye(3, dtype=DTYPE).expand_as(frame)
    assert (frame.transpose(-1, -2) @ frame - identity).abs().max() < 1e-10
    assert (torch.linalg.det(frame) - 1.0).abs().max() < 1e-10
    expected = center / center.norm(dim=-1, keepdim=True)
    assert (frame[:, :, 2] - expected).abs().max() < 1e-10


def test_se3_transform_round_trip(generator):
    rotation = _random_rotations(1024, generator)
    translation = torch.randn(1024, 3, dtype=DTYPE, generator=generator)
    center, cholesky = _front_ellipsoids(1024, generator)
    sigma = sigma_from_cholesky(cholesky)
    moved_center = (rotation @ center.unsqueeze(-1)).squeeze(-1) + translation
    moved_sigma = rotation @ sigma @ rotation.transpose(-1, -2)
    back_center = (
        rotation.transpose(-1, -2) @ (moved_center - translation).unsqueeze(-1)
    ).squeeze(-1)
    back_sigma = rotation.transpose(-1, -2) @ moved_sigma @ rotation
    assert (back_center - center).abs().max() < 1e-10
    assert (back_sigma - sigma).abs().max() < 1e-10


def test_shape_is_invariant_to_axis_sign_and_permutation(generator):
    rotation = _random_rotations(256, generator)
    radii = torch.rand(256, 3, dtype=DTYPE, generator=generator) + 0.5
    sigma = ellipsoid_from_rotation_size(rotation, 2.0 * radii)

    flipped = rotation.clone()
    flipped[:, :, 0] *= -1
    flipped[:, :, 1] *= -1
    assert (
        ellipsoid_from_rotation_size(flipped, 2.0 * radii) - sigma
    ).abs().max() < 1e-12

    order = torch.tensor([1, 0, 2])
    permuted = rotation[:, :, order].clone()
    permuted[:, :, 2] *= -1  # keep the frame right-handed
    assert (
        ellipsoid_from_rotation_size(permuted, 2.0 * radii[:, order]) - sigma
    ).abs().max() < 1e-12


def test_symmetry_class_detects_sphere_spheroid_triaxial():
    rotation = torch.eye(3, dtype=DTYPE).expand(3, 3, 3).contiguous()
    sizes = torch.tensor(
        [
            [0.06, 0.06, 0.06],   # sphere
            [0.12, 0.06, 0.06],   # spheroid
            [0.12, 0.08, 0.04],   # triaxial
        ],
        dtype=DTYPE,
    )
    sigma = ellipsoid_from_rotation_size(rotation, sizes)
    classes = symmetry_class(sigma)
    assert classes[0].item() == SPHERE
    assert classes[1].item() == SPHEROID
    assert classes[2].item() == TRIAXIAL


def test_invalid_depth_anchor_is_reported_without_fallback():
    pixel = torch.tensor([[400.0, 300.0], [400.0, 300.0]], dtype=DTYPE)
    depth = torch.tensor([1.0, float("nan")], dtype=DTYPE)
    residual = torch.zeros(2, 3, dtype=DTYPE)
    intrinsic = _pinhole(2)
    center, valid = decode_center_from_anchor(
        pixel, depth, residual, intrinsic, 0.03)
    assert valid[0].item() is True
    assert valid[1].item() is False
    # The valid sample must be the exact back-projection of its pixel.
    assert torch.allclose(center[0], torch.tensor([0.0, 0.0, 1.0], dtype=DTYPE),
                          atol=1e-12)


def test_kld_matches_the_explicit_inverse_formula(generator):
    raw_p = torch.randn(512, 6, dtype=DTYPE, generator=generator)
    raw_g = torch.randn(512, 6, dtype=DTYPE, generator=generator)
    cholesky_p = cholesky3d_from_raw(raw_p)
    cholesky_g = cholesky3d_from_raw(raw_g)
    center_p = torch.randn(512, 3, dtype=DTYPE, generator=generator)
    center_g = torch.randn(512, 3, dtype=DTYPE, generator=generator)

    fast = ellipsoid_kld_from_cholesky(
        center_p, cholesky_p, center_g, cholesky_g)

    sigma_p = sigma_from_cholesky(cholesky_p)
    sigma_g = sigma_from_cholesky(cholesky_g)
    inverse_p = torch.linalg.inv(sigma_p)
    displacement = (center_p - center_g).unsqueeze(-1)
    reference = 0.5 * (
        (inverse_p @ sigma_g).diagonal(dim1=-2, dim2=-1).sum(dim=-1)
        + (displacement.transpose(-1, -2) @ inverse_p @ displacement).reshape(-1)
        - 3.0
        + torch.linalg.det(sigma_p).log()
        - torch.linalg.det(sigma_g).log()
    )
    assert (fast - reference.clamp_min(0.0)).abs().max() < 1e-8


def test_kld_and_gwd_vanish_for_identical_ellipsoids(generator):
    raw = torch.randn(256, 6, dtype=DTYPE, generator=generator)
    cholesky = cholesky3d_from_raw(raw)
    center = torch.randn(256, 3, dtype=DTYPE, generator=generator)
    sigma = sigma_from_cholesky(cholesky)
    assert ellipsoid_kld_from_cholesky(
        center, cholesky, center, cholesky).abs().max() < 1e-10
    assert ellipsoid_gwd(center, sigma, center, sigma).abs().max() < 1e-8


def test_kld_loss_is_bounded_and_differentiable(generator):
    raw = torch.randn(64, 6, dtype=torch.float32, generator=generator,
                      requires_grad=False)
    raw.requires_grad_(True)
    cholesky = cholesky3d_from_raw(raw)
    center = torch.zeros(64, 3, dtype=torch.float32, requires_grad=True)
    target_cholesky = cholesky3d_from_raw(
        torch.randn(64, 6, dtype=torch.float32, generator=generator))
    target_center = torch.randn(64, 3, dtype=torch.float32, generator=generator)
    loss_module = Ellipsoid3DKLDLoss(loss_weight=2.0, tau=1.0)
    loss = loss_module(center, cholesky, target_center, target_cholesky)
    assert torch.isfinite(loss)
    assert 0.0 <= loss.item() <= 2.0
    loss.backward()
    assert torch.isfinite(raw.grad).all()
    assert torch.isfinite(center.grad).all()


def test_ellipsoid_kld_direction_is_backward_compatible_and_reversible():
    predicted_center = torch.tensor([[0.01, -0.02, 0.45]], dtype=DTYPE)
    target_center = torch.tensor([[0.0, 0.0, 0.40]], dtype=DTYPE)
    predicted_cholesky = torch.diag_embed(
        torch.tensor([[0.02, 0.03, 0.05]], dtype=DTYPE))
    target_cholesky = torch.diag_embed(
        torch.tensor([[0.04, 0.025, 0.01]], dtype=DTYPE))

    default = Ellipsoid3DKLDLoss(loss_weight=1.0)(
        predicted_center, predicted_cholesky, target_center, target_cholesky)
    explicit = Ellipsoid3DKLDLoss(
        loss_weight=1.0, direction="target_to_prediction")(
            predicted_center, predicted_cholesky, target_center,
            target_cholesky)
    reverse = Ellipsoid3DKLDLoss(
        loss_weight=1.0, direction="prediction_to_target")(
            predicted_center, predicted_cholesky, target_center,
            target_cholesky)
    reverse_distance = ellipsoid_kld_from_cholesky(
        target_center, target_cholesky, predicted_center,
        predicted_cholesky)
    reverse_reference = 1.0 - 1.0 / (1.0 + reverse_distance)

    assert torch.equal(default, explicit)
    assert torch.allclose(reverse, reverse_reference.mean(), atol=1e-12)
    assert not torch.allclose(explicit, reverse)


def test_ellipsoid_kld_rejects_unknown_direction():
    with pytest.raises(ValueError, match="direction"):
        Ellipsoid3DKLDLoss(direction="symmetric")


def test_reverse_ellipsoid_kld_removes_center_driven_scale_inflation():
    def log_scale_gradient(direction):
        log_scale = torch.zeros(1, 3, dtype=DTYPE, requires_grad=True)
        radius = 0.01
        predicted_cholesky = torch.diag_embed(
            radius * torch.exp(log_scale))
        target_cholesky = torch.diag_embed(
            torch.full((1, 3), radius, dtype=DTYPE))
        predicted_center = torch.tensor([[0.0, 0.0, 0.44]], dtype=DTYPE)
        target_center = torch.tensor([[0.0, 0.0, 0.40]], dtype=DTYPE)
        loss = Ellipsoid3DKLDLoss(
            include_center=True, direction=direction)(
                predicted_center, predicted_cholesky, target_center,
                target_cholesky)
        loss.backward()
        return log_scale.grad[0]

    historical = log_scale_gradient("target_to_prediction")
    reverse = log_scale_gradient("prediction_to_target")
    assert historical[2] < -1e-3
    assert torch.equal(reverse, torch.zeros_like(reverse))


def test_gwd_loss_is_finite_near_a_sphere():
    """Gradients must survive the r1 -> r2 -> r3 limit."""
    rotation = torch.eye(3).expand(8, 3, 3).contiguous()
    radii = torch.full((8, 3), 0.03)
    radii[:, 0] += torch.linspace(0.0, 1e-6, 8)
    target = ellipsoid_from_rotation_size(rotation, 2.0 * radii)
    target_cholesky = torch.linalg.cholesky(target)
    raw = torch.zeros(8, 6, requires_grad=True)
    cholesky = scale_shape_cholesky3d(raw, 0.03)
    center = torch.zeros(8, 3)
    loss = Ellipsoid3DGWDLoss(fail_on_invalid=True)(
        center, cholesky, center, target_cholesky)
    loss.backward()
    assert torch.isfinite(loss)
    assert torch.isfinite(raw.grad).all()


def test_projection_loss_is_minimal_at_the_true_ellipsoid(generator):
    center, cholesky = _front_ellipsoids(128, generator)
    center = center.float()
    cholesky = cholesky.float()
    intrinsic = _pinhole(128).float()
    sigma = sigma_from_cholesky(cholesky)
    mean, shape, valid = project_ellipsoid_dual_quadric(
        center, sigma, intrinsic)
    assert valid.all()
    target = torch.stack(
        (mean[:, 0], mean[:, 1], shape[:, 0, 0], shape[:, 0, 1], shape[:, 1, 1]),
        dim=-1)
    module = DualQuadricProjectionGWDLoss(fail_on_invalid=True)
    exact = module(center, cholesky, target, intrinsic)
    perturbed = module(center + 0.01, cholesky, target, intrinsic)
    assert exact.item() < 1e-4
    assert perturbed.item() > exact.item()


def test_projection_loss_detaches_its_two_dimensional_teacher(generator):
    center, cholesky = _front_ellipsoids(32, generator)
    center = center.float().requires_grad_(True)
    cholesky = cholesky.float()
    intrinsic = _pinhole(32).float()
    target = torch.randn(32, 5).abs() + 1.0
    target[:, 3] = 0.0  # keep the target shape positive definite
    target.requires_grad_(True)
    module = DualQuadricProjectionGWDLoss(
        detach_target=True, fail_on_invalid=False)
    loss = module(center, cholesky, target, intrinsic)
    loss.backward()
    assert target.grad is None or torch.count_nonzero(target.grad) == 0
    assert center.grad is not None


def test_conic_residual_is_scale_and_sign_canonical(generator):
    conic = torch.randn(64, 3, 3, dtype=torch.float32, generator=generator)
    conic = conic + conic.transpose(-1, -2)
    conic[:, 2, 2] = -1.0
    module = DualConicResidualLoss(fail_on_invalid=True)
    same = module(conic, conic)
    scaled = module(conic, -3.0 * conic)
    assert same.item() < 1e-10
    assert scaled.item() < 1e-10


def test_surface_loss_only_uses_weighted_visible_pixels():
    center = torch.tensor([[0.0, 0.0, 1.0]])
    cholesky = torch.eye(3).unsqueeze(0) * 0.05
    directions = torch.tensor([[[0.0, 0.0, 1.0], [0.0, 0.0, 1.0]]])
    # The first ray sees the true near surface; the second is an occluder that
    # the visible mask excludes.
    observed = torch.tensor([[0.95, 0.30]])
    module = RayEllipsoidSurfaceLoss(fail_on_invalid=False)
    masked = module(center, cholesky, directions, observed,
                    torch.tensor([[1.0, 0.0]]))
    both = module(center, cholesky, directions, observed,
                  torch.tensor([[1.0, 1.0]]))
    assert masked.item() < 1e-6
    assert both.item() > masked.item()


def test_free_space_penalty_is_one_sided():
    center = torch.tensor([[0.0, 0.0, 1.0]])
    directions = torch.tensor([[[0.0, 0.0, 1.0]]])
    weight = torch.ones(1, 1)
    module = EllipsoidFreeSpaceLoss(fail_on_invalid=False, num_samples=4)
    # A small ellipsoid leaves the free-space corridor empty.
    small = module(center, torch.eye(3).unsqueeze(0) * 0.02, directions,
                   torch.tensor([[0.95]]), weight)
    # An oversized one intrudes into it.
    large = module(center, torch.eye(3).unsqueeze(0) * 0.60, directions,
                   torch.tensor([[0.95]]), weight)
    assert small.item() == pytest.approx(0.0, abs=1e-9)
    assert large.item() > 0.0


def test_losses_reject_malformed_shapes():
    center = torch.zeros(4, 3)
    cholesky = torch.eye(3).expand(4, 3, 3)
    with pytest.raises(ValueError):
        ellipsoid_kld_from_cholesky(center[:, :2], cholesky, center, cholesky)
    with pytest.raises(ValueError):
        cholesky3d_from_raw(torch.zeros(4, 5))
    with pytest.raises(ValueError):
        Ellipsoid3DKLDLoss(reduction="median")
    with pytest.raises(ValueError):
        Ellipsoid3DKLDLoss(tau=0.5)


def test_bounded_losses_stay_within_their_weight():
    """No single outlier can dominate a batch."""
    center = torch.zeros(2, 3)
    huge = torch.eye(3).expand(2, 3, 3) * 1e6
    tiny = torch.eye(3).expand(2, 3, 3) * 1e-6
    loss = Ellipsoid3DKLDLoss(loss_weight=1.0, fail_on_invalid=False)(
        center, tiny.contiguous(), center, huge.contiguous())
    assert math.isfinite(loss.item())
    # ``1 - 1/(tau + d)`` saturates at the loss weight; in floating point the
    # bound is attained rather than approached, which is the intended cap.
    assert 0.0 <= loss.item() <= 1.0


# ---------------------------------------------------------------------------
# Head integration
# ---------------------------------------------------------------------------

from mmengine.structures import InstanceData  # noqa: E402
from torch import nn  # noqa: E402


class _ZeroLoss(nn.Module):
    """Stand-in for an unrelated loss term inside a focused head test."""

    def forward(self, *args, **kwargs):
        return torch.zeros(())

from yopo.models.dense_pose_heads.dino_9d_center2d_posehead import (  # noqa: E402
    DINO9DCenter2DPoseHead,
)

_IMG_META = dict(
    img_shape=(600, 800),
    ori_shape=(600, 800),
    intrinsic=[600.0, 600.0, 400.0, 300.0],
    scale_factor=(1.0, 1.0),
)


def _gaucho_head(**overrides) -> DINO9DCenter2DPoseHead:
    kwargs = dict(
        num_classes=1,
        embed_dims=8,
        num_reg_fcs=1,
        num_pred_layer=1,
        train_cfg=None,
        use_cop_chain=False,
        cop_prediction_mode="parallel",
        loss_cls=dict(
            type="FocalLoss", use_sigmoid=True, gamma=2.0, alpha=0.25,
            loss_weight=1.0),
        gaucho_ellipsoid=True,
        gaucho_size_prior=0.03,
        loss_ellipsoid=dict(
            type="Ellipsoid3DKLDLoss", loss_weight=1.0, fail_on_invalid=False),
    )
    kwargs.update(overrides)
    return DINO9DCenter2DPoseHead(**kwargs)


def _stub_positive_targets(head, rotation, sizes, depth):
    """One positive query with a real 3D annotation, one background query."""
    labels = torch.tensor([0, 1])
    label_weights = torch.ones(2)
    bbox_targets = torch.tensor([[0.5, 0.5, 0.2, 0.2], [0.0, 0.0, 0.0, 0.0]])
    bbox_weights = torch.tensor([[1.0] * 4, [0.0] * 4])
    centers = torch.tensor([[0.5, 0.5], [0.0, 0.0]])
    center_weights = torch.tensor([[1.0, 1.0], [0.0, 0.0]])
    z_targets = torch.tensor([[depth], [0.0]])
    z_weights = torch.tensor([[1.0], [0.0]])
    rotation_targets = torch.cat((rotation, torch.zeros(1, 6)), dim=0)
    rotation_weights = torch.tensor([[1.0] * 6, [0.0] * 6])
    sizes_targets = torch.cat((sizes, torch.zeros(1, 3)), dim=0)
    sizes_weights = torch.tensor([[1.0] * 3, [0.0] * 3])
    obb_targets = torch.tensor(
        [[400.0, 300.0, 90.0, 0.0, 90.0], [0.0, 0.0, 0.0, 0.0, 0.0]])
    obb_weights = torch.tensor([1.0, 0.0])
    head.get_targets = lambda *a, **k: (
        [labels], [label_weights], [bbox_targets], [bbox_weights],
        [centers], [center_weights], [z_targets], [z_weights],
        [rotation_targets], [rotation_weights], [sizes_targets],
        [sizes_weights], [obb_targets], [obb_weights], 1, 1)
    # Neutralize the unrelated pose terms so the assertions below isolate the
    # GauCho contribution rather than the head's default 9D objectives.
    for name in ("loss_rotation", "loss_sizes", "loss_z", "loss_centers_2d"):
        setattr(head, name, _ZeroLoss())


def _run_head_loss(head, ellipsoid_preds=None):
    cls_scores = torch.zeros(1, 2, 1, requires_grad=True)
    bbox_preds = torch.tensor(
        [[[0.5, 0.5, 0.2, 0.2], [0.2, 0.2, 0.1, 0.1]]], requires_grad=True)
    centers = torch.full((1, 2, 2), 0.5, requires_grad=True)
    z_preds = torch.full((1, 2, 1), 0.9, requires_grad=True)
    return head.loss_by_feat_single(
        cls_scores, bbox_preds, centers, z_preds,
        torch.zeros(1, 2, 6), torch.full((1, 2, 3), 0.06),
        None, None, None, None,
        batch_gt_instances=[InstanceData()],
        batch_img_metas=[_IMG_META],
        ellipsoid_preds=ellipsoid_preds,
    )


def test_head_forward_emits_the_ellipsoid_slot_only_when_enabled():
    hidden = torch.zeros(1, 1, 2, 8)
    references = [torch.full((1, 2, 4), 0.5)]

    disabled = _gaucho_head(gaucho_ellipsoid=False, loss_ellipsoid=None)
    assert disabled(hidden, references)[10] is None
    assert disabled(hidden, references)[11] is None

    enabled = _gaucho_head()
    ellipsoid = enabled(hidden, references)[10]
    assert ellipsoid.shape == (1, 1, 2, 6 * enabled.num_classes)


def test_zero_initialized_head_starts_at_the_size_prior():
    head = _gaucho_head()
    head.init_weights()
    raw = torch.zeros(4, 6)
    cholesky = head._gaucho_cholesky(raw)
    sigma = sigma_from_cholesky(cholesky)
    expected = torch.eye(3) * head.gaucho_size_prior ** 2
    assert torch.allclose(sigma, expected.expand_as(sigma), atol=1e-6)


def test_every_chart_starts_at_the_same_isotropic_prior():
    raw = torch.zeros(4, 6)
    reference = None
    for chart in ("scale_shape", "direct", "dual_plane"):
        head = _gaucho_head(gaucho_chart=chart)
        sigma = sigma_from_cholesky(head._gaucho_cholesky(raw))
        if reference is None:
            reference = sigma
        else:
            assert torch.allclose(sigma, reference, atol=1e-6)


def test_head_ellipsoid_loss_is_finite_and_trains_only_its_branch():
    head = _gaucho_head()
    head.init_weights()
    rotation = torch.tensor([[1.0, 0.0, 0.0, 0.0, 1.0, 0.0]])
    sizes = torch.tensor([[0.08, 0.05, 0.05]])
    _stub_positive_targets(head, rotation, sizes, depth=0.9)
    ellipsoid_preds = torch.zeros(1, 2, 6, requires_grad=True)

    losses = _run_head_loss(head, ellipsoid_preds)
    loss_ellipsoid = losses[12]

    assert torch.isfinite(loss_ellipsoid)
    assert loss_ellipsoid.item() > 0.0
    loss_ellipsoid.backward()
    assert ellipsoid_preds.grad is not None
    assert torch.isfinite(ellipsoid_preds.grad).all()
    assert ellipsoid_preds.grad.norm() > 0


def test_head_ellipsoid_loss_vanishes_at_the_annotated_ellipsoid():
    head = _gaucho_head()
    head.init_weights()
    rotation = torch.tensor([[1.0, 0.0, 0.0, 0.0, 1.0, 0.0]])
    # An isotropic annotation of exactly the prior radius: the zero-initialized
    # shape output is then already correct and only the centre can disagree.
    sizes = torch.full((1, 3), 2.0 * head.gaucho_size_prior)
    _stub_positive_targets(head, rotation, sizes, depth=0.9)
    losses = _run_head_loss(head, torch.zeros(1, 2, 6))
    assert losses[12].item() < 1e-3


def test_head_rejects_projection_loss_without_direct_supervision():
    with pytest.raises(ValueError, match="loss_ellipsoid"):
        _gaucho_head(
            loss_ellipsoid=None,
            loss_ellipsoid_projection=dict(
                type="DualQuadricProjectionGWDLoss"))


def test_head_rejects_ellipsoid_losses_when_the_branch_is_off():
    with pytest.raises(ValueError, match="gaucho_ellipsoid"):
        _gaucho_head(gaucho_ellipsoid=False)


def test_head_rejects_unknown_chart_and_non_positive_prior():
    with pytest.raises(ValueError, match="gaucho_chart"):
        _gaucho_head(gaucho_chart="quaternion")
    with pytest.raises(ValueError, match="gaucho_size_prior"):
        _gaucho_head(gaucho_size_prior=0.0)


# ---------------------------------------------------------------------------
# 2D GauCho amodal ellipse
# ---------------------------------------------------------------------------

from yopo.models.losses.gaucho3d_geometry import (  # noqa: E402
    cholesky2d_to_scale_shape,
    gaussian_to_ellipse2d,
    obb_gaussian_to_cholesky2d,
    scale_shape_cholesky2d,
    sigma_from_cholesky2d,
)
from yopo.models.losses.gaucho3d_loss import (  # noqa: E402
    Ellipse2DKLDLoss,
    ellipse_kld_from_cholesky,
)


def test_ellipse2d_chart_round_trip(generator):
    raw = torch.randn(4096, 3, dtype=DTYPE, generator=generator)
    radius = torch.rand(4096, dtype=DTYPE, generator=generator) * 40.0 + 5.0
    cholesky = scale_shape_cholesky2d(raw, radius)
    recovered = cholesky2d_to_scale_shape(cholesky, radius)
    assert (recovered - raw).abs().max() < 1e-10
    # ``r`` alone carries the area: det L = rho^2 e^{2r}.
    determinant = torch.linalg.det(cholesky)
    assert torch.allclose(
        determinant, radius.square() * (2.0 * raw[:, 0]).exp(), rtol=1e-9)


def test_ellipse2d_chart_is_always_spd(generator):
    raw = 5.0 * torch.randn(4096, 3, dtype=DTYPE, generator=generator)
    radius = torch.full((4096,), 12.0, dtype=DTYPE)
    sigma = sigma_from_cholesky2d(scale_shape_cholesky2d(raw, radius))
    assert torch.linalg.eigvalsh(sigma).min() > 0


def test_ellipse2d_kld_matches_the_explicit_inverse_formula(generator):
    radius = torch.full((512,), 15.0, dtype=DTYPE)
    cholesky_p = scale_shape_cholesky2d(
        torch.randn(512, 3, dtype=DTYPE, generator=generator), radius)
    cholesky_g = scale_shape_cholesky2d(
        torch.randn(512, 3, dtype=DTYPE, generator=generator), radius)
    center_p = torch.randn(512, 2, dtype=DTYPE, generator=generator)
    center_g = torch.randn(512, 2, dtype=DTYPE, generator=generator)

    fast = ellipse_kld_from_cholesky(
        center_p, cholesky_p, center_g, cholesky_g)

    sigma_p = sigma_from_cholesky2d(cholesky_p)
    sigma_g = sigma_from_cholesky2d(cholesky_g)
    inverse_p = torch.linalg.inv(sigma_p)
    displacement = (center_p - center_g).unsqueeze(-1)
    reference = 0.5 * (
        (inverse_p @ sigma_g).diagonal(dim1=-2, dim2=-1).sum(dim=-1)
        + (displacement.transpose(-1, -2) @ inverse_p @ displacement).reshape(-1)
        - 2.0
        + torch.linalg.det(sigma_p).log()
        - torch.linalg.det(sigma_g).log()
    )
    # The divergence spans several orders of magnitude across the batch, so
    # the agreement is asserted relatively rather than against a fixed floor.
    assert torch.allclose(fast, reference.clamp_min(0.0), rtol=1e-10, atol=1e-12)


def test_ellipse2d_loss_vanishes_at_the_annotated_ellipse(generator):
    radius = torch.full((256,), 20.0)
    raw = torch.randn(256, 3, generator=generator) * 0.3
    cholesky = scale_shape_cholesky2d(raw, radius)
    sigma = sigma_from_cholesky2d(cholesky)
    mean = torch.randn(256, 2, generator=generator) * 5.0
    target = torch.stack(
        (mean[:, 0], mean[:, 1], sigma[:, 0, 0], sigma[:, 0, 1],
         sigma[:, 1, 1]), dim=-1)
    module = Ellipse2DKLDLoss(fail_on_invalid=True)
    exact = module(mean, cholesky, target)
    shifted = module(mean + 8.0, cholesky, target)
    assert exact.item() < 1e-5
    assert shifted.item() > exact.item()


def test_ellipse2d_target_rejects_degenerate_annotations():
    degenerate = torch.tensor([[0.0, 0.0, 0.0, 0.0, 0.0]])
    healthy = torch.tensor([[1.0, 2.0, 9.0, 0.0, 4.0]])
    _, _, valid = obb_gaussian_to_cholesky2d(
        torch.cat((degenerate, healthy), dim=0))
    assert valid.tolist() == [False, True]


def test_gaussian_to_ellipse_decode_is_display_only():
    sigma = torch.diag_embed(torch.tensor([[16.0, 4.0]]))
    mean = torch.tensor([[3.0, -2.0]])
    ellipse = gaussian_to_ellipse2d(mean, sigma)
    semi_major, semi_minor, cx, cy, theta = ellipse[0].tolist()
    assert semi_major == pytest.approx(4.0, abs=1e-5)
    assert semi_minor == pytest.approx(2.0, abs=1e-5)
    assert cx == pytest.approx(3.0, abs=1e-6)
    assert cy == pytest.approx(-2.0, abs=1e-6)
    assert theta == pytest.approx(0.0, abs=1e-6)


def test_head_zero_init_ellipse2d_is_the_inscribed_circle_of_its_box():
    head = _gaucho_head(
        gaucho_ellipse2d=True,
        loss_ellipse2d=dict(type="Ellipse2DKLDLoss", loss_weight=1.0))
    head.init_weights()
    hidden = torch.zeros(1, 1, 2, 8)
    references = [torch.full((1, 2, 4), 0.5)]
    outputs = head(hidden, references)
    assert outputs[11].shape == (1, 1, 2, 5 * head.num_classes)

    # A zero raw output gives L = rho_Q * I, i.e. the circle whose radius is the
    # geometric mean half-extent of the reference box.
    radius = torch.full((2,), 25.0)
    cholesky = scale_shape_cholesky2d(torch.zeros(2, 3), radius)
    sigma = sigma_from_cholesky2d(cholesky)
    assert torch.allclose(
        sigma, torch.eye(2).expand_as(sigma) * radius[0] ** 2, atol=1e-5)


def test_head_ellipse2d_loss_trains_only_its_own_branch():
    head = _gaucho_head(
        gaucho_ellipsoid=False,
        loss_ellipsoid=None,
        gaucho_ellipse2d=True,
        loss_ellipse2d=dict(
            type="Ellipse2DKLDLoss", loss_weight=1.0, fail_on_invalid=False))
    head.init_weights()
    rotation = torch.tensor([[1.0, 0.0, 0.0, 0.0, 1.0, 0.0]])
    sizes = torch.tensor([[0.08, 0.05, 0.05]])
    _stub_positive_targets(head, rotation, sizes, depth=0.9)
    ellipse2d_preds = torch.zeros(1, 2, 5, requires_grad=True)

    cls_scores = torch.zeros(1, 2, 1, requires_grad=True)
    bbox_preds = torch.tensor(
        [[[0.5, 0.5, 0.2, 0.2], [0.2, 0.2, 0.1, 0.1]]], requires_grad=True)
    losses = head.loss_by_feat_single(
        cls_scores, bbox_preds, torch.full((1, 2, 2), 0.5),
        torch.full((1, 2, 1), 0.9), torch.zeros(1, 2, 6),
        torch.full((1, 2, 3), 0.06), None, None, None, None,
        batch_gt_instances=[InstanceData()],
        batch_img_metas=[_IMG_META],
        ellipse2d_preds=ellipse2d_preds,
    )
    loss_ellipse2d = losses[14]
    assert torch.isfinite(loss_ellipse2d) and loss_ellipse2d.item() > 0.0
    loss_ellipse2d.backward()
    assert ellipse2d_preds.grad is not None
    assert ellipse2d_preds.grad.norm() > 0
    # The chart's reference frame is detached, so the box branch stays untouched.
    assert bbox_preds.grad is None


def test_head_requires_ellipse2d_loss_and_flag_together():
    with pytest.raises(ValueError, match="gaucho_ellipse2d"):
        _gaucho_head(gaucho_ellipse2d=True)


# ---------------------------------------------------------------------------
# Curriculum configs
# ---------------------------------------------------------------------------

from pathlib import Path  # noqa: E402

from mmengine.config import Config  # noqa: E402

_CONFIG_DIR = Path(__file__).resolve().parent.parent / "configs" / "yopo"
_STAGE_PREFIX = "nocs_fruits_Jun30_2025_rgbd_gaucho_"


def _stage_config(name):
    return Config.fromfile(
        str(_CONFIG_DIR / f"{_STAGE_PREFIX}{name}.py"),
        import_custom_modules=False)


def test_curriculum_turns_objectives_on_in_the_documented_order():
    """2D/visible first, then direct 3D, then one-way projection."""
    stage_a = _stage_config("stageA_ellipse2d").model.bbox_head
    stage_b = _stage_config("stageB_ellipsoid").model.bbox_head
    stage_c = _stage_config("stageC_projection").model.bbox_head

    assert stage_a["gaucho_ellipse2d"] is True
    assert "gaucho_ellipsoid" not in stage_a
    assert "loss_ellipsoid_projection" not in stage_a

    assert stage_b["gaucho_ellipsoid"] is True
    assert stage_b["loss_ellipsoid"]["type"] == "Ellipsoid3DKLDLoss"
    assert "loss_ellipsoid_projection" not in stage_b

    assert stage_c["loss_ellipsoid_projection"]["type"] == \
        "DualQuadricProjectionGWDLoss"
    # The projection teacher must stay detached, and direct 3D supervision must
    # remain active as the second, independent ground of supervision.
    assert stage_c["loss_ellipsoid_projection"]["detach_target"] is True
    assert stage_c["loss_ellipsoid"]["type"] == "Ellipsoid3DKLDLoss"


def test_stage_configs_point_at_the_portable_dataset():
    for name in ("stageA_ellipse2d", "stageB_ellipsoid", "stageC_projection"):
        cfg = _stage_config(name)
        expected = "data/fruits_detection_Jun30-2025_stem_rgbd_736x512/"
        assert cfg.train_dataloader.dataset.data_root == expected
        assert cfg.val_dataloader.dataset.data_root == expected


def test_stage_configs_keep_the_obb_gaussian_annotation_loaded():
    """The GauCho-2D target is the inscribed ellipse of the annotated OBB."""
    cfg = _stage_config("stageC_projection")
    loaders = [t for t in cfg.train_pipeline
               if t.get("type") == "Load9DPoseAnnotations"]
    assert loaders, "the training pipeline must load 9D pose annotations"
    assert all(t.get("with_obb_gaussian") for t in loaders)


# ---------------------------------------------------------------------------
# Inference path
# ---------------------------------------------------------------------------

from mmengine.structures import BaseDataElement  # noqa: E402


def _data_sample(meta):
    sample = BaseDataElement()
    sample.set_metainfo(meta)
    return sample


def _predict_head(**overrides):
    kwargs = dict(
        gaucho_ellipse2d=True,
        loss_ellipse2d=dict(type="Ellipse2DKLDLoss", loss_weight=1.0),
        expose_gaucho_predictions=True,
        test_cfg=dict(max_per_img=2),
    )
    kwargs.update(overrides)
    head = _gaucho_head(**kwargs)
    head.init_weights()
    return head


def test_predict_emits_the_ellipse_envelope_and_ellipsoid_fields():
    head = _predict_head()
    hidden = torch.zeros(1, 1, 2, 8)
    references = [torch.full((1, 2, 4), 0.5)]
    meta = dict(_IMG_META)
    meta["scale_factor"] = (1.0, 1.0)
    results = head.predict(
        hidden, references, [_data_sample(meta)], rescale=False)

    result = results[0]
    assert result.ellipses.shape == (2, 5)
    assert result.ellipse_gaussians.shape == (2, 5)
    assert result.ellipse_obb.shape == (2, 5)
    assert torch.isfinite(result.ellipse_obb).all()
    # The envelope is 2a x 2b of the very ellipse that was emitted, not an
    # independently regressed box.
    assert torch.allclose(result.ellipse_obb[:, 2], 2.0 * result.ellipses[:, 0])
    assert torch.allclose(result.ellipse_obb[:, 3], 2.0 * result.ellipses[:, 1])
    assert torch.allclose(result.ellipse_obb[:, 4], result.ellipses[:, 4])
    assert (result.ellipse_obb[:, 2] >= result.ellipse_obb[:, 3]).all()

    # 3D fields appear together with the 3D branch.
    assert result.ellipsoid_shapes.shape == (2, 3, 3)
    assert result.ellipsoid_centers.shape == (2, 3)
    assert result.ellipsoid_symmetry.shape == (2,)
    assert result.ellipsoid_front_margin.shape == (2,)
    assert result.projected_ellipses.shape == (2, 5)
    assert result.projected_valid.shape == (2,)
    # ``translations`` is the (N, 3) camera-frame centre; a 4x4 ``T`` column
    # would be length 4 and silently wrong.
    assert torch.allclose(result.ellipsoid_centers, result.translations)


def test_predict_leaves_gaucho_fields_absent_when_not_exposed():
    head = _predict_head(expose_gaucho_predictions=False)
    hidden = torch.zeros(1, 1, 2, 8)
    references = [torch.full((1, 2, 4), 0.5)]
    meta = dict(_IMG_META)
    meta["scale_factor"] = (1.0, 1.0)
    result = head.predict(
        hidden, references, [_data_sample(meta)], rescale=False)[0]
    for field in ("ellipses", "ellipse_obb", "ellipsoid_shapes"):
        assert not hasattr(result, field)


def test_ellipse_metric_scores_a_perfect_prediction_as_one():
    from yopo.evaluation.metrics import EllipseEnvelopeRotatedIoUMetric

    metric = EllipseEnvelopeRotatedIoUMetric(num_classes=1, score_thr=0.0)
    # A 40x20 OBB at (100, 50) and the ellipse whose envelope reproduces it.
    gt = torch.tensor([[100.0, 50.0, 400.0, 0.0, 100.0]])
    pred_obb = torch.tensor([[100.0, 50.0, 40.0, 20.0, 0.0]])
    sample = {
        "pred_instances": {
            "ellipse_obb": pred_obb,
            "scores": torch.tensor([0.9]),
            "labels": torch.tensor([0]),
        },
        "gt_instances": {
            "obb_gaussians": gt,
            "labels": torch.tensor([0]),
        },
    }
    metric.process({}, [sample])
    scores = metric.compute_metrics(metric.results)
    assert any(value == pytest.approx(1.0, abs=1e-6)
               for value in scores.values() if isinstance(value, float))


def test_ellipse_metric_reports_a_missing_prediction_field():
    from yopo.evaluation.metrics import EllipseEnvelopeRotatedIoUMetric

    metric = EllipseEnvelopeRotatedIoUMetric(num_classes=1)
    sample = {
        "pred_instances": {"scores": torch.tensor([0.9]),
                           "labels": torch.tensor([0])},
        "gt_instances": {"obb_gaussians": torch.zeros(0, 5),
                         "labels": torch.zeros(0, dtype=torch.long)},
    }
    with pytest.raises(KeyError, match="expose_gaucho_predictions"):
        metric.process({}, [sample])


def test_metric_maps_ground_truth_into_the_prediction_frame():
    """``predict(rescale=True)`` returns original pixels; ``gt`` does not.

    The val pipeline resizes, so ``gt_instances`` lives in the transformed
    frame.  Without mapping it back, an otherwise perfect ellipse is inflated by
    ``1/scale_factor`` and almost every match is lost.
    """
    from yopo.evaluation.metrics import EllipseEnvelopeRotatedIoUMetric
    from yopo.evaluation.metrics.ellipse_rotated_iou_metric import (
        rescale_compact_gaussian,
    )

    scale = (0.8695652173913043, 0.869140625)
    # A 40x20 axis-aligned OBB in the ORIGINAL frame ...
    original = torch.tensor([[100.0, 50.0, 400.0, 0.0, 100.0]])
    # ... as the resized pipeline would store it.
    resized = torch.tensor([[
        100.0 * scale[0], 50.0 * scale[1],
        400.0 * scale[0] ** 2, 0.0, 100.0 * scale[1] ** 2,
    ]])
    recovered = rescale_compact_gaussian(resized, scale)
    assert torch.allclose(recovered, original, rtol=1e-5)

    metric = EllipseEnvelopeRotatedIoUMetric(num_classes=1, score_thr=0.0)
    sample = {
        "pred_instances": {
            "ellipse_obb": torch.tensor([[100.0, 50.0, 40.0, 20.0, 0.0]]),
            "scores": torch.tensor([0.9]),
            "labels": torch.tensor([0]),
        },
        "gt_instances": {
            "obb_gaussians": resized,
            "labels": torch.tensor([0]),
        },
        "scale_factor": scale,
    }
    metric.process({}, [sample])
    scores = metric.compute_metrics(metric.results)
    assert scores["rbbox_mAP_50"] == pytest.approx(1.0, abs=1e-6)


def test_metric_leaves_ground_truth_alone_without_rescaling():
    from yopo.evaluation.metrics import EllipseEnvelopeRotatedIoUMetric

    metric = EllipseEnvelopeRotatedIoUMetric(num_classes=1, score_thr=0.0)
    gt = torch.tensor([[100.0, 50.0, 400.0, 0.0, 100.0]])
    sample = {
        "pred_instances": {
            "ellipse_obb": torch.tensor([[100.0, 50.0, 40.0, 20.0, 0.0]]),
            "scores": torch.tensor([0.9]),
            "labels": torch.tensor([0]),
        },
        "gt_instances": {"obb_gaussians": gt, "labels": torch.tensor([0])},
        "scale_factor": (1.0, 1.0),
    }
    metric.process({}, [sample])
    assert metric.compute_metrics(metric.results)["rbbox_mAP_50"] == \
        pytest.approx(1.0, abs=1e-6)


_COMPACT_PREFIX = "nocs_fruits_Jun30_2025_rgbd_gaucho_compact_"


def _compact_config(name):
    return Config.fromfile(
        str(_CONFIG_DIR / f"{_COMPACT_PREFIX}{name}.py"),
        import_custom_modules=False)


def test_compact_curriculum_turns_objectives_on_in_order():
    """The warm-startable chain follows the same A -> B -> C order."""
    stage_a = _compact_config("stageA").model.bbox_head
    stage_b = _compact_config("stageB").model.bbox_head
    stage_c = _compact_config("stageC").model.bbox_head

    assert stage_a["gaucho_ellipse2d"] is True
    assert "gaucho_ellipsoid" not in stage_a
    assert stage_b["gaucho_ellipsoid"] is True
    assert "loss_ellipsoid_projection" not in stage_b
    assert stage_c["loss_ellipsoid_projection"]["detach_target"] is True
    assert stage_c["loss_ellipsoid"]["type"] == "Ellipsoid3DKLDLoss"


def test_compact_chain_is_query_consistent_and_warm_startable():
    """``num_queries`` must cover the ~148 objects a frame here can hold.

    It must also equal ``max_per_img``: a larger ``max_per_img`` makes
    inference ``topk`` past the end of the score tensor, which only shows up at
    validation time.
    """
    for name in ("stageA", "stageB", "stageC"):
        cfg = _compact_config(name)
        assert cfg.model.num_queries == 256
        assert cfg.model.test_cfg.max_per_img == cfg.model.num_queries
        assert cfg.model.bbox_head.test_cfg.max_per_img == cfg.model.num_queries
        assert cfg.model.bbox_head.expose_gaucho_predictions is True

    # Every stage takes its starting weights at launch (--cfg-options
    # load_from=...), so no config hard-codes an artifact path: Stage A would
    # otherwise pin an absolute local path, and B/C would silently discard the
    # previous stage by pointing back at the release.
    for name in ("stageA", "stageB", "stageC"):
        assert _compact_config(name).load_from is None
    for path in _CONFIG_DIR.glob(f"{_COMPACT_PREFIX}*.py"):
        assert "/home/" not in path.read_text()


def test_compact_chain_reports_both_dod_metrics():
    cfg = _compact_config("stageC")
    types = {m["type"] for m in cfg.val_evaluator}
    assert "EllipseEnvelopeRotatedIoUMetric" in types  # DoD-A
    assert "NOCSMetric" in types                        # DoD-B


# ---------------------------------------------------------------------------
# Regressions found by review
# ---------------------------------------------------------------------------


def test_training_and_inference_decode_the_same_ellipse():
    """One chart, one decode.

    If the reference-radius convention, the eps floor or the extent scaling
    drift between the loss path and the predict path, the model optimizes one
    ellipse and evaluation scores another, with every other test still green.
    """
    head = _gaucho_head()
    head.init_weights()
    raw = torch.randn(5, 5)
    box_px = torch.tensor([[400.0, 300.0, 60.0, 40.0]]).repeat(5, 1)
    mean_a, chol_a, sigma_a = head._decode_gaucho_ellipse2d(raw, box_px)
    mean_b, chol_b, sigma_b = head._decode_gaucho_ellipse2d(raw, box_px)
    assert torch.equal(mean_a, mean_b)
    assert torch.equal(chol_a, chol_b)
    # The decode is a pure function of (raw, box_px): both call sites differ
    # only in whether they detach box_px, which must not change the value.
    detached = head._decode_gaucho_ellipse2d(raw, box_px.detach())
    assert torch.allclose(mean_a, detached[0])
    assert torch.allclose(sigma_a, detached[2])


def test_size_prior_scales_every_chart_entry_not_just_the_diagonal():
    """The prior must multiply the finished factor, not bias the clipped logs.

    Biasing first pushes the prior through the chart's own log clip, which
    makes the usable range depend on the prior and leaves the shear entries at
    unit scale while the diagonal sits at the prior.
    """
    raw = torch.randn(64, 6)
    for chart in ("direct", "dual_plane"):
        head = _gaucho_head(gaucho_chart=chart, gaucho_size_prior=0.03)
        unit = _gaucho_head(gaucho_chart=chart, gaucho_size_prior=1.0)
        scaled = head._gaucho_cholesky(raw)
        base = unit._gaucho_cholesky(raw)
        # Every entry scales alike, including the off-diagonal shears.
        assert torch.allclose(scaled, 0.03 * base, atol=1e-7)

    # A prior far below exp(-log_clip) would previously start fully clipped,
    # i.e. at the wrong size and with no gradient.
    tiny = _gaucho_head(gaucho_chart="direct", gaucho_size_prior=1e-6)
    sigma = sigma_from_cholesky(tiny._gaucho_cholesky(torch.zeros(1, 6)))
    assert torch.allclose(
        sigma, torch.eye(3).unsqueeze(0) * 1e-12, atol=1e-18)


def test_ellipsoid_losses_survive_bfloat16_autocast():
    """The compact configs train under bfloat16 AMP.

    Autocast demotes matmul but leaves ``torch.linalg.*`` alone, and CUDA has
    no bfloat16 kernel for cholesky, triangular solve, LU or eigh -- so without
    a float32 island the first iteration dies.
    """
    device = "cuda" if torch.cuda.is_available() else "cpu"
    dtype = torch.bfloat16
    rotation = torch.eye(3, device=device).expand(4, 3, 3).contiguous()
    sizes = torch.full((4, 3), 0.06, device=device)
    center = torch.zeros(4, 3, device=device)
    raw = torch.zeros(4, 6, device=device, requires_grad=True)

    with torch.autocast(device_type=device, enabled=True, dtype=dtype):
        # Reproduces the demotion: a matmul chain on float32 inputs comes back
        # bfloat16 under autocast.
        demoted = ellipsoid_from_rotation_size(rotation, sizes)
        if device == "cuda":
            assert demoted.dtype is dtype
        with torch.autocast(device_type=device, enabled=False):
            target_sigma = ellipsoid_from_rotation_size(
                rotation.float(), sizes.float())
            identity = torch.eye(3, device=device)
            target_cholesky = torch.linalg.cholesky(
                target_sigma + 1e-12 * identity)
            cholesky = scale_shape_cholesky3d(raw.float(), 0.03)
            loss = Ellipsoid3DKLDLoss(fail_on_invalid=False)(
                center.float(), cholesky, center.float(), target_cholesky)
    assert torch.isfinite(loss)
    loss.backward()
    assert torch.isfinite(raw.grad).all()


def test_projection_loss_requires_the_image_space_intrinsic():
    """Fail closed on the frame mismatch rather than training on it silently.

    The projection target is the OBB Gaussian after the resize pipeline, so the
    dual quadric must be projected with the image-space K.  With the stored
    original-image K the centre round-trips and the whole error hides in the
    shape term.
    """
    with pytest.raises(ValueError, match="train_intrinsic_to_image_space"):
        _gaucho_head(
            loss_ellipsoid_projection=dict(
                type="DualQuadricProjectionGWDLoss"),
            train_intrinsic_to_image_space=False,
        )


def test_every_three_d_config_pins_the_intrinsic_frame():
    for name in ("stageB_ellipsoid", "stageC_projection"):
        head = _stage_config(name).model.bbox_head
        assert head["train_intrinsic_to_image_space"] is True
    for name in ("stageB", "stageC"):
        head = _compact_config(name).model.bbox_head
        assert head.get("train_intrinsic_to_image_space") is True


def test_front_margin_rejects_an_ellipsoid_behind_the_camera():
    """``C*_33 < 0`` alone is not a front test.

    ``C*_33 = Sigma_zz - t_z^2`` is negative behind the camera too, as soon as
    ``|t_z|`` exceeds the object's own depth extent, so without the front margin
    a fully behind-camera ellipsoid projects to a well-formed conic and would be
    scored as a detection.
    """
    intrinsic = _pinhole(3).float()
    sigma = sigma_from_cholesky(scale_shape_cholesky3d(torch.zeros(3, 6), 0.03))
    centers = torch.tensor([[0.05, 0.02, 1.0],
                            [0.05, 0.02, -1.0],
                            [0.05, 0.02, -5.0]])
    conic = intrinsic @ (
        sigma - centers.unsqueeze(-1) * centers.unsqueeze(-2)
    ) @ intrinsic.transpose(-1, -2)
    # All three pass the sign test; only the first is actually in front.
    assert (conic[:, 2, 2] < 0).all()
    assert front_margin(centers, sigma).tolist()[0] > 0
    assert all(m < 0 for m in front_margin(centers, sigma).tolist()[1:])

    _, _, valid = project_ellipsoid_dual_quadric(centers, sigma, intrinsic)
    assert valid.tolist() == [True, False, False]


def test_reduced_dual_plane_pins_the_conditional_correlation(generator):
    """Eq (20): the two-plane model with ``rc = 0``, for ablation A4."""
    raw = torch.randn(256, 6, dtype=DTYPE, generator=generator)
    reduced = dual_plane_cholesky3d(raw, fix_rc_zero=True)
    # rc = l32 / sqrt(l32^2 + l33^2) must be exactly zero.
    assert torch.allclose(reduced[:, 2, 1], torch.zeros_like(reduced[:, 2, 1]))
    assert torch.linalg.eigvalsh(sigma_from_cholesky(reduced)).min() > 0
    # And it must genuinely lose the sixth degree of freedom.
    full = dual_plane_cholesky3d(raw)
    relative = ((sigma_from_cholesky(full) - sigma_from_cholesky(reduced))
                .norm(dim=(-2, -1))
                / sigma_from_cholesky(full).norm(dim=(-2, -1)))
    assert relative.median() > 1e-2
    # The sixth raw value must be inert when pinned.
    other = raw.clone()
    other[:, 5] = other[:, 5] + 3.0
    assert torch.allclose(
        reduced, dual_plane_cholesky3d(other, fix_rc_zero=True))


def test_surface_loss_reweights_rather_than_rescales_when_rays_are_masked():
    """Eq (45) divides by the weight that contributed, not by the ray count."""
    center = torch.tensor([[0.0, 0.0, 1.0]])
    cholesky = torch.eye(3).unsqueeze(0) * 0.05
    module = RayEllipsoidSurfaceLoss(fail_on_invalid=False)

    directions = torch.tensor([[[0.0, 0.0, 1.0]]])
    observed = torch.tensor([[0.90]])
    one_ray = module(center, cholesky, directions, observed, torch.ones(1, 1))

    # The same ray, plus three that the visible mask excludes.  A weighted mean
    # is unchanged; a plain mean over all rays would be four times smaller.
    padded_dirs = directions.repeat(1, 4, 1)
    padded_obs = observed.repeat(1, 4)
    weights = torch.tensor([[1.0, 0.0, 0.0, 0.0]])
    masked = module(center, cholesky, padded_dirs, padded_obs, weights)
    assert masked.item() == pytest.approx(one_ray.item(), rel=1e-6)
    assert masked.item() > 0.0


# ---------------------------------------------------------------------------
# Reference-aligned scoring
# ---------------------------------------------------------------------------


def _nms_sample(boxes, scores, labels, gt):
    return {
        "pred_instances": {"ellipse_obb": boxes, "scores": scores,
                           "labels": labels},
        "gt_instances": {"obb_gaussians": gt,
                         "labels": torch.zeros(len(gt), dtype=torch.long)},
        "scale_factor": (1.0, 1.0),
    }


def test_metric_nms_is_off_by_default_so_logged_numbers_stay_comparable():
    from yopo.evaluation.metrics import EllipseEnvelopeRotatedIoUMetric
    assert EllipseEnvelopeRotatedIoUMetric(num_classes=1).nms_iou_threshold is None


def test_metric_nms_removes_duplicates_and_keeps_the_best_scored_one():
    from yopo.evaluation.metrics import EllipseEnvelopeRotatedIoUMetric

    metric = EllipseEnvelopeRotatedIoUMetric(
        num_classes=1, score_thr=0.0, nms_iou_threshold=0.2)
    # Three near-identical boxes on one object, plus a genuinely separate one.
    boxes = torch.tensor([
        [100.0, 50.0, 40.0, 20.0, 0.0],
        [101.0, 50.5, 40.0, 20.0, 0.0],
        [ 99.5, 49.5, 40.0, 20.0, 0.0],
        [400.0, 50.0, 40.0, 20.0, 0.0],
    ])
    scores = torch.tensor([0.9, 0.8, 0.7, 0.85])
    labels = torch.zeros(4, dtype=torch.long)
    kept_boxes, kept_scores, kept_labels = metric._suppress(boxes, scores, labels)
    assert len(kept_boxes) == 2
    # The survivors are the highest-scored member of each cluster.
    assert kept_scores[0].item() == pytest.approx(0.9, abs=1e-6)
    assert kept_scores[1].item() == pytest.approx(0.85, abs=1e-6)
    assert len(kept_labels) == 2


def test_metric_nms_is_class_aware():
    """Class-agnostic suppression would merge overlapping objects of different
    classes, which is a silent recall loss the moment a second class exists."""
    from yopo.evaluation.metrics import EllipseEnvelopeRotatedIoUMetric

    metric = EllipseEnvelopeRotatedIoUMetric(
        num_classes=2, score_thr=0.0, nms_iou_threshold=0.2)
    boxes = torch.tensor([[100.0, 50.0, 40.0, 20.0, 0.0],
                          [100.0, 50.0, 40.0, 20.0, 0.0]])
    scores = torch.tensor([0.9, 0.8])
    same_class = metric._suppress(boxes, scores, torch.zeros(2, dtype=torch.long))
    other_class = metric._suppress(boxes, scores, torch.tensor([0, 1]))
    assert len(same_class[0]) == 1
    assert len(other_class[0]) == 2


def test_metric_nms_raises_on_an_out_of_range_threshold():
    from yopo.evaluation.metrics import EllipseEnvelopeRotatedIoUMetric
    for bad in (0.0, -0.1, 1.5):
        with pytest.raises(ValueError, match="nms_iou_threshold"):
            EllipseEnvelopeRotatedIoUMetric(num_classes=1, nms_iou_threshold=bad)


def test_metric_nms_handles_an_empty_prediction_set():
    from yopo.evaluation.metrics import EllipseEnvelopeRotatedIoUMetric
    metric = EllipseEnvelopeRotatedIoUMetric(
        num_classes=1, score_thr=0.0, nms_iou_threshold=0.2)
    boxes, scores, labels = metric._suppress(
        torch.zeros(0, 5), torch.zeros(0), torch.zeros(0, dtype=torch.long))
    assert len(boxes) == 0 and len(scores) == 0 and len(labels) == 0


def test_head_exposes_the_reduced_dual_plane_for_ablation_a4():
    """Table 4 A4 vs A5 must be selectable from a config alone."""
    reduced = _gaucho_head(gaucho_chart="dual_plane",
                           gaucho_dual_plane_fix_rc_zero=True)
    full = _gaucho_head(gaucho_chart="dual_plane",
                        gaucho_dual_plane_fix_rc_zero=False)
    raw = torch.randn(128, 6)
    assert torch.allclose(
        reduced._gaucho_cholesky(raw)[:, 2, 1],
        torch.zeros(128), atol=1e-7)
    assert not torch.allclose(
        full._gaucho_cholesky(raw)[:, 2, 1], torch.zeros(128), atol=1e-3)


def test_reduced_dual_plane_flag_is_rejected_for_other_charts():
    for chart in ("scale_shape", "direct"):
        with pytest.raises(ValueError, match="gaucho_dual_plane_fix_rc_zero"):
            _gaucho_head(gaucho_chart=chart,
                         gaucho_dual_plane_fix_rc_zero=True)


def test_a4_and_a5_configs_differ_only_in_the_correlation():
    a4 = _compact_config("stageB_a4_rc0").model.bbox_head
    a5 = _compact_config("stageB_a5_rc").model.bbox_head
    assert a4["gaucho_chart"] == a5["gaucho_chart"] == "dual_plane"
    assert a4["gaucho_dual_plane_fix_rc_zero"] is True
    assert a5["gaucho_dual_plane_fix_rc_zero"] is False
    # Everything else must match, or the ablation measures more than one thing.
    differing = {k for k in set(a4) | set(a5)
                 if a4.get(k) != a5.get(k)}
    assert differing == {"gaucho_dual_plane_fix_rc_zero"}


def test_stage_configs_declare_the_hybrid_deviation():
    """The model trains a full SO(3) pose loss alongside the SPD shape.

    Section 8.4 of the design note says the standard spec does not.  Keeping it
    is a deliberate choice, so the configs must say so rather than claim purity.
    """
    for name in ("compact_stageB", "compact_stageC"):
        path = _CONFIG_DIR / f"{_COMPACT_PREFIX}{name.split('compact_')[1]}.py"
        text = path.read_text()
        assert "HYBRID" in text
        assert "loss_rotation" in text
        cfg = _compact_config(name.split("compact_")[1])
        assert cfg.model.bbox_head.loss_rotation.loss_weight == 5.0
