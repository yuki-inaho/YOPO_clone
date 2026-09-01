"""Losses for the GauCho-3D ellipsoid head.

Design contract
---------------
The canonical 3D state is ``Ellipsoid3D(t, Sigma)`` with ``Sigma = L L^T``.
Every loss here consumes the Cholesky factor directly wherever that keeps the
computation inverse-free, mirroring ``_gaussian_kld_from_cholesky`` in
``rotated_rtmdet_jax``: for a 3x3 factor the triangular solve is both cheaper
and better conditioned than materializing ``Sigma^{-1}``, and it degrades
gracefully as the ellipsoid approaches a sphere.

Two independent grounds of supervision are required at all times.  Optimizing
only ``2D head <-> projected 3D head`` lets both collapse onto the same wrong
ellipse, so the projection loss detaches its teacher and the caller is expected
to keep at least one of 3D direct GT, visible-depth surface, or cross-view
active alongside it.
"""

from __future__ import annotations

import math
from typing import Optional

import torch
from torch import Tensor, nn

from yopo.registry import MODELS
from .gaucho3d_geometry import (
    DEFAULT_EPS,
    obb_gaussian_to_cholesky2d,
    project_ellipsoid_dual_quadric,
    ray_ellipsoid_roots,
    sigma_from_cholesky,
)
from .projected_ellipsoid_loss import (
    compact_gaussian,
    gaussian_wasserstein_distance,
)
from .utils import weight_reduce_loss

__all__ = [
    "ellipse_kld_from_cholesky",
    "Ellipse2DKLDLoss",
    "ellipsoid_kld_from_cholesky",
    "ellipsoid_gwd",
    "Ellipsoid3DKLDLoss",
    "Ellipsoid3DGWDLoss",
    "DualQuadricProjectionGWDLoss",
    "DualConicResidualLoss",
    "RayEllipsoidSurfaceLoss",
    "EllipsoidFreeSpaceLoss",
]


def _check_reduction(reduction: str) -> None:
    if reduction not in {"none", "mean", "sum"}:
        raise ValueError(f"unsupported reduction: {reduction}")


def _bounded(distance: Tensor, tau: float) -> Tensor:
    """Map a non-negative distance into ``[0, 1)``.

    ``1 - 1/(tau + d)`` is the bounded form already used by
    :class:`GaussianGWDLoss`, so 2D and 3D terms stay on a comparable scale and
    a single outlier cannot dominate the batch gradient.
    """
    return 1.0 - 1.0 / (tau + distance)


def ellipsoid_kld_from_cholesky(
    predicted_center: Tensor,
    predicted_cholesky: Tensor,
    target_center: Tensor,
    target_cholesky: Tensor,
    *,
    include_center: bool = True,
    eps: float = DEFAULT_EPS,
) -> Tensor:
    """``D_KL(target || prediction)`` for 3D Gaussians, without any inverse.

    With ``X = L_p^{-1} L_g`` and ``y = L_p^{-1} (t_p - t_g)`` obtained by
    triangular solves,

        2 KL = ||X||_F^2 + ||y||^2 - 3 + 2 (sum log diag L_p - sum log diag L_g).

    The centre term is a Mahalanobis norm under the predicted shape, so the
    whole expression is already invariant to the metric size of the object; no
    extra normalization is applied or needed.  ``include_center=False`` drops it
    entirely for shape-only stages.
    """
    if predicted_center.shape[-1] != 3 or target_center.shape[-1] != 3:
        raise ValueError("Gaussian centers must end in three values")
    if predicted_cholesky.shape[-2:] != (3, 3) or \
            target_cholesky.shape[-2:] != (3, 3):
        raise ValueError("Cholesky factors must end in shape (3, 3)")

    solved_shape = torch.linalg.solve_triangular(
        predicted_cholesky, target_cholesky, upper=False)
    trace_term = solved_shape.square().sum(dim=(-2, -1))

    if include_center:
        displacement = (predicted_center - target_center).unsqueeze(-1)
        solved_center = torch.linalg.solve_triangular(
            predicted_cholesky, displacement, upper=False).squeeze(-1)
        center_term = solved_center.square().sum(dim=-1)
    else:
        center_term = torch.zeros_like(trace_term)

    predicted_diag = predicted_cholesky.diagonal(dim1=-2, dim2=-1).clamp_min(eps)
    target_diag = target_cholesky.diagonal(dim1=-2, dim2=-1).clamp_min(eps)
    log_determinant_ratio = 2.0 * (
        predicted_diag.log().sum(dim=-1) - target_diag.log().sum(dim=-1))

    return (0.5 * (trace_term + center_term - 3.0 + log_determinant_ratio)
            ).clamp_min(0.0)


def _symmetric_sqrt(matrix: Tensor, *, eps: float) -> Tensor:
    """SPD matrix square root via ``eigh``, evaluated without gradient.

    ``eigh`` backward divides by eigenvalue gaps, so it produces NaN for a
    sphere or spheroid -- precisely the shapes tomatoes take.  Only the target
    ellipsoid needs this root, and a target is a constant, so the whole
    decomposition is detached and the singularity never reaches a gradient.
    """
    with torch.no_grad():
        eigenvalues, eigenvectors = torch.linalg.eigh(matrix.detach())
        root = eigenvalues.clamp_min(eps).sqrt()
        return (eigenvectors @ torch.diag_embed(root)
                @ eigenvectors.transpose(-1, -2))


def ellipsoid_gwd(
    predicted_center: Tensor,
    predicted_sigma: Tensor,
    target_center: Tensor,
    target_sigma: Tensor,
    *,
    normalize: bool = True,
    include_center: bool = True,
    eps: float = DEFAULT_EPS,
) -> Tensor:
    """Squared 3D Gaussian Wasserstein distance (closed form, 3x3).

    ``W2^2 = ||t_p - t_g||^2 + tr(Sigma_p + Sigma_g - 2 (Sg^{1/2} Sp Sg^{1/2})^{1/2})``.
    When ``normalize`` is set the result is divided by ``r_g^2`` with
    ``r_g = (det Sigma_g)^{1/6}``, which removes the metric size of the object
    so a 2 cm and a 10 cm fruit contribute comparable gradients.
    """
    center_distance = (predicted_center - target_center).square().sum(dim=-1)
    if not include_center:
        center_distance = torch.zeros_like(center_distance)
    target_root = _symmetric_sqrt(target_sigma, eps=eps)
    cross = target_root @ predicted_sigma @ target_root
    # ``clamp_min(eps)`` rather than ``clamp_min(0)``: the derivative of
    # ``sqrt`` is unbounded at zero, and a matched prediction sits exactly
    # there.
    cross_eigenvalues = torch.linalg.eigvalsh(cross).clamp_min(eps)
    trace_term = (
        predicted_sigma.diagonal(dim1=-2, dim2=-1).sum(dim=-1)
        + target_sigma.diagonal(dim1=-2, dim2=-1).sum(dim=-1)
        - 2.0 * cross_eigenvalues.sqrt().sum(dim=-1)
    ).clamp_min(0.0)
    distance = center_distance + trace_term
    if normalize:
        radius_squared = torch.linalg.det(
            target_sigma).clamp_min(eps ** 6).pow(1.0 / 3.0)
        distance = distance / radius_squared.clamp_min(eps)
    return distance


class _WeightedLoss(nn.Module):
    """Shared validity bookkeeping for the GauCho-3D losses."""

    def __init__(
        self,
        loss_weight: float,
        reduction: str,
        tau: float,
        fail_on_invalid: bool,
        eps: float,
    ) -> None:
        super().__init__()
        _check_reduction(reduction)
        if tau < 1.0:
            raise ValueError("tau must be >= 1 for bounded postprocessing")
        if eps <= 0.0:
            raise ValueError("eps must be positive")
        self.loss_weight = float(loss_weight)
        self.reduction = reduction
        self.tau = float(tau)
        self.fail_on_invalid = bool(fail_on_invalid)
        self.eps = float(eps)
        self.last_invalid_count = None
        self.last_positive_count = None

    def _finalize(
        self,
        per_sample: Tensor,
        weight: Tensor,
        valid: Tensor,
        reduction: str,
        avg_factor: Optional[int],
        name: str,
    ) -> Tensor:
        invalid_count = (~valid).sum()
        self.last_invalid_count = invalid_count.detach()
        if self.fail_on_invalid:
            # ``.item()`` forces a device sync, so it is only paid when the
            # caller has asked for a hard failure on invalid geometry.
            invalid = int(invalid_count.item())
            if invalid:
                raise RuntimeError(
                    f"{name} received {invalid}/{int(valid.numel())} invalid "
                    "positive samples")
        masked_weight = weight * valid.to(weight.dtype)
        return self.loss_weight * weight_reduce_loss(
            per_sample, masked_weight, reduction=reduction,
            avg_factor=avg_factor)


@MODELS.register_module()
class Ellipsoid3DKLDLoss(_WeightedLoss):
    """Primary 3D geometry objective: inverse-free Cholesky KLD.

    Both prediction and target are supplied as ``(center, cholesky)``.  The
    target factor comes from ``cholesky(Sigma_g)`` where ``Sigma_g`` is built
    from the annotation's rotation and extent; the network itself never
    regresses a rotation.
    """

    def __init__(
        self,
        loss_weight: float = 1.0,
        reduction: str = "mean",
        tau: float = 1.0,
        include_center: bool = True,
        fail_on_invalid: bool = True,
        eps: float = DEFAULT_EPS,
    ) -> None:
        super().__init__(loss_weight, reduction, tau, fail_on_invalid, eps)
        self.include_center = bool(include_center)

    def forward(
        self,
        predicted_center: Tensor,
        predicted_cholesky: Tensor,
        target_center: Tensor,
        target_cholesky: Tensor,
        weight: Optional[Tensor] = None,
        avg_factor: Optional[int] = None,
        reduction_override: Optional[str] = None,
    ) -> Tensor:
        reduction = reduction_override or self.reduction
        _check_reduction(reduction)
        if weight is None:
            weight = predicted_center.new_ones(predicted_center.shape[:-1])
        if weight.ndim > predicted_center.ndim - 1:
            weight = weight.mean(dim=-1)
        self.last_positive_count = (weight > 0).sum().detach()

        distance = ellipsoid_kld_from_cholesky(
            predicted_center, predicted_cholesky, target_center,
            target_cholesky, include_center=self.include_center, eps=self.eps)
        valid = (
            torch.isfinite(distance)
            & torch.isfinite(predicted_cholesky).all(dim=-1).all(dim=-1)
            & torch.isfinite(target_cholesky).all(dim=-1).all(dim=-1)
            & (predicted_cholesky.diagonal(dim1=-2, dim2=-1) > 0).all(dim=-1)
            & (target_cholesky.diagonal(dim1=-2, dim2=-1) > 0).all(dim=-1)
        )
        safe_distance = torch.where(
            valid, distance, torch.zeros_like(distance))
        per_sample = _bounded(safe_distance, self.tau)
        return self._finalize(
            per_sample, weight, valid, reduction, avg_factor,
            "Ellipsoid3DKLDLoss")


@MODELS.register_module()
class Ellipsoid3DGWDLoss(_WeightedLoss):
    """Secondary 3D geometry objective: normalized Gaussian Wasserstein."""

    def __init__(
        self,
        loss_weight: float = 1.0,
        reduction: str = "mean",
        tau: float = 1.0,
        normalize: bool = True,
        include_center: bool = True,
        fail_on_invalid: bool = True,
        eps: float = DEFAULT_EPS,
    ) -> None:
        super().__init__(loss_weight, reduction, tau, fail_on_invalid, eps)
        self.normalize = bool(normalize)
        self.include_center = bool(include_center)

    def forward(
        self,
        predicted_center: Tensor,
        predicted_cholesky: Tensor,
        target_center: Tensor,
        target_cholesky: Tensor,
        weight: Optional[Tensor] = None,
        avg_factor: Optional[int] = None,
        reduction_override: Optional[str] = None,
    ) -> Tensor:
        reduction = reduction_override or self.reduction
        _check_reduction(reduction)
        if weight is None:
            weight = predicted_center.new_ones(predicted_center.shape[:-1])
        if weight.ndim > predicted_center.ndim - 1:
            weight = weight.mean(dim=-1)
        self.last_positive_count = (weight > 0).sum().detach()

        predicted_sigma = sigma_from_cholesky(predicted_cholesky)
        target_sigma = sigma_from_cholesky(target_cholesky)
        distance = ellipsoid_gwd(
            predicted_center, predicted_sigma, target_center, target_sigma,
            normalize=self.normalize, include_center=self.include_center,
            eps=self.eps)
        valid = torch.isfinite(distance) & (distance >= 0)
        safe_distance = torch.where(
            valid, distance, torch.zeros_like(distance))
        # Bound the *squared* distance, as in the design note.  Taking a square
        # root first would put a cusp with infinite slope exactly at a perfect
        # match, which is where a converged model lives.
        per_sample = _bounded(safe_distance, self.tau)
        return self._finalize(
            per_sample, weight, valid, reduction, avg_factor,
            "Ellipsoid3DGWDLoss")


@MODELS.register_module()
class DualQuadricProjectionGWDLoss(_WeightedLoss):
    """2D--3D consistency through the exact dual-quadric silhouette.

    The 3D ellipsoid is projected with ``C* = P Q* P^T`` and compared against
    the 2D observation as a Gaussian.  The 2D side is detached by default: at
    the point this loss is switched on the 2D branch is the better-trained of
    the two, and letting gradients flow both ways is exactly the head-collusion
    failure the design forbids.
    """

    def __init__(
        self,
        loss_weight: float = 1.0,
        reduction: str = "mean",
        tau: float = 1.0,
        normalize: bool = True,
        include_center: bool = True,
        detach_target: bool = True,
        fail_on_invalid: bool = True,
        eps: float = DEFAULT_EPS,
    ) -> None:
        super().__init__(loss_weight, reduction, tau, fail_on_invalid, eps)
        self.normalize = bool(normalize)
        self.include_center = bool(include_center)
        self.detach_target = bool(detach_target)

    def forward(
        self,
        predicted_center_3d: Tensor,
        predicted_cholesky: Tensor,
        target_gaussians: Tensor,
        intrinsics: Tensor,
        weight: Optional[Tensor] = None,
        avg_factor: Optional[int] = None,
        reduction_override: Optional[str] = None,
    ) -> Tensor:
        reduction = reduction_override or self.reduction
        _check_reduction(reduction)
        if target_gaussians.shape[-1] != 5:
            raise ValueError("target_gaussians must end in five values")
        if weight is None:
            weight = predicted_center_3d.new_ones(
                predicted_center_3d.shape[:-1])
        if weight.ndim > predicted_center_3d.ndim - 1:
            weight = weight.mean(dim=-1)
        self.last_positive_count = (weight > 0).sum().detach()

        targets = target_gaussians.detach() if self.detach_target \
            else target_gaussians
        sigma = sigma_from_cholesky(predicted_cholesky)
        mean, shape, projection_valid = project_ellipsoid_dual_quadric(
            predicted_center_3d, sigma, intrinsics, eps=self.eps)
        predicted_compact = compact_gaussian(mean, shape)
        distance, gaussian_valid = gaussian_wasserstein_distance(
            predicted_compact, targets, normalize=self.normalize,
            include_center=self.include_center, eps=self.eps)
        valid = projection_valid & gaussian_valid
        safe_distance = torch.where(
            valid, distance, torch.zeros_like(distance))
        per_sample = _bounded(safe_distance, self.tau)
        return self._finalize(
            per_sample, weight, valid, reduction, avg_factor,
            "DualQuadricProjectionGWDLoss")


@MODELS.register_module()
class DualConicResidualLoss(_WeightedLoss):
    """Auxiliary Frobenius residual between canonicalized dual conics.

    Both conics are normalized by their ``(3, 3)`` entry first, because a dual
    conic is only defined up to projective scale and sign; comparing raw
    matrices would penalize a gauge choice rather than a geometric difference.
    """

    def __init__(
        self,
        loss_weight: float = 1.0,
        reduction: str = "mean",
        tau: float = 1.0,
        detach_target: bool = True,
        fail_on_invalid: bool = True,
        eps: float = DEFAULT_EPS,
    ) -> None:
        super().__init__(loss_weight, reduction, tau, fail_on_invalid, eps)
        self.detach_target = bool(detach_target)

    @staticmethod
    def _canonicalize(conic: Tensor, eps: float) -> tuple[Tensor, Tensor]:
        scale = conic[..., 2, 2]
        valid = scale.abs() > eps
        safe = torch.where(valid, scale, torch.ones_like(scale))
        return conic / safe[..., None, None], valid

    def forward(
        self,
        predicted_conic: Tensor,
        target_conic: Tensor,
        weight: Optional[Tensor] = None,
        avg_factor: Optional[int] = None,
        reduction_override: Optional[str] = None,
    ) -> Tensor:
        reduction = reduction_override or self.reduction
        _check_reduction(reduction)
        if predicted_conic.shape[-2:] != (3, 3) or \
                target_conic.shape[-2:] != (3, 3):
            raise ValueError("conics must end in shape (3, 3)")
        if weight is None:
            weight = predicted_conic.new_ones(predicted_conic.shape[:-2])
        self.last_positive_count = (weight > 0).sum().detach()

        target = target_conic.detach() if self.detach_target else target_conic
        predicted_canonical, predicted_valid = self._canonicalize(
            predicted_conic, self.eps)
        target_canonical, target_valid = self._canonicalize(target, self.eps)
        residual = (predicted_canonical - target_canonical).square().sum(
            dim=(-2, -1))
        valid = predicted_valid & target_valid & torch.isfinite(residual)
        safe_residual = torch.where(
            valid, residual, torch.zeros_like(residual))
        per_sample = _bounded(safe_residual, self.tau)
        return self._finalize(
            per_sample, weight, valid, reduction, avg_factor,
            "DualConicResidualLoss")


@MODELS.register_module()
class RayEllipsoidSurfaceLoss(_WeightedLoss):
    """Pull the ellipsoid's near surface onto observed visible-fruit depth.

    Only pixels inside the visible fruit mask may contribute.  Leaf, branch,
    background and neighbouring-fruit depth would otherwise shrink the ellipsoid
    toward the occluder, and the amodal hidden region has no depth evidence at
    all, so no equality is imposed there.
    """

    def __init__(
        self,
        loss_weight: float = 1.0,
        reduction: str = "mean",
        beta: float = 0.02,
        depth_scale: float = 0.05,
        fail_on_invalid: bool = False,
        eps: float = DEFAULT_EPS,
    ) -> None:
        super().__init__(loss_weight, reduction, 1.0, fail_on_invalid, eps)
        if beta <= 0.0 or depth_scale <= 0.0:
            raise ValueError("beta and depth_scale must be positive")
        self.beta = float(beta)
        self.depth_scale = float(depth_scale)

    def forward(
        self,
        predicted_center: Tensor,
        predicted_cholesky: Tensor,
        ray_directions: Tensor,
        observed_depth: Tensor,
        pixel_weight: Tensor,
        avg_factor: Optional[int] = None,
        reduction_override: Optional[str] = None,
    ) -> Tensor:
        reduction = reduction_override or self.reduction
        _check_reduction(reduction)
        if ray_directions.shape[-1] != 3:
            raise ValueError("ray_directions must end in three values")
        if observed_depth.shape != ray_directions.shape[:-1] or \
                pixel_weight.shape != ray_directions.shape[:-1]:
            raise ValueError(
                "observed_depth and pixel_weight must match the ray batch")

        center = predicted_center.unsqueeze(-2).expand_as(ray_directions)
        cholesky = predicted_cholesky.unsqueeze(-3).expand(
            *ray_directions.shape[:-1], 3, 3)
        z_near, _, hit = ray_ellipsoid_roots(
            ray_directions, center, cholesky, eps=self.eps)
        depth_valid = torch.isfinite(observed_depth) & (observed_depth > self.eps)
        valid = hit & depth_valid & (pixel_weight > 0)
        residual = (z_near - observed_depth) / self.depth_scale
        safe_residual = torch.where(
            valid, residual, torch.zeros_like(residual))
        absolute = safe_residual.abs()
        huber = torch.where(
            absolute < self.beta,
            0.5 * absolute.square() / self.beta,
            absolute - 0.5 * self.beta,
        )
        if reduction == "mean" and avg_factor is None:
            # Eq (45) is a weighted mean over *contributing* rays,
            # ``sum_j w_j rho_H / (sum_j w_j + eps)``.  Deferring to the generic
            # mean would divide by the total ray count, so masking an occluder
            # would quietly rescale the whole term instead of reweighting it --
            # the more rays the visible mask excludes, the smaller the loss.
            invalid_count = (~valid).sum()
            self.last_invalid_count = invalid_count.detach()
            if self.fail_on_invalid and int(invalid_count.item()):
                raise RuntimeError(
                    "RayEllipsoidSurfaceLoss received "
                    f"{int(invalid_count.item())}/{int(valid.numel())} invalid "
                    "rays")
            weight = pixel_weight * valid.to(pixel_weight.dtype)
            total = weight.sum()
            return self.loss_weight * (
                (huber * weight).sum() / (total + self.eps))
        return self._finalize(
            huber, pixel_weight, valid, reduction, avg_factor,
            "RayEllipsoidSurfaceLoss")


@MODELS.register_module()
class EllipsoidFreeSpaceLoss(_WeightedLoss):
    """Penalize ellipsoid mass in front of the observed visible surface.

    Samples taken at ``z < z_obs - margin`` along a visible ray are known empty;
    any of them falling inside the ellipsoid is a violation.  The penalty is
    one-sided so it can only shrink an over-large ellipsoid, never inflate one.
    """

    def __init__(
        self,
        loss_weight: float = 1.0,
        reduction: str = "mean",
        num_samples: int = 4,
        margin: float = 0.01,
        fail_on_invalid: bool = False,
        eps: float = DEFAULT_EPS,
    ) -> None:
        super().__init__(loss_weight, reduction, 1.0, fail_on_invalid, eps)
        if num_samples < 1:
            raise ValueError("num_samples must be positive")
        if margin < 0.0 or not math.isfinite(margin):
            raise ValueError("margin must be finite and non-negative")
        self.num_samples = int(num_samples)
        self.margin = float(margin)

    def forward(
        self,
        predicted_center: Tensor,
        predicted_cholesky: Tensor,
        ray_directions: Tensor,
        observed_depth: Tensor,
        pixel_weight: Tensor,
        avg_factor: Optional[int] = None,
        reduction_override: Optional[str] = None,
    ) -> Tensor:
        reduction = reduction_override or self.reduction
        _check_reduction(reduction)
        if ray_directions.shape[-1] != 3:
            raise ValueError("ray_directions must end in three values")

        free_depth = (observed_depth - self.margin).clamp_min(0.0)
        fractions = torch.linspace(
            1.0 / (self.num_samples + 1), self.num_samples / (self.num_samples + 1.0),
            self.num_samples, device=ray_directions.device,
            dtype=ray_directions.dtype)
        # (..., rays, samples)
        depths = free_depth.unsqueeze(-1) * fractions
        points = depths.unsqueeze(-1) * ray_directions.unsqueeze(-2)
        center = predicted_center[..., None, None, :]
        delta = (points - center).unsqueeze(-1)
        cholesky = predicted_cholesky[..., None, None, :, :].expand(
            *delta.shape[:-2], 3, 3)
        solved = torch.linalg.solve_triangular(
            cholesky, delta, upper=False).squeeze(-1)
        mahalanobis = solved.square().sum(dim=-1)
        violation = (1.0 - mahalanobis).clamp_min(0.0)

        depth_valid = torch.isfinite(observed_depth) & (observed_depth > self.margin)
        valid = (depth_valid & (pixel_weight > 0)).unsqueeze(-1).expand_as(
            violation)
        weight = pixel_weight.unsqueeze(-1).expand_as(violation)
        return self._finalize(
            violation, weight, valid, reduction, avg_factor,
            "EllipsoidFreeSpaceLoss")


def ellipse_kld_from_cholesky(
    predicted_mean: Tensor,
    predicted_cholesky: Tensor,
    target_mean: Tensor,
    target_cholesky: Tensor,
    *,
    eps: float = DEFAULT_EPS,
) -> Tensor:
    """``D_KL(target || prediction)`` for 2D Gaussians, without any inverse.

    The 2x2 case is written out as explicit forward substitutions rather than a
    batched solve: it is cheaper, and it makes the absence of any angle,
    axis-sort, or covariance-inverse step obvious by inspection.
    """
    if predicted_mean.shape[-1] != 2 or target_mean.shape[-1] != 2:
        raise ValueError("Gaussian means must end in two values")
    if predicted_cholesky.shape[-2:] != (2, 2) or \
            target_cholesky.shape[-2:] != (2, 2):
        raise ValueError("Cholesky factors must end in shape (2, 2)")

    pred_l11 = predicted_cholesky[..., 0, 0].clamp_min(eps)
    pred_l21 = predicted_cholesky[..., 1, 0]
    pred_l22 = predicted_cholesky[..., 1, 1].clamp_min(eps)
    target_l11 = target_cholesky[..., 0, 0].clamp_min(eps)
    target_l21 = target_cholesky[..., 1, 0]
    target_l22 = target_cholesky[..., 1, 1].clamp_min(eps)

    # X = L_pred^{-1} L_target; the (0, 1) entry is identically zero.
    x00 = target_l11 / pred_l11
    x10 = (target_l21 - pred_l21 * x00) / pred_l22
    x11 = target_l22 / pred_l22
    trace_term = x00.square() + x10.square() + x11.square()

    displacement = predicted_mean - target_mean
    y0 = displacement[..., 0] / pred_l11
    y1 = (displacement[..., 1] - pred_l21 * y0) / pred_l22
    center_term = y0.square() + y1.square()

    log_determinant_ratio = 2.0 * (
        pred_l11.log() + pred_l22.log()
        - target_l11.log() - target_l22.log())
    return (0.5 * (trace_term + center_term - 2.0 + log_determinant_ratio)
            ).clamp_min(0.0)


@MODELS.register_module()
class Ellipse2DKLDLoss(_WeightedLoss):
    """Angle-free 2D GauCho objective for the amodal ellipse head.

    Targets arrive as the stored compact ``(cx, cy, xx, xy, yy)`` Gaussian of
    the annotated oriented box.  Degenerate annotations are dropped through the
    validity mask instead of being repaired.
    """

    def __init__(
        self,
        loss_weight: float = 1.0,
        reduction: str = "mean",
        tau: float = 1.0,
        include_center: bool = True,
        fail_on_invalid: bool = False,
        eps: float = DEFAULT_EPS,
    ) -> None:
        super().__init__(loss_weight, reduction, tau, fail_on_invalid, eps)
        self.include_center = bool(include_center)

    def forward(
        self,
        predicted_mean: Tensor,
        predicted_cholesky: Tensor,
        target_gaussians: Tensor,
        weight: Optional[Tensor] = None,
        avg_factor: Optional[int] = None,
        reduction_override: Optional[str] = None,
    ) -> Tensor:
        reduction = reduction_override or self.reduction
        _check_reduction(reduction)
        if weight is None:
            weight = predicted_mean.new_ones(predicted_mean.shape[:-1])
        if weight.ndim > predicted_mean.ndim - 1:
            weight = weight.mean(dim=-1)
        self.last_positive_count = (weight > 0).sum().detach()

        target_mean, target_cholesky, target_valid = obb_gaussian_to_cholesky2d(
            target_gaussians, eps=self.eps)
        center = predicted_mean if self.include_center else target_mean
        distance = ellipse_kld_from_cholesky(
            center, predicted_cholesky, target_mean, target_cholesky,
            eps=self.eps)
        valid = (
            target_valid
            & torch.isfinite(distance)
            & (predicted_cholesky.diagonal(dim1=-2, dim2=-1) > 0).all(dim=-1)
        )
        safe_distance = torch.where(
            valid, distance, torch.zeros_like(distance))
        per_sample = _bounded(safe_distance, self.tau)
        return self._finalize(
            per_sample, weight, valid, reduction, avg_factor,
            "Ellipse2DKLDLoss")
