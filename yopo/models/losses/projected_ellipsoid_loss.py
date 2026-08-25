"""Differentiable 3D ellipsoid to 2D OBB projection consistency loss."""

from __future__ import annotations

import math
from typing import Optional

import torch
from torch import Tensor, nn

from yopo.registry import MODELS
from .utils import weight_reduce_loss


def _rotation_6d_to_matrix(rotation: Tensor, eps: float = 1e-8) -> Tensor:
    """Convert column-major 6D rotations to right-handed matrices."""
    first, second = torch.split(rotation, 3, dim=-1)
    first = first / first.norm(dim=-1, keepdim=True).clamp_min(eps)
    second = second - (first * second).sum(
        dim=-1, keepdim=True) * first
    second = second / second.norm(dim=-1, keepdim=True).clamp_min(eps)
    third = torch.cross(first, second, dim=-1)
    return torch.stack((first, second, third), dim=-1)


def compact_gaussian(center: Tensor, sigma: Tensor) -> Tensor:
    """Pack ``(center, symmetric 2x2 shape)`` as ``(x,y,xx,xy,yy)``."""
    return torch.stack(
        (center[..., 0], center[..., 1], sigma[..., 0, 0],
         sigma[..., 0, 1], sigma[..., 1, 1]),
        dim=-1,
    )


def expand_compact_gaussian(compact: Tensor) -> tuple[Tensor, Tensor]:
    """Unpack ``(x,y,xx,xy,yy)`` into center and symmetric shape."""
    if compact.shape[-1] != 5:
        raise ValueError(
            f"compact Gaussian must end in 5 values, got {compact.shape}")
    sigma = torch.stack(
        (compact[..., 2], compact[..., 3],
         compact[..., 3], compact[..., 4]),
        dim=-1,
    ).reshape(*compact.shape[:-1], 2, 2)
    return compact[..., :2], sigma


def gaussian_wasserstein_distance(
    predicted: Tensor,
    target: Tensor,
    *,
    normalize: bool = True,
    include_center: bool = True,
    eps: float = 1e-7,
) -> tuple[Tensor, Tensor]:
    """Closed-form 2D Gaussian Wasserstein distance and validity mask."""
    predicted_center, predicted_sigma = expand_compact_gaussian(predicted)
    target_center, target_sigma = expand_compact_gaussian(target)
    predicted_eigenvalues = torch.linalg.eigvalsh(predicted_sigma)
    target_eigenvalues = torch.linalg.eigvalsh(target_sigma)
    valid = (
        torch.isfinite(predicted_center).all(dim=-1)
        & torch.isfinite(predicted_sigma).all(dim=(-2, -1))
        & torch.isfinite(target_center).all(dim=-1)
        & torch.isfinite(target_sigma).all(dim=(-2, -1))
        & (predicted_eigenvalues > eps).all(dim=-1)
        & (target_eigenvalues > eps).all(dim=-1)
    )
    xy_distance = (predicted_center - target_center).square().sum(dim=-1)
    if not include_center:
        xy_distance = torch.zeros_like(xy_distance)
    trace_product = (predicted_sigma @ target_sigma).diagonal(
        dim1=-2, dim2=-1).sum(dim=-1)
    determinant_product = (
        predicted_sigma.det() * target_sigma.det()).clamp_min(eps**4)
    covariance_distance = (
        predicted_sigma.diagonal(dim1=-2, dim2=-1).sum(dim=-1)
        + target_sigma.diagonal(dim1=-2, dim2=-1).sum(dim=-1)
        - 2.0 * (trace_product + 2.0 * determinant_product.sqrt())
        .clamp_min(eps).sqrt())
    distance = (xy_distance + covariance_distance).clamp_min(eps).sqrt()
    if normalize:
        scale = 2.0 * determinant_product.pow(0.125)
        distance = distance / scale.clamp_min(eps)
    return distance, valid


def normalized_gaussian_anisotropy_weights(
    targets: Tensor,
    *,
    power: float = 1.0,
    eps: float = 1e-7,
) -> tuple[Tensor, Tensor, Tensor]:
    """Return unit-mean continuous weights from target ellipse anisotropy.

    Anisotropy is ``(lambda_max-lambda_min)/(lambda_max+lambda_min)``.
    Normalizing over valid targets preserves the mean projection-loss scale.
    ``power=0`` is the exact legacy uniform-weighting behavior.
    """
    power = float(power)
    if not math.isfinite(power) or power < 0.0:
        raise ValueError(f"power must be finite and non-negative, got {power}")
    if eps <= 0.0:
        raise ValueError("eps must be positive")
    target_center, target_sigma = expand_compact_gaussian(targets)
    eigenvalues = torch.linalg.eigvalsh(target_sigma)
    valid = (
        torch.isfinite(target_center).all(dim=-1)
        & torch.isfinite(target_sigma).all(dim=(-2, -1))
        & (eigenvalues > eps).all(dim=-1)
    )
    safe_eigenvalues = torch.nan_to_num(eigenvalues).clamp_min(eps)
    anisotropy = (
        (safe_eigenvalues[..., 1] - safe_eigenvalues[..., 0])
        / (safe_eigenvalues.sum(dim=-1).clamp_min(eps))
    ).clamp(min=0.0, max=1.0)
    raw_weight = (
        torch.ones_like(anisotropy) if power == 0.0
        else anisotropy.pow(power)
    )
    raw_weight = raw_weight * valid.to(raw_weight.dtype)
    valid_count = valid.sum().to(raw_weight.dtype)
    normalizer = raw_weight.sum() / valid_count.clamp_min(1.0)
    weights = raw_weight / normalizer.clamp_min(eps)
    return weights, anisotropy, valid


@MODELS.register_module()
class GaussianGWDLoss(nn.Module):
    """Bounded GWD for compact 2D Gaussian predictions and targets."""

    def __init__(
        self,
        loss_weight: float = 1.0,
        reduction: str = "mean",
        tau: float = 1.0,
        normalize: bool = True,
        include_center: bool = False,
        fail_on_invalid: bool = True,
        eps: float = 1e-7,
    ) -> None:
        super().__init__()
        if reduction not in {"none", "mean", "sum"}:
            raise ValueError(f"unsupported reduction: {reduction}")
        if tau < 1.0:
            raise ValueError("tau must be >= 1 for bounded GWD postprocessing")
        self.loss_weight = float(loss_weight)
        self.reduction = reduction
        self.tau = float(tau)
        self.normalize = bool(normalize)
        self.include_center = bool(include_center)
        self.fail_on_invalid = bool(fail_on_invalid)
        self.eps = float(eps)

    def forward(
        self,
        predicted: Tensor,
        target: Tensor,
        weight: Optional[Tensor] = None,
        avg_factor: Optional[int] = None,
        reduction_override: Optional[str] = None,
    ) -> Tensor:
        if predicted.shape != target.shape or predicted.shape[-1] != 5:
            raise ValueError(
                "Gaussian GWD expects equal (..., 5) tensors, got "
                f"{predicted.shape} and {target.shape}")
        reduction = reduction_override or self.reduction
        if weight is None:
            weight = predicted.new_ones(predicted.shape[:-1])
        if weight.ndim == predicted.ndim:
            weight = weight.mean(dim=-1)
        positive = weight > 0
        if not torch.any(positive):
            return predicted.sum() * 0.0
        distance, valid = gaussian_wasserstein_distance(
            predicted[positive], target[positive], normalize=self.normalize,
            include_center=self.include_center, eps=self.eps)
        if self.fail_on_invalid and bool(torch.any(~valid).item()):
            raise RuntimeError(
                "Gaussian GWD received "
                f"{int((~valid).sum().item())}/{int(valid.numel())} invalid "
                "positive pairs")
        selected_weight = weight[positive] * valid.to(weight.dtype)
        loss = 1.0 - 1.0 / (self.tau + distance)
        return self.loss_weight * weight_reduce_loss(
            loss, selected_weight, reduction=reduction,
            avg_factor=avg_factor)


def project_ellipsoid_to_gaussian(
    translation: Tensor,
    rotation: Tensor,
    size: Tensor,
    intrinsic: Tensor,
    eps: float = 1e-7,
) -> tuple[Tensor, Tensor, Tensor]:
    """Project camera-frame ellipsoids exactly through their dual quadrics.

    Args:
        translation: Camera-frame centers with shape ``(N, 3)``.
        rotation: 6D rotations ``(N, 6)`` or matrices ``(N, 3, 3)``.
        size: Full local axis lengths in metres, shape ``(N, 3)``.
        intrinsic: Camera matrices, shape ``(N, 3, 3)``.
        eps: Positive numerical boundary.

    Returns:
        Pixel centers, 2x2 ellipse shape matrices, and a validity mask.
    """
    if translation.ndim != 2 or translation.shape[-1] != 3:
        raise ValueError(
            f"translation must have shape (N, 3), got {translation.shape}")
    if size.shape != translation.shape:
        raise ValueError(
            f"size must match translation shape, got {size.shape} and "
            f"{translation.shape}")
    if intrinsic.shape != (len(translation), 3, 3):
        raise ValueError(
            f"intrinsic must have shape (N, 3, 3), got {intrinsic.shape}")
    if rotation.shape == (len(translation), 6):
        rotation_matrix = _rotation_6d_to_matrix(rotation, eps=eps)
    elif rotation.shape == (len(translation), 3, 3):
        rotation_matrix = rotation
    else:
        raise ValueError(
            "rotation must have shape (N, 6) or (N, 3, 3), got "
            f"{rotation.shape}")

    input_finite = (
        torch.isfinite(translation).all(dim=-1)
        & torch.isfinite(rotation_matrix).all(dim=(-2, -1))
        & torch.isfinite(size).all(dim=-1)
        & torch.isfinite(intrinsic).all(dim=(-2, -1))
    )
    input_valid = (
        input_finite
        & (size > 0).all(dim=-1)
        & (translation[:, 2] > size.max(dim=-1).values * 0.5 + eps)
    )

    # Keep invalid samples numerically evaluable so the caller can zero their
    # weights and report them without poisoning the remaining batch.
    safe_size = torch.nan_to_num(size, nan=eps, posinf=eps, neginf=eps).clamp_min(eps)
    safe_translation = torch.nan_to_num(
        translation, nan=0.0, posinf=0.0, neginf=0.0)
    minimum_z = safe_size.max(dim=-1).values * 0.5 + eps
    safe_translation = torch.cat(
        (safe_translation[:, :2],
         torch.maximum(safe_translation[:, 2], minimum_z).unsqueeze(-1)),
        dim=-1,
    )
    safe_rotation = torch.nan_to_num(rotation_matrix)
    safe_intrinsic = torch.nan_to_num(intrinsic)

    radii_squared = (safe_size * 0.5).square()
    shape = safe_rotation @ torch.diag_embed(radii_squared) @ \
        safe_rotation.transpose(-1, -2)
    dual_conic = safe_intrinsic @ (
        shape - safe_translation.unsqueeze(-1)
        * safe_translation.unsqueeze(-2)) @ safe_intrinsic.transpose(-1, -2)
    conic, conic_info = torch.linalg.inv_ex(dual_conic, check_errors=False)
    matrix = conic[:, :2, :2]
    vector = conic[:, :2, 2]
    matrix_inv, matrix_info = torch.linalg.inv_ex(matrix, check_errors=False)
    center = -(matrix_inv @ vector.unsqueeze(-1)).squeeze(-1)
    centered_constant = conic[:, 2, 2] - torch.einsum(
        "ni,nij,nj->n", vector, matrix_inv, vector)
    sigma = -centered_constant[:, None, None] * matrix_inv
    sigma = (sigma + sigma.transpose(-1, -2)) * 0.5
    sigma_eigenvalues = torch.linalg.eigvalsh(sigma)

    output_valid = (
        (conic_info == 0)
        & (matrix_info == 0)
        & torch.isfinite(center).all(dim=-1)
        & torch.isfinite(sigma).all(dim=(-2, -1))
        & (sigma_eigenvalues > eps).all(dim=-1)
    )
    return center, sigma, input_valid & output_valid


@MODELS.register_module()
class ProjectedEllipsoidGWDLoss(nn.Module):
    """GWD between a projected 3D ellipsoid and an observed 2D OBB Gaussian.

    The default rotation-only mode detaches the already trained 2D center,
    depth and size branches, preventing them from absorbing rotation error.
    """

    def __init__(
        self,
        loss_weight: float = 1.0,
        reduction: str = "mean",
        tau: float = 1.0,
        normalize: bool = True,
        detach_center: bool = True,
        detach_depth: bool = True,
        detach_size: bool = True,
        target_anisotropy_power: float = 0.0,
        fail_on_invalid: bool = True,
        eps: float = 1e-7,
    ) -> None:
        super().__init__()
        if reduction not in {"none", "mean", "sum"}:
            raise ValueError(f"unsupported reduction: {reduction}")
        if tau < 1.0:
            raise ValueError("tau must be >= 1 for bounded GWD postprocessing")
        if eps <= 0.0:
            raise ValueError("eps must be positive")
        if not math.isfinite(target_anisotropy_power) or \
                target_anisotropy_power < 0.0:
            raise ValueError(
                "target_anisotropy_power must be finite and non-negative")
        self.loss_weight = float(loss_weight)
        self.reduction = reduction
        self.tau = float(tau)
        self.normalize = bool(normalize)
        self.detach_center = bool(detach_center)
        self.detach_depth = bool(detach_depth)
        self.detach_size = bool(detach_size)
        self.target_anisotropy_power = float(target_anisotropy_power)
        self.fail_on_invalid = bool(fail_on_invalid)
        self.eps = float(eps)
        self.last_invalid_count = None
        self.last_positive_count = None
        self.last_target_anisotropy_mean = None
        self.last_anisotropy_weight_max = None

    def forward(
        self,
        centers_2d_px: Tensor,
        depth: Tensor,
        rotations: Tensor,
        sizes: Tensor,
        target_gaussians: Tensor,
        intrinsics: Tensor,
        weight: Optional[Tensor] = None,
        avg_factor: Optional[int] = None,
        reduction_override: Optional[str] = None,
    ) -> Tensor:
        reduction = reduction_override or self.reduction
        if reduction not in {"none", "mean", "sum"}:
            raise ValueError(f"unsupported reduction: {reduction}")
        if centers_2d_px.shape[-1] != 2 or depth.shape[-1] != 1 or \
                rotations.shape[-1] != 6 or sizes.shape[-1] != 3 or \
                target_gaussians.shape[-1] != 5:
            raise ValueError(
                "projection loss expects (...,2), (...,1), (...,6), "
                "(...,3), (...,5) inputs")
        count = len(centers_2d_px)
        if any(len(value) != count for value in
               (depth, rotations, sizes, target_gaussians, intrinsics)):
            raise ValueError("all projection loss inputs must share N")
        if weight is None:
            weight = centers_2d_px.new_ones(count)
        if weight.ndim > 1:
            weight = weight.mean(dim=-1)
        positive = weight > 0
        self.last_positive_count = positive.sum().detach()
        if not torch.any(positive):
            self.last_invalid_count = positive.sum().detach()
            self.last_target_anisotropy_mean = positive.sum().detach()
            self.last_anisotropy_weight_max = positive.sum().detach()
            return rotations.sum() * 0.0

        centers = centers_2d_px[positive]
        selected_depth = depth[positive]
        selected_sizes = sizes[positive]
        if self.detach_center:
            centers = centers.detach()
        if self.detach_depth:
            selected_depth = selected_depth.detach()
        if self.detach_size:
            selected_sizes = selected_sizes.detach()
        selected_rotations = rotations[positive]
        selected_targets = target_gaussians[positive]
        selected_intrinsics = intrinsics[positive]
        selected_weight = weight[positive]

        computation_dtype = (
            torch.float64 if selected_rotations.dtype == torch.float64
            else torch.float32)
        device_type = rotations.device.type
        context = torch.autocast(device_type=device_type, enabled=False)
        with context:
            centers = centers.to(computation_dtype)
            selected_depth = selected_depth.to(computation_dtype)
            selected_sizes = selected_sizes.to(computation_dtype)
            selected_rotations = selected_rotations.to(computation_dtype)
            selected_targets = selected_targets.to(computation_dtype)
            selected_intrinsics = selected_intrinsics.to(computation_dtype)
            selected_weight = selected_weight.to(computation_dtype)

            homogeneous = torch.cat(
                (centers, torch.ones_like(centers[:, :1])), dim=-1)
            rays = torch.linalg.solve(
                selected_intrinsics, homogeneous.unsqueeze(-1)).squeeze(-1)
            translation = selected_depth * rays
            predicted_center, predicted_sigma, valid = \
                project_ellipsoid_to_gaussian(
                    translation, selected_rotations, selected_sizes,
                    selected_intrinsics, eps=self.eps)
            target_center, target_sigma = expand_compact_gaussian(
                selected_targets)
            target_eigenvalues = torch.linalg.eigvalsh(target_sigma)
            valid = (
                valid
                & torch.isfinite(target_center).all(dim=-1)
                & torch.isfinite(target_sigma).all(dim=(-2, -1))
                & (target_eigenvalues > self.eps).all(dim=-1)
            )
            self.last_invalid_count = (~valid).sum().detach()
            if self.fail_on_invalid and bool(torch.any(~valid).item()):
                raise RuntimeError(
                    "projection GWD received "
                    f"{int(self.last_invalid_count.item())}/"
                    f"{int(self.last_positive_count.item())} invalid positive "
                    "ellipsoid/OBB pairs")

            anisotropy_weight, target_anisotropy, anisotropy_valid = \
                normalized_gaussian_anisotropy_weights(
                    selected_targets,
                    power=self.target_anisotropy_power,
                    eps=self.eps,
                )
            anisotropy_count = anisotropy_valid.sum().clamp_min(1)
            self.last_target_anisotropy_mean = (
                (target_anisotropy * anisotropy_valid).sum()
                / anisotropy_count
            ).detach()
            self.last_anisotropy_weight_max = anisotropy_weight.max().detach()

            predicted_center = torch.nan_to_num(predicted_center)
            predicted_sigma = torch.nan_to_num(predicted_sigma)
            target_center = torch.nan_to_num(target_center)
            target_sigma = torch.nan_to_num(target_sigma)
            selected_weight = (
                selected_weight
                * valid.to(selected_weight.dtype)
                * anisotropy_weight.to(selected_weight.dtype)
            )

            xy_distance = (predicted_center - target_center).square().sum(dim=-1)
            trace_product = (predicted_sigma @ target_sigma).diagonal(
                dim1=-2, dim2=-1).sum(dim=-1)
            determinant_product = (
                predicted_sigma.det() * target_sigma.det()).clamp_min(
                    self.eps**4)
            determinant_sqrt = determinant_product.sqrt()
            covariance_distance = (
                predicted_sigma.diagonal(dim1=-2, dim2=-1).sum(dim=-1)
                + target_sigma.diagonal(dim1=-2, dim2=-1).sum(dim=-1)
                - 2.0 * (trace_product + 2.0 * determinant_sqrt)
                .clamp_min(self.eps).sqrt())
            distance = (xy_distance + covariance_distance).clamp_min(
                self.eps).sqrt()
            if self.normalize:
                scale = 2.0 * determinant_product.pow(0.125)
                distance = distance / scale.clamp_min(self.eps)
            loss = 1.0 - 1.0 / (self.tau + distance)
            return self.loss_weight * weight_reduce_loss(
                loss,
                selected_weight,
                reduction=reduction,
                avg_factor=avg_factor,
            )
