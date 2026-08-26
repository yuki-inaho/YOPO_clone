"""Kalman-filter IoU loss for compact 2D Gaussian OBBs.

The compact representation is ``(x, y, sigma_xx, sigma_xy, sigma_yy)``.
This loss deliberately ignores the center and compares only the covariance
(shape and orientation), so translation remains the responsibility of the
detector's box/center objective.
"""

from __future__ import annotations

import math
from typing import Optional

import torch
from torch import Tensor, nn

from yopo.registry import MODELS

from .projected_ellipsoid_loss import expand_compact_gaussian
from .utils import weight_reduce_loss


def gaussian_kfiou_similarity(
    predicted: Tensor,
    target: Tensor,
    *,
    eps: float = 1e-7,
) -> tuple[Tensor, Tensor]:
    r"""Compute normalized covariance-only KFIoU similarity.

    For positive-definite Gaussian covariances :math:`\Sigma_p` and
    :math:`\Sigma_t`, Kalman fusion gives

    .. math::

       \Sigma_\cap=(\Sigma_p^{-1}+\Sigma_t^{-1})^{-1}.

    Only its volume is required.  The determinant identity

    .. math::

       |\Sigma_\cap|=\frac{|\Sigma_p||\Sigma_t|}
                              {|\Sigma_p+\Sigma_t|}

    avoids explicitly constructing either inverse.  Cholesky log
    determinants keep the calculation stable across covariance scales.

    Raw 2D KFIoU reaches only ``1/3`` for identical covariances.  Dividing by
    that theoretical maximum maps exact equality to similarity one and loss
    zero.

    Args:
        predicted: Compact Gaussians with shape ``(..., 5)``.
        target: Compact Gaussians with the same shape as ``predicted``.
        eps: Strict lower numerical boundary for covariance eigenvalues.

    Returns:
        A pair ``(similarity, valid)`` with shape ``predicted.shape[:-1]``.
        Invalid entries are numerically safe and have similarity one; callers
        must use ``valid`` either to reject or mask them.
    """
    if predicted.shape != target.shape or predicted.ndim < 1 or \
            predicted.shape[-1] != 5:
        raise ValueError(
            "Gaussian KFIoU expects equal (..., 5) tensors, got "
            f"{predicted.shape} and {target.shape}")
    if predicted.device != target.device:
        raise ValueError("predicted and target must be on the same device")
    if not predicted.is_floating_point() or not target.is_floating_point():
        raise TypeError("predicted and target must be floating-point tensors")
    if not math.isfinite(eps) or eps <= 0.0:
        raise ValueError(f"eps must be finite and positive, got {eps}")

    computation_dtype = (
        torch.float64
        if predicted.dtype == torch.float64 or target.dtype == torch.float64
        else torch.float32
    )
    device_type = predicted.device.type
    context = torch.autocast(device_type=device_type, enabled=False)
    with context:
        work_predicted = predicted.to(computation_dtype)
        work_target = target.to(computation_dtype)
        _, predicted_sigma = expand_compact_gaussian(work_predicted)
        _, target_sigma = expand_compact_gaussian(work_target)

        matrix_shape = predicted_sigma.shape
        eye = torch.eye(
            2, dtype=computation_dtype, device=predicted.device).expand(
                matrix_shape)
        predicted_finite = torch.isfinite(work_predicted).all(dim=-1)
        target_finite = torch.isfinite(work_target).all(dim=-1)

        # Validation is intentionally detached.  Invalid matrices are replaced
        # before the differentiable Cholesky calls below, preventing NaN
        # gradients even when ``fail_on_invalid=False``.
        with torch.no_grad():
            validation_predicted = torch.where(
                predicted_finite[..., None, None], predicted_sigma, eye)
            validation_target = torch.where(
                target_finite[..., None, None], target_sigma, eye)
            _, predicted_info = torch.linalg.cholesky_ex(
                validation_predicted - eps * eye, check_errors=False)
            _, target_info = torch.linalg.cholesky_ex(
                validation_target - eps * eye, check_errors=False)
            valid = (
                predicted_finite
                & target_finite
                & (predicted_info == 0)
                & (target_info == 0)
            )

        safe_predicted_sigma = torch.where(
            valid[..., None, None], predicted_sigma, eye)
        safe_target_sigma = torch.where(
            valid[..., None, None], target_sigma, eye)
        predicted_cholesky = torch.linalg.cholesky(safe_predicted_sigma)
        target_cholesky = torch.linalg.cholesky(safe_target_sigma)
        sum_cholesky = torch.linalg.cholesky(
            safe_predicted_sigma + safe_target_sigma)

        predicted_logdet = 2.0 * torch.log(
            predicted_cholesky.diagonal(dim1=-2, dim2=-1)).sum(dim=-1)
        target_logdet = 2.0 * torch.log(
            target_cholesky.diagonal(dim1=-2, dim2=-1)).sum(dim=-1)
        sum_logdet = 2.0 * torch.log(
            sum_cholesky.diagonal(dim1=-2, dim2=-1)).sum(dim=-1)

        predicted_log_volume = 0.5 * predicted_logdet
        target_log_volume = 0.5 * target_logdet
        intersection_log_volume = 0.5 * (
            predicted_logdet + target_logdet - sum_logdet)

        # A common log-volume offset prevents overflow/underflow without
        # altering the IoU ratio.
        log_scale = torch.maximum(
            predicted_log_volume, target_log_volume)
        predicted_volume = torch.exp(predicted_log_volume - log_scale)
        target_volume = torch.exp(target_log_volume - log_scale)
        intersection_volume = torch.exp(
            intersection_log_volume - log_scale)
        union_volume = (
            predicted_volume + target_volume - intersection_volume
        ).clamp_min(eps)
        raw_similarity = intersection_volume / union_volume

        kappa_max_2d = 1.0 / 3.0
        similarity = (raw_similarity / kappa_max_2d).clamp(0.0, 1.0)
        # Invalid pairs are masked or rejected by the module.  Giving them a
        # finite neutral value here makes the functional API safe as well.
        similarity = torch.where(valid, similarity, torch.ones_like(similarity))
        return similarity, valid


@MODELS.register_module()
class GaussianKFIoULoss(nn.Module):
    """Normalized covariance-only KFIoU for compact 2D Gaussian OBBs.

    The center coordinates are validated as finite but do not affect the loss.
    Only positive-weight pairs are evaluated, matching the positive-sample
    behavior of :class:`GaussianGWDLoss`.
    """

    def __init__(
        self,
        loss_weight: float = 1.0,
        reduction: str = "mean",
        fail_on_invalid: bool = True,
        eps: float = 1e-7,
    ) -> None:
        super().__init__()
        if reduction not in {"none", "mean", "sum"}:
            raise ValueError(f"unsupported reduction: {reduction}")
        if not math.isfinite(loss_weight) or loss_weight < 0.0:
            raise ValueError(
                "loss_weight must be finite and non-negative, got "
                f"{loss_weight}")
        if not math.isfinite(eps) or eps <= 0.0:
            raise ValueError(f"eps must be finite and positive, got {eps}")
        self.loss_weight = float(loss_weight)
        self.reduction = reduction
        self.fail_on_invalid = bool(fail_on_invalid)
        self.eps = float(eps)

    def forward(
        self,
        predicted: Tensor,
        target: Tensor,
        weight: Optional[Tensor] = None,
        avg_factor: Optional[float] = None,
        reduction_override: Optional[str] = None,
    ) -> Tensor:
        if predicted.shape != target.shape or predicted.ndim < 1 or \
                predicted.shape[-1] != 5:
            raise ValueError(
                "Gaussian KFIoU expects equal (..., 5) tensors, got "
                f"{predicted.shape} and {target.shape}")
        reduction = reduction_override or self.reduction
        if reduction not in {"none", "mean", "sum"}:
            raise ValueError(f"unsupported reduction: {reduction}")

        sample_shape = predicted.shape[:-1]
        if weight is None:
            weight = predicted.new_ones(sample_shape)
        else:
            if weight.device != predicted.device:
                raise ValueError(
                    "weight and predicted must be on the same device")
            if weight.shape == predicted.shape:
                weight = weight.to(dtype=predicted.dtype).mean(dim=-1)
            elif weight.shape != sample_shape:
                raise ValueError(
                    "weight must have shape equal to predicted.shape[:-1] "
                    f"or predicted.shape, got {weight.shape}")
            else:
                weight = weight.to(dtype=predicted.dtype)
        if not bool(torch.isfinite(weight).all().item()):
            raise ValueError("weight must contain only finite values")
        if bool(torch.any(weight < 0).item()):
            raise ValueError("weight must be non-negative")

        positive = weight > 0
        computation_dtype = (
            torch.float64
            if predicted.dtype == torch.float64 or target.dtype == torch.float64
            else torch.float32
        )
        # Preserve the complete sample shape before reduction.  In
        # particular, ``reduction='none'`` must return one entry per input and
        # the default mean must retain zero-weight entries in its denominator,
        # just like ``weight_reduce_loss`` used by the other detector losses.
        loss = predicted.to(computation_dtype).sum(dim=-1) * 0.0
        effective_weight = weight.to(computation_dtype)
        if bool(torch.any(positive).item()):
            similarity, valid = gaussian_kfiou_similarity(
                predicted[positive], target[positive], eps=self.eps)
            if self.fail_on_invalid and bool(torch.any(~valid).item()):
                raise RuntimeError(
                    "Gaussian KFIoU received "
                    f"{int((~valid).sum().item())}/{int(valid.numel())} invalid "
                    "positive pairs")

            selected_loss = (1.0 - similarity).clamp_min(0.0)
            selected_weight = weight[positive].to(similarity.dtype)
            selected_weight = selected_weight * valid.to(
                selected_weight.dtype)
            loss = loss.masked_scatter(positive, selected_loss)
            effective_weight = effective_weight.masked_scatter(
                positive, selected_weight)
        return self.loss_weight * weight_reduce_loss(
            loss,
            effective_weight,
            reduction=reduction,
            avg_factor=avg_factor,
        )
