"""Frame-validity regularization for raw 6D rotation predictions."""

from __future__ import annotations

import math
from typing import Optional

import torch
import torch.nn as nn
from torch import Tensor

from yopo.registry import MODELS

from .utils import weight_reduce_loss


def _validate_raw_rotation_6d(pred: Tensor) -> None:
    if not pred.is_floating_point():
        raise TypeError(
            'raw 6D rotation predictions must be floating point, '
            f'got {pred.dtype}')
    if pred.ndim == 0 or pred.shape[-1] != 6:
        raise ValueError(
            'raw 6D rotation predictions must have shape (..., 6), '
            f'got {tuple(pred.shape)}')


def rotation_6d_stiefel_loss(pred: Tensor, beta: float = 1.0) -> Tensor:
    """Return a per-prediction Smooth-L1 penalty on ``A.T @ A - I``.

    The first and last three values of ``pred`` are interpreted as the two
    columns of a raw 3-by-2 frame ``A``.  The returned tensor has shape
    ``pred.shape[:-1]``.  Both off-diagonal Gram entries are retained so the
    loss gives the two unit-length constraints and twice the orthogonality
    constraint equal elementwise treatment.

    Geometry is always evaluated in float32.  This keeps the squared norms
    and dot product stable when the detector forward runs under AMP.
    """
    _validate_raw_rotation_6d(pred)
    if not math.isfinite(beta) or beta <= 0.0:
        raise ValueError(f'beta must be finite and positive, got {beta}')

    device_type = pred.device.type
    with torch.autocast(device_type=device_type, enabled=False):
        columns = pred.float().unflatten(-1, (2, 3))
        gram = columns @ columns.transpose(-1, -2)
        identity = torch.eye(2, dtype=gram.dtype, device=gram.device)
        residual = (gram - identity).abs()
        elementwise = torch.where(
            residual < beta,
            0.5 * residual.square() / beta,
            residual - 0.5 * beta,
        )
        return elementwise.mean(dim=(-2, -1))


def _sample_weight(weight: Tensor, pred: Tensor) -> Tensor:
    """Convert common raw-rotation weights to one weight per frame."""
    leading_shape = pred.shape[:-1]
    if weight.shape == pred.shape:
        return weight.float().mean(dim=-1)
    if weight.shape == (*leading_shape, 1):
        return weight.float().squeeze(-1)
    if weight.shape == leading_shape or weight.ndim == 0:
        return weight.float()
    raise ValueError(
        'weight must be scalar, per-frame (...), (..., 1), or (..., 6); '
        f'got pred shape {tuple(pred.shape)} and weight shape '
        f'{tuple(weight.shape)}')


@MODELS.register_module()
class Rotation6DStiefelLoss(nn.Module):
    """Regularize a raw 6D rotation frame toward the Stiefel manifold.

    This target-free auxiliary loss constrains the two raw 3D vectors before
    Gram--Schmidt conversion.  It complements, rather than replaces, a pose
    loss evaluated after conversion to SO(3).

    Args:
        beta: Smooth-L1 transition point for each Gram-matrix residual.
        reduction: Reduction over predictions: ``none``, ``mean``, or ``sum``.
        loss_weight: Scalar multiplier applied after reduction.
    """

    def __init__(
        self,
        beta: float = 1.0,
        reduction: str = 'mean',
        loss_weight: float = 1.0,
    ) -> None:
        super().__init__()
        if not math.isfinite(beta) or beta <= 0.0:
            raise ValueError(f'beta must be finite and positive, got {beta}')
        if reduction not in ('none', 'mean', 'sum'):
            raise ValueError(f'unsupported reduction: {reduction}')
        if not math.isfinite(loss_weight) or loss_weight < 0.0:
            raise ValueError(
                'loss_weight must be finite and non-negative, '
                f'got {loss_weight}')
        self.beta = beta
        self.reduction = reduction
        self.loss_weight = loss_weight

    def forward(
        self,
        pred: Tensor,
        target: Optional[Tensor] = None,
        weight: Optional[Tensor] = None,
        avg_factor: Optional[float] = None,
        reduction_override: Optional[str] = None,
        **kwargs,
    ) -> Tensor:
        """Calculate the target-free frame-validity loss.

        ``target`` is accepted and ignored so the module follows the common
        detection-loss call signature.  A six-component rotation weight is
        reduced to its mean, preserving the usual all-ones/all-zeros mask
        semantics.
        """
        del target, kwargs
        if reduction_override not in (None, 'none', 'mean', 'sum'):
            raise ValueError(
                f'unsupported reduction_override: {reduction_override}')
        reduction = reduction_override or self.reduction
        loss = rotation_6d_stiefel_loss(pred, beta=self.beta)
        if weight is not None:
            weight = _sample_weight(weight, pred)

        # ``mean`` over an empty set is NaN, whereas an empty positive set
        # should contribute a differentiable zero like other detector losses.
        if loss.numel() == 0 and reduction == 'mean' and avg_factor is None:
            return pred.float().sum() * 0.0

        return self.loss_weight * weight_reduce_loss(
            loss,
            weight,
            reduction=reduction,
            avg_factor=avg_factor,
        )
