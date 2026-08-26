"""Matchability-aware classification loss used by DEIM.

This module intentionally has no dependency on a particular detector head.
Callers provide class labels and an aligned localization-quality value for
each prediction.  The labels, rather than ``quality > 0``, identify matched
positives so that a zero-overlap Hungarian match remains a positive sample.
"""

from __future__ import annotations

import math
from collections.abc import Sequence

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch import Tensor

from yopo.models.losses.utils import weighted_loss
from yopo.registry import MODELS


def _validate_target(pred: Tensor,
                     target: Sequence[Tensor]) -> tuple[Tensor, Tensor]:
    """Validate and unpack ``(labels, quality)`` MAL targets."""
    if not isinstance(target, (tuple, list)) or len(target) != 2:
        raise TypeError(
            'target must be a (labels, quality) tuple or list with two '
            'tensors')
    labels, quality = target
    if not isinstance(labels, Tensor) or not isinstance(quality, Tensor):
        raise TypeError('labels and quality must both be torch.Tensor values')
    expected_shape = pred.shape[:-1]
    if labels.shape != expected_shape:
        raise ValueError(
            f'labels shape must be {expected_shape}, got {labels.shape}')
    if quality.shape != expected_shape:
        raise ValueError(
            f'quality shape must be {expected_shape}, got {quality.shape}')
    if labels.device != pred.device or quality.device != pred.device:
        raise ValueError('pred, labels, and quality must be on the same device')
    return labels, quality


@weighted_loss
def matchability_aware_loss(
        pred: Tensor,
        target: Sequence[Tensor],
        gamma: float = 1.5,
        force_float32: bool = True) -> Tensor:
    r"""Compute per-sample Matchability-Aware Loss (MAL).

    For a matched class with detached localization quality :math:`q`, MAL
    uses :math:`q^\gamma` as the binary-cross-entropy target and a unit loss
    weight.  Every unmatched class uses a zero target and the detached weight
    :math:`\sigma(p)^\gamma`.  Consequently, a matched positive with
    :math:`q=0` is not confused with an unmatched negative.

    Args:
        pred: Classification logits with shape ``(..., num_classes)``.
        target: ``(labels, quality)``.  Both tensors have shape ``pred.shape``
            without the final class dimension.  Foreground labels are in
            ``[0, num_classes - 1]``; other labels are treated as background.
            Ignored labels should additionally receive zero external weight.
        gamma: Exponent applied to positive quality targets and detached
            negative probabilities.  Defaults to 1.5, as in DEIM.
        force_float32: Compute sigmoid, powers, and BCE in float32 when logits
            are float16 or bfloat16.  Float32/float64 inputs retain their dtype.

    Returns:
        Per-sample loss with shape ``pred.shape[:-1]``.  The ``weighted_loss``
        wrapper supplies ``weight``, ``reduction``, and ``avg_factor``.
    """
    if pred.ndim < 1 or pred.shape[-1] < 1:
        raise ValueError('pred must have a non-empty final class dimension')
    if not math.isfinite(gamma) or gamma <= 0:
        raise ValueError(f'gamma must be finite and positive, got {gamma}')

    labels, quality = _validate_target(pred, target)
    num_classes = pred.shape[-1]
    sample_shape = pred.shape[:-1]

    compute_in_float32 = force_float32 and pred.dtype in (
        torch.float16, torch.bfloat16)
    work_pred = pred.float() if compute_in_float32 else pred
    flat_pred = work_pred.reshape(-1, num_classes)
    flat_labels = labels.detach().reshape(-1).long()
    flat_quality = quality.detach().reshape(-1).to(dtype=work_pred.dtype)

    positive_rows = (flat_labels >= 0) & (flat_labels < num_classes)
    positive_indices = positive_rows.nonzero(as_tuple=False).squeeze(1)

    positive_mask = torch.zeros_like(flat_pred, dtype=torch.bool)
    quality_target = torch.zeros_like(flat_pred)
    if positive_indices.numel() > 0:
        positive_labels = flat_labels[positive_indices]
        positive_mask[positive_indices, positive_labels] = True
        quality_target[positive_indices, positive_labels] = flat_quality[
            positive_indices].pow(gamma)

    # DEIM treats both the IoU-derived target and the negative modulation as
    # constants.  Keeping these detach operations explicit also prevents a
    # future differentiable OBB-IoU target from opening a degenerate gradient
    # path from classification into box regression.
    quality_target = quality_target.detach()
    negative_weight = flat_pred.sigmoid().detach().pow(gamma)
    modulation = torch.where(
        positive_mask, torch.ones_like(flat_pred), negative_weight)

    loss = F.binary_cross_entropy_with_logits(
        flat_pred, quality_target, reduction='none') * modulation
    return loss.sum(dim=-1).reshape(sample_shape)


@MODELS.register_module()
class MatchabilityAwareLoss(nn.Module):
    """MMEngine-compatible wrapper for Matchability-Aware Loss.

    Args:
        use_sigmoid: MAL only supports sigmoid classification.
        gamma: Matchability and negative-modulation exponent.
        reduction: One of ``'none'``, ``'mean'``, or ``'sum'``.
        loss_weight: Scalar multiplier for the reduced loss.
        force_float32: Promote float16/bfloat16 logits to float32 internally.
    """

    # Detector heads can dispatch quality-target losses through this small
    # protocol instead of importing every concrete loss class.  This keeps
    # the loss reusable and lets future quality-aware losses opt in without
    # adding another head-side ``isinstance`` branch.
    requires_quality_target = True

    def __init__(self,
                 use_sigmoid: bool = True,
                 gamma: float = 1.5,
                 reduction: str = 'mean',
                 loss_weight: float = 1.0,
                 force_float32: bool = True) -> None:
        super().__init__()
        if use_sigmoid is not True:
            raise ValueError('MatchabilityAwareLoss only supports sigmoid')
        if not math.isfinite(gamma) or gamma <= 0:
            raise ValueError(
                f'gamma must be finite and positive, got {gamma}')
        if reduction not in ('none', 'mean', 'sum'):
            raise ValueError(
                f'reduction must be none, mean, or sum, got {reduction!r}')
        if not math.isfinite(loss_weight):
            raise ValueError(
                f'loss_weight must be finite, got {loss_weight}')
        self.use_sigmoid = use_sigmoid
        self.gamma = float(gamma)
        self.reduction = reduction
        self.loss_weight = float(loss_weight)
        self.force_float32 = bool(force_float32)

    def forward(self,
                pred: Tensor,
                target: Sequence[Tensor],
                weight: Tensor | None = None,
                avg_factor: float | None = None,
                reduction_override: str | None = None) -> Tensor:
        """Calculate MAL using the standard OpenMMLab loss interface."""
        if reduction_override not in (None, 'none', 'mean', 'sum'):
            raise ValueError(
                'reduction_override must be none, mean, sum, or None')
        reduction = reduction_override or self.reduction
        return self.loss_weight * matchability_aware_loss(
            pred,
            target,
            weight=weight,
            gamma=self.gamma,
            force_float32=self.force_float32,
            reduction=reduction,
            avg_factor=avg_factor)
