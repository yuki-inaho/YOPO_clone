"""Configurable geometry-quality targets for matchability-aware losses."""

from __future__ import annotations

import math

import torch
import torch.nn as nn
from torch import Tensor

from yopo.models.losses.gaucho3d_geometry import (
    obb_gaussian_to_cholesky2d,
)
from yopo.models.losses.gaucho3d_loss import ellipse_kld_from_cholesky
from yopo.models.losses.projected_ellipsoid_loss import (
    gaussian_wasserstein_distance,
)
from yopo.registry import MODELS
from yopo.structures.bbox import bbox_cxcywh_to_xyxy, bbox_overlaps


def aligned_hbb_iou_quality_targets(
        labels: Tensor,
        bbox_predictions: Tensor,
        bbox_targets: Tensor,
        num_classes: int) -> Tensor:
    """Build detached, aligned HBB IoU targets for matched queries.

    Geometry-quality construction belongs to this policy module rather than a
    detector head.  The old head-level function remains as a forwarding
    wrapper so existing imports keep working.
    """
    if labels.ndim != 1:
        raise ValueError(f'labels must be 1D, got {labels.shape}')
    if (bbox_predictions.shape != bbox_targets.shape
            or bbox_predictions.ndim != 2
            or bbox_predictions.shape[-1] != 4
            or len(labels) != len(bbox_predictions)):
        raise ValueError(
            'bbox predictions/targets must share shape (N, 4) and match '
            f'labels, got {bbox_predictions.shape}/{bbox_targets.shape}/'
            f'{labels.shape}')
    quality = bbox_predictions.new_zeros(len(labels))
    positive = _positive_mask(labels, num_classes)
    if positive.any():
        predicted_xyxy = bbox_cxcywh_to_xyxy(
            bbox_predictions[positive].detach())
        target_xyxy = bbox_cxcywh_to_xyxy(
            bbox_targets[positive].detach())
        quality[positive] = bbox_overlaps(
            predicted_xyxy, target_xyxy, is_aligned=True
        ).clamp(min=0.0, max=1.0)
    return quality.detach()


def _positive_mask(labels: Tensor, num_classes: int) -> Tensor:
    return (labels >= 0) & (labels < num_classes)


def _low_precision_to_float(tensor: Tensor) -> Tensor:
    if tensor.dtype in (torch.float16, torch.bfloat16):
        return tensor.float()
    return tensor


@MODELS.register_module()
class MatchabilityQualityPolicy(nn.Module):
    r"""Build detached per-query geometry quality in ``[0, 1]``.

    This module is a stateless policy: it owns no learned parameters and does
    not embed geometry choices in a detector head.  HBB inputs are normalized
    ``(cx, cy, width, height)`` boxes.  Optional OBB inputs are compact
    Gaussians ``(x, y, sigma_xx, sigma_xy, sigma_yy)``.

    Args:
        num_classes: Foreground class count.  Labels in
            ``[0, num_classes - 1]`` are matched positives.
        source: ``'hbb_iou'``, ``'obb_gwd'``, ``'blend'``,
            ``'ellipse_kld'``, or ``'ellipse_kld_blend'``.
        obb_weight: OBB fraction in ``'blend'`` mode.
        normalize: Normalize Gaussian Wasserstein distance by Gaussian scale.
        include_center: Include compact-Gaussian center displacement in GWD.
        ellipse_symmetric: Average both KLD directions for ellipse quality.
        tau: GWD quality denominator in ``1 / (tau + distance)``.  It must be
            at least one so the quality cannot exceed one.
        missing_obb: Behavior when an OBB-dependent source receives no OBB
            pair.  ``'error'`` fails or ``'hbb_iou'`` explicitly falls back.
        fail_on_invalid: Raise for invalid positive HBB/Gaussian geometry.
            If false, the corresponding positive quality is set to zero.
        eps: Numerical threshold forwarded to GWD validity checks.
    """

    _SOURCES = frozenset({
        'hbb_iou', 'obb_gwd', 'blend',
        'ellipse_kld', 'ellipse_kld_blend'})
    _MISSING_OBB = frozenset({'error', 'hbb_iou'})

    def __init__(self,
                 num_classes: int,
                 source: str = 'hbb_iou',
                 obb_weight: float = 0.5,
                 normalize: bool = True,
                 include_center: bool = False,
                 ellipse_symmetric: bool = True,
                 tau: float = 1.0,
                 missing_obb: str = 'error',
                 fail_on_invalid: bool = True,
                 eps: float = 1e-7) -> None:
        super().__init__()
        if not isinstance(num_classes, int) or num_classes < 1:
            raise ValueError(
                f'num_classes must be a positive integer, got {num_classes}')
        if source not in self._SOURCES:
            raise ValueError(
                f'source must be one of {sorted(self._SOURCES)}, got {source!r}')
        if not math.isfinite(obb_weight) or not 0.0 <= obb_weight <= 1.0:
            raise ValueError(
                f'obb_weight must be finite and in [0, 1], got {obb_weight}')
        if not math.isfinite(tau) or tau < 1.0:
            raise ValueError(f'tau must be finite and >= 1, got {tau}')
        if missing_obb not in self._MISSING_OBB:
            raise ValueError(
                'missing_obb must be error or hbb_iou, got '
                f'{missing_obb!r}')
        if not math.isfinite(eps) or eps <= 0.0:
            raise ValueError(f'eps must be finite and positive, got {eps}')

        self.num_classes = num_classes
        self.source = source
        self.obb_weight = float(obb_weight)
        self.normalize = bool(normalize)
        self.include_center = bool(include_center)
        self.ellipse_symmetric = bool(ellipse_symmetric)
        self.tau = float(tau)
        self.missing_obb = missing_obb
        self.fail_on_invalid = bool(fail_on_invalid)
        self.eps = float(eps)

    def _validate_required_inputs(self,
                                  labels: Tensor,
                                  bbox_predictions: Tensor,
                                  bbox_targets: Tensor) -> None:
        if labels.ndim != 1:
            raise ValueError(f'labels must be 1D, got {labels.shape}')
        expected = (len(labels), 4)
        if (bbox_predictions.shape != expected
                or bbox_targets.shape != expected):
            raise ValueError(
                'HBB predictions and targets must both have shape '
                f'{expected}, got {bbox_predictions.shape} and '
                f'{bbox_targets.shape}')
        if (labels.device != bbox_predictions.device
                or bbox_targets.device != bbox_predictions.device):
            raise ValueError('labels and all HBB tensors must share a device')

    def _invalid_hbb_positives(self,
                               labels: Tensor,
                               bbox_predictions: Tensor,
                               bbox_targets: Tensor) -> Tensor:
        positive = _positive_mask(labels, self.num_classes)
        finite = (
            torch.isfinite(bbox_predictions).all(dim=-1)
            & torch.isfinite(bbox_targets).all(dim=-1)
        )
        positive_extent = (
            (bbox_predictions[:, 2:] > 0).all(dim=-1)
            & (bbox_targets[:, 2:] > 0).all(dim=-1)
        )
        return positive & ~(finite & positive_extent)

    def _hbb_quality(self,
                     labels: Tensor,
                     bbox_predictions: Tensor,
                     bbox_targets: Tensor) -> Tensor:
        work_predictions = _low_precision_to_float(
            bbox_predictions.detach())
        work_targets = _low_precision_to_float(bbox_targets.detach())
        quality = aligned_hbb_iou_quality_targets(
            labels.detach(), work_predictions, work_targets,
            self.num_classes)
        invalid = self._invalid_hbb_positives(
            labels.detach(), work_predictions, work_targets)
        if self.fail_on_invalid and bool(invalid.any().item()):
            raise RuntimeError(
                'matchability quality received '
                f'{int(invalid.sum().item())}/{int(invalid.numel())} invalid '
                'positive HBB pairs')
        quality = torch.where(invalid, torch.zeros_like(quality), quality)
        quality = torch.nan_to_num(
            quality, nan=0.0, posinf=1.0, neginf=0.0)
        positive = _positive_mask(labels.detach(), self.num_classes)
        return torch.where(
            positive, quality.clamp(min=0.0, max=1.0),
            torch.zeros_like(quality)).detach()

    def _validate_obb_pair(self,
                           labels: Tensor,
                           obb_predictions: Tensor,
                           obb_targets: Tensor) -> None:
        expected = (len(labels), 5)
        if (obb_predictions.shape != expected or obb_targets.shape != expected):
            raise ValueError(
                'compact Gaussian predictions and targets must both have '
                f'shape {expected}, got {obb_predictions.shape} and '
                f'{obb_targets.shape}')
        if (obb_predictions.device != labels.device
                or obb_targets.device != labels.device):
            raise ValueError('labels and all compact Gaussians must share a device')

    def _obb_quality(self,
                     labels: Tensor,
                     hbb_quality: Tensor,
                     obb_predictions: Tensor | None,
                     obb_targets: Tensor | None) -> Tensor:
        if (obb_predictions is None) != (obb_targets is None):
            raise ValueError(
                'obb_predictions and obb_targets must be provided together')
        if obb_predictions is None:
            if self.missing_obb == 'hbb_iou':
                return hbb_quality
            raise ValueError(
                f'source={self.source!r} requires compact Gaussian OBB inputs; '
                'set missing_obb="hbb_iou" for an explicit fallback')

        self._validate_obb_pair(labels, obb_predictions, obb_targets)
        positive = _positive_mask(labels.detach(), self.num_classes)
        quality = hbb_quality.new_zeros(len(labels))
        if not bool(positive.any().item()):
            return quality.detach()

        work_predictions = _low_precision_to_float(
            obb_predictions.detach()[positive])
        work_targets = _low_precision_to_float(
            obb_targets.detach()[positive])
        distance, valid = gaussian_wasserstein_distance(
            work_predictions,
            work_targets,
            normalize=self.normalize,
            include_center=self.include_center,
            eps=self.eps,
        )
        if self.fail_on_invalid and bool((~valid).any().item()):
            raise RuntimeError(
                'matchability quality received '
                f'{int((~valid).sum().item())}/{int(valid.numel())} invalid '
                'positive compact Gaussian pairs')

        selected_quality = 1.0 / (self.tau + distance)
        selected_quality = torch.nan_to_num(
            selected_quality, nan=0.0, posinf=1.0, neginf=0.0)
        selected_quality = torch.where(
            valid, selected_quality.clamp(min=0.0, max=1.0),
            torch.zeros_like(selected_quality))
        quality[positive] = selected_quality.to(dtype=quality.dtype)
        return quality.detach()

    def _ellipse_quality(self,
                         labels: Tensor,
                         hbb_quality: Tensor,
                         predictions: Tensor | None,
                         targets: Tensor | None) -> Tensor:
        """Symmetric GauCho KLD quality on aligned positive queries."""
        if (predictions is None) != (targets is None):
            raise ValueError(
                'obb_predictions and obb_targets must be provided together')
        if predictions is None:
            if self.missing_obb == 'hbb_iou':
                return hbb_quality
            raise ValueError(
                f'source={self.source!r} requires compact Gaussian ellipse '
                'inputs; set missing_obb="hbb_iou" for an explicit fallback')

        self._validate_obb_pair(labels, predictions, targets)
        positive = _positive_mask(labels.detach(), self.num_classes)
        quality = hbb_quality.new_zeros(len(labels))
        if not bool(positive.any().item()):
            return quality.detach()

        work_predictions = _low_precision_to_float(
            predictions.detach()[positive])
        work_targets = _low_precision_to_float(targets.detach()[positive])
        pred_mean, pred_cholesky, pred_valid = \
            obb_gaussian_to_cholesky2d(work_predictions, eps=self.eps)
        target_mean, target_cholesky, target_valid = \
            obb_gaussian_to_cholesky2d(work_targets, eps=self.eps)
        if not self.include_center:
            pred_mean = target_mean
        distance = ellipse_kld_from_cholesky(
            pred_mean, pred_cholesky, target_mean, target_cholesky,
            eps=self.eps)
        if self.ellipse_symmetric:
            reverse = ellipse_kld_from_cholesky(
                target_mean, target_cholesky, pred_mean, pred_cholesky,
                eps=self.eps)
            distance = 0.5 * (distance + reverse)
        valid = pred_valid & target_valid & torch.isfinite(distance)
        if self.fail_on_invalid and bool((~valid).any().item()):
            raise RuntimeError(
                'matchability quality received '
                f'{int((~valid).sum().item())}/{int(valid.numel())} invalid '
                'positive compact Gaussian ellipse pairs')

        selected_quality = 1.0 / (self.tau + distance)
        selected_quality = torch.nan_to_num(
            selected_quality, nan=0.0, posinf=1.0, neginf=0.0)
        selected_quality = torch.where(
            valid, selected_quality.clamp(min=0.0, max=1.0),
            torch.zeros_like(selected_quality))
        quality[positive] = selected_quality.to(dtype=quality.dtype)
        return quality.detach()

    def forward(self,
                labels: Tensor,
                bbox_predictions: Tensor,
                bbox_targets: Tensor,
                obb_predictions: Tensor | None = None,
                obb_targets: Tensor | None = None) -> Tensor:
        """Return detached matchability quality for every query."""
        self._validate_required_inputs(
            labels, bbox_predictions, bbox_targets)
        labels = labels.detach()
        hbb_quality = self._hbb_quality(
            labels, bbox_predictions, bbox_targets)
        if self.source == 'hbb_iou':
            return hbb_quality
        if self.source == 'blend' and self.obb_weight == 0.0:
            return hbb_quality

        ellipse_source = self.source in {
            'ellipse_kld', 'ellipse_kld_blend'}
        quality_from_geometry = (
            self._ellipse_quality(
                labels, hbb_quality, obb_predictions, obb_targets)
            if ellipse_source else
            self._obb_quality(
                labels, hbb_quality, obb_predictions, obb_targets))
        if obb_predictions is None:
            # Explicit HBB fallback is a full policy fallback, rather than an
            # artificial blend that scales valid HBB quality by 1 - weight.
            return quality_from_geometry
        if self.source in {'obb_gwd', 'ellipse_kld'}:
            return quality_from_geometry

        quality = (
            (1.0 - self.obb_weight) * hbb_quality
            + self.obb_weight * quality_from_geometry)
        positive = _positive_mask(labels, self.num_classes)
        quality = torch.where(
            positive, quality.clamp(min=0.0, max=1.0),
            torch.zeros_like(quality))
        return quality.detach()


__all__ = [
    'MatchabilityQualityPolicy', 'aligned_hbb_iou_quality_targets'
]
