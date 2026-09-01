"""Eccentricity-weighted, pi-periodic angle supervision for the GauCho ellipse.

The gap this fills
------------------
Decomposing the 2D residual over matched pairs put essentially all of it in the
orientation: centre within 8.6% of object scale, extents within 3%, and an
angle error of 17 degrees median with a 67 degree p90.  Tripling the ellipse
KLD weight moved the matched IoU by 0.0006, because the KLD reduces to the
Frobenius norm of ``L_p^{-1} L_g``, which for a near-circular shape is nearly a
rotation and nearly constant -- there is no gradient in the rotation direction
to recruit.  These objects have a median aspect ratio of 1.20.

A corner-set Chamfer term was tried and made it worse (angle p90 67 -> 75
degrees): comparing corners as a *set* makes a quarter turn nearly free, so it
opens a spurious basin at 90 degrees.

The reference implementation this work is measured against solves it with an
explicit angle term rather than relying on the KLD, and weights that term by the
target's eccentricity -- ``rotated_rtmdet_jax/losses.py`` builds
``angle_weight = floor + (1 - floor) * (1 - (b/a)^2)`` and applies it to both a
cosine and a geodesic form of the double-angle error.  This module is that
construction, expressed on ``Sigma`` instead of on ``(a, b, theta)``.

Why each piece
--------------
*Double angle.*  ``(cos 2t, sin 2t)`` is continuous and pi-periodic, so an
ellipse and the same ellipse relabelled by a half turn are identical and no
boundary discontinuity is reintroduced -- the property GauCho exists to keep.

*From Sigma directly.*  ``(cos 2t, sin 2t)`` is proportional to
``(xx - yy, 2 xy)``, so no eigendecomposition is needed.  That matters here:
``eigh``'s backward divides by the eigenvalue gap and would produce NaN for
exactly the near-circular shapes this dataset is made of.  The vector's own
norm is the discriminant, so it shrinks to zero as the shape becomes a circle,
which is the correct behaviour rather than a numerical accident.

*Geodesic, not cosine.*  ``1 - cos`` has zero gradient at a 90 degree error,
which is where the observed tail sits.  ``|atan2(cross, dot)| / pi`` does not
saturate there.

*Eccentricity weight.*  A circle has no identifiable orientation, so charging
for its angle is charging for annotation noise.  The weight uses the *target*
eccentricity, which is a constant, so it re-weights without adding a way to
reduce the loss by predicting rounder shapes.
"""

from __future__ import annotations

import math
from typing import Optional

import torch
from torch import Tensor, nn

from yopo.registry import MODELS

from .utils import weight_reduce_loss

__all__ = [
    "double_angle_direction",
    "shape_eccentricity",
    "EllipseAngleGeodesicLoss",
]

DEFAULT_EPS = 1e-6


def double_angle_direction(sigma: Tensor, *, eps: float = DEFAULT_EPS):
    """Return the unit ``(cos 2t, sin 2t)`` of a 2x2 shape, and its discriminant.

    ``Sigma = R diag(l1, l2) R^T`` gives ``xx - yy = (l1 - l2) cos 2t`` and
    ``2 xy = (l1 - l2) sin 2t``, so the raw vector is the direction scaled by
    the eigenvalue gap.  Normalising it recovers the direction; the gap itself
    is returned because it is what says how meaningful that direction is.
    """
    if sigma.shape[-2:] != (2, 2):
        raise ValueError(
            f"sigma must end in shape (2, 2), got {tuple(sigma.shape)}")
    xx = sigma[..., 0, 0]
    yy = sigma[..., 1, 1]
    xy = 0.5 * (sigma[..., 0, 1] + sigma[..., 1, 0])
    raw = torch.stack((xx - yy, 2.0 * xy), dim=-1)
    gap = raw.square().sum(dim=-1).clamp_min(eps * eps).sqrt()
    # An exactly circular shape has ``raw = 0`` and no direction at all, and
    # ``atan2(0, 0)`` -- which the caller reaches -- has a NaN backward.  These
    # objects are near-circular, so that is a live case, not a corner one.  A
    # tie-break proportional to the trace picks a definite direction there
    # while staying negligible against any real anisotropy: at aspect 1.2 the
    # gap is already 30% of the trace.
    # Proportional to the trace so it is negligible against real anisotropy,
    # plus an absolute term so an all-zero matrix -- which every negative query
    # carries -- still gets a definite direction instead of ``atan2(0, 0)``.
    tie_break = eps * (xx + yy).abs() + eps
    resolved = torch.stack((raw[..., 0] + tie_break, raw[..., 1]), dim=-1)
    norm = resolved.square().sum(dim=-1).clamp_min(eps * eps).sqrt()
    return resolved / norm.unsqueeze(-1), gap


def shape_eccentricity(sigma: Tensor, *, eps: float = DEFAULT_EPS) -> Tensor:
    """``1 - (b/a)^2 = (l_max - l_min) / l_max``, in ``[0, 1)``.

    Zero for a circle, approaching one for a needle.  Written from the trace
    and the eigenvalue gap so no eigendecomposition is taken.
    """
    _, gap = double_angle_direction(sigma, eps=eps)
    trace = (sigma[..., 0, 0] + sigma[..., 1, 1]).clamp_min(eps)
    largest = (trace + gap) * 0.5
    return (gap / largest.clamp_min(eps)).clamp(0.0, 1.0)


@MODELS.register_module()
class EllipseAngleGeodesicLoss(nn.Module):
    """Wrapped double-angle error, weighted by how identifiable the angle is.

    ``eccentricity_floor`` sets how much angle supervision a perfectly circular
    target still receives.  Zero -- the reference default -- means none, which
    is the honest choice when the annotation itself carries no orientation
    there.
    """

    def __init__(
        self,
        loss_weight: float = 1.0,
        reduction: str = "mean",
        eccentricity_floor: float = 0.0,
        eps: float = DEFAULT_EPS,
    ) -> None:
        super().__init__()
        if reduction not in {"none", "mean", "sum"}:
            raise ValueError(f"unsupported reduction: {reduction}")
        if not 0.0 <= eccentricity_floor <= 1.0:
            raise ValueError(
                f"eccentricity_floor must lie in [0, 1], got {eccentricity_floor}")
        self.loss_weight = float(loss_weight)
        self.reduction = reduction
        self.eccentricity_floor = float(eccentricity_floor)
        self.eps = float(eps)
        self.last_mean_angle_deg = None

    def forward(
        self,
        predicted_sigma: Tensor,
        target_compact: Tensor,
        weight: Optional[Tensor] = None,
        avg_factor: Optional[int] = None,
        reduction_override: Optional[str] = None,
    ) -> Tensor:
        reduction = reduction_override or self.reduction
        target_sigma = torch.stack(
            (target_compact[..., 2], target_compact[..., 3],
             target_compact[..., 3], target_compact[..., 4]),
            dim=-1).reshape(*target_compact.shape[:-1], 2, 2)

        # ``torch.where(valid, f(x), 0)`` does not stop a NaN: the backward
        # multiplies the NaN by zero and gets NaN.  Every negative query
        # carries an all-zero target here, so the invalid case is the common
        # one -- the inputs are made safe *before* the computation instead.
        valid_target = (
            torch.isfinite(target_sigma).all(dim=-1).all(dim=-1)
            & (target_sigma[..., 0, 0] > 0)
            & (target_sigma[..., 1, 1] > 0))
        identity = torch.eye(2, dtype=target_sigma.dtype,
                             device=target_sigma.device)
        safe_target = torch.where(valid_target[..., None, None],
                                  target_sigma, identity).detach()
        safe_predicted = torch.where(
            torch.isfinite(predicted_sigma).all(dim=-1).all(dim=-1)[
                ..., None, None],
            predicted_sigma, identity)

        predicted_direction, _ = double_angle_direction(
            safe_predicted, eps=self.eps)
        target_direction, _ = double_angle_direction(
            safe_target, eps=self.eps)

        dot = (predicted_direction * target_direction).sum(dim=-1).clamp(
            -1.0, 1.0)
        cross = (predicted_direction[..., 0] * target_direction[..., 1]
                 - predicted_direction[..., 1] * target_direction[..., 0])
        # Wrapped to (-pi, pi] in double-angle space, i.e. (-pi/2, pi/2] in the
        # ellipse's own angle, and normalised so a quarter-turn error is 1.
        geodesic = torch.atan2(cross, dot).abs() / math.pi

        eccentricity = shape_eccentricity(safe_target, eps=self.eps)
        angle_weight = (self.eccentricity_floor
                        + (1.0 - self.eccentricity_floor) * eccentricity)
        per_sample = geodesic * angle_weight

        valid = valid_target & torch.isfinite(per_sample)
        per_sample = torch.where(valid, per_sample,
                                 torch.zeros_like(per_sample))
        with torch.no_grad():
            selected = geodesic[valid]
            self.last_mean_angle_deg = (
                (selected.mean() * 180.0).detach() if selected.numel()
                else None)

        if weight is None:
            weight = per_sample.new_ones(per_sample.shape)
        if weight.ndim > per_sample.ndim:
            weight = weight.mean(dim=-1)
        return self.loss_weight * weight_reduce_loss(
            per_sample, weight * valid.to(weight.dtype), reduction=reduction,
            avg_factor=avg_factor)
