"""Forward KLD between two oriented 3D boxes, read as Gaussians.

This is the objective of ``KLD_3D_Gaussian_SO3_Derivation_JA_v2_tensor``: map an
oriented box ``(mu, R, d)`` to ``N(mu, R diag(alpha d^2) R^T)`` and optimise
``KL(N_p || N_t)`` in closed form.  It exists for the *parallel* 9D head, whose
centre, depth, rotation and extent are currently trained by four independent
losses that never see each other's error.

Why this direction, and not the one the GauCho ellipsoid branch uses
-------------------------------------------------------------------
``Ellipsoid3DKLDLoss`` optimises ``KL(target || prediction)``, whose centre term
is a Mahalanobis norm under the *predicted* covariance.  A displaced centre can
therefore be paid for by growing the prediction along the displacement -- the
closed-form optimum at a fixed centre error ``d`` is ``Sigma_t + d d^T``, which
on this data predicted a 2.74x optical-axis bloat against a measured 2.87x.
That branch now runs with ``include_center=False`` for exactly this reason.

Reversing the direction removes the escape route structurally rather than by
switching a term off: in ``KL(prediction || target)`` the centre term is
normalised by the *target* extent (eq. centerClosed), so it does not move at all
when the prediction grows, while the log-determinant term charges for the growth.
The centre can then stay in the objective, coupled to shape and orientation.

Closed form
-----------
With ``r = R_t^T (mu_p - mu_t)`` and ``Q = R_t^T R_p``,

    2 KL = (1/alpha) sum_i r_i^2 / d_{t,i}^2
         + sum_ij (d_{p,j}^2 / d_{t,i}^2) q_ij^2
         + 2 sum_i log(d_{t,i} / d_{p,i})
         - 3

No inverse, no Cholesky, no eigendecomposition: every division is by a target
extent, which is a constant, and the only logarithm is of the predicted extent.
That matters here because the near-spherical shapes this data is full of make
``eigh``-based formulations produce NaN in the backward pass.

``q_ij^2`` is invariant to the sign of a principal axis, so the objective does
not distinguish a box from the same box flipped -- which is correct for an
extent-only description, and is also why a near-spherical object contributes
almost no rotation gradient.  A semantic pose objective still has to supply
heading; this loss is an auxiliary, not a replacement for ``loss_rotation``.
"""

from __future__ import annotations

from typing import Optional

import torch
from torch import Tensor, nn

from yopo.registry import MODELS

from .utils import weight_reduce_loss

__all__ = ["oriented_box3d_kld", "ParallelOrientedBox3DKLDLoss"]

#: Extent-to-variance factor.  ``alpha = 1/4`` makes ``d`` a full side length,
#: matching the inscribed-ellipsoid convention used everywhere else here.
DEFAULT_ALPHA = 0.25
DEFAULT_EPS = 1e-9


def oriented_box3d_kld(
    predicted_center: Tensor,
    predicted_rotation: Tensor,
    predicted_size: Tensor,
    target_center: Tensor,
    target_rotation: Tensor,
    target_size: Tensor,
    *,
    alpha: float = DEFAULT_ALPHA,
    eps: float = DEFAULT_EPS,
) -> Tensor:
    """``KL(N_p || N_t)`` for two oriented boxes, in closed form.

    ``predicted_size`` and ``target_size`` are full side lengths and must be
    positive; they are floored at ``eps`` so a head without a positive
    parameterisation cannot produce a NaN, but a head that needs this loss
    should emit ``d = d0 * exp(delta)`` rather than rely on the floor.
    """
    for name, value in (("predicted_center", predicted_center),
                        ("target_center", target_center),
                        ("predicted_size", predicted_size),
                        ("target_size", target_size)):
        if value.shape[-1] != 3:
            raise ValueError(f"{name} must end in three values, got "
                             f"{tuple(value.shape)}")
    for name, value in (("predicted_rotation", predicted_rotation),
                        ("target_rotation", target_rotation)):
        if value.shape[-2:] != (3, 3):
            raise ValueError(f"{name} must end in shape (3, 3), got "
                             f"{tuple(value.shape)}")

    displacement = predicted_center - target_center
    # r = R_t^T delta: the centre error expressed in the target's own axes, so
    # each component is charged against the extent along that axis.
    relative = torch.einsum(
        "...ij,...i->...j", target_rotation, displacement)
    # Q = R_t^T R_p
    relative_rotation = target_rotation.transpose(-1, -2) @ predicted_rotation

    target_squared = target_size.square().clamp_min(eps)
    predicted_squared = predicted_size.square().clamp_min(eps)

    center_term = (relative.square() / target_squared).sum(dim=-1) / alpha
    # [i, j] weight is d_{p,j}^2 / d_{t,i}^2.
    ratio = (predicted_squared.unsqueeze(-2)
             / target_squared.unsqueeze(-1))
    trace_term = (relative_rotation.square() * ratio).sum(dim=(-2, -1))
    log_ratio = 2.0 * (target_size.clamp_min(eps).log()
                       - predicted_size.clamp_min(eps).log()).sum(dim=-1)

    return (0.5 * (center_term + trace_term + log_ratio - 3.0)).clamp_min(0.0)


@MODELS.register_module()
class ParallelOrientedBox3DKLDLoss(nn.Module):
    """Couple the parallel head's centre, extent and orientation in one term.

    Intended as an *auxiliary* alongside the existing ``loss_centers_2d``,
    ``loss_z``, ``loss_rotation`` and ``loss_sizes``, not as a replacement:
    the rotation gradient of this objective vanishes for a sphere, and these
    objects are close to spherical.

    ``tau`` bounds the distance into ``[0, 1)`` the same way the 2D and 3D
    GauCho losses bound theirs, so one badly-placed query cannot dominate the
    batch gradient.
    """

    def __init__(
        self,
        loss_weight: float = 0.25,
        reduction: str = "mean",
        tau: float = 1.0,
        alpha: float = DEFAULT_ALPHA,
        fail_on_invalid: bool = False,
        eps: float = DEFAULT_EPS,
    ) -> None:
        super().__init__()
        if reduction not in {"none", "mean", "sum"}:
            raise ValueError(f"unsupported reduction: {reduction}")
        if not tau > 0.0:
            raise ValueError(f"tau must be positive, got {tau}")
        if not alpha > 0.0:
            raise ValueError(f"alpha must be positive, got {alpha}")
        self.loss_weight = float(loss_weight)
        self.reduction = reduction
        self.tau = float(tau)
        self.alpha = float(alpha)
        self.fail_on_invalid = bool(fail_on_invalid)
        self.eps = float(eps)
        self.last_invalid_count = None

    def forward(
        self,
        predicted_center: Tensor,
        predicted_rotation: Tensor,
        predicted_size: Tensor,
        target_center: Tensor,
        target_rotation: Tensor,
        target_size: Tensor,
        weight: Optional[Tensor] = None,
        avg_factor: Optional[int] = None,
        reduction_override: Optional[str] = None,
    ) -> Tensor:
        reduction = reduction_override or self.reduction
        distance = oriented_box3d_kld(
            predicted_center, predicted_rotation, predicted_size,
            target_center, target_rotation, target_size,
            alpha=self.alpha, eps=self.eps)

        valid = (
            torch.isfinite(distance)
            & torch.isfinite(predicted_size).all(dim=-1)
            & (predicted_size > 0).all(dim=-1)
            & (target_size > 0).all(dim=-1)
        )
        self.last_invalid_count = (~valid).sum().detach()
        if self.fail_on_invalid and bool(self.last_invalid_count.item()):
            raise RuntimeError(
                f"{type(self).__name__} received "
                f"{int(self.last_invalid_count.item())}/{int(valid.numel())} "
                "invalid samples")

        safe = torch.where(valid, distance, torch.zeros_like(distance))
        # ``d / (d + tau)``, not ``1 - 1/(tau + d)``.  The two agree at
        # ``tau = 1`` -- which is why the error was invisible -- but the latter
        # returns ``1 - 1/tau`` at zero distance, so a perfect prediction costs
        # 0.5 at ``tau = 2`` and *minus one* at ``tau = 0.5``.  This form is
        # zero at zero and approaches one, for every positive tau.
        bounded = safe / (safe + self.tau)
        if weight is None:
            weight = bounded.new_ones(bounded.shape)
        if weight.ndim > bounded.ndim:
            weight = weight.mean(dim=-1)
        return self.loss_weight * weight_reduce_loss(
            bounded, weight * valid.to(weight.dtype), reduction=reduction,
            avg_factor=avg_factor)
