"""Corner loss on the ellipse's oriented envelope -- the thing that is scored.

Why this is needed
------------------
The reported 2D metric is rotated IoU between the *envelope* of the predicted
ellipse and the annotated OBB.  The training objective is a Gaussian KLD.  Those
two disagree about orientation, and measurably so.

Decomposing the residual over matched pairs gives::

    centre_error_rel   median 0.086     long_ratio  median 0.989
    angle_error_deg    median 17.2      short_ratio median 1.035
    aspect_ratio_gt    median 1.200

Centre and extent are within a few percent; essentially all of the missing
overlap is orientation.  Reproducing that distribution numerically costs
0.20 of IoU from the angle alone and 0.12 from centre and extent together,
which lands on the measured 0.706 matched IoU.

And the angle does not respond to optimisation pressure: tripling the ellipse
loss weight moved the matched IoU by 0.0006.  That is structural.  The KLD
reduces to the Frobenius norm of ``L_p^{-1} L_g``; for a near-circular shape
that matrix is nearly a rotation, its norm is nearly constant, and the gradient
in the rotation direction nearly vanishes -- the same property that makes the
3D forward KLD unable to supply heading for a sphere.  Meanwhile the metric,
which compares *corners*, stays strongly angle-sensitive: at aspect 1.2 a 17
degree error still costs 0.20 of IoU.

So this loss compares corners, not covariances.  It is an auxiliary for the
KLD, not a replacement: KLD keeps the scale and shape well-posed, and this
supplies the orientation gradient the KLD structurally cannot.

Numerical care
--------------
The envelope needs the eigenvalues and eigenvectors of a 2x2 SPD matrix, which
have a closed form.  Both go singular at an exact circle -- ``sqrt`` of a zero
discriminant has infinite derivative and ``atan2(0, 0)`` is undefined -- and
these objects are near-circular, so the singularity is not a corner case here.
The discriminant is therefore softened to ``sqrt(d^2 + eps^2)``, which is exact
away from the circle and finite at it.

The four corners are compared as a *set* (Chamfer), because an oriented box is
invariant to flipping either axis: a formulation that matched corners by index
would charge for a relabelling that changes nothing.
"""

from __future__ import annotations

from typing import Optional

import torch
from torch import Tensor, nn

from yopo.registry import MODELS

from .utils import weight_reduce_loss

__all__ = ["ellipse_envelope_corners", "EllipseEnvelopeCornerLoss"]

#: Softening of the eigen-decomposition, as a *fraction of the trace*.  An
#: absolute floor is wrong here: Sigma is in squared pixels, so a 1e-6 floor
#: leaves the discriminant six orders of magnitude below the data and the
#: half-angle derivative explodes to ~1e7 at a circle.  One percent of the
#: trace is negligible for a real aspect ratio -- at aspect 1.2 the true
#: discriminant is already 20% of the trace -- and bounds the derivative where
#: the orientation is genuinely unidentified.
DEFAULT_EPS = 1e-2


def ellipse_envelope_corners(mean: Tensor, sigma: Tensor, *,
                             eps: float = DEFAULT_EPS) -> Tensor:
    """Return the four corners ``(..., 4, 2)`` of the ellipse's envelope box.

    ``sigma``'s eigenvalues are the squared semi-axes and its eigenvectors the
    box axes, so the smallest oriented box containing the ellipse has corners
    ``mean +- a u +- b v``.
    """
    if mean.shape[-1] != 2:
        raise ValueError(f"mean must end in two values, got {tuple(mean.shape)}")
    if sigma.shape[-2:] != (2, 2):
        raise ValueError(
            f"sigma must end in shape (2, 2), got {tuple(sigma.shape)}")
    xx = sigma[..., 0, 0]
    yy = sigma[..., 1, 1]
    xy = 0.5 * (sigma[..., 0, 1] + sigma[..., 1, 0])
    trace = (xx + yy).clamp_min(0.0)
    # Softened *relative to the trace* so the derivative stays bounded for a
    # circle, where the true discriminant is zero and its square root is not
    # differentiable.
    floor = eps * trace + eps
    discriminant = ((xx - yy).square() + 4.0 * xy.square()
                    + floor.square()).sqrt()
    major = ((trace + discriminant) * 0.5).clamp_min(floor).sqrt()
    minor = ((trace - discriminant) * 0.5).clamp_min(floor).sqrt()
    # The principal axis, built directly rather than through an angle.
    # ``atan2(0, 0)`` is defined but its backward is 0/0, and a circular shape
    # sits exactly on that point; the half-angle route instead needs the square
    # root of a quantity that vanishes for an axis-aligned shape.  Both are
    # avoided by taking the eigenvector itself:
    #     n1 = (lambda_max - yy, xy)      n2 = (xy, lambda_max - xx)
    # Each is an eigenvector for ``lambda_max`` and each degenerates on a
    # different configuration, so the one with the larger norm is used.  They
    # agree in direction wherever both are well conditioned, so the choice
    # changes no value -- only which expression carries the gradient.
    largest = (trace + discriminant) * 0.5
    first = torch.stack((largest - yy, xy), dim=-1)
    second = torch.stack((xy, largest - xx), dim=-1)
    first_norm = first.square().sum(dim=-1)
    second_norm = second.square().sum(dim=-1)
    chosen = torch.where((first_norm >= second_norm).unsqueeze(-1), first,
                         second)
    # The max-norm choice already bounds this away from zero -- the larger of
    # the two candidates has norm of order the (softened) discriminant -- so the
    # floor here only has to prevent an exact 0/0, and must stay far below the
    # true norm or it would shorten the axis and shrink the box.
    chosen = chosen / chosen.square().sum(
        dim=-1, keepdim=True).clamp_min(1e-12).sqrt()
    cos, sin = chosen[..., 0], chosen[..., 1]
    axis_u = torch.stack((cos, sin), dim=-1) * major.unsqueeze(-1)
    axis_v = torch.stack((-sin, cos), dim=-1) * minor.unsqueeze(-1)
    return torch.stack(
        (mean + axis_u + axis_v, mean + axis_u - axis_v,
         mean - axis_u + axis_v, mean - axis_u - axis_v), dim=-2)


@MODELS.register_module()
class EllipseEnvelopeCornerLoss(nn.Module):
    """Symmetric Chamfer distance between envelope corner sets, scale-normalised.

    Normalised by the target's own size so a 2 cm and a 5 cm object contribute
    comparably, and so the value is directly readable as a fraction of object
    extent.
    """

    def __init__(
        self,
        loss_weight: float = 1.0,
        reduction: str = "mean",
        eps: float = DEFAULT_EPS,
    ) -> None:
        super().__init__()
        if reduction not in {"none", "mean", "sum"}:
            raise ValueError(f"unsupported reduction: {reduction}")
        self.loss_weight = float(loss_weight)
        self.reduction = reduction
        self.eps = float(eps)

    def forward(
        self,
        predicted_mean: Tensor,
        predicted_sigma: Tensor,
        target_compact: Tensor,
        weight: Optional[Tensor] = None,
        avg_factor: Optional[int] = None,
        reduction_override: Optional[str] = None,
    ) -> Tensor:
        reduction = reduction_override or self.reduction
        target_mean = target_compact[..., :2]
        target_sigma = torch.stack(
            (target_compact[..., 2], target_compact[..., 3],
             target_compact[..., 3], target_compact[..., 4]),
            dim=-1).reshape(*target_compact.shape[:-1], 2, 2)

        predicted_corners = ellipse_envelope_corners(
            predicted_mean, predicted_sigma, eps=self.eps)
        target_corners = ellipse_envelope_corners(
            target_mean, target_sigma, eps=self.eps).detach()

        # (..., 4, 4) pairwise squared distances between the two corner sets.
        distance = (predicted_corners.unsqueeze(-2)
                    - target_corners.unsqueeze(-3)).square().sum(dim=-1)
        chamfer = (distance.min(dim=-1).values.sum(dim=-1)
                   + distance.min(dim=-2).values.sum(dim=-1))

        scale = (target_sigma[..., 0, 0]
                 + target_sigma[..., 1, 1]).clamp_min(1e-9)
        per_sample = chamfer / (8.0 * scale)

        valid = torch.isfinite(per_sample) & (
            target_sigma[..., 0, 0] > 0) & (target_sigma[..., 1, 1] > 0)
        per_sample = torch.where(valid, per_sample,
                                 torch.zeros_like(per_sample))
        if weight is None:
            weight = per_sample.new_ones(per_sample.shape)
        if weight.ndim > per_sample.ndim:
            weight = weight.mean(dim=-1)
        return self.loss_weight * weight_reduce_loss(
            per_sample, weight * valid.to(weight.dtype), reduction=reduction,
            avg_factor=avg_factor)
