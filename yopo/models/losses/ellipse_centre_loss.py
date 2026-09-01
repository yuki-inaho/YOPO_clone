"""Explicit, scale-normalised supervision for the 2D ellipse centre.

Why the centre and not the angle
--------------------------------
Decomposing the 2D residual by *IoU* said it was almost all orientation, and
two orientation losses were built on that reading.  Decomposing it by *AP* --
replacing one part of each matched prediction's geometry with the annotation's
and re-scoring -- says something different::

    replace nothing   mAP50 0.7312
    replace centre          0.7949   (+0.0637)
    replace extent          0.7503   (+0.0191)
    replace angle           0.7331   (+0.0019)
    replace everything      0.8215   (+0.0903)

The angle is worth two thousandths.  Its errors sit on near-circular objects,
where a wrong angle costs little overlap; the IoU view weighted those heavily
because it averaged over pairs rather than over what changes a match.  The
centre carries 70% of the available headroom.

Why a new term rather than more weight on the KLD
-------------------------------------------------
The ellipse's centre is supervised only by the KLD's centre term, which is a
Mahalanobis norm under the *predicted* shape -- the same structure that let the
3D branch pay for a displaced centre by inflating its ellipsoid, and the reason
that branch now runs with ``include_center=False``.  Normalised that way, a
centre error of 8.6% of object scale produces very little gradient, which is
why tripling the ellipse weight moved the matched overlap by 0.0006.

So this charges the centre directly, in units of the target's own size, where
the gradient does not depend on how large the prediction has decided to be.
"""

from __future__ import annotations

from typing import Optional

import torch
from torch import Tensor, nn

from yopo.registry import MODELS

from .utils import weight_reduce_loss

__all__ = ["EllipseCentreLoss"]

DEFAULT_EPS = 1e-9


@MODELS.register_module()
class EllipseCentreLoss(nn.Module):
    """Smooth-L1 on the ellipse centre, in units of the target's own extent.

    Normalising by ``sqrt(trace(Sigma_t))`` makes a 20 px object and a 50 px
    object contribute comparably and makes the value readable as a fraction of
    object size.  ``beta`` is expressed in those same units, so the quadratic
    region covers errors small enough that pushing them further is noise.
    """

    def __init__(
        self,
        loss_weight: float = 1.0,
        beta: float = 0.05,
        reduction: str = "mean",
        eps: float = DEFAULT_EPS,
    ) -> None:
        super().__init__()
        if reduction not in {"none", "mean", "sum"}:
            raise ValueError(f"unsupported reduction: {reduction}")
        if not beta > 0.0:
            raise ValueError(f"beta must be positive, got {beta}")
        self.loss_weight = float(loss_weight)
        self.beta = float(beta)
        self.reduction = reduction
        self.eps = float(eps)

    def forward(
        self,
        predicted_mean: Tensor,
        target_compact: Tensor,
        weight: Optional[Tensor] = None,
        avg_factor: Optional[int] = None,
        reduction_override: Optional[str] = None,
    ) -> Tensor:
        reduction = reduction_override or self.reduction
        target_mean = target_compact[..., :2]
        trace = target_compact[..., 2] + target_compact[..., 4]
        valid = (torch.isfinite(target_compact).all(dim=-1)
                 & torch.isfinite(predicted_mean).all(dim=-1)
                 & (trace > 0))
        # Computed on sanitised inputs rather than masked afterwards: a NaN
        # multiplied by a zero mask is still a NaN in the backward pass, and
        # background queries carry all-zero targets.
        scale = torch.where(valid, trace, torch.ones_like(trace)).clamp_min(
            self.eps).sqrt()
        offset = (predicted_mean - target_mean).norm(dim=-1) / scale
        offset = torch.where(valid, offset, torch.zeros_like(offset))

        beta = self.beta
        per_sample = torch.where(offset < beta,
                                 0.5 * offset.square() / beta,
                                 offset - 0.5 * beta)
        if weight is None:
            weight = per_sample.new_ones(per_sample.shape)
        if weight.ndim > per_sample.ndim:
            weight = weight.mean(dim=-1)
        return self.loss_weight * weight_reduce_loss(
            per_sample, weight * valid.to(weight.dtype), reduction=reduction,
            avg_factor=avg_factor)
