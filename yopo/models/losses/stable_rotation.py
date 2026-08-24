"""Numerically stable rotation loss variants for mixed-precision training."""

from typing import Optional

import torch
from torch import Tensor

from yopo.registry import MODELS
from .pose_loss import Rotation3DLoss, _to_rotation_matrix_6d, _to_rotation_matrix_9d
from .utils import weight_reduce_loss


def _safe_acos_linear_extrapolation(cosine: Tensor, clamp: float) -> Tensor:
    """acos with finite, tangent-continuous gradients near +/- 1."""
    if not 0.0 < clamp < 1.0:
        raise ValueError(f'acos clamp must be in (0, 1), got {clamp}')
    interior = torch.acos(cosine.clamp(-clamp, clamp))
    slope = (1.0 - clamp * clamp) ** -0.5
    upper = torch.acos(cosine.new_tensor(clamp)) - slope * (cosine - clamp)
    lower = torch.acos(cosine.new_tensor(-clamp)) - slope * (cosine + clamp)
    return torch.where(cosine > clamp, upper,
                       torch.where(cosine < -clamp, lower, interior))


def _safe_rotation_matrix_6d(v6d: Tensor, normalize_eps: float) -> Tensor:
    """6D Gram--Schmidt rotation with bounded gradients at the origin."""
    r1, r2 = torch.split(v6d, 3, dim=1)
    r1 = r1 / r1.norm(p=2, dim=1, keepdim=True).clamp_min(normalize_eps)
    r2 = r2 - torch.sum(r1 * r2, dim=1, keepdim=True) * r1
    r2 = r2 / r2.norm(p=2, dim=1, keepdim=True).clamp_min(normalize_eps)
    r3 = torch.cross(r1, r2, dim=1)
    return torch.stack([r1, r2, r3], dim=2)


@MODELS.register_module()
class AMPStableRotation3DLoss(Rotation3DLoss):
    """Evaluate geodesic rotation geometry in fp32 under an AMP forward.

    ``acos`` has an infinite derivative at one. With fp16 inputs, the base
    loss safety margin rounds to one and its remaining near-boundary gradient
    can overflow on the cast back to fp16. This variant evaluates geometry in
    fp32 and applies a tangent-continuous linear extension at ``|cos|=0.9999``.
    Only this numeric singularity is changed; detector AMP, targets, weights
    and reductions remain unchanged.
    """

    def __init__(
        self,
        acos_clamp: float = 0.9999,
        normalize_eps: float = 1e-2,
        **kwargs,
    ) -> None:
        super().__init__(**kwargs)
        self.acos_clamp = acos_clamp
        if normalize_eps <= 0.0:
            raise ValueError(f'normalize_eps must be positive, got {normalize_eps}')
        self.normalize_eps = normalize_eps

    def forward(
        self,
        pred: Tensor,
        target: Tensor,
        weight: Optional[Tensor] = None,
        avg_factor: Optional[int] = None,
        reduction_override: Optional[str] = None,
        **kwargs,
    ) -> Tensor:
        if self.symmetric_classes is not None:
            raise ValueError(
                'AMPStableRotation3DLoss currently requires '
                'symmetric_classes=None')
        if reduction_override not in (None, 'none', 'mean', 'sum'):
            raise ValueError(f'unsupported reduction: {reduction_override}')
        if weight is not None and not torch.any(weight > 0) and \
                (reduction_override or self.reduction) != 'none':
            return (pred * weight).sum()

        device_type = pred.device.type
        context = torch.autocast(device_type='cuda', enabled=False) \
            if device_type == 'cuda' else torch.autocast(
                device_type=device_type, enabled=False)
        with context:
            pred32 = pred.float()
            target32 = target.float()
            if pred32.size(1) == 6:
                rotation_pred = _safe_rotation_matrix_6d(
                    pred32, self.normalize_eps)
            elif pred32.size(1) == 9:
                rotation_pred = _to_rotation_matrix_9d(pred32)
            else:
                raise ValueError(f'rotation prediction width must be 6 or 9, got {pred32.size(1)}')
            rotation_target = _safe_rotation_matrix_6d(
                target32, self.normalize_eps)
            rotation_error = torch.bmm(
                rotation_pred, rotation_target.transpose(1, 2))
            cosine = (torch.einsum('bii->b', rotation_error) - 1.0) * 0.5
            loss = _safe_acos_linear_extrapolation(cosine, self.acos_clamp)
            if weight is not None:
                weight = weight.float()
                if weight.dim() > 1 and weight.shape[1] in (6, 9):
                    weight = weight.mean(-1)
            reduction = reduction_override or self.reduction
            return self.loss_weight * weight_reduce_loss(
                loss, weight, reduction=reduction, avg_factor=avg_factor)
