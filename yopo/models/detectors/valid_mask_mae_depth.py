"""Validity-aware MAE-style depth pretraining for raw RGB-D captures."""

from __future__ import annotations

import torch
from torch import Tensor

from yopo.registry import MODELS

from .mae_depth import MAEDepth


@MODELS.register_module()
class ValidMaskMAEDepth(MAEDepth):
    """MAE depth model that excludes invalid raw-depth pixels from its loss.

    Input channel 3 is the raw depth channel and input channel 4 is its
    validity mask after preprocessing (positive = raw depth was valid).
    RGB remains part of the batch format but is never read by this model.
    """

    @staticmethod
    def masked_valid_l1(
        prediction: Tensor,
        target: Tensor,
        valid_mask: Tensor,
        masked_mask: Tensor,
    ) -> Tensor:
        """Mean absolute error over the intersection of valid and masked pixels."""
        contributing = valid_mask.bool() & masked_mask.bool()
        if not contributing.any():
            # Keep a differentiable, finite zero for all-invalid batches.
            return prediction.sum() * 0.0
        return (prediction[contributing] - target[contributing]).abs().mean()

    def loss(self, batch_inputs: Tensor, batch_data_samples):
        if batch_inputs.shape[1] != 5:
            raise ValueError(
                "ValidMaskMAEDepth expects RGB(3)+depth(1)+valid_mask(1), "
                f"got {batch_inputs.shape[1]} channels"
            )

        depth = batch_inputs[:, 3:4, :, :]
        valid_mask = batch_inputs[:, 4:5, :, :] > 0.0
        masked_mask = torch.zeros_like(valid_mask)
        for batch_index in range(depth.shape[0]):
            valid_indices = valid_mask[batch_index, 0].flatten().nonzero().flatten()
            num_masked = int(self.mask_ratio * valid_indices.numel())
            if num_masked:
                chosen = valid_indices[
                    torch.randperm(valid_indices.numel(), device=depth.device)[:num_masked]
                ]
                masked_mask[batch_index, 0].flatten()[chosen] = True

        masked_depth = depth.masked_fill(masked_mask, 0.0)
        prediction = self._reconstruct(masked_depth)
        loss = self.masked_valid_l1(prediction, depth, valid_mask, masked_mask)
        return {"loss_depth": self.loss_weight * loss}
