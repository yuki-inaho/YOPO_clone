"""Numerical contract for validity-aware masked depth reconstruction."""

from __future__ import annotations

import torch


def test_depth_mae_valid_mask():
    """All-invalid depth is finite; only valid masked pixels contribute."""
    from yopo.models.detectors.valid_mask_mae_depth import ValidMaskMAEDepth

    prediction = torch.tensor([[[[2.0, 9.0], [5.0, -3.0]]]], requires_grad=True)
    target = torch.tensor([[[[1.0, 7.0], [3.0, 11.0]]]])
    valid = torch.tensor([[[[True, False], [True, False]]]])
    masked = torch.tensor([[[[True, True], [False, True]]]])

    loss = ValidMaskMAEDepth.masked_valid_l1(prediction, target, valid, masked)
    assert torch.isfinite(loss)
    assert torch.isclose(loss, torch.tensor(1.0))  # |2 - 1| at [0, 0] only
    loss.backward()
    assert prediction.grad[0, 0, 0, 0] != 0
    assert prediction.grad[0, 0, 0, 1] == 0
    assert prediction.grad[0, 0, 1, 0] == 0

    all_invalid = torch.zeros_like(valid, dtype=torch.bool)
    zero_loss = ValidMaskMAEDepth.masked_valid_l1(
        prediction.detach().requires_grad_(True), target, all_invalid, masked
    )
    assert torch.isfinite(zero_loss)
    assert zero_loss.item() == 0.0
