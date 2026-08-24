"""Numerical contract for the fp16-safe 3D rotation objective."""

import pytest
import torch


@pytest.mark.skipif(not torch.cuda.is_available(), reason='requires CUDA AMP')
def test_amp_stable_rotation_loss_has_finite_gradient_at_identity():
    from yopo.models.losses.stable_rotation import AMPStableRotation3DLoss

    # In fp16, acos(1) has an infinite derivative.  Identity-vs-identity is
    # therefore the boundary case the loss must evaluate in fp32.
    prediction = torch.tensor(
        [[1., 0., 0., 0., 1., 0.]], device='cuda', dtype=torch.float16,
        requires_grad=True)
    target = prediction.detach().clone()
    weights = torch.ones_like(prediction)

    loss_fn = AMPStableRotation3DLoss(loss_weight=5.0)
    with torch.autocast(device_type='cuda', dtype=torch.float16):
        loss = loss_fn(prediction, target, weights, avg_factor=1)
    loss.backward()

    assert torch.isfinite(loss)
    assert torch.isfinite(prediction.grad).all()


@pytest.mark.skipif(not torch.cuda.is_available(), reason='requires CUDA AMP')
def test_amp_stable_rotation_loss_has_finite_gradient_for_zero_6d_prediction():
    from yopo.models.losses.stable_rotation import AMPStableRotation3DLoss

    # Detection heads commonly start the rotation branch near zero.  The
    # 6D-to-SO(3) normalization denominator must not amplify this to inf.
    prediction = torch.zeros(
        (1, 6), device='cuda', dtype=torch.float16, requires_grad=True)
    target = torch.tensor(
        [[1., 0., 0., 0., 1., 0.]], device='cuda', dtype=torch.float16)
    weights = torch.ones_like(prediction)

    loss_fn = AMPStableRotation3DLoss(loss_weight=5.0)
    with torch.autocast(device_type='cuda', dtype=torch.float16):
        loss = loss_fn(prediction, target, weights, avg_factor=1)
    loss.backward()

    assert torch.isfinite(loss)
    assert torch.isfinite(prediction.grad).all()
