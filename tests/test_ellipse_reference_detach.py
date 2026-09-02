"""The ellipse chart's reference box, and whether its gradient reaches the box.

The 2D ellipse decodes as ``box_centre + offset * extent``.  With the reference
detached -- the behaviour every run in this work trained under -- the ellipse
objective cannot reach the box branch, so the offset must absorb the box
centre's error without observing it.  It measurably does not: the predicted
ellipse centre carries the same 0.0866 relative error as the box centre, and a
partial oracle says replacing that centre alone is worth +0.0637 mAP50, more
than extent and orientation combined.

These pin the flag that opens the path: that the gradient genuinely reaches the
box tensor when it is off, genuinely does not when it is on, and that the
decoded ellipse itself is identical either way -- the flag must change where
gradients go, never what is predicted.
"""
import torch

from yopo.models.dense_pose_heads.dino_9d_center2d_posehead import (
    DINO9DCenter2DPoseHead)


class _Chart:
    """Just the decode path, without building a whole head."""

    def __init__(self, detach: bool):
        self.gaucho_eps = 1e-7
        self.gaucho_ellipse2d_reference_detach = detach

    _ellipse_reference = DINO9DCenter2DPoseHead._ellipse_reference
    _decode_gaucho_ellipse2d = DINO9DCenter2DPoseHead._decode_gaucho_ellipse2d


def _run(detach: bool):
    torch.manual_seed(0)
    box = torch.tensor([[100.0, 80.0, 40.0, 30.0],
                        [200.0, 150.0, 20.0, 60.0]], requires_grad=True)
    raw = torch.tensor([[0.10, -0.05, 0.3, 0.1, 0.2],
                        [-0.20, 0.15, 0.1, 0.0, 0.4]], requires_grad=True)
    chart = _Chart(detach)
    reference = chart._ellipse_reference(box)
    mean, cholesky, sigma = chart._decode_gaucho_ellipse2d(raw, reference)
    return box, raw, mean, cholesky, sigma


def test_detached_reference_blocks_the_box_branch():
    box, raw, mean, _, _ = _run(detach=True)
    mean.sum().backward()
    assert box.grad is None or torch.all(box.grad == 0), \
        "the box received gradient although the reference was detached"
    assert raw.grad is not None and torch.any(raw.grad != 0)


def test_undetached_reference_reaches_the_box_branch():
    box, raw, mean, _, _ = _run(detach=False)
    mean.sum().backward()
    assert box.grad is not None and torch.any(box.grad != 0), \
        "the box received no gradient although the reference was live"
    # The centre term passes straight through, so each box centre takes
    # exactly the upstream gradient of its own ellipse centre.
    assert torch.allclose(box.grad[:, :2], torch.ones(2, 2))
    # ...and the width/height enter through the chart's radius, so they are
    # reached too rather than only the centre.
    assert torch.any(box.grad[:, 2:] != 0)


def test_the_flag_does_not_change_the_prediction():
    _, _, mean_on, chol_on, sigma_on = _run(detach=True)
    _, _, mean_off, chol_off, sigma_off = _run(detach=False)
    for a, b in ((mean_on, mean_off), (chol_on, chol_off),
                 (sigma_on, sigma_off)):
        assert torch.equal(a.detach(), b.detach()), \
            "the flag altered the decoded ellipse, not just the gradient path"


def test_default_preserves_the_trained_behaviour():
    import inspect
    signature = inspect.signature(DINO9DCenter2DPoseHead.__init__)
    default = signature.parameters[
        "gaucho_ellipse2d_reference_detach"].default
    assert default is True, \
        "the default must keep every existing config decoding as it trained"
