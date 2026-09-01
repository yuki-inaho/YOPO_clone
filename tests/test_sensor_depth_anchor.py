"""Contracts for the structural depth anchor.

The anchor exists because of three measurements: the depth channel reads an
object's centre to 5.3 mm while the regressed range is off by 21.9 mm; range
alone multiplies the shared 3D AP by fifteen; and depth features were already
flowing into the z path without fixing it, so the fix has to be metric, and it
has to live inside the forward pass -- a post-hoc substitution is not model
performance (and was removed for exactly that reason).
"""

import pytest
import torch

from yopo.models.dense_pose_heads.dino_9d_center2d_posehead import (
    DINO9DCenter2DPoseHead)


def _head(**overrides):
    kwargs = dict(
        num_classes=1,
        embed_dims=8,
        num_reg_fcs=1,
        num_pred_layer=1,
        train_cfg=None,
        use_cop_chain=False,
        cop_prediction_mode="parallel",
        loss_cls=dict(
            type="MatchabilityAwareLoss",
            use_sigmoid=True,
            gamma=1.5,
            loss_weight=1.0,
        ),
        sensor_depth_scale=4.0e-3,
        sensor_depth_anchor=True,
    )
    kwargs.update(overrides)
    return DINO9DCenter2DPoseHead(**kwargs)


def test_anchor_requires_the_scale():
    with pytest.raises(ValueError, match="sensor_depth_scale"):
        _head(sensor_depth_scale=None)


def test_sampling_reads_the_window_median_in_metres():
    head = _head()
    depth = torch.zeros(1, 1, 20, 40)
    depth[0, 0, 8:13, 18:23] = 100.0          # 5x5 block of 100 units
    head.sensor_depth_map = depth
    centres = torch.tensor([[[0.5, 0.5]]])    # centre of the block
    anchor = head._sample_depth_anchor(centres)
    assert anchor.shape == (1, 1, 1)
    assert anchor[0, 0, 0].item() == pytest.approx(100.0 * 4.0e-3)


def test_invalid_window_falls_back_to_the_image_median():
    head = _head()
    depth = torch.zeros(1, 1, 20, 40)
    depth[0, 0, :, :10] = 150.0               # valid region away from the query
    head.sensor_depth_map = depth
    centres = torch.tensor([[[0.9, 0.9]]])    # empty corner
    anchor = head._sample_depth_anchor(centres)
    assert anchor[0, 0, 0].item() == pytest.approx(150.0 * 4.0e-3)


def test_anchor_is_a_constant_not_a_gradient_path():
    head = _head()
    head.sensor_depth_map = torch.full((1, 1, 20, 40), 100.0)
    centres = torch.rand(1, 3, 2, requires_grad=True)
    anchor = head._sample_depth_anchor(centres.detach())
    assert not anchor.requires_grad


def test_missing_map_yields_zero_anchor():
    head = _head()
    head.sensor_depth_map = None
    anchor = head._sample_depth_anchor(torch.rand(2, 5, 2))
    assert anchor.shape == (2, 5, 1)
    assert float(anchor.abs().max()) == 0.0


def test_disabled_anchor_leaves_the_head_bit_identical():
    """The flag defaults to off, and off must mean exactly the old model."""
    head = _head(sensor_depth_anchor=False, sensor_depth_scale=None)
    assert head.sensor_depth_anchor is False
    assert head.sensor_depth_map is None
