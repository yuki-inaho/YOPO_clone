"""Behavioral contracts for raw s/n backbone-plus-encoder transfer."""

import pytest
import torch

from yopo.models.backbones.yolo26 import YOLO26Backbone
from yopo.models.backbones.yolo26_features import (
    EncodedPyramidNeck,
    YOLO26FeatureBackbone,
)
from yopo.models.backbones.dual_rgbd import RGBDResidualBackbone
from yopo.models.necks.portable_hybrid_encoder import PortableHybridEncoderNeck


def test_depth_stem_sum_matches_repeated_rgb_input():
    torch.manual_seed(7)
    rgb = YOLO26Backbone(scale="n").eval()
    depth = YOLO26Backbone(scale="n", in_channels=1).eval()
    state = rgb.state_dict()
    state["layers.0.conv.weight"] = state["layers.0.conv.weight"].sum(1, keepdim=True)
    depth.load_state_dict(state, strict=True)
    image = torch.rand(1, 1, 80, 96)
    with torch.no_grad():
        expected = rgb(image.repeat(1, 3, 1, 1))
        actual = depth(image)
    for left, right in zip(actual, expected):
        torch.testing.assert_close(left, right, atol=2e-6, rtol=2e-5)


def test_portable_encoder_three_outputs_have_no_unused_p6_weights():
    encoder = PortableHybridEncoderNeck([128, 128, 256], num_outs=3)
    assert not any(key.startswith("derived_p6.") for key in encoder.state_dict())
    assert (
        len(
            encoder(
                (
                    torch.rand(1, 128, 8, 12),
                    torch.rand(1, 128, 4, 6),
                    torch.rand(1, 256, 2, 3),
                )
            )
        )
        == 3
    )


def test_asymmetric_feature_branches_train_backbones_and_encoders():
    torch.manual_seed(10)
    model = RGBDResidualBackbone(
        rgb_backbone=dict(type="YOLO26FeatureBackbone", scale="s", in_channels=3),
        depth_backbone=dict(type="YOLO26FeatureBackbone", scale="n", in_channels=1),
        beta_init=0.1,
    ).train()
    fused, depth = model.forward_with_depth_features(torch.rand(1, 4, 80, 96))
    assert [tuple(value.shape) for value in fused] == [
        (1, 256, 10, 12),
        (1, 256, 5, 6),
        (1, 256, 3, 3),
    ]
    assert [value.shape for value in depth] == [value.shape for value in fused]
    sum(value.square().mean() for value in fused).backward()
    for branch in (model.rgb_backbone, model.depth_backbone):
        for parameter in (
            branch.backbone.layers["0"].conv.weight,
            branch.encoder.projections[0].weight,
            branch.encoder.pan[-1].weight,
        ):
            assert parameter.grad is not None
            assert torch.isfinite(parameter.grad).all()
            assert parameter.grad.abs().sum() > 0


def test_encoded_pyramid_preserves_three_levels_and_adds_fourth():
    neck = EncodedPyramidNeck()
    inputs = (
        torch.rand(1, 256, 10, 12),
        torch.rand(1, 256, 5, 6),
        torch.rand(1, 256, 3, 3),
    )
    outputs = neck(inputs)
    for actual, expected in zip(outputs[:3], inputs):
        torch.testing.assert_close(actual, expected, atol=0, rtol=0)
    assert outputs[3].shape == (1, 256, 2, 2)


def test_feature_branch_rejects_invalid_input_channels():
    with pytest.raises(ValueError, match="channels"):
        YOLO26FeatureBackbone(scale="n", in_channels=2)


def test_depth_channel_projection_can_be_folded_after_roi_sampling():
    """Exercise the real sampler, including bilinear sampling and level concat."""
    from copy import deepcopy
    from torch import nn
    from yopo.models.dense_pose_heads.depth_query_context import (
        MultiScaleDepthQuerySampler,
    )

    torch.manual_seed(23)
    teacher = MultiScaleDepthQuerySampler(embed_dims=8, in_channels=(8, 8, 8))
    teacher.input_projections = nn.ModuleList(
        nn.Conv2d(8, 8, 1, bias=False) for _ in range(3)
    )
    student = deepcopy(teacher)
    weight = teacher.output_projection[0].weight.detach()
    folded = torch.cat(
        [
            weight[:, level * 8 : (level + 1) * 8] @ layer.weight.detach()[:, :, 0, 0]
            for level, layer in enumerate(teacher.input_projections)
        ],
        dim=1,
    )
    with torch.no_grad():
        student.output_projection[0].weight.copy_(folded)
    student.input_projections = nn.ModuleList(nn.Identity() for _ in range(3))
    features = (
        torch.rand(2, 8, 15, 20),
        torch.rand(2, 8, 8, 10),
        torch.rand(2, 8, 4, 5),
    )
    boxes = torch.rand(2, 10, 4)
    torch.testing.assert_close(
        student(features, boxes), teacher(features, boxes), atol=1e-6, rtol=1e-5
    )


def test_refinement_changes_only_learning_rate_and_run_boundaries():
    from copy import deepcopy
    from pathlib import Path
    from mmengine.config import Config

    config_dir = Path(__file__).resolve().parents[1] / "configs/yopo"
    base = Config.fromfile(
        config_dir / "nocs_fruits_2026_rgbd_yolo26s_n_raw_features_calibrated_full.py"
    ).to_dict()
    actual = Config.fromfile(
        config_dir / "nocs_fruits_2026_rgbd_yolo26s_n_raw_features_refine.py"
    ).to_dict()
    expected = deepcopy(base)
    expected["optim_wrapper"]["optimizer"].update(lr=1e-5, aux_lr=1e-5)
    expected["max_epochs"] = 15
    expected["train_cfg"]["max_epochs"] = 15
    expected["default_hooks"]["checkpoint"]["max_keep_ckpts"] = 3
    expected["load_from"] = (
        "work_dirs/yopo_yolo26s_n_raw_features_calibrated_full_20260906/"
        "best_ellipsoid_shared_AP_25_epoch_20.pth"
    )
    expected["work_dir"] = "work_dirs/yopo_yolo26s_n_raw_features_refine_20260906"
    assert actual == expected
    assert actual["resume"] is False
    assert all(
        group.get("lr_mult", 1.0) > 0
        for group in actual["optim_wrapper"]["paramwise_cfg"]["custom_keys"].values()
    )
