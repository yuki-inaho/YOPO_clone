"""Contracts for precise YOPO geometry and controlled shape adaptation."""

from types import SimpleNamespace

import pytest
import torch
from torch import nn

from yopo.engine.optimizers.amuse import AmuseOptimizer
from yopo.models.dense_pose_heads.dino_9d_center2d_posehead import (
    DINO9DCenter2DPoseHead,
)


def _head(**overrides) -> DINO9DCenter2DPoseHead:
    kwargs = dict(
        num_classes=1,
        embed_dims=8,
        num_reg_fcs=1,
        num_pred_layer=1,
        train_cfg=None,
        use_cop_chain=True,
        cop_prediction_mode="chain",
        cop_chain_order=("z", "size", "rotation"),
        cop_use_bbox_conditioning=True,
        cop_fusion_mode="depth_dense",
        cop_depth_context=dict(
            num_levels=3,
            roi_size=3,
            in_channels=[8, 8, 8],
        ),
        gaucho_ellipsoid=True,
        gaucho_ellipse2d=True,
        loss_ellipsoid=dict(
            type="Ellipsoid3DKLDLoss",
            loss_weight=1.0,
        ),
        loss_ellipse2d=dict(
            type="Ellipse2DKLDLoss",
            loss_weight=1.0,
        ),
        loss_cls=dict(
            type="FocalLoss",
            use_sigmoid=True,
            gamma=2.0,
            alpha=0.25,
            loss_weight=1.0,
        ),
    )
    kwargs.update(overrides)
    return DINO9DCenter2DPoseHead(**kwargs)


def _inputs(dtype: torch.dtype = torch.float32):
    hidden = torch.randn(1, 1, 3, 8, dtype=dtype)
    references = [torch.full((1, 3, 4), 0.5, dtype=dtype)]
    depth = [
        torch.randn(1, 8, height, width, dtype=dtype)
        for height, width in ((12, 16), (6, 8), (3, 4))
    ]
    return hidden, references, depth


def test_geometry_float32_keeps_pose_values_out_of_bfloat16() -> None:
    head = _head(
        geometry_float32=True,
        sensor_depth_scale=1.0,
        sensor_depth_anchor=True,
    )
    head.sensor_depth_map = torch.full((1, 1, 24, 32), 0.731)
    hidden, references, depth = _inputs()

    with torch.no_grad(), torch.autocast("cpu", dtype=torch.bfloat16):
        outputs = head(hidden, references, depth_features=depth)
        anchor = head._sample_depth_anchor(outputs[2][-1].detach())

    assert outputs[0].dtype == torch.bfloat16
    for output in (*outputs[1:6], outputs[10], outputs[11]):
        assert output.dtype == torch.float32
    assert anchor.dtype == torch.float32
    assert anchor[0, 0, 0].item() == pytest.approx(0.731)


def test_geometry_output_weights_use_amuse_auxiliary_path() -> None:
    head = _head(geometry_output_aux_optimizer=True)
    optimizer = AmuseOptimizer(head.parameters(), lr=1e-3, warmup_steps=2)
    update_type = {
        id(parameter): group["update_type"]
        for group in optimizer.param_groups
        for parameter in group["params"]
    }

    geometry_output_weights = [
        head.reg_branches[0][-1].weight,
        head.reg_centers_2d_branch[0][-1].weight,
        head.cop_z_out[0].weight,
        head.cop_size_out[0].weight,
        head.cop_rotation_out[0].weight,
        head.reg_ellipsoid_branch[0][-1].weight,
        head.reg_ellipse2d_branch[0][-1].weight,
    ]
    assert all(update_type[id(weight)] == "adamw"
               for weight in geometry_output_weights)
    assert update_type[id(head.cop_z_net[0][0].weight)] == "muon"


def test_freeze_except_hook_leaves_only_exact_allowlist_trainable() -> None:
    from yopo.engine.hooks.freeze_except import FreezeExceptHook

    class ToyModel(nn.Module):
        def __init__(self) -> None:
            super().__init__()
            self.backbone = nn.Linear(4, 4)
            self.bbox_head = nn.Module()
            self.bbox_head.reg_ellipsoid_branch = nn.Linear(4, 6)
            self.bbox_head.reg_ellipse2d_branch = nn.Linear(4, 5)

    model = ToyModel()
    hook = FreezeExceptHook(
        trainable_patterns=(r"^bbox_head\.reg_ellipsoid_branch\..+$",),
    )
    hook.before_train(SimpleNamespace(model=model, logger=None))
    trainable = {name for name, parameter in model.named_parameters()
                 if parameter.requires_grad}

    assert trainable == {
        "bbox_head.reg_ellipsoid_branch.weight",
        "bbox_head.reg_ellipsoid_branch.bias",
    }
    with pytest.raises(RuntimeError, match="matched no parameters"):
        FreezeExceptHook(
            trainable_patterns=(r"^missing\.",),
        ).before_train(SimpleNamespace(model=model, logger=None))


def test_gaucho_depth_adapter_is_zero_init_and_receives_gradient() -> None:
    head = _head(gaucho_depth_context=True)
    hidden, references, depth_a = _inputs()
    depth_b = [feature + 3.0 for feature in depth_a]

    output_a = head(hidden, references, depth_features=depth_a)[10]
    output_b = head(hidden, references, depth_features=depth_b)[10]

    torch.testing.assert_close(output_a, output_b, rtol=0.0, atol=0.0)
    output_a.sum().backward()
    gradients = [parameter.grad for parameter in head.gaucho_depth_adapter.parameters()]
    assert gradients
    assert all(gradient is not None and torch.isfinite(gradient).all()
               for gradient in gradients)
    assert any(gradient.abs().sum() > 0 for gradient in gradients)


def test_new_precision_features_default_to_legacy_structure() -> None:
    head = _head()

    assert head.geometry_float32 is False
    assert head.geometry_output_aux_optimizer is False
    assert head.gaucho_depth_context is False
    assert not hasattr(head, "gaucho_depth_adapter")
