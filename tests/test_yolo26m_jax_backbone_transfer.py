"""YOLO26m JAX-backbone transfer contracts for the YOPO RGB branch."""

from __future__ import annotations

from collections.abc import Mapping

import numpy as np
import pytest
import torch
from mmengine.config import Config

from yopo.models.backbones.yolo26 import YOLO26MBackbone
from yopo.utils.jax_yolo26_transfer import convert_jax_yolo26_backbone_arrays
from yopo.utils.yolo26_rgbd_initialization import select_stage8_reuse_state


TARGET_PREFIX = "backbone.rgb_backbone."


def _target_state() -> dict[str, torch.Tensor]:
    model = YOLO26MBackbone()
    return {
        TARGET_PREFIX + key: value.detach().cpu()
        for key, value in model.state_dict().items()
    }


def _source_key(target_key: str) -> tuple[str, bool] | None:
    relative = target_key.removeprefix(TARGET_PREFIX)
    relative = relative.removeprefix("layers.")
    if relative.endswith(".conv.weight"):
        path = relative.removesuffix(".conv.weight").replace(".", "/")
        return f"ema::backbone/{path}/conv/kernel", True
    if ".norm." not in relative or relative.endswith("num_batches_tracked"):
        return None
    path, field = relative.rsplit(".norm.", 1)
    return f"ema::backbone/{path.replace('.', '/')}/norm/{field}", False


def _canonical_source(
    target_state: Mapping[str, torch.Tensor],
) -> dict[str, np.ndarray]:
    source: dict[str, np.ndarray] = {}
    for target_key, target in target_state.items():
        mapped = _source_key(target_key)
        if mapped is None:
            continue
        source_key, transpose = mapped
        array = target.numpy()
        if transpose:
            array = array.transpose(2, 3, 1, 0)
        source[source_key] = np.ascontiguousarray(array)
    return source


def test_yolo26m_backbone_outputs_expected_three_level_pyramid() -> None:
    model = YOLO26MBackbone()

    outputs = model(torch.randn(1, 3, 128, 160))

    assert model.return_idx == (1, 2, 3)
    assert model.out_layer_indices == (4, 6, 10)
    assert model._out_channels == {1: 512, 2: 512, 3: 512}
    assert tuple(output.shape for output in outputs) == (
        (1, 512, 16, 20),
        (1, 512, 8, 10),
        (1, 512, 4, 5),
    )


def test_yolo26m_backbone_supports_native_600_style_non_aligned_height() -> None:
    model = YOLO26MBackbone()

    outputs = model(torch.randn(1, 3, 80, 96))

    assert tuple(output.shape[-2:] for output in outputs) == (
        (10, 12),
        (5, 6),
        (3, 3),
    )


def test_yolo26m_backbone_freezes_bn_statistics_but_trains_affine() -> None:
    model = YOLO26MBackbone().train()
    norms = [module for module in model.modules() if isinstance(module, torch.nn.BatchNorm2d)]
    assert len(norms) == 50
    assert all(not norm.training for norm in norms)
    before = [(norm.running_mean.clone(), norm.running_var.clone()) for norm in norms]

    sum(output.mean() for output in model(torch.randn(2, 3, 64, 64))).backward()

    assert all(norm.weight.requires_grad and norm.weight.grad is not None for norm in norms)
    assert all(
        torch.equal(norm.running_mean, mean) and torch.equal(norm.running_var, var)
        for norm, (mean, var) in zip(norms, before)
    )


def test_strict_converter_maps_exactly_250_jax_backbone_leaves() -> None:
    target = _target_state()
    source = _canonical_source(target)

    converted, report = convert_jax_yolo26_backbone_arrays(source, target)

    assert report.ok
    assert len(source) == len(converted) == len(report.mapped) == 250
    assert len(report.excluded_target) == 50
    assert all(key.endswith("num_batches_tracked") for key in report.excluded_target)
    assert report.excluded_source == ()
    assert set(converted) == {
        key for key in target if not key.endswith("num_batches_tracked")
    }


@pytest.mark.parametrize("failure", ["missing", "extra", "shape"])
def test_strict_converter_fails_closed(failure: str) -> None:
    target = _target_state()
    source = _canonical_source(target)
    first_key = sorted(source)[0]
    if failure == "missing":
        source.pop(first_key)
    elif failure == "extra":
        source["ema::backbone/unexpected/kernel"] = np.zeros(1, np.float32)
    else:
        source[first_key] = np.zeros(1, np.float32)

    with pytest.raises(ValueError, match=failure):
        convert_jax_yolo26_backbone_arrays(source, target, strict=True)


def _initialization_states() -> tuple[dict[str, torch.Tensor], dict[str, torch.Tensor]]:
    target = {
        "backbone.rgb_backbone.layers.0.conv.weight": torch.empty(2, 3, 3, 3),
        "backbone.depth_backbone.stage.weight": torch.empty(2, 2),
        "backbone.depth_adapters.0.weight": torch.empty(2, 2, 1, 1),
        "backbone.depth_beta": torch.empty(3),
        "neck.projections.0.weight": torch.empty(2, 2, 1, 1),
        "neck.aifi.0.weight": torch.empty(2, 2),
        "encoder.layers.0.weight": torch.empty(2, 2),
        "decoder.layers.0.weight": torch.empty(2, 2),
        "bbox_head.cls_branches.0.weight": torch.empty(2, 2),
        "level_embed": torch.empty(4, 2),
        "query_embedding.weight": torch.empty(4, 2),
        "memory_trans_fc.weight": torch.empty(2, 2),
        "memory_trans_norm.weight": torch.empty(2),
        "dn_query_generator.label_embedding.weight": torch.empty(2, 2),
    }
    source = {key: torch.ones_like(value) for key, value in target.items()}
    source["backbone.rgb_backbone.old.weight"] = torch.ones(1)
    return source, target


def test_stage8_reuse_is_explicit_and_resets_changed_fusion_boundary() -> None:
    source, target = _initialization_states()

    selected, report = select_stage8_reuse_state(source, target)

    assert report.ok
    assert set(selected) == {
        "backbone.depth_backbone.stage.weight",
        "neck.aifi.0.weight",
        "encoder.layers.0.weight",
        "decoder.layers.0.weight",
        "bbox_head.cls_branches.0.weight",
        "level_embed",
        "query_embedding.weight",
        "memory_trans_fc.weight",
        "memory_trans_norm.weight",
        "dn_query_generator.label_embedding.weight",
    }
    assert {
        "backbone.rgb_backbone.layers.0.conv.weight",
        "backbone.depth_adapters.0.weight",
        "backbone.depth_beta",
        "neck.projections.0.weight",
    }.issubset(report.fresh_target)


@pytest.mark.parametrize("failure", ["missing", "shape"])
def test_stage8_reuse_fails_closed_on_contract_drift(failure: str) -> None:
    source, target = _initialization_states()
    key = "decoder.layers.0.weight"
    if failure == "missing":
        source.pop(key)
    else:
        source[key] = torch.empty(1)

    with pytest.raises(ValueError, match=failure):
        select_stage8_reuse_state(source, target, strict=True)


def test_yolo26m_rgbd_full_config_replaces_only_the_rgb_interface() -> None:
    config = Config.fromfile(
        "configs/yopo/nocs_fruits_2026_rgbd_yolo26m_stage1_full.py"
    )

    assert config.model.backbone.type == "RGBDResidualBackbone"
    assert config.model.backbone.rgb_backbone.type == "YOLO26MBackbone"
    assert config.model.backbone.rgb_backbone.return_idx == [1, 2, 3]
    assert config.model.backbone.depth_backbone.type == "HGNetV2"
    assert config.model.backbone.depth_backbone.name == "B0"
    assert config.model.neck.in_channels == [512, 512, 512]
    assert config.load_from.endswith("yolo26m_rgbd_stage1_initial.pth")
    assert config.resume is False


def test_yolo26m_rgbd_capacity_and_resume_gate_configs_are_bounded() -> None:
    capacity = Config.fromfile(
        "configs/yopo/nocs_fruits_2026_rgbd_yolo26m_stage1_capacity.py"
    )
    gate = Config.fromfile(
        "configs/yopo/nocs_fruits_2026_rgbd_yolo26m_stage1_gate200.py"
    )

    assert capacity.train_cfg.type == "IterBasedTrainLoop"
    assert capacity.train_cfg.max_iters == 2
    assert capacity.val_cfg is None
    assert capacity.default_hooks.checkpoint is None
    assert gate.train_cfg.type == "IterBasedTrainLoop"
    assert gate.train_cfg.max_iters == 200
    assert gate.train_cfg.val_interval == 100
    assert gate.default_hooks.checkpoint.by_epoch is False
    assert gate.default_hooks.checkpoint.interval == 100
