"""YOLO26m JAX-backbone transfer contracts for the YOPO RGB branch."""

from __future__ import annotations

from collections.abc import Mapping

import numpy as np
import pytest
import torch
from mmengine.config import Config

from tools.model_converters.calibrate_yolo26m_rgbd_frontend import (
    BOUNDARY_KEYS,
    _deterministic_train_loader,
)
from yopo.models.backbones.yolo26 import YOLO26Backbone, YOLO26MBackbone
from yopo.registry import MODELS
from yopo.utils import register_all_modules
from yopo.utils.jax_yolo26_transfer import convert_jax_yolo26_backbone_arrays
from yopo.utils.rotated_yolo26_transfer import (
    convert_rotated_yolo26_backbone_arrays,
)
from yopo.utils.yolo26_frontend_calibration import (
    factor_depth_adapter,
    fit_ridge_projection,
    fit_ridge_projection_from_moments,
)
from yopo.utils.yolo26_rgbd_initialization import select_stage8_reuse_state


TARGET_PREFIX = "backbone.rgb_backbone."


def _target_state() -> dict[str, torch.Tensor]:
    model = YOLO26MBackbone()
    return {
        TARGET_PREFIX + key: value.detach().cpu()
        for key, value in model.state_dict().items()
    }


def _scale_target_state(scale: str) -> dict[str, torch.Tensor]:
    model = YOLO26Backbone(scale=scale)
    return {
        TARGET_PREFIX + key: value.detach().cpu()
        for key, value in model.state_dict().items()
    }


def _rotated_path(relative: str) -> str:
    relative = relative.removeprefix("layers.")
    layer, _, remainder = relative.partition(".")
    if layer == "10":
        remainder = remainder.replace("block.attention.projection", "m.0.attn.proj")
        remainder = remainder.replace("block.attention", "m.0.attn")
        remainder = remainder.replace("block.ffn", "m.0.ffn")
    else:
        remainder = remainder.replace("block.bottleneck", "m.0.bottleneck")
        remainder = remainder.replace("block.c3k.blocks", "m.0.c3k.m")
        remainder = remainder.replace("block.c3k", "m.0.c3k")
    return f"yolo26/model/{layer}" + (
        f"/{remainder.replace('.', '/')}" if remainder else ""
    )


def _canonical_rotated_source(
    target_state: Mapping[str, torch.Tensor],
) -> dict[str, np.ndarray]:
    source: dict[str, np.ndarray] = {}
    for target_key, target in target_state.items():
        relative = target_key.removeprefix(TARGET_PREFIX)
        if relative.endswith("num_batches_tracked"):
            continue
        if relative.endswith(".conv.weight"):
            path = _rotated_path(relative.removesuffix(".conv.weight"))
            source[f"params/{path}/conv/kernel"] = np.ascontiguousarray(
                target.numpy().transpose(2, 3, 1, 0)
            )
            continue
        path, field = relative.rsplit(".norm.", 1)
        collection = "batch_stats" if field.startswith("running_") else "params"
        source_field = {
            "weight": "scale",
            "bias": "bias",
            "running_mean": "mean",
            "running_var": "var",
        }[field]
        source[f"{collection}/{_rotated_path(path)}/norm/{source_field}"] = (
            np.ascontiguousarray(target.numpy())
        )
    return source


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


@pytest.mark.parametrize(
    ("scale", "channels", "mapped_leaves", "batch_norms"),
    [
        ("n", (128, 128, 256), 200, 40),
        ("s", (256, 256, 512), 200, 40),
        ("m", (512, 512, 512), 250, 50),
    ],
)
def test_yolo26_rgb_backbone_outputs_scale_specific_channels(
    scale: str,
    channels: tuple[int, int, int],
    mapped_leaves: int,
    batch_norms: int,
) -> None:
    model = YOLO26Backbone(scale=scale)

    outputs = model(torch.randn(1, 3, 64, 96))

    assert model.scale == scale
    assert tuple(model._out_channels.values()) == channels
    assert tuple(output.shape for output in outputs) == (
        (1, channels[0], 8, 12),
        (1, channels[1], 4, 6),
        (1, channels[2], 2, 3),
    )
    state = model.state_dict()
    assert (
        sum(not key.endswith("num_batches_tracked") for key in state) == mapped_leaves
    )
    assert sum(key.endswith("num_batches_tracked") for key in state) == batch_norms


def test_yolo26_generic_backbone_rejects_implicit_scale_fallback() -> None:
    with pytest.raises(ValueError, match="scale"):
        YOLO26Backbone(scale="x")


@pytest.mark.parametrize(("scale", "leaf_count"), [("n", 200), ("s", 200), ("m", 250)])
def test_rotated_backbone_transfer_maps_exact_scale_leaf_count(
    scale: str, leaf_count: int
) -> None:
    target = _scale_target_state(scale)
    source = _canonical_rotated_source(target)
    source["params/yolo26/model/23/head/kernel"] = np.zeros(1, np.float32)

    converted, report = convert_rotated_yolo26_backbone_arrays(
        source, target, weights="params"
    )

    assert report.ok
    assert len(converted) == len(report.mapped) == leaf_count
    assert report.excluded_source == ()
    assert report.ignored_non_backbone == ("params/yolo26/model/23/head/kernel",)


@pytest.mark.parametrize("failure", ["missing", "extra", "shape", "nonfinite"])
def test_rotated_backbone_transfer_fails_closed(failure: str) -> None:
    target = _scale_target_state("n")
    source = _canonical_rotated_source(target)
    first_key = sorted(source)[0]
    if failure == "missing":
        source.pop(first_key)
    elif failure == "extra":
        source["params/yolo26/model/10/unexpected/kernel"] = np.zeros(1, np.float32)
    elif failure == "shape":
        source[first_key] = np.zeros(1, np.float32)
    else:
        source[first_key] = source[first_key].copy()
        source[first_key].flat[0] = np.nan

    with pytest.raises(ValueError, match=failure):
        convert_rotated_yolo26_backbone_arrays(source, target, strict=True)


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
    norms = [
        module for module in model.modules() if isinstance(module, torch.nn.BatchNorm2d)
    ]
    assert len(norms) == 50
    assert all(not norm.training for norm in norms)
    before = [(norm.running_mean.clone(), norm.running_var.clone()) for norm in norms]

    sum(output.mean() for output in model(torch.randn(2, 3, 64, 64))).backward()

    assert all(
        norm.weight.requires_grad and norm.weight.grad is not None for norm in norms
    )
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


@pytest.mark.parametrize(
    ("scale", "channels"),
    [
        ("n", [128, 128, 256]),
        ("s", [256, 256, 512]),
        ("m", [512, 512, 512]),
    ],
)
def test_yolo26_rgbd_full_config_replaces_only_the_rgb_interface(
    scale: str, channels: list[int]
) -> None:
    config = Config.fromfile(
        f"configs/yopo/nocs_fruits_2026_rgbd_yolo26{scale}_stage1_full.py"
    )

    assert config.model.backbone.type == "RGBDResidualBackbone"
    assert config.model.backbone.rgb_backbone.type == "YOLO26Backbone"
    assert config.model.backbone.rgb_backbone.scale == scale
    assert config.model.backbone.rgb_backbone.return_idx == [1, 2, 3]
    assert config.model.backbone.depth_backbone.type == "HGNetV2"
    assert config.model.backbone.depth_backbone.name == "B0"
    assert config.model.neck.in_channels == channels
    assert config.load_from.endswith(f"yolo26{scale}_rgbd_stage1_initial.pth")
    assert config.resume is False
    assert (
        config.optim_wrapper.optimizer.type
        == "yopo.engine.optimizers.amuse.AmuseOptimizer"
    )
    assert (
        config.optim_wrapper.type == "yopo.engine.optimizers.amuse.AmpAmuseOptimWrapper"
    )
    assert config.optim_wrapper.dtype == "bfloat16"


@pytest.mark.parametrize("scale", ["n", "s", "m"])
def test_yolo26_rgbd_capacity_and_resume_gate_configs_are_bounded(
    scale: str,
) -> None:
    capacity = Config.fromfile(
        f"configs/yopo/nocs_fruits_2026_rgbd_yolo26{scale}_stage1_capacity.py"
    )
    gate = Config.fromfile(
        f"configs/yopo/nocs_fruits_2026_rgbd_yolo26{scale}_stage1_gate200.py"
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


@pytest.mark.parametrize("scale", ["n", "s", "m"])
def test_calibrated_full_changes_only_the_weight_source(scale: str) -> None:
    original = Config.fromfile(
        f"configs/yopo/nocs_fruits_2026_rgbd_yolo26{scale}_stage1_full.py"
    ).to_dict()
    calibrated = Config.fromfile(
        f"configs/yopo/nocs_fruits_2026_rgbd_yolo26{scale}_stage2_calibrated_full.py"
    ).to_dict()

    assert calibrated["load_from"].endswith(
        f"yolo26{scale}_rgbd_frontend_calibrated_train64.pth"
    )
    calibrated["load_from"] = original["load_from"]

    assert calibrated == original


@pytest.mark.parametrize(
    ("scale", "rgb_channels"),
    [
        ("n", (128, 128, 256)),
        ("s", (256, 256, 512)),
        ("m", (512, 512, 512)),
    ],
)
def test_scale_specific_calibration_changes_only_seven_leaves(
    scale: str, rgb_channels: tuple[int, int, int]
) -> None:
    register_all_modules()
    config = Config.fromfile(
        f"configs/yopo/nocs_fruits_2026_rgbd_yolo26{scale}_stage1_full.py"
    )
    backbone = MODELS.build(config.model.backbone)
    neck = MODELS.build(config.model.neck)

    assert BOUNDARY_KEYS == {
        "backbone.depth_beta",
        *(f"backbone.depth_adapters.{level}.weight" for level in range(3)),
        *(f"neck.projections.{level}.weight" for level in range(3)),
    }
    assert len(BOUNDARY_KEYS) == 7
    assert tuple(layer.weight.shape[0] for layer in backbone.depth_adapters) == (
        rgb_channels
    )
    assert tuple(layer.weight.shape[1] for layer in neck.projections) == rgb_channels
    assert tuple(layer.weight.shape[0] for layer in neck.projections) == (256, 256, 256)


def test_frontend_calibration_rejects_validation_split_before_loading() -> None:
    config = Config(
        dict(
            train_dataloader=dict(
                batch_size=1,
                num_workers=0,
                dataset=dict(split="valid", pipeline=[]),
            )
        )
    )

    with pytest.raises(ValueError, match="train split"):
        _deterministic_train_loader(config, batch_size=1, num_workers=0, seed=7)


def test_ridge_projection_recovers_a_known_channel_map() -> None:
    generator = torch.Generator().manual_seed(20260906)
    source = torch.randn(512, 5, generator=generator, dtype=torch.float64)
    expected = torch.randn(3, 5, generator=generator, dtype=torch.float64)
    target = source @ expected.T

    fitted = fit_ridge_projection(source, target, ridge=1e-12)
    fitted_from_moments = fit_ridge_projection_from_moments(
        source.T @ source,
        source.T @ target,
        ridge=1e-12,
    )

    torch.testing.assert_close(fitted, expected, rtol=1e-8, atol=1e-8)
    torch.testing.assert_close(fitted_from_moments, fitted, rtol=0, atol=0)


def test_depth_adapter_factorization_preserves_teacher_projected_depth() -> None:
    generator = torch.Generator().manual_seed(3407)
    student_projection = torch.randn(3, 5, generator=generator, dtype=torch.float64)
    teacher_projection = torch.randn(3, 4, generator=generator, dtype=torch.float64)
    teacher_adapter = torch.randn(4, 2, generator=generator, dtype=torch.float64)

    student_adapter = factor_depth_adapter(
        student_projection,
        teacher_projection,
        teacher_adapter,
        ridge=1e-12,
    )

    torch.testing.assert_close(
        student_projection @ student_adapter,
        teacher_projection @ teacher_adapter,
        rtol=1e-8,
        atol=1e-8,
    )


@pytest.mark.parametrize("failure", ["shape", "nonfinite", "ridge"])
def test_ridge_projection_fails_closed(failure: str) -> None:
    source = torch.ones(8, 4, dtype=torch.float64)
    target = torch.ones(8, 3, dtype=torch.float64)
    ridge = 1e-4
    if failure == "shape":
        target = target[:-1]
    elif failure == "nonfinite":
        source[0, 0] = torch.nan
    else:
        ridge = -1.0

    with pytest.raises(ValueError):
        fit_ridge_projection(source, target, ridge=ridge)
