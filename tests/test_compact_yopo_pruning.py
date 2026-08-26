"""Contracts for compact weight transfer and native Group Fisher collection."""

import json
from pathlib import Path

import pytest
import torch
from mmengine.config import Config
from torch import nn

from yopo.pruning import (
    FFNGroupSpec,
    GroupGateImportanceCollector,
    build_compact_yopo_transfer_state,
    discover_transformer_ffn_groups,
    load_pruning_choices,
)
from yopo.registry import MODELS
from yopo.utils import register_all_modules


ROOT = Path(__file__).resolve().parents[1]
COMPACT_CONFIG = ROOT / "configs/yopo/nocs_fruits_2025_2026_rgbd_3dbbox_" \
    "b1b0_e4d4_ffn1024_native_800x600.py"


def test_compact_config_builds_expected_architecture() -> None:
    register_all_modules()
    cfg = Config.fromfile(COMPACT_CONFIG)

    assert cfg.model.backbone.rgb_backbone.name == "B1"
    assert cfg.model.backbone.depth_backbone.name == "B0"
    assert cfg.model.backbone.depth_backbone.init_cfg is None
    assert cfg.model.neck.in_channels == [256, 512, 1024]
    assert cfg.model.encoder.num_layers == 4
    assert cfg.model.decoder.num_layers == 4
    assert cfg.model.encoder.layer_cfg.ffn_cfg.feedforward_channels == 1024
    assert cfg.model.decoder.layer_cfg.ffn_cfg.feedforward_channels == 1024
    assert cfg.model.num_queries == 256
    assert cfg.train_dataloader.batch_size == 12

    model = MODELS.build(cfg.model)
    assert sum(parameter.numel() for parameter in model.parameters()) == 24_589_045
    groups = discover_transformer_ffn_groups(model)
    assert len(groups) == 8
    assert {group.hidden_channels for group in groups} == {1024}


def test_group_fisher_sums_tied_sites_before_square() -> None:
    first = nn.Linear(3, 3, bias=False)
    second = nn.Linear(3, 3, bias=False)
    with torch.no_grad():
        first.weight.copy_(torch.eye(3))
        second.weight.copy_(torch.eye(3))
    group = FFNGroupSpec(
        name="tied.hidden",
        hidden_channels=3,
        sites=(first, second),
        first_weight="unused.first.weight",
        first_bias=None,
        second_weight="unused.second.weight",
    )
    inputs = torch.tensor(
        [[1.0, 2.0, 3.0], [-1.0, 1.0, 2.0]],
        requires_grad=True,
    )
    left_scale = torch.tensor([1.0, 2.0, -1.0])
    right_scale = torch.tensor([2.0, -1.0, 3.0])

    with GroupGateImportanceCollector((group,)) as collector:
        collector.begin_batch()
        loss = (first(inputs) * left_scale + second(inputs) * right_scale).sum()
        loss.backward()
        collector.end_batch()
        taylor = collector.scores("taylor")[group.name]
        fisher = collector.scores("fisher")[group.name]

    per_example = inputs.detach() * (left_scale + right_scale)
    expected_taylor = per_example.abs().mean(dim=0)
    expected_fisher = 0.5 * per_example.square().mean(dim=0)
    torch.testing.assert_close(taylor, expected_taylor)
    torch.testing.assert_close(fisher, expected_fisher)


def _tiny_source_and_target():
    source = {
        "level_embed": torch.tensor([7.0]),
        "backbone.rgb_backbone.stem.weight": torch.tensor([99.0]),
        "encoder.layers.0.ffn.layers.0.0.weight": torch.tensor(
            [[1.0, 0.0], [2.0, 0.0], [3.0, 0.0], [4.0, 0.0]]
        ),
        "encoder.layers.0.ffn.layers.0.0.bias": torch.tensor(
            [10.0, 11.0, 12.0, 13.0]
        ),
        "encoder.layers.0.ffn.layers.1.weight": torch.tensor(
            [[1.0, 2.0, 3.0, 4.0], [5.0, 6.0, 7.0, 8.0]]
        ),
        "encoder.layers.0.ffn.layers.1.bias": torch.tensor([20.0, 21.0]),
    }
    for index in range(3):
        source[f"bbox_head.cls_branches.{index}.weight"] = torch.tensor(
            [float(index)]
        )
    target = {
        "level_embed": torch.zeros(1),
        "backbone.rgb_backbone.stem.weight": torch.zeros(1),
        "encoder.layers.0.ffn.layers.0.0.weight": torch.zeros(2, 2),
        "encoder.layers.0.ffn.layers.0.0.bias": torch.zeros(2),
        "encoder.layers.0.ffn.layers.1.weight": torch.zeros(2, 2),
        "encoder.layers.0.ffn.layers.1.bias": torch.zeros(2),
        "bbox_head.cls_branches.0.weight": torch.zeros(1),
        "bbox_head.cls_branches.1.weight": torch.zeros(1),
    }
    return source, target


def test_compact_transfer_slices_ffn_and_remaps_encoder_proposal_branch() -> None:
    source, target = _tiny_source_and_target()
    result = build_compact_yopo_transfer_state(
        source,
        target,
        pruning_choices={"encoder.layers.0.ffn.hidden": (1, 3)},
    )
    state = result.state_dict

    torch.testing.assert_close(
        state["encoder.layers.0.ffn.layers.0.0.weight"],
        source["encoder.layers.0.ffn.layers.0.0.weight"][[1, 3]],
    )
    torch.testing.assert_close(
        state["encoder.layers.0.ffn.layers.1.weight"],
        source["encoder.layers.0.ffn.layers.1.weight"][:, [1, 3]],
    )
    torch.testing.assert_close(
        state["bbox_head.cls_branches.1.weight"],
        source["bbox_head.cls_branches.2.weight"],
    )
    assert "backbone.rgb_backbone.stem.weight" not in state
    assert result.report["sliced_ffn_groups"][0]["selection"] \
        == "importance_plan"


def test_physical_ffn_slice_matches_dense_channel_mask() -> None:
    source, target = _tiny_source_and_target()
    indices = (1, 3)
    result = build_compact_yopo_transfer_state(
        source,
        target,
        pruning_choices={"encoder.layers.0.ffn.hidden": indices},
    )
    state = result.state_dict
    inputs = torch.tensor([[0.5, -1.0], [2.0, 1.0]])

    hidden = torch.relu(
        torch.nn.functional.linear(
            inputs,
            source["encoder.layers.0.ffn.layers.0.0.weight"],
            source["encoder.layers.0.ffn.layers.0.0.bias"],
        )
    )
    dense_masked = torch.nn.functional.linear(
        hidden[:, indices],
        source["encoder.layers.0.ffn.layers.1.weight"][:, indices],
        source["encoder.layers.0.ffn.layers.1.bias"],
    )
    physical = torch.nn.functional.linear(
        torch.relu(
            torch.nn.functional.linear(
                inputs,
                state["encoder.layers.0.ffn.layers.0.0.weight"],
                state["encoder.layers.0.ffn.layers.0.0.bias"],
            )
        ),
        state["encoder.layers.0.ffn.layers.1.weight"],
        state["encoder.layers.0.ffn.layers.1.bias"],
    )

    torch.testing.assert_close(physical, dense_masked)


def test_synthetic_plan_requires_explicit_opt_in(tmp_path: Path) -> None:
    plan = tmp_path / "synthetic.json"
    plan.write_text(
        json.dumps({
            "choices": {"encoder.layers.0.ffn.hidden": [1, 3]},
            "metadata": {
                "calibration": "synthetic_smoke_only",
                "purpose": "smoke",
                "estimator": "group_fisher",
            },
        }),
        encoding="utf-8",
    )

    try:
        load_pruning_choices(plan)
    except ValueError as error:
        assert "smoke pruning" in str(error)
    else:
        raise AssertionError("synthetic plan must fail closed")

    assert load_pruning_choices(plan, allow_smoke=True) == {
        "encoder.layers.0.ffn.hidden": (1, 3)
    }


def test_calibrated_plan_cannot_fall_back_for_a_missing_group() -> None:
    source, target = _tiny_source_and_target()

    with pytest.raises(KeyError, match="misses required target group"):
        build_compact_yopo_transfer_state(
            source,
            target,
            pruning_choices={"unrelated.hidden": (1, 3)},
        )
