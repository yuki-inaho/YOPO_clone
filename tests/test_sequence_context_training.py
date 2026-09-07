"""CPU contracts for the standalone sequence-context trainer."""

from __future__ import annotations

import numpy as np
import pytest
import torch
from mmengine.config import Config

from tools.train_sequence_context import (
    _feature_contract,
    _model_input,
    _run_kind,
    _update_budget,
    _validate_feature_cache,
    probe_feature_batch_size,
)
from yopo.engine.optimizers.amuse import AmuseOptimizer
from yopo.models.tracking.sequence_training import (
    FEATURE_CACHE_SCHEMA,
    PairExample,
    evaluate_model,
    load_feature_cache,
    make_amuse_optimizer,
    make_identity_head,
    sample_pyramid_at_centers,
    sha256_file,
    train_epoch,
    train_modes,
    warm_start_residual_context,
)
from yopo.models.tracking.sequence_training import _select_context_candidate


def test_model_input_uses_rgb_unit_interval_and_metric_depth() -> None:
    image_bgr = np.array([[[0, 127, 255], [255, 0, 127]]], dtype=np.uint8)
    depth_mm = np.array([[0, 1250]], dtype=np.uint16)

    actual = _model_input({"image_bgr": image_bgr, "depth_mm": depth_mm})

    expected = torch.tensor(
        [
            [[1.0, 127.0 / 255.0]],
            [[127.0 / 255.0, 0.0]],
            [[0.0, 1.0]],
            [[0.0, 1.25]],
        ],
        dtype=torch.float32,
    )
    torch.testing.assert_close(actual, expected)


def test_legacy_feature_cache_is_rejected(tmp_path) -> None:
    legacy = tmp_path / "legacy.pth"
    torch.save({"schema": "yopo_g10_sequence_features_v1", "frames": {}}, legacy)

    with pytest.raises(ValueError, match="unsupported sequence feature cache schema"):
        load_feature_cache(legacy)

    current = tmp_path / "current.pth"
    torch.save({"schema": FEATURE_CACHE_SCHEMA, "frames": {}}, current)
    assert load_feature_cache(current)["schema"] == FEATURE_CACHE_SCHEMA


def test_run_kind_requires_explicit_bounded_pilot() -> None:
    from argparse import Namespace

    assert (
        _run_kind(
            Namespace(smoke=True, pilot=False, full=False, max_pairs=None, epochs=None)
        )
        == "smoke"
    )
    assert (
        _run_kind(
            Namespace(smoke=False, pilot=True, full=False, max_pairs=64, epochs=3)
        )
        == "pilot"
    )
    assert (
        _run_kind(
            Namespace(smoke=False, pilot=False, full=True, max_pairs=None, epochs=None)
        )
        == "full"
    )
    with pytest.raises(ValueError, match="pilot requires"):
        _run_kind(
            Namespace(smoke=False, pilot=True, full=False, max_pairs=None, epochs=3)
        )
    with pytest.raises(ValueError, match="full rejects"):
        _run_kind(
            Namespace(smoke=False, pilot=False, full=True, max_pairs=64, epochs=3)
        )
    with pytest.raises(ValueError, match="max-pairs requires"):
        _run_kind(
            Namespace(smoke=False, pilot=False, full=False, max_pairs=64, epochs=3)
        )
    with pytest.raises(ValueError, match="select exactly one"):
        _run_kind(
            Namespace(smoke=False, pilot=False, full=False, max_pairs=None, epochs=None)
        )


def test_update_budget_exposes_warmup_coverage() -> None:
    insufficient = _update_budget(
        pair_count=64,
        pair_batch_size=32,
        epochs=3,
        modes=["B0", "B1", "B2", "C0", "C1"],
        warmup_steps=100,
    )
    full = _update_budget(
        pair_count=1068,
        pair_batch_size=32,
        epochs=20,
        modes=["B0", "B1", "B2", "C0", "C1"],
        warmup_steps=100,
    )

    assert insufficient["planned_updates_per_trainable_mode"] == 6
    assert not insufficient["warmup_can_complete"]
    assert full["updates_per_epoch_per_mode"] == 34
    assert full["planned_updates_per_trainable_mode"] == 680
    assert full["planned_updates_all_trainable_modes"] == 2720
    assert full["warmup_can_complete"]

    curriculum = _update_budget(
        pair_count=512,
        pair_batch_size=32,
        epochs=8,
        modes=["B0", "B1", "B2", "C0", "C1"],
        warmup_steps=100,
        training_strategy="residual_context_curriculum_v1",
    )
    assert curriculum["trajectory_ids"] == ["appearance", "context_adapter"]
    assert curriculum["planned_updates_per_trainable_mode"] == 128
    assert curriculum["planned_updates_all_trainable_modes"] == 256
    assert curriculum["warmup_can_complete"]
    with pytest.raises(ValueError, match="unsupported sequence training strategy"):
        _update_budget(
            pair_count=512,
            pair_batch_size=32,
            epochs=8,
            modes=["B0", "B1", "B2", "C0", "C1"],
            warmup_steps=100,
            training_strategy="implicit_fallback",
        )


def test_feature_contract_ignores_head_but_tracks_feature_producer() -> None:
    base = {
        "model": {"backbone": {"type": "DetectorA", "depth": 18}},
        "sequence_context": {
            "feature_extraction": {
                "source": "frozen_g10_detector_prediction_hbb_center"
            },
            "prediction_matching": {"score_threshold": 0.3},
            "head": {"fusion_strategy": "legacy_concat_v1"},
            "training": {"strategy": "independent_v1"},
        },
    }
    changed_head = Config(base)
    changed_head.sequence_context.head.fusion_strategy = "residual_gated_v1"
    changed_head.sequence_context.training.strategy = "residual_context_curriculum_v1"
    baseline = Config(base)
    changed_model = Config(base)
    changed_model.model.backbone.depth = 34

    assert _feature_contract(changed_head) == _feature_contract(baseline)
    assert _feature_contract(changed_model) != _feature_contract(baseline)


def test_external_feature_cache_validation_is_fail_closed(tmp_path) -> None:
    manifest = tmp_path / "manifest.json"
    checkpoint = tmp_path / "g10.pth"
    cache_path = tmp_path / "features.pth"
    manifest.write_text("manifest", encoding="utf-8")
    checkpoint.write_bytes(b"checkpoint")
    cache_path.write_bytes(b"cache")
    config = Config({"model": {"backbone": {"type": "unused-in-contract-test"}}})
    cache = {
        "schema": FEATURE_CACHE_SCHEMA,
        "manifest_sha256": sha256_file(manifest),
        "g10_checkpoint_sha256": sha256_file(checkpoint),
        "annotation_kind": "pseudo",
        "splits": ["train", "val", "smoke"],
        "appearance_dim": 2,
        "feature_contract": _feature_contract(config),
        "frames": {
            "scene/000000": {
                "appearance": torch.ones(1, 2),
                "centers_world": torch.zeros(1, 3),
            }
        },
    }

    report = _validate_feature_cache(
        cache,
        cache_path=cache_path,
        manifest_path=manifest,
        checkpoint_path=checkpoint,
        config=config,
        required_splits=("train", "val", "smoke"),
    )

    assert report["validated"] is True
    assert report["frame_count"] == 1
    bad_provenance = {**cache, "manifest_sha256": "wrong"}
    with pytest.raises(ValueError, match="provenance"):
        _validate_feature_cache(
            bad_provenance,
            cache_path=cache_path,
            manifest_path=manifest,
            checkpoint_path=checkpoint,
            config=config,
            required_splits=("train",),
        )
    bad_frame = {
        **cache,
        "frames": {
            "scene/000000": {
                "appearance": torch.tensor([[float("nan"), 1.0]]),
                "centers_world": torch.zeros(1, 3),
            }
        },
    }
    with pytest.raises(FloatingPointError, match="nonfinite"):
        _validate_feature_cache(
            bad_frame,
            cache_path=cache_path,
            manifest_path=manifest,
            checkpoint_path=checkpoint,
            config=config,
            required_splits=("train",),
        )
    legacy_contract = {
        **cache,
        "feature_contract": {
            **cache["feature_contract"],
            "resolved_config_sha256": "legacy-whole-config-hash",
        },
    }
    legacy_contract["feature_contract"].pop("producer_config_sha256")
    with pytest.raises(ValueError, match="contract differs"):
        _validate_feature_cache(
            legacy_contract,
            cache_path=cache_path,
            manifest_path=manifest,
            checkpoint_path=checkpoint,
            config=config,
            required_splits=("train",),
        )


def _cache() -> dict:
    generator = torch.Generator().manual_seed(3)
    frames = {}
    for index in range(4):
        frames[f"scene/{index:06d}"] = {
            "appearance": torch.randn(3, 6, generator=generator),
            "centers_world": torch.tensor(
                [[0.0, 0.0, 0.0], [0.03, 0.0, 0.0], [0.06, 0.0, 0.0]]
            )
            + index * 0.002,
        }
    return {"appearance_dim": 6, "frames": frames}


def _examples() -> tuple[PairExample, ...]:
    matches = torch.tensor([[0, 0], [1, 1], [2, 2]])
    return tuple(
        PairExample(f"scene/{index:06d}", f"scene/{index + 1:06d}", matches)
        for index in range(3)
    )


def test_pyramid_sampler_preserves_level_and_center_order() -> None:
    first = torch.arange(16, dtype=torch.float32).reshape(1, 1, 4, 4)
    second = (100 + torch.arange(4, dtype=torch.float32)).reshape(1, 1, 2, 2)
    centers = torch.tensor([[0.0, 0.0], [3.0, 3.0]])

    sampled = sample_pyramid_at_centers(
        (first, second), batch_index=0, centers_px=centers, image_size=(4, 4)
    )

    torch.testing.assert_close(sampled, torch.tensor([[0.0, 100.0], [15.0, 103.0]]))


def test_cpu_amuse_step_and_proxy_metrics_are_finite() -> None:
    cache = _cache()
    examples = _examples()
    model = make_identity_head(
        mode="C1",
        appearance_dim=6,
        head_config={"embedding_dim": 8, "hidden_dim": 12, "context_radius_m": 0.15},
    )
    optimizer = make_amuse_optimizer(
        model,
        {
            "type": "AmuseOptimizer",
            "lr": 1e-3,
            "aux_lr": 1e-3,
            "weight_decay": 0.0,
            "warmup_steps": 2,
        },
    )

    trained = train_epoch(
        model,
        optimizer,
        examples,
        cache,
        device=torch.device("cpu"),
        pair_batch_size=2,
        seed=1,
    )
    metrics = evaluate_model(
        model,
        examples,
        cache,
        device=torch.device("cpu"),
        embedding_gate=2.0,
        geometry_gate_m=0.075,
    )

    assert trained["updates"] == 2
    assert trained["learning_rate_start"] == pytest.approx(5e-4)
    assert trained["learning_rate_end"] == pytest.approx(1e-3)
    assert trained["gradient_norm_max"] > 0.0
    assert torch.isfinite(
        torch.tensor(list(metrics.values()), dtype=torch.float64)
    ).all()
    assert metrics["proxy_pseudo_pairs"] == 9


def test_feature_probe_reduces_only_batch_after_oom() -> None:
    class BatchLimitedBackbone(torch.nn.Module):
        def forward(self, inputs: torch.Tensor) -> tuple[torch.Tensor]:
            if len(inputs) > 2:
                raise torch.OutOfMemoryError("synthetic batch limit")
            return (inputs[:, :1],)

    selected, attempts, peak = probe_feature_batch_size(
        BatchLimitedBackbone(),
        device=torch.device("cpu"),
        image_size=(8, 8),
        candidates=[4, 2, 1],
    )

    assert selected == 2
    assert attempts == [
        {"batch_size": 4, "status": "oom"},
        {"batch_size": 2, "status": "success", "peak_bytes": 0},
    ]
    assert peak == 0


def test_b1_is_not_a_trainable_absolute_position_descriptor() -> None:
    with pytest.raises(ValueError, match="geometry-only baseline"):
        make_identity_head(
            mode="B1",
            appearance_dim=6,
            head_config={
                "embedding_dim": 8,
                "hidden_dim": 12,
                "context_radius_m": 0.15,
            },
        )


def test_train_modes_reports_epoch0_and_fair_geometry_baselines(
    tmp_path, monkeypatch
) -> None:
    eval_calls = 0
    original_eval = AmuseOptimizer.eval

    def recording_eval(optimizer):
        nonlocal eval_calls
        eval_calls += 1
        return original_eval(optimizer)

    monkeypatch.setattr(AmuseOptimizer, "eval", recording_eval)
    report = train_modes(
        modes=("B0", "B1", "B2", "C0", "C1"),
        train_examples=_examples(),
        validation_examples=_examples(),
        cache=_cache(),
        config={
            "seed": 17,
            "head": {
                "embedding_dim": 8,
                "hidden_dim": 12,
                "context_radius_m": 0.15,
            },
            "optimizer": {
                "type": "AmuseOptimizer",
                "lr": 1e-3,
                "aux_lr": 1e-3,
                "weight_decay": 0.0,
                "warmup_steps": 2,
            },
            "pair_batch_size": 2,
            "embedding_gate": 2.0,
            "association": {"max_distance_m": 0.075},
        },
        provenance={"test": True},
        work_dir=tmp_path,
        device=torch.device("cpu"),
        epochs=1,
        patience=1,
    )

    assert report["modes"]["B1"]["kind"] == "non_learned_geometry_only"
    assert report["modes"]["B1"]["best_checkpoint"] is None
    assert all(item["initialization_seed"] == 17 for item in report["modes"].values())
    assert all("epoch0_metrics" in item for item in report["modes"].values())
    assert "proxy_candidate_passes_baselines" in report["selection"]
    assert report["selection"]["adoption_requires_independently_checked_correspondence"]
    assert eval_calls == 4
    for mode in ("B0", "B2", "C0", "C1"):
        item = report["modes"][mode]
        assert item["actual_updates"] == 2
        assert item["warmup_completed"] is True
        assert item["history"][0]["cumulative_updates"] == 2
        candidates = [(0, item["epoch0_metrics"]["proxy_association_f1"])] + [
            (row["epoch"], row["proxy_association_f1"]) for row in item["history"]
        ]
        expected_epoch, expected_score = max(candidates, key=lambda row: row[1])
        checkpoint = torch.load(
            item["best_checkpoint"], map_location="cpu", weights_only=False
        )
        assert item["best_epoch"] == expected_epoch
        assert checkpoint["epoch"] == expected_epoch
        assert item["reload_metrics"]["proxy_association_f1"] == pytest.approx(
            expected_score
        )
    b0 = torch.load(
        report["modes"]["B0"]["best_checkpoint"], map_location="cpu", weights_only=False
    )
    b2 = torch.load(
        report["modes"]["B2"]["best_checkpoint"], map_location="cpu", weights_only=False
    )
    for key in b0["model_state_dict"]:
        torch.testing.assert_close(
            b0["model_state_dict"][key], b2["model_state_dict"][key]
        )


def test_residual_context_warm_start_freezes_appearance_and_routes_gradient() -> None:
    head_config = {
        "embedding_dim": 8,
        "hidden_dim": 12,
        "context_radius_m": 0.15,
        "fusion_strategy": "residual_gated_v1",
    }
    torch.manual_seed(24)
    parent = make_identity_head(mode="B0", appearance_dim=6, head_config=head_config)
    torch.manual_seed(25)
    context = make_identity_head(mode="C0", appearance_dim=6, head_config=head_config)

    metadata = warm_start_residual_context(context, parent)
    before = context(
        _cache()["frames"]["scene/000000"]["appearance"],
        _cache()["frames"]["scene/000000"]["centers_world"],
    )
    expected = parent(
        _cache()["frames"]["scene/000000"]["appearance"],
        _cache()["frames"]["scene/000000"]["centers_world"],
    )
    torch.testing.assert_close(before, expected, atol=0.0, rtol=0.0)
    assert metadata["parent_parity_verified"] is True
    assert metadata["frozen_parameter_names"]
    assert metadata["trainable_parameter_names"]
    assert all(
        not parameter.requires_grad
        for name, parameter in context.named_parameters()
        if name in metadata["frozen_parameter_names"]
    )

    source = _cache()["frames"]["scene/000000"]
    target = _cache()["frames"]["scene/000001"]
    loss = torch.nn.functional.mse_loss(
        context(source["appearance"], source["centers_world"]),
        context(target["appearance"], target["centers_world"]),
    )
    loss.backward()

    assert context.context_gate.grad is not None
    assert torch.isfinite(context.context_gate.grad)
    assert all(
        parameter.grad is None
        for name, parameter in context.named_parameters()
        if name in metadata["frozen_parameter_names"]
    )


def test_residual_context_warm_start_rejects_parent_shape_mismatch() -> None:
    parent = make_identity_head(
        mode="B0",
        appearance_dim=6,
        head_config={
            "embedding_dim": 8,
            "hidden_dim": 12,
            "context_radius_m": 0.15,
            "fusion_strategy": "residual_gated_v1",
        },
    )
    context = make_identity_head(
        mode="C0",
        appearance_dim=6,
        head_config={
            "embedding_dim": 9,
            "hidden_dim": 12,
            "context_radius_m": 0.15,
            "fusion_strategy": "residual_gated_v1",
        },
    )

    with pytest.raises(RuntimeError, match="size mismatch"):
        warm_start_residual_context(context, parent)


def test_residual_curriculum_uses_two_shared_trajectories(tmp_path) -> None:
    report = train_modes(
        modes=("B0", "B1", "B2", "C0", "C1"),
        train_examples=_examples(),
        validation_examples=_examples(),
        cache=_cache(),
        config={
            "seed": 26,
            "training_strategy": "residual_context_curriculum_v1",
            "head": {
                "embedding_dim": 8,
                "hidden_dim": 12,
                "context_radius_m": 0.15,
                "fusion_strategy": "residual_gated_v1",
            },
            "optimizer": {
                "type": "AmuseOptimizer",
                "lr": 1e-3,
                "aux_lr": 1e-3,
                "weight_decay": 0.0,
                "warmup_steps": 2,
            },
            "pair_batch_size": 2,
            "embedding_gate": 2.0,
            "association": {"max_distance_m": 0.075},
        },
        provenance={"test": True},
        work_dir=tmp_path,
        device=torch.device("cpu"),
        epochs=2,
        patience=2,
    )

    assert report["training_strategy"] == "residual_context_curriculum_v1"
    assert set(report["trajectories"]) == {"appearance", "context_adapter"}
    assert report["modes"]["B0"]["trajectory_id"] == "appearance"
    assert report["modes"]["B2"]["trajectory_id"] == "appearance"
    assert report["modes"]["C0"]["trajectory_id"] == "context_adapter"
    assert report["modes"]["C1"]["trajectory_id"] == "context_adapter"
    assert (
        report["modes"]["C0"]["parent_checkpoint_sha256"]
        == report["modes"]["B0"]["best_checkpoint_sha256"]
    )
    assert (
        report["modes"]["C0"]["epoch0_metrics"]
        == report["modes"]["B0"]["reload_metrics"]
    )
    assert (
        report["modes"]["C1"]["epoch0_metrics"]
        == report["modes"]["B2"]["reload_metrics"]
    )
    assert report["trajectories"]["appearance"]["actual_updates"] == 4
    assert report["trajectories"]["context_adapter"]["actual_updates"] == 4
    assert report["trajectories"]["context_adapter"]["parent_parity_verified"]


def test_context_candidate_uses_best_absolute_score_after_gain_gate() -> None:
    candidate, basis = _select_context_candidate(
        {
            "C0": {"proxy_association_f1": 0.1265},
            "C1": {"proxy_association_f1": 0.1552},
        },
        c0_improves=True,
        c1_improves=True,
    )

    assert candidate == "C1"
    assert basis == "highest_absolute_proxy_f1_among_improved_context_modes"
