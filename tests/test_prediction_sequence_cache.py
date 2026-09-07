"""Contracts for detector-prediction-derived sequence supervision."""

from __future__ import annotations

import numpy as np
import pytest
import torch
from mmengine.config import Config
from mmengine.structures import InstanceData

from tools.train_sequence_context import _feature_contract, _validate_feature_cache
from yopo.models.tracking.causal_inference import build_prediction_observation_batch
from yopo.models.tracking.prediction_cache import (
    PREDICTION_FEATURE_CACHE_SCHEMA,
    assign_predictions_to_teachers,
    teacher_obb_envelopes_xyxy,
)
from yopo.models.tracking.sequence_training import (
    build_pair_examples,
    load_feature_cache,
    sha256_file,
)


def test_teacher_envelopes_and_partial_assignment_keep_unknown_explicit() -> None:
    teacher_obbs = np.asarray(
        [[2.0, 2.0, 4.0, 4.0, 0.0], [10.0, 2.0, 4.0, 4.0, 0.0]],
        dtype=np.float32,
    )
    envelopes = teacher_obb_envelopes_xyxy(teacher_obbs, coordinate_scale=1.0)
    assignment = assign_predictions_to_teachers(
        prediction_boxes_xyxy=torch.tensor(
            [[0.0, 0.0, 4.0, 4.0], [8.0, 0.0, 12.0, 4.0], [20.0, 0.0, 24.0, 4.0]]
        ),
        prediction_labels=torch.tensor([0, 0, 0]),
        teacher_obbs=teacher_obbs,
        teacher_class_ids=np.asarray([1, 1]),
        teacher_coordinate_scale=1.0,
        class_id_mapping={0: 1},
        min_iou=0.30,
        max_center_distance_px=16.0,
    )

    np.testing.assert_allclose(
        envelopes, np.asarray([[0, 0, 4, 4], [8, 0, 12, 4]], dtype=np.float32)
    )
    assert assignment.teacher_indices.tolist() == [0, 1, -1]
    assert assignment.matched_prediction_indices.tolist() == [0, 1]
    assert assignment.matched_teacher_indices.tolist() == [0, 1]
    assert torch.all(assignment.ious[:2] == 1.0)
    assert torch.isnan(assignment.ious[2])


def test_prediction_pair_examples_transfer_teacher_matches_to_prediction_rows(
    tmp_path,
) -> None:
    manifest = tmp_path / "sequence_manifest.json"
    manifest.write_text(
        """{
          "schema":"yopo_rgbd_sequence_manifest_v1","schema_version":1,
          "frame_contract":{"rgb_dtype":"uint8","rgb_channels":3,"depth_dtype":"uint16","depth_unit":"mm","depth_invalid_value":0,"width":8,"height":6},
          "annotation":{"kind":"pseudo"},
          "frames":[
            {"scene":"s","frame_id":0,"source_stem":"a","split":"train","color":"a.png","depth":"a_d.png","label":"a.pkl","annotation_kind":"pseudo","intrinsic":[[1,0,0],[0,1,0],[0,0,1]],"extrinsic_w2c":[[1,0,0,0],[0,1,0,0],[0,0,1,0]]},
            {"scene":"s","frame_id":1,"source_stem":"b","split":"train","color":"b.png","depth":"b_d.png","label":"b.pkl","annotation_kind":"pseudo","intrinsic":[[1,0,0],[0,1,0],[0,0,1]],"extrinsic_w2c":[[1,0,0,0],[0,1,0,0],[0,0,1,0]]}
          ],
          "windows":[{"sequence_id":"w","scene":"s","source_row":0,"split":"train","frame_ids":[0,1],"length":2}]
        }""",
        encoding="utf-8",
    )
    nan = float("nan")
    cache = {
        "schema": PREDICTION_FEATURE_CACHE_SCHEMA,
        "frames": {
            "s/000000": {
                "appearance": torch.ones(3, 2),
                "centers_world": torch.tensor([[0.0, 0, 1], [1.0, 0, 1], [0.1, 0, 1]]),
                "teacher_indices": torch.tensor([4, -1, 9]),
                "teacher_centers_world": torch.tensor(
                    [[0.0, 0, 1], [nan, nan, nan], [0.1, 0, 1]]
                ),
            },
            "s/000001": {
                "appearance": torch.ones(3, 2),
                "centers_world": torch.tensor(
                    [[1.0, 0, 1], [0.102, 0, 1], [0.002, 0, 1]]
                ),
                "teacher_indices": torch.tensor([-1, 3, 8]),
                "teacher_centers_world": torch.tensor(
                    [[nan, nan, nan], [0.102, 0, 1], [0.002, 0, 1]]
                ),
            },
        },
    }

    examples = build_pair_examples(
        manifest,
        split="train",
        cache=cache,
        max_distance_m=0.075,
        ambiguity_margin_m=0.003,
        min_matches=2,
    )

    assert len(examples) == 1
    assert examples[0].matches.tolist() == [[0, 2], [2, 1]]


def test_raw_prediction_observation_batch_is_the_causal_sampling_boundary() -> None:
    predictions = InstanceData(
        bboxes=torch.tensor([[0.0, 0.0, 4.0, 4.0], [4.0, 2.0, 8.0, 6.0]]),
        scores=torch.tensor([0.9, 0.2]),
        labels=torch.tensor([0, 0]),
        translations=torch.tensor([[0.1, 0.2, 1.0], [0.0, 0.0, 1.0]]),
    )
    batch = build_prediction_observation_batch(
        predictions,
        (torch.ones(1, 3, 3, 4),),
        image_size=(6, 8),
        extrinsic_w2c=torch.eye(4)[:3],
        score_threshold=0.30,
        max_detections=128,
    )

    assert batch.prediction_indices.tolist() == [0]
    torch.testing.assert_close(batch.centers_px, torch.tensor([[2.0, 2.0]]))
    torch.testing.assert_close(batch.centers_camera, torch.tensor([[0.1, 0.2, 1.0]]))
    torch.testing.assert_close(batch.centers_world, torch.tensor([[0.1, 0.2, 1.0]]))
    assert batch.appearance.shape == (1, 3)


def test_prediction_observation_rejects_malformed_detector_fields() -> None:
    predictions = InstanceData(
        bboxes=torch.ones(2, 5),
        scores=torch.ones(2),
        labels=torch.zeros(2, dtype=torch.long),
        translations=torch.ones(2, 3),
    )
    with pytest.raises(ValueError, match="bboxes.*shape"):
        build_prediction_observation_batch(
            predictions,
            (torch.ones(1, 3, 3, 4),),
            image_size=(6, 8),
            extrinsic_w2c=torch.eye(4)[:3],
            score_threshold=0.30,
            max_detections=128,
        )


def test_prediction_teacher_assignment_rejects_unmapped_class() -> None:
    with pytest.raises(ValueError, match="no teacher class mapping"):
        assign_predictions_to_teachers(
            prediction_boxes_xyxy=torch.tensor([[0.0, 0.0, 4.0, 4.0]]),
            prediction_labels=torch.tensor([7]),
            teacher_obbs=np.asarray([[2.0, 2.0, 4.0, 4.0, 0.0]]),
            teacher_class_ids=np.asarray([1]),
            teacher_coordinate_scale=1.0,
            class_id_mapping={0: 1},
            min_iou=0.30,
            max_center_distance_px=16.0,
        )


def test_prediction_cache_schema_is_explicit_and_legacy_is_not_relabelled(
    tmp_path,
) -> None:
    prediction = tmp_path / "prediction.pth"
    torch.save({"schema": PREDICTION_FEATURE_CACHE_SCHEMA, "frames": {}}, prediction)
    assert load_feature_cache(prediction)["schema"] == PREDICTION_FEATURE_CACHE_SCHEMA

    unknown = tmp_path / "unknown.pth"
    torch.save({"schema": "renamed_old_cache", "frames": {}}, unknown)
    with pytest.raises(ValueError, match="unsupported sequence feature cache schema"):
        load_feature_cache(unknown)


def test_prediction_cache_validation_checks_source_and_unknown_contract(
    tmp_path,
) -> None:
    manifest = tmp_path / "manifest.json"
    checkpoint = tmp_path / "g10.pth"
    cache_path = tmp_path / "features.pth"
    manifest.write_text("manifest", encoding="utf-8")
    checkpoint.write_bytes(b"checkpoint")
    cache_path.write_bytes(b"cache")
    config = {
        "model": {"backbone": {"type": "unused"}},
        "sequence_context": {
            "feature_extraction": {
                "source": "frozen_g10_detector_prediction_hbb_center"
            }
        },
    }
    resolved = Config(config)
    nan = float("nan")
    frame = {
        "appearance": torch.ones(2, 3),
        "centers_world": torch.tensor([[0.0, 0.0, 1.0], [nan, nan, nan]]),
        "prediction_indices": torch.tensor([2, 5]),
        "labels": torch.tensor([0, 0]),
        "bboxes_xyxy": torch.tensor([[0.0, 0.0, 2.0, 2.0], [3.0, 0.0, 5.0, 2.0]]),
        "scores": torch.tensor([0.8, 0.7]),
        "geometry_valid": torch.tensor([True, False]),
        "teacher_indices": torch.tensor([4, -1]),
        "teacher_centers_world": torch.tensor([[0.0, 0.0, 1.0], [nan, nan, nan]]),
        "teacher_ious": torch.tensor([0.8, nan]),
        "teacher_center_distances_px": torch.tensor([1.0, nan]),
    }
    cache = {
        "schema": PREDICTION_FEATURE_CACHE_SCHEMA,
        "manifest_sha256": sha256_file(manifest),
        "g10_checkpoint_sha256": sha256_file(checkpoint),
        "annotation_kind": "pseudo",
        "splits": ["train"],
        "appearance_dim": 3,
        "feature_contract": _feature_contract(resolved),
        "frames": {"s/000000": frame},
    }

    report = _validate_feature_cache(
        cache,
        cache_path=cache_path,
        manifest_path=manifest,
        checkpoint_path=checkpoint,
        config=resolved,
        required_splits=("train",),
    )
    assert report["schema"] == PREDICTION_FEATURE_CACHE_SCHEMA

    legacy = {**cache, "schema": "yopo_g10_sequence_features_v2"}
    with pytest.raises(ValueError, match="schema differs"):
        _validate_feature_cache(
            legacy,
            cache_path=cache_path,
            manifest_path=manifest,
            checkpoint_path=checkpoint,
            config=resolved,
            required_splits=("train",),
        )
    bad_unknown = {
        **cache,
        "frames": {
            "s/000000": {
                **frame,
                "teacher_ious": torch.tensor([0.8, 0.0]),
            }
        },
    }
    with pytest.raises(FloatingPointError, match="teacher IoU"):
        _validate_feature_cache(
            bad_unknown,
            cache_path=cache_path,
            manifest_path=manifest,
            checkpoint_path=checkpoint,
            config=resolved,
            required_splits=("train",),
        )
