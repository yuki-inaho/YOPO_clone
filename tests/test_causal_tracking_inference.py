"""Detector-output-only contracts for causal RGB-D tracking inference."""

from __future__ import annotations

import pytest
import torch
from mmengine.structures import InstanceData
from torch.nn import functional as F

from yopo.models.tracking.causal_inference import (
    build_detection_observations,
    load_sequence_descriptor,
)
from yopo.models.tracking.sequence_training import CHECKPOINT_SCHEMA


class _AppearanceOnlyHead:
    def __call__(
        self, appearance: torch.Tensor, positions_world: torch.Tensor
    ) -> torch.Tensor:
        assert appearance.shape[0] == positions_world.shape[0]
        return F.normalize(appearance, dim=1)


def test_observations_use_predictions_features_and_pose_without_labels() -> None:
    predictions = InstanceData(
        bboxes=torch.tensor(
            [[0.0, 0.0, 4.0, 4.0], [4.0, 2.0, 8.0, 6.0], [1.0, 1.0, 2.0, 2.0]]
        ),
        scores=torch.tensor([0.90, 0.80, 0.10]),
        labels=torch.tensor([2, 3, 4]),
        translations=torch.tensor(
            [[0.10, 0.20, 1.0], [float("nan"), 0.0, 0.0], [0.0, 0.0, 1.0]]
        ),
    )
    features = (
        torch.cat([torch.ones(1, 1, 3, 4), torch.full((1, 1, 3, 4), 0.5)], dim=1),
    )

    observations = build_detection_observations(
        predictions,
        features,
        _AppearanceOnlyHead(),
        image_size=(6, 8),
        extrinsic_w2c=torch.eye(4)[:3],
        score_threshold=0.50,
        max_detections=8,
    )

    assert [item.prediction_index for item in observations] == [0, 1]
    assert [item.label for item in observations] == [2, 3]
    torch.testing.assert_close(
        observations[0].detection.center_world, torch.tensor([0.10, 0.20, 1.0])
    )
    assert observations[1].detection.center_world is None
    assert all(torch.isfinite(item.detection.embedding).all() for item in observations)


def test_observation_builder_handles_an_empty_detector_result() -> None:
    predictions = InstanceData(
        bboxes=torch.empty(0, 4),
        scores=torch.empty(0),
        labels=torch.empty(0, dtype=torch.long),
        translations=torch.empty(0, 3),
    )
    features = (torch.ones(1, 2, 3, 4),)

    observations = build_detection_observations(
        predictions,
        features,
        _AppearanceOnlyHead(),
        image_size=(6, 8),
        extrinsic_w2c=torch.eye(4)[:3],
        score_threshold=0.50,
        max_detections=8,
    )

    assert observations == ()


def test_descriptor_loading_has_no_implicit_random_or_provenance_fallback(
    tmp_path,
) -> None:
    head_config = {
        "embedding_dim": 4,
        "hidden_dim": 6,
        "context_radius_m": 0.15,
    }
    with torch.no_grad(), pytest.raises(ValueError, match="checkpoint is required"):
        load_sequence_descriptor(
            None,
            requested_mode="C1",
            allow_untrained=False,
            appearance_dim=2,
            head_config=head_config,
            detector_sha256="correct",
            seed=1,
            device=torch.device("cpu"),
        )

    wrong = tmp_path / "wrong_g10.pth"
    torch.save(
        {
            "schema": CHECKPOINT_SCHEMA,
            "mode": "C1",
            "appearance_dim": 2,
            "provenance": {"g10_checkpoint_sha256": "different"},
        },
        wrong,
    )
    with pytest.raises(ValueError, match="not trained from this G10"):
        load_sequence_descriptor(
            wrong,
            requested_mode="C1",
            allow_untrained=False,
            appearance_dim=2,
            head_config=head_config,
            detector_sha256="correct",
            seed=1,
            device=torch.device("cpu"),
        )
