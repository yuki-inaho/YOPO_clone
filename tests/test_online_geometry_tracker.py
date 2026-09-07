"""Tests for the causal online geometry/appearance tracker."""

from __future__ import annotations

import numpy as np
import pytest
import torch

import yopo.models.tracking.online_tracker as tracker_module
from yopo.models.tracking.online_tracker import (
    AssignmentSolverUnavailable,
    Detection,
    DuplicateAssignmentError,
    OnlineGeometryTracker,
    TrackState,
    TrackerConfig,
)


def _detection(
    x: float,
    *,
    embedding: tuple[float, float] = (1.0, 0.0),
    confidence: float = 0.95,
) -> Detection:
    return Detection(
        center_world=torch.tensor([x, 0.0, 0.0]),
        embedding=torch.tensor(embedding),
        confidence=confidence,
    )


def test_state_transitions_preserve_id_through_short_occlusion() -> None:
    tracker = OnlineGeometryTracker(
        TrackerConfig(min_confirmed_hits=2, max_unobserved_frames=1)
    )

    first = tracker.update([_detection(0.0)], frame_index=0)
    track_id = first.detection_track_ids[0]
    assert tracker.get_track(track_id).state is TrackState.TENTATIVE

    second = tracker.update([_detection(0.01)], frame_index=1)
    assert second.detection_track_ids == (track_id,)
    assert tracker.get_track(track_id).state is TrackState.TRACKING

    missed = tracker.update([], frame_index=2)
    assert missed.unmatched_track_ids == (track_id,)
    assert tracker.get_track(track_id).state is TrackState.TEMPORARILY_UNOBSERVED

    recovered = tracker.update([_detection(0.02)], frame_index=3)
    assert recovered.detection_track_ids == (track_id,)
    assert tracker.get_track(track_id).state is TrackState.TRACKING


def test_unmatched_detection_creates_new_track_and_old_track_terminates() -> None:
    tracker = OnlineGeometryTracker(
        TrackerConfig(geometry_gate_m=0.075, max_unobserved_frames=1)
    )
    old_id = tracker.update([_detection(0.0)], frame_index=0).detection_track_ids[0]

    result = tracker.update([_detection(0.20)], frame_index=1)
    new_id = result.detection_track_ids[0]

    assert new_id != old_id
    assert result.unmatched_track_ids == (old_id,)
    assert tracker.get_track(old_id).state is TrackState.TEMPORARILY_UNOBSERVED
    tracker.update([], frame_index=2)
    assert tracker.get_track(old_id).state is TrackState.TERMINATED
    assert tracker.get_track(old_id).state is not TrackState.HARVEST_CONFIRMED


def test_forbidden_low_cost_edge_does_not_hide_valid_match() -> None:
    tracker = OnlineGeometryTracker()
    track_id = tracker.update([_detection(0.0)], frame_index=0).detection_track_ids[0]

    result = tracker.update(
        [
            _detection(0.0, embedding=(0.49, 0.871722)),
            _detection(0.074, embedding=(0.9, 0.43589)),
        ],
        frame_index=1,
    )

    assert result.matched_track_detection_pairs == ((track_id, 1),)
    assert result.detection_track_ids[1] == track_id
    assert result.detection_track_ids[0] != track_id


def test_missing_geometry_uses_appearance_only_and_retains_last_position() -> None:
    tracker = OnlineGeometryTracker(TrackerConfig(embedding_gate=0.2))
    track_id = tracker.update([_detection(0.03)], frame_index=0).detection_track_ids[0]

    result = tracker.update(
        [
            Detection(
                center_world=None, embedding=torch.tensor([1.0, 0.0]), confidence=0.9
            )
        ],
        frame_index=1,
    )

    assert result.detection_track_ids == (track_id,)
    assert result.created_track_ids == ()
    torch.testing.assert_close(
        tracker.get_track(track_id).center_world, torch.tensor([0.03, 0.0, 0.0])
    )


def test_default_assignment_strategy_remains_legacy_linear_cost() -> None:
    config = TrackerConfig()

    assert config.assignment_strategy == "linear_cost_v1"


def test_explicit_legacy_strategy_matches_default_updates() -> None:
    default = OnlineGeometryTracker(TrackerConfig())
    explicit = OnlineGeometryTracker(
        TrackerConfig(assignment_strategy="linear_cost_v1")
    )
    sequence = [
        [_detection(0.0), _detection(0.04, embedding=(0.0, 1.0))],
        [_detection(0.01), _detection(0.05, embedding=(0.0, 1.0))],
        [_detection(0.02), _detection(0.06, embedding=(0.0, 1.0))],
    ]

    default_updates = [
        default.update(detections, frame_index=frame_index)
        for frame_index, detections in enumerate(sequence)
    ]
    explicit_updates = [
        explicit.update(detections, frame_index=frame_index)
        for frame_index, detections in enumerate(sequence)
    ]

    assert explicit_updates == default_updates
    assert explicit.tracks == default.tracks


def test_soft_affinity_prioritizes_a_near_candidate_over_far_appearance() -> None:
    tracker = OnlineGeometryTracker(
        TrackerConfig(
            assignment_strategy="soft_affinity_v1",
            geometry_gate_m=0.15,
            geometry_weight=0.75,
            embedding_weight=0.25,
            geometry_affinity_sigma_m=0.04,
            embedding_affinity_temperature=0.20,
        )
    )
    track_id = tracker.update([_detection(0.0)], frame_index=0).detection_track_ids[0]

    update = tracker.update(
        [
            _detection(0.02, embedding=(0.8, 0.6)),
            _detection(0.10, embedding=(1.0, 0.0)),
        ],
        frame_index=1,
    )

    assert update.matched_track_detection_pairs == ((track_id, 0),)
    assert update.detection_track_ids[0] == track_id
    assert update.detection_track_ids[1] != track_id


def test_soft_affinity_can_match_beyond_legacy_gate_with_explicit_wider_gate() -> None:
    tracker = OnlineGeometryTracker(
        TrackerConfig(
            assignment_strategy="soft_affinity_v1",
            geometry_gate_m=0.15,
            geometry_affinity_sigma_m=0.05,
            embedding_affinity_temperature=0.25,
        )
    )
    track_id = tracker.update([_detection(0.0)], frame_index=0).detection_track_ids[0]

    update = tracker.update([_detection(0.10)], frame_index=1)

    assert update.detection_track_ids == (track_id,)
    assert update.created_track_ids == ()


def test_soft_affinity_missing_geometry_is_appearance_only() -> None:
    tracker = OnlineGeometryTracker(
        TrackerConfig(
            assignment_strategy="soft_affinity_v1",
            geometry_gate_m=0.15,
            geometry_affinity_sigma_m=0.05,
            embedding_affinity_temperature=0.25,
        )
    )
    track_id = tracker.update([_detection(0.0)], frame_index=0).detection_track_ids[0]

    update = tracker.update(
        [Detection(None, torch.tensor([1.0, 0.0]), 0.9)], frame_index=1
    )

    assert update.detection_track_ids == (track_id,)
    assert update.created_track_ids == ()


@pytest.mark.parametrize(
    "override",
    [
        {"assignment_strategy": "unknown"},
        {"geometry_affinity_sigma_m": 0.0},
        {"embedding_affinity_temperature": 0.0},
    ],
)
def test_soft_affinity_config_is_validated(override: dict) -> None:
    with pytest.raises(ValueError, match="strategy|sigma|temperature"):
        TrackerConfig(**override)


def test_low_confidence_match_does_not_update_prototype() -> None:
    tracker = OnlineGeometryTracker(
        TrackerConfig(
            prototype_update_confidence=0.80,
            prototype_momentum=0.50,
            embedding_gate=2.0,
        )
    )
    track_id = tracker.update([_detection(0.0)], frame_index=0).detection_track_ids[0]
    original = tracker.get_track(track_id).prototype.clone()

    tracker.update(
        [_detection(0.01, embedding=(0.8, 0.2), confidence=0.20)], frame_index=1
    )
    torch.testing.assert_close(tracker.get_track(track_id).prototype, original)

    tracker.update(
        [_detection(0.02, embedding=(0.8, 0.2), confidence=0.90)], frame_index=2
    )
    assert not torch.allclose(tracker.get_track(track_id).prototype, original)


def test_harvest_requires_an_explicit_event() -> None:
    tracker = OnlineGeometryTracker(TrackerConfig(max_unobserved_frames=0))
    disappeared_id = tracker.update(
        [_detection(0.0)], frame_index=0
    ).detection_track_ids[0]
    tracker.update([], frame_index=1)
    assert tracker.get_track(disappeared_id).state is TrackState.TERMINATED

    harvested_id = tracker.update([_detection(0.3)], frame_index=2).detection_track_ids[
        0
    ]
    tracker.confirm_harvest(harvested_id)
    assert tracker.get_track(harvested_id).state is TrackState.HARVEST_CONFIRMED


def test_config_controls_confirmation_and_occlusion_lifetime() -> None:
    config = TrackerConfig(
        min_confirmed_hits=3,
        max_unobserved_frames=2,
        geometry_gate_m=0.05,
        embedding_gate=0.4,
        prototype_update_confidence=0.7,
        prototype_momentum=0.8,
    )
    tracker = OnlineGeometryTracker(config)
    track_id = tracker.update([_detection(0.0)], frame_index=0).detection_track_ids[0]
    tracker.update([_detection(0.01)], frame_index=1)
    assert tracker.get_track(track_id).state is TrackState.TENTATIVE
    tracker.update([_detection(0.02)], frame_index=2)
    assert tracker.get_track(track_id).state is TrackState.TRACKING
    tracker.update([], frame_index=3)
    tracker.update([], frame_index=4)
    assert tracker.get_track(track_id).state is TrackState.TEMPORARILY_UNOBSERVED
    tracker.update([], frame_index=5)
    assert tracker.get_track(track_id).state is TrackState.TERMINATED


def test_missing_assignment_solver_fails_without_greedy_fallback(monkeypatch) -> None:
    tracker = OnlineGeometryTracker()
    tracker.update([_detection(0.0)], frame_index=0)
    monkeypatch.setattr(tracker_module, "_linear_sum_assignment", None)

    with pytest.raises(AssignmentSolverUnavailable, match="scipy"):
        tracker.update([_detection(0.01)], frame_index=1)

    assert tracker.last_frame_index == 0


def test_duplicate_solver_assignment_rejects_entire_frame() -> None:
    def duplicate_solver(_cost: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
        return np.array([0, 0]), np.array([0, 1])

    tracker = OnlineGeometryTracker(assignment_solver=duplicate_solver)
    initial = tracker.update([_detection(0.0), _detection(0.03)], frame_index=0)
    before = tuple(
        tracker.get_track(track_id) for track_id in initial.created_track_ids
    )

    with pytest.raises(DuplicateAssignmentError, match="duplicate"):
        tracker.update([_detection(0.01), _detection(0.04)], frame_index=1)

    after = tuple(tracker.get_track(track_id) for track_id in initial.created_track_ids)
    assert after == before
    assert tracker.last_frame_index == 0


@pytest.mark.parametrize(
    "detection",
    [
        Detection(torch.tensor([float("nan"), 0.0, 0.0]), torch.ones(2), 0.9),
        Detection(torch.zeros(3), torch.tensor([float("inf"), 0.0]), 0.9),
    ],
)
def test_nonfinite_detection_rejects_frame(detection: Detection) -> None:
    tracker = OnlineGeometryTracker()
    with pytest.raises(ValueError, match="finite"):
        tracker.update([detection], frame_index=0)
    assert tracker.last_frame_index is None
