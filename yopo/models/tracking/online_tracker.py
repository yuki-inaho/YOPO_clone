"""Causal online tracker with explicit lifecycle and harvest semantics."""

from __future__ import annotations

from dataclasses import dataclass
from enum import Enum
from typing import Callable, Sequence

import numpy as np
import torch
from torch import Tensor
from torch.nn import functional as F

from .geometry_context import stable_pairwise_distance

try:
    from scipy.optimize import linear_sum_assignment as _linear_sum_assignment
except ImportError:  # pragma: no cover - exercised by monkeypatch in tests
    _linear_sum_assignment = None


class AssignmentSolverUnavailable(RuntimeError):
    """Raised when the configured one-to-one solver cannot be loaded."""


class DuplicateAssignmentError(RuntimeError):
    """Raised before mutation when a solver violates one-to-one assignment."""


class TrackState(str, Enum):
    TENTATIVE = "tentative"
    TRACKING = "tracking"
    TEMPORARILY_UNOBSERVED = "temporarily_unobserved"
    TERMINATED = "terminated"
    HARVEST_CONFIRMED = "harvest_confirmed"


@dataclass(frozen=True)
class TrackerConfig:
    """All state, gate and prototype policies for :class:`OnlineGeometryTracker`."""

    geometry_gate_m: float = 0.075
    embedding_gate: float = 0.50
    geometry_weight: float = 0.50
    embedding_weight: float = 0.50
    miss_cost: float = 0.55
    new_cost: float = 0.55
    min_confirmed_hits: int = 2
    max_unobserved_frames: int = 2
    prototype_update_confidence: float = 0.70
    prototype_momentum: float = 0.90

    def __post_init__(self) -> None:
        if self.geometry_gate_m <= 0:
            raise ValueError("geometry_gate_m must be positive")
        if not 0 < self.embedding_gate <= 2:
            raise ValueError("embedding_gate must be in (0, 2]")
        if self.geometry_weight < 0 or self.embedding_weight < 0:
            raise ValueError("assignment weights must be nonnegative")
        if self.geometry_weight + self.embedding_weight <= 0:
            raise ValueError("at least one assignment weight must be positive")
        if self.miss_cost < 0 or self.new_cost < 0:
            raise ValueError("miss and new costs must be nonnegative")
        if self.min_confirmed_hits < 1 or self.max_unobserved_frames < 0:
            raise ValueError("hit and unobserved-frame limits are invalid")
        if not 0 <= self.prototype_update_confidence <= 1:
            raise ValueError("prototype_update_confidence must be in [0, 1]")
        if not 0 <= self.prototype_momentum < 1:
            raise ValueError("prototype_momentum must be in [0, 1)")


@dataclass(frozen=True)
class Detection:
    center_world: Tensor | None
    embedding: Tensor
    confidence: float


@dataclass(frozen=True, eq=False)
class TrackSnapshot:
    track_id: int
    state: TrackState
    center_world: Tensor | None
    prototype: Tensor
    hits: int
    misses: int
    last_frame_index: int

    def __eq__(self, other: object) -> bool:
        if not isinstance(other, TrackSnapshot):
            return NotImplemented
        return (
            self.track_id == other.track_id
            and self.state is other.state
            and (
                self.center_world is None
                and other.center_world is None
                or self.center_world is not None
                and other.center_world is not None
                and torch.equal(self.center_world, other.center_world)
            )
            and torch.equal(self.prototype, other.prototype)
            and self.hits == other.hits
            and self.misses == other.misses
            and self.last_frame_index == other.last_frame_index
        )


@dataclass(frozen=True)
class TrackerUpdate:
    frame_index: int
    detection_track_ids: tuple[int, ...]
    matched_track_detection_pairs: tuple[tuple[int, int], ...]
    created_track_ids: tuple[int, ...]
    unmatched_track_ids: tuple[int, ...]


@dataclass
class _Track:
    track_id: int
    state: TrackState
    center_world: Tensor | None
    prototype: Tensor
    hits: int
    misses: int
    last_frame_index: int


AssignmentSolver = Callable[[np.ndarray], tuple[np.ndarray, np.ndarray]]


def partial_linear_assignment(
    cost: Tensor,
    valid: Tensor,
    *,
    miss_cost: float,
    new_cost: float,
    assignment_solver: AssignmentSolver | None = None,
) -> tuple[tuple[int, int], ...]:
    """Solve gated one-to-one matching with explicit miss and new choices."""

    if cost.ndim != 2 or valid.shape != cost.shape or valid.dtype != torch.bool:
        raise ValueError("cost and valid must have equal [tracks,detections] shapes")
    if not torch.isfinite(cost).all() or miss_cost < 0 or new_cost < 0:
        raise ValueError("assignment costs must be finite and nonnegative")
    track_count, detection_count = cost.shape
    if track_count == 0 or detection_count == 0:
        return ()
    size = track_count + detection_count
    cost_numpy = cost.detach().cpu().numpy().astype(np.float64, copy=False)
    valid_numpy = valid.detach().cpu().numpy()
    scale = max(float(np.abs(cost_numpy).max()), miss_cost, new_cost, 1.0)
    forbidden = scale * (size + 1)
    augmented = np.full((size, size), forbidden, dtype=np.float64)
    augmented[:track_count, :detection_count] = np.where(
        valid_numpy, cost_numpy, forbidden
    )
    for track_index in range(track_count):
        augmented[track_index, detection_count + track_index] = miss_cost
    for detection_index in range(detection_count):
        augmented[track_count + detection_index, detection_index] = new_cost
    augmented[track_count:, detection_count:] = 0.0

    solver = assignment_solver or _linear_sum_assignment
    if solver is None:
        raise AssignmentSolverUnavailable(
            "scipy.optimize.linear_sum_assignment is required; "
            "no greedy fallback is permitted"
        )
    rows, columns = solver(augmented)
    rows = np.asarray(rows)
    columns = np.asarray(columns)
    if rows.ndim != 1 or columns.ndim != 1 or rows.shape != columns.shape:
        raise DuplicateAssignmentError(
            "assignment solver returned invalid index arrays"
        )
    if len(set(rows.tolist())) != len(rows) or len(set(columns.tolist())) != len(
        columns
    ):
        raise DuplicateAssignmentError(
            "assignment solver returned duplicate assignments"
        )
    if (
        np.any(rows < 0)
        or np.any(rows >= size)
        or np.any(columns < 0)
        or np.any(columns >= size)
    ):
        raise DuplicateAssignmentError(
            "assignment solver returned out-of-range indices"
        )
    matches = tuple(
        (int(row), int(column))
        for row, column in zip(rows.tolist(), columns.tolist())
        if row < track_count and column < detection_count
    )
    if any(not bool(valid[row, column]) for row, column in matches):
        raise RuntimeError("partial assignment selected a forbidden edge")
    return matches


class OnlineGeometryTracker:
    """Track detections causally with gated geometry/appearance assignment."""

    def __init__(
        self,
        config: TrackerConfig | None = None,
        *,
        assignment_solver: AssignmentSolver | None = None,
    ) -> None:
        self.config = config or TrackerConfig()
        self._assignment_solver = assignment_solver
        self._tracks: dict[int, _Track] = {}
        self._next_track_id = 1
        self._last_frame_index: int | None = None

    @property
    def last_frame_index(self) -> int | None:
        return self._last_frame_index

    @property
    def tracks(self) -> tuple[TrackSnapshot, ...]:
        return tuple(self.get_track(track_id) for track_id in sorted(self._tracks))

    def get_track(self, track_id: int) -> TrackSnapshot:
        if track_id not in self._tracks:
            raise KeyError(f"unknown track_id {track_id}")
        track = self._tracks[track_id]
        return TrackSnapshot(
            track_id=track.track_id,
            state=track.state,
            center_world=(
                track.center_world.detach().clone()
                if track.center_world is not None
                else None
            ),
            prototype=track.prototype.detach().clone(),
            hits=track.hits,
            misses=track.misses,
            last_frame_index=track.last_frame_index,
        )

    def confirm_harvest(self, track_id: int) -> TrackSnapshot:
        if track_id not in self._tracks:
            raise KeyError(f"unknown track_id {track_id}")
        track = self._tracks[track_id]
        if track.state is TrackState.TERMINATED:
            raise ValueError("a terminated track cannot be confirmed as harvested")
        track.state = TrackState.HARVEST_CONFIRMED
        return self.get_track(track_id)

    def update(
        self, detections: Sequence[Detection], *, frame_index: int
    ) -> TrackerUpdate:
        """Apply one frame atomically after validating solver output."""

        if self._last_frame_index is not None and frame_index <= self._last_frame_index:
            raise ValueError("frame_index must increase strictly")
        normalized = self._validate_detections(detections)
        active_tracks = [
            track
            for track in self._tracks.values()
            if track.state not in (TrackState.TERMINATED, TrackState.HARVEST_CONFIRMED)
        ]
        active_tracks.sort(key=lambda track: track.track_id)

        assignments = self._plan_assignments(active_tracks, normalized)
        matched_track_rows = {track_row for track_row, _ in assignments}
        matched_detection_rows = {detection_row for _, detection_row in assignments}
        detection_track_ids: list[int | None] = [None] * len(normalized)

        for track_row, detection_row in assignments:
            track = active_tracks[track_row]
            detection = normalized[detection_row]
            if detection.center_world is not None:
                track.center_world = detection.center_world
            if detection.confidence >= self.config.prototype_update_confidence:
                momentum = self.config.prototype_momentum
                track.prototype = F.normalize(
                    momentum * track.prototype + (1.0 - momentum) * detection.embedding,
                    dim=0,
                )
            track.hits += 1
            track.misses = 0
            track.last_frame_index = frame_index
            if track.hits >= self.config.min_confirmed_hits:
                track.state = TrackState.TRACKING
            detection_track_ids[detection_row] = track.track_id

        unmatched_track_ids: list[int] = []
        for track_row, track in enumerate(active_tracks):
            if track_row in matched_track_rows:
                continue
            track.misses += 1
            track.last_frame_index = frame_index
            if track.misses > self.config.max_unobserved_frames:
                track.state = TrackState.TERMINATED
            else:
                track.state = TrackState.TEMPORARILY_UNOBSERVED
            unmatched_track_ids.append(track.track_id)

        created_track_ids: list[int] = []
        for detection_row, detection in enumerate(normalized):
            if detection_row in matched_detection_rows:
                continue
            track_id = self._next_track_id
            self._next_track_id += 1
            state = (
                TrackState.TRACKING
                if self.config.min_confirmed_hits == 1
                else TrackState.TENTATIVE
            )
            self._tracks[track_id] = _Track(
                track_id=track_id,
                state=state,
                center_world=detection.center_world,
                prototype=detection.embedding,
                hits=1,
                misses=0,
                last_frame_index=frame_index,
            )
            detection_track_ids[detection_row] = track_id
            created_track_ids.append(track_id)

        self._last_frame_index = frame_index
        return TrackerUpdate(
            frame_index=frame_index,
            detection_track_ids=tuple(
                int(track_id) for track_id in detection_track_ids
            ),
            matched_track_detection_pairs=tuple(
                (active_tracks[track_row].track_id, detection_row)
                for track_row, detection_row in assignments
            ),
            created_track_ids=tuple(created_track_ids),
            unmatched_track_ids=tuple(unmatched_track_ids),
        )

    def _validate_detections(
        self, detections: Sequence[Detection]
    ) -> tuple[Detection, ...]:
        normalized: list[Detection] = []
        embedding_dim: int | None = None
        for index, detection in enumerate(detections):
            if (
                detection.center_world is not None
                and detection.center_world.shape != (3,)
                or detection.embedding.ndim != 1
            ):
                raise ValueError(f"detection {index} has invalid tensor shape")
            if (
                detection.center_world is not None
                and not torch.isfinite(detection.center_world).all()
                or not torch.isfinite(detection.embedding).all()
            ):
                raise ValueError(f"detection {index} tensors must be finite")
            if torch.linalg.vector_norm(detection.embedding) <= 0:
                raise ValueError(f"detection {index} embedding must have nonzero norm")
            if not 0 <= detection.confidence <= 1:
                raise ValueError(f"detection {index} confidence must be in [0, 1]")
            if embedding_dim is None:
                embedding_dim = detection.embedding.numel()
            elif detection.embedding.numel() != embedding_dim:
                raise ValueError(
                    "all detection embeddings must have the same dimension"
                )
            normalized.append(
                Detection(
                    center_world=(
                        detection.center_world.detach().clone()
                        if detection.center_world is not None
                        else None
                    ),
                    embedding=F.normalize(detection.embedding.detach().clone(), dim=0),
                    confidence=float(detection.confidence),
                )
            )
        active_dimensions = {
            track.prototype.numel()
            for track in self._tracks.values()
            if track.state not in (TrackState.TERMINATED, TrackState.HARVEST_CONFIRMED)
        }
        if normalized and active_dimensions and active_dimensions != {embedding_dim}:
            raise ValueError("detection embedding dimension differs from active tracks")
        return tuple(normalized)

    def _plan_assignments(
        self, active_tracks: Sequence[_Track], detections: Sequence[Detection]
    ) -> tuple[tuple[int, int], ...]:
        if not active_tracks or not detections:
            return ()
        prototypes = torch.stack([track.prototype for track in active_tracks])
        detection_embeddings = torch.stack([item.embedding for item in detections])
        centers = torch.stack(
            [
                track.center_world
                if track.center_world is not None
                else track.prototype.new_zeros(3)
                for track in active_tracks
            ]
        )
        detection_centers = torch.stack(
            [
                item.center_world
                if item.center_world is not None
                else item.embedding.new_zeros(3)
                for item in detections
            ]
        )
        geometry_available = torch.tensor(
            [track.center_world is not None for track in active_tracks],
            dtype=torch.bool,
            device=prototypes.device,
        ).unsqueeze(1) & torch.tensor(
            [item.center_world is not None for item in detections],
            dtype=torch.bool,
            device=prototypes.device,
        ).unsqueeze(0)
        geometry_distance = stable_pairwise_distance(centers, detection_centers)
        embedding_distance = 1.0 - prototypes @ detection_embeddings.transpose(0, 1)
        valid = (embedding_distance <= self.config.embedding_gate) & (
            ~geometry_available | (geometry_distance <= self.config.geometry_gate_m)
        )
        geometry_weight = geometry_available * self.config.geometry_weight
        total_weight = geometry_weight + self.config.embedding_weight
        valid &= total_weight > 0
        cost_dtype = geometry_distance.dtype
        cost = (
            geometry_weight * geometry_distance / self.config.geometry_gate_m
            + self.config.embedding_weight
            * embedding_distance
            / self.config.embedding_gate
        ) / total_weight.clamp_min(torch.finfo(cost_dtype).eps)
        return partial_linear_assignment(
            cost.to(cost_dtype),
            valid,
            miss_cost=self.config.miss_cost,
            new_cost=self.config.new_cost,
            assignment_solver=self._assignment_solver,
        )
