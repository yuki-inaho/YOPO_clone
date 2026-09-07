"""Geometry pseudo-label association and order-invariant ID context.

This module is deliberately independent of the detector runner.  It consumes
metric camera/world centres and per-instance appearance features, which keeps
the sequence experiment reusable without changing the frame detector.
"""

from __future__ import annotations

from dataclasses import dataclass

import torch
from torch import Tensor, nn
from torch.nn import functional as F


@dataclass(frozen=True)
class GeometryExclusion:
    """An input excluded before matching, with a machine-readable reason."""

    index: int
    reason: str


@dataclass(frozen=True)
class AssociationResult:
    """One-to-one pseudo associations and inputs not assigned to a pair."""

    matches: Tensor
    distances_m: Tensor
    unmatched_source: Tensor
    unmatched_target: Tensor
    excluded_source: tuple[GeometryExclusion, ...]
    excluded_target: tuple[GeometryExclusion, ...]


def _validate_points(points: Tensor, name: str) -> None:
    if points.ndim != 2 or points.shape[1] != 3:
        raise ValueError(f"{name} must have shape [N, 3], got {tuple(points.shape)}")
    if not points.is_floating_point():
        raise ValueError(f"{name} must be floating point")


def camera_to_world_points(points_camera: Tensor, world_to_camera: Tensor) -> Tensor:
    """Transform row-vector camera points using a metric W2C extrinsic.

    The accepted convention is ``p_camera = R @ p_world + t``.  Invalid or
    non-rigid extrinsics fail explicitly instead of producing pseudo labels.
    """

    _validate_points(points_camera, "points_camera")
    if world_to_camera.shape != (4, 4):
        raise ValueError(
            "world_to_camera must have shape [4, 4], "
            f"got {tuple(world_to_camera.shape)}"
        )
    if (
        not world_to_camera.is_floating_point()
        or not torch.isfinite(world_to_camera).all()
    ):
        raise ValueError("world_to_camera must be finite floating point")
    expected_tail = world_to_camera.new_tensor([0.0, 0.0, 0.0, 1.0])
    if not torch.allclose(world_to_camera[3], expected_tail, atol=1e-6, rtol=0.0):
        raise ValueError("world_to_camera must have homogeneous final row [0,0,0,1]")
    rotation = world_to_camera[:3, :3]
    identity = torch.eye(3, dtype=rotation.dtype, device=rotation.device)
    # Geometry retains its declared precision even when frozen feature
    # extraction runs inside a surrounding BF16 autocast context.
    with torch.autocast(device_type=world_to_camera.device.type, enabled=False):
        proper_rotation = torch.allclose(
            rotation.transpose(0, 1) @ rotation,
            identity,
            atol=1e-5,
            rtol=1e-5,
        ) and torch.allclose(
            torch.linalg.det(rotation),
            rotation.new_tensor(1.0),
            atol=1e-5,
            rtol=1e-5,
        )
        if not proper_rotation:
            raise ValueError(
                "world_to_camera rotation must be a proper orthonormal matrix"
            )
        translation = world_to_camera[:3, 3]
        return (points_camera - translation) @ rotation


def _empty_indices(reference: Tensor) -> Tensor:
    return torch.empty(0, dtype=torch.long, device=reference.device)


def stable_pairwise_distance(source: Tensor, target: Tensor) -> Tensor:
    """Compute Euclidean distances without the cancellation-prone MM identity."""

    if source.ndim != 2 or target.ndim != 2 or source.shape[1] != target.shape[1]:
        raise ValueError("source and target must have shape [N,D] and [M,D]")
    if source.device != target.device or source.dtype != target.dtype:
        raise ValueError("source and target must use the same device and dtype")
    if not source.is_floating_point():
        raise ValueError("source and target must be floating point")
    compute_dtype = (
        torch.float32
        if source.dtype in {torch.float16, torch.bfloat16}
        else source.dtype
    )
    with torch.autocast(device_type=source.device.type, enabled=False):
        delta = source.to(compute_dtype).unsqueeze(1) - target.to(
            compute_dtype
        ).unsqueeze(0)
        return torch.linalg.vector_norm(delta, dim=-1)


def mutual_nearest_association(
    source_world: Tensor,
    target_world: Tensor,
    *,
    max_distance_m: float = 0.075,
    ambiguity_margin_m: float = 0.0,
) -> AssociationResult:
    """Associate finite centres by gated, unambiguous mutual nearest neighbor.

    A pair is retained only when each endpoint selects the other and both
    nearest-neighbor gaps exceed ``ambiguity_margin_m``.  Rejected valid points
    remain unmatched; nonfinite points are reported separately and never used
    as implicit negatives.
    """

    _validate_points(source_world, "source_world")
    _validate_points(target_world, "target_world")
    if source_world.device != target_world.device:
        raise ValueError("source_world and target_world must use the same device")
    if max_distance_m <= 0 or ambiguity_margin_m < 0:
        raise ValueError(
            "distance gate must be positive and ambiguity margin nonnegative"
        )

    source_finite = torch.isfinite(source_world).all(dim=1)
    target_finite = torch.isfinite(target_world).all(dim=1)
    source_indices = source_finite.nonzero(as_tuple=False).flatten()
    target_indices = target_finite.nonzero(as_tuple=False).flatten()
    excluded_source = tuple(
        GeometryExclusion(int(index), "nonfinite_geometry")
        for index in (~source_finite).nonzero(as_tuple=False).flatten().tolist()
    )
    excluded_target = tuple(
        GeometryExclusion(int(index), "nonfinite_geometry")
        for index in (~target_finite).nonzero(as_tuple=False).flatten().tolist()
    )

    if source_indices.numel() == 0 or target_indices.numel() == 0:
        empty = _empty_indices(source_world)
        return AssociationResult(
            matches=empty.reshape(0, 2),
            distances_m=source_world.new_empty(0),
            unmatched_source=source_indices,
            unmatched_target=target_indices,
            excluded_source=excluded_source,
            excluded_target=excluded_target,
        )

    distances = stable_pairwise_distance(
        source_world[source_indices], target_world[target_indices]
    )
    source_values, source_nearest = distances.min(dim=1)
    _, target_nearest = distances.min(dim=0)

    source_gap = torch.full_like(source_values, torch.inf)
    if distances.shape[1] > 1:
        source_two = distances.topk(k=2, dim=1, largest=False).values
        source_gap = source_two[:, 1] - source_two[:, 0]
    target_gap = torch.full(
        (distances.shape[1],),
        torch.inf,
        dtype=distances.dtype,
        device=distances.device,
    )
    if distances.shape[0] > 1:
        target_two = distances.topk(k=2, dim=0, largest=False).values
        target_gap = target_two[1] - target_two[0]

    local_source = torch.arange(distances.shape[0], device=distances.device)
    mutual = target_nearest[source_nearest] == local_source
    accepted = (
        mutual
        & (source_values <= max_distance_m)
        & (source_gap > ambiguity_margin_m)
        & (target_gap[source_nearest] > ambiguity_margin_m)
    )
    accepted_source = local_source[accepted]
    accepted_target = source_nearest[accepted]
    matches = torch.stack(
        [source_indices[accepted_source], target_indices[accepted_target]], dim=1
    )
    matched_source_mask = torch.zeros_like(source_finite)
    matched_target_mask = torch.zeros_like(target_finite)
    if matches.numel() > 0:
        matched_source_mask[matches[:, 0]] = True
        matched_target_mask[matches[:, 1]] = True

    return AssociationResult(
        matches=matches,
        distances_m=source_values[accepted],
        unmatched_source=(source_finite & ~matched_source_mask)
        .nonzero(as_tuple=False)
        .flatten(),
        unmatched_target=(target_finite & ~matched_target_mask)
        .nonzero(as_tuple=False)
        .flatten(),
        excluded_source=excluded_source,
        excluded_target=excluded_target,
    )


class LocalGeometryContext(nn.Module):
    """Aggregate neighbor messages with a permutation-invariant masked mean."""

    def __init__(
        self,
        appearance_dim: int,
        context_dim: int,
        *,
        radius_m: float = 0.15,
    ) -> None:
        super().__init__()
        if appearance_dim <= 0 or context_dim <= 0 or radius_m <= 0:
            raise ValueError("dimensions and radius_m must be positive")
        self.radius_m = float(radius_m)
        self.message = nn.Sequential(
            nn.Linear(appearance_dim + 1, context_dim),
            nn.ReLU(),
            nn.Linear(context_dim, context_dim),
        )

    def forward(self, appearance: Tensor, positions_world: Tensor) -> Tensor:
        _validate_points(positions_world, "positions_world")
        if appearance.ndim != 2 or appearance.shape[0] != positions_world.shape[0]:
            raise ValueError("appearance must have shape [N, D] matching positions")
        count = positions_world.shape[0]
        if count == 0:
            return appearance.new_empty((0, self.message[-1].out_features))
        point_valid = torch.isfinite(positions_world).all(dim=1)
        safe_positions = torch.where(
            point_valid.unsqueeze(1), positions_world, torch.zeros_like(positions_world)
        )
        relative = safe_positions.unsqueeze(0) - safe_positions.unsqueeze(1)
        distance = torch.linalg.vector_norm(relative, dim=-1, keepdim=True)
        neighbors = appearance.unsqueeze(0).expand(count, -1, -1)
        messages = self.message(torch.cat([neighbors, distance], dim=-1))
        valid = (
            (distance[..., 0] > 0)
            & (distance[..., 0] <= self.radius_m)
            & point_valid.unsqueeze(1)
            & point_valid.unsqueeze(0)
        )
        weights = valid.unsqueeze(-1).to(messages.dtype)
        denominator = weights.sum(dim=1).clamp_min(1.0)
        return (messages * weights).sum(dim=1) / denominator


class GeometryContextIdentityHead(nn.Module):
    """Per-object appearance descriptor with optional invariant local context."""

    _MODE_CONTEXT = {
        "B0": False,
        "B2": False,
        "C0": True,
        "C1": True,
    }

    def __init__(
        self,
        appearance_dim: int,
        embedding_dim: int,
        *,
        mode: str = "C1",
        hidden_dim: int = 128,
        context_radius_m: float = 0.15,
        fusion_strategy: str = "legacy_concat_v1",
    ) -> None:
        super().__init__()
        if mode not in self._MODE_CONTEXT:
            raise ValueError(
                f"mode must be one of {tuple(self._MODE_CONTEXT)}, got {mode!r}"
            )
        if min(appearance_dim, embedding_dim, hidden_dim) <= 0:
            raise ValueError("feature dimensions must be positive")
        if fusion_strategy not in {"legacy_concat_v1", "residual_gated_v1"}:
            raise ValueError(f"unsupported fusion strategy: {fusion_strategy!r}")
        self.mode = mode
        self.fusion_strategy = fusion_strategy
        self.appearance_encoder = nn.Sequential(
            nn.Linear(appearance_dim, hidden_dim), nn.ReLU()
        )
        self.context_encoder = LocalGeometryContext(
            appearance_dim, hidden_dim, radius_m=context_radius_m
        )
        use_context = self._MODE_CONTEXT[mode]
        if fusion_strategy == "legacy_concat_v1":
            fusion_dim = hidden_dim * (1 + int(use_context))
            self.embedding = nn.Linear(fusion_dim, embedding_dim)
        else:
            # Both appearance-only and context modes own the same state layout,
            # which permits a strict B0 -> C0 warm start.  A bias-free context
            # projection preserves an exact zero residual for isolated objects.
            self.appearance_projection = nn.Linear(hidden_dim, embedding_dim)
            self.context_projection = nn.Linear(hidden_dim, embedding_dim, bias=False)
            self.context_gate = nn.Parameter(torch.zeros(()))

    def forward(self, appearance: Tensor, positions_world: Tensor) -> Tensor:
        if appearance.ndim != 2:
            raise ValueError("appearance must have shape [N, D]")
        _validate_points(positions_world, "positions_world")
        if appearance.shape[0] != positions_world.shape[0]:
            raise ValueError("appearance and positions_world must have equal length")
        use_context = self._MODE_CONTEXT[self.mode]
        encoded_appearance = self.appearance_encoder(appearance)
        if self.fusion_strategy == "legacy_concat_v1":
            features = [encoded_appearance]
            if use_context:
                # Context is intentionally independent of ``self.training`` and
                # is therefore retained during evaluation/inference.
                features.append(self.context_encoder(appearance, positions_world))
            return F.normalize(self.embedding(torch.cat(features, dim=-1)), dim=-1)
        embedding = self.appearance_projection(encoded_appearance)
        if use_context:
            context = self.context_encoder(appearance, positions_world)
            embedding = embedding + torch.tanh(self.context_gate) * (
                self.context_projection(context)
            )
        return F.normalize(embedding, dim=-1)


def matched_identity_loss(
    source_embeddings: Tensor,
    target_embeddings: Tensor,
    matches: Tensor,
    *,
    temperature: float = 0.07,
    negative_scope: str = "matched_only",
) -> Tensor:
    """Bidirectional InfoNCE with an explicit in-frame negative scope."""

    if matches.ndim != 2 or matches.shape[1] != 2 or matches.shape[0] == 0:
        raise ValueError("matches must have non-empty shape [M, 2]")
    if temperature <= 0:
        raise ValueError("temperature must be positive")
    if negative_scope not in {"matched_only", "all_detections"}:
        raise ValueError("negative_scope must be 'matched_only' or 'all_detections'")
    source = F.normalize(source_embeddings, dim=-1)
    target = F.normalize(target_embeddings, dim=-1)
    if negative_scope == "matched_only":
        source = source[matches[:, 0]]
        target = target[matches[:, 1]]
        logits = source @ target.transpose(0, 1) / temperature
        labels = torch.arange(matches.shape[0], device=matches.device)
        return 0.5 * (
            F.cross_entropy(logits, labels) + F.cross_entropy(logits.T, labels)
        )
    logits = source @ target.transpose(0, 1) / temperature
    source_loss = F.cross_entropy(logits[matches[:, 0]], matches[:, 1])
    target_loss = F.cross_entropy(logits.T[matches[:, 1]], matches[:, 0])
    return 0.5 * (source_loss + target_loss)
