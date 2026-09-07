"""Geometry-derived pseudo association and identity-context contracts."""

from __future__ import annotations

import pytest
import torch

from yopo.models.tracking.geometry_context import (
    GeometryContextIdentityHead,
    LocalGeometryContext,
    camera_to_world_points,
    matched_identity_loss,
    mutual_nearest_association,
    stable_pairwise_distance,
)


def test_camera_to_world_points_uses_world_to_camera_extrinsic() -> None:
    world_to_camera = torch.eye(4, dtype=torch.float64)
    world_to_camera[:3, 3] = torch.tensor([1.0, -2.0, 0.5])
    camera_points = torch.tensor(
        [[2.0, 0.0, 1.5], [1.0, -2.0, 0.5]], dtype=torch.float64
    )

    world_points = camera_to_world_points(camera_points, world_to_camera)

    torch.testing.assert_close(
        world_points,
        torch.tensor([[1.0, 2.0, 1.0], [0.0, 0.0, 0.0]], dtype=torch.float64),
    )


def test_camera_to_world_points_retains_float32_inside_autocast() -> None:
    points = torch.tensor([[0.1, 0.2, 0.3]], dtype=torch.float32)
    extrinsic = torch.eye(4, dtype=torch.float32)

    with torch.autocast(device_type="cpu", dtype=torch.bfloat16):
        result = camera_to_world_points(points, extrinsic)

    assert result.dtype is torch.float32
    torch.testing.assert_close(result, points)


@pytest.mark.parametrize(
    "extrinsic",
    [torch.zeros(3, 3), torch.full((4, 4), float("nan")), torch.zeros(4, 4)],
)
def test_camera_to_world_rejects_invalid_extrinsic(extrinsic: torch.Tensor) -> None:
    with pytest.raises(ValueError, match="world_to_camera"):
        camera_to_world_points(torch.zeros(1, 3), extrinsic)


def test_mutual_nearest_enforces_75mm_gate_and_unmatched() -> None:
    source = torch.tensor([[0.0, 0.0, 0.0], [0.20, 0.0, 0.0]])
    target = torch.tensor([[0.074, 0.0, 0.0], [0.40, 0.0, 0.0]])

    result = mutual_nearest_association(source, target, max_distance_m=0.075)

    assert result.matches.tolist() == [[0, 0]]
    torch.testing.assert_close(result.distances_m, torch.tensor([0.074]))
    assert result.unmatched_source.tolist() == [1]
    assert result.unmatched_target.tolist() == [1]


def test_stable_distance_matches_float64_at_large_world_origin() -> None:
    source = torch.tensor(
        [[-36.62561, -41.490505, -13.612418], [-36.60061, -41.490505, -13.612418]],
        dtype=torch.float32,
    )
    target = torch.tensor(
        [[-36.62061, -41.490505, -13.612418], [-36.52561, -41.490505, -13.612418]],
        dtype=torch.float32,
    )
    reference = torch.linalg.vector_norm(
        source.double().unsqueeze(1) - target.double().unsqueeze(0), dim=-1
    )

    actual = stable_pairwise_distance(source, target)

    torch.testing.assert_close(actual.double(), reference, atol=1e-7, rtol=1e-5)
    translated = stable_pairwise_distance(source + 20.0, target + 20.0)
    torch.testing.assert_close(translated, actual, atol=5e-6, rtol=1e-5)


@pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA is unavailable")
def test_stable_distance_is_cpu_cuda_consistent() -> None:
    generator = torch.Generator().manual_seed(20260907)
    source = torch.randn(64, 3, generator=generator) * 0.1 - 30.0
    target = torch.randn(71, 3, generator=generator) * 0.1 - 30.0

    cpu = stable_pairwise_distance(source, target)
    cuda = stable_pairwise_distance(source.cuda(), target.cuda()).cpu()

    torch.testing.assert_close(cuda, cpu, atol=2e-6, rtol=1e-5)


def test_mutual_nearest_is_one_to_one_and_ambiguous_pairs_are_unmatched() -> None:
    source = torch.tensor([[0.0, 0.0, 0.0], [0.010, 0.0, 0.0]])
    target = torch.tensor([[0.004, 0.0, 0.0]])

    result = mutual_nearest_association(
        source, target, max_distance_m=0.075, ambiguity_margin_m=0.003
    )

    assert result.matches.shape == (0, 2)
    assert result.unmatched_source.tolist() == [0, 1]
    assert result.unmatched_target.tolist() == [0]


def test_nonfinite_geometry_is_excluded_with_an_explicit_reason() -> None:
    source = torch.tensor([[0.0, 0.0, 0.0], [float("nan"), 0.0, 0.0]])
    target = torch.tensor([[0.01, 0.0, 0.0], [float("inf"), 0.0, 0.0]])

    result = mutual_nearest_association(source, target)

    assert result.matches.tolist() == [[0, 0]]
    assert [(item.index, item.reason) for item in result.excluded_source] == [
        (1, "nonfinite_geometry")
    ]
    assert [(item.index, item.reason) for item in result.excluded_target] == [
        (1, "nonfinite_geometry")
    ]


def test_local_context_is_permutation_equivariant() -> None:
    torch.manual_seed(4)
    module = LocalGeometryContext(appearance_dim=3, context_dim=5, radius_m=0.20)
    appearance = torch.randn(4, 3)
    positions = torch.tensor(
        [[0.0, 0.0, 0.0], [0.03, 0.0, 0.0], [0.08, 0.01, 0.0], [1.0, 0.0, 0.0]]
    )
    permutation = torch.tensor([2, 0, 3, 1])

    original = module(appearance, positions)
    permuted = module(appearance[permutation], positions[permutation])

    torch.testing.assert_close(permuted, original[permutation])


def test_local_context_is_rigid_transform_invariant() -> None:
    torch.manual_seed(5)
    module = LocalGeometryContext(appearance_dim=3, context_dim=5, radius_m=0.20)
    appearance = torch.randn(4, 3)
    positions = torch.tensor(
        [[0.0, 0.0, 0.0], [0.03, 0.0, 0.0], [0.08, 0.01, 0.0], [1.0, 0.0, 0.0]]
    )
    rotation = torch.tensor([[0.0, -1.0, 0.0], [1.0, 0.0, 0.0], [0.0, 0.0, 1.0]])
    transformed = positions @ rotation.T + torch.tensor([10.0, -4.0, 2.0])

    original = module(appearance, positions)
    moved = module(appearance, transformed)

    torch.testing.assert_close(moved, original, atol=2e-6, rtol=1e-5)


@pytest.mark.parametrize("mode", ["B0", "B2", "C0", "C1"])
def test_identity_head_modes_produce_distinct_normalized_embeddings(mode: str) -> None:
    torch.manual_seed(7)
    head = GeometryContextIdentityHead(
        appearance_dim=4, embedding_dim=8, mode=mode, hidden_dim=12
    )
    appearance = torch.tensor([[1.0, 0.0, 0.0, 0.0], [0.0, 1.0, 0.0, 0.0]])
    positions = torch.tensor([[0.0, 0.0, 0.0], [0.01, 0.0, 0.0]])

    embeddings = head(appearance, positions)

    assert embeddings.shape == (2, 8)
    torch.testing.assert_close(embeddings.norm(dim=-1), torch.ones(2))
    assert not torch.allclose(embeddings[0], embeddings[1])


def test_b0_and_b2_descriptors_are_identical_with_the_same_seed() -> None:
    appearance = torch.randn(3, 4)
    positions = torch.randn(3, 3)
    torch.manual_seed(8)
    b0 = GeometryContextIdentityHead(4, 8, mode="B0", hidden_dim=12)
    torch.manual_seed(8)
    b2 = GeometryContextIdentityHead(4, 8, mode="B2", hidden_dim=12)

    torch.testing.assert_close(b0(appearance, positions), b2(appearance, positions))


def test_context_remains_active_during_inference() -> None:
    torch.manual_seed(9)
    head = GeometryContextIdentityHead(
        appearance_dim=2, embedding_dim=4, mode="C0", hidden_dim=8
    ).eval()
    positions = torch.tensor([[0.0, 0.0, 0.0], [0.02, 0.0, 0.0]])
    first = torch.tensor([[1.0, 0.0], [0.0, 1.0]])
    changed_neighbor = torch.tensor([[1.0, 0.0], [1.0, 0.0]])

    with torch.no_grad():
        baseline = head(first, positions)
        changed = head(changed_neighbor, positions)

    assert not torch.allclose(baseline[0], changed[0])


def test_missing_geometry_does_not_poison_appearance_or_valid_context() -> None:
    torch.manual_seed(10)
    head = GeometryContextIdentityHead(
        appearance_dim=2, embedding_dim=4, mode="C1", hidden_dim=8
    )
    appearance = torch.tensor([[1.0, 0.0], [0.0, 1.0], [0.5, 0.5]])
    positions = torch.tensor(
        [[0.0, 0.0, 0.0], [float("nan"), 0.0, 0.0], [0.03, 0.0, 0.0]]
    )

    context = head.context_encoder(appearance, positions)
    embeddings = head(appearance, positions)

    assert torch.isfinite(context).all()
    assert torch.isfinite(embeddings).all()
    torch.testing.assert_close(context[1], torch.zeros_like(context[1]))


def test_matched_identity_loss_is_finite_and_routes_gradients() -> None:
    torch.manual_seed(12)
    head = GeometryContextIdentityHead(
        appearance_dim=3, embedding_dim=6, mode="C1", hidden_dim=10
    )
    source_appearance = torch.randn(3, 3, requires_grad=True)
    target_appearance = torch.randn(3, 3, requires_grad=True)
    source_positions = torch.randn(3, 3, requires_grad=True) * 0.02
    target_positions = torch.randn(3, 3, requires_grad=True) * 0.02
    matches = torch.tensor([[0, 0], [1, 1], [2, 2]])

    loss = matched_identity_loss(
        head(source_appearance, source_positions),
        head(target_appearance, target_positions),
        matches,
    )
    loss.backward()

    assert torch.isfinite(loss)
    assert source_appearance.grad is not None
    assert target_appearance.grad is not None
    assert any(
        parameter.grad is not None and torch.isfinite(parameter.grad).all()
        for parameter in head.context_encoder.parameters()
    )


def test_all_detection_negatives_include_unmatched_candidates() -> None:
    source = torch.tensor([[1.0, 0.0], [0.0, 1.0]], requires_grad=True)
    target = torch.tensor([[1.0, 0.0], [0.0, 1.0], [1.0, 0.0]], requires_grad=True)
    matches = torch.tensor([[0, 0], [1, 1]])

    matched_only = matched_identity_loss(
        source, target, matches, negative_scope="matched_only"
    )
    all_detections = matched_identity_loss(
        source, target, matches, negative_scope="all_detections"
    )
    all_detections.backward()

    assert all_detections > matched_only
    assert target.grad is not None
    assert target.grad[2].abs().sum() > 0


def test_identity_loss_rejects_unknown_negative_scope() -> None:
    with pytest.raises(ValueError, match="negative_scope"):
        matched_identity_loss(
            torch.eye(2),
            torch.eye(2),
            torch.tensor([[0, 0], [1, 1]]),
            negative_scope="implicit_fallback",
        )


def test_b0_does_not_route_gradients_to_geometry_or_context() -> None:
    head = GeometryContextIdentityHead(
        appearance_dim=3, embedding_dim=4, mode="B0", hidden_dim=8
    )
    appearance = torch.randn(3, 3, requires_grad=True)
    positions = torch.randn(3, 3, requires_grad=True)

    head(appearance, positions).square().sum().backward()

    assert appearance.grad is not None
    assert positions.grad is None
    assert not hasattr(head, "geometry_encoder")
    assert all(
        parameter.grad is None for parameter in head.context_encoder.parameters()
    )


def test_residual_context_zero_gate_preserves_appearance_descriptor() -> None:
    appearance = torch.randn(4, 6)
    positions = torch.tensor(
        [
            [0.0, 0.0, 0.0],
            [0.02, 0.0, 0.0],
            [0.07, 0.01, 0.0],
            [1.0, 0.0, 0.0],
        ]
    )
    torch.manual_seed(21)
    appearance_only = GeometryContextIdentityHead(
        6,
        8,
        mode="B0",
        hidden_dim=12,
        fusion_strategy="residual_gated_v1",
    )
    torch.manual_seed(21)
    context = GeometryContextIdentityHead(
        6,
        8,
        mode="C0",
        hidden_dim=12,
        fusion_strategy="residual_gated_v1",
    )

    context.load_state_dict(appearance_only.state_dict(), strict=True)

    torch.testing.assert_close(
        context(appearance, positions),
        appearance_only(appearance, positions),
        atol=0.0,
        rtol=0.0,
    )
    assert context.context_gate.item() == 0.0
    assert context.context_projection.bias is None


def test_legacy_concat_state_dict_still_loads_strictly() -> None:
    torch.manual_seed(22)
    original = GeometryContextIdentityHead(
        6, 8, mode="C0", hidden_dim=12, fusion_strategy="legacy_concat_v1"
    )
    torch.manual_seed(23)
    reloaded = GeometryContextIdentityHead(6, 8, mode="C0", hidden_dim=12)

    reloaded.load_state_dict(original.state_dict(), strict=True)

    assert reloaded.fusion_strategy == "legacy_concat_v1"
