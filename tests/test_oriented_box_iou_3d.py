from __future__ import annotations

import numpy as np
import pytest
from scipy.spatial.transform import Rotation

from yopo.evaluation.metrics.oriented_box_iou_3d import (
    oriented_box_aabbs_3d,
    oriented_box_intersection_volume_3d,
    oriented_box_iou_3d,
    pairwise_oriented_box_iou_3d,
    pairwise_oriented_box_iou_upper_bound_3d,
    pairwise_oriented_box_iou_upper_bound_from_boxes_3d,
)


IDENTITY = np.eye(3, dtype=np.float64)


def _rotation_xyz(*angles: float) -> np.ndarray:
    return Rotation.from_euler("xyz", angles).as_matrix()


def test_identical_arbitrary_so3_box_has_exact_unit_iou():
    center = np.array([1.5, -2.0, 7.0])
    size = np.array([0.8, 1.7, 2.3])
    rotation = _rotation_xyz(0.31, -0.72, 1.17)

    assert oriented_box_iou_3d(center, size, rotation, center, size, rotation) == 1.0
    assert oriented_box_intersection_volume_3d(
        center, size, rotation, center, size, rotation
    ) == pytest.approx(np.prod(size), rel=1e-12, abs=1e-12)


def test_disjoint_boxes_have_zero_iou():
    assert (
        oriented_box_iou_3d(
            np.zeros(3),
            np.ones(3),
            _rotation_xyz(0.2, 0.3, 0.4),
            np.array([10.0, 0.0, 0.0]),
            np.ones(3),
            _rotation_xyz(-0.7, 0.1, 1.2),
        )
        == 0.0
    )


def test_axis_aligned_half_overlap_has_known_volume_and_iou():
    intersection = oriented_box_intersection_volume_3d(
        np.zeros(3),
        np.full(3, 2.0),
        IDENTITY,
        np.array([1.0, 0.0, 0.0]),
        np.full(3, 2.0),
        IDENTITY,
    )
    iou = oriented_box_iou_3d(
        np.zeros(3),
        np.full(3, 2.0),
        IDENTITY,
        np.array([1.0, 0.0, 0.0]),
        np.full(3, 2.0),
        IDENTITY,
    )

    assert intersection == pytest.approx(4.0, abs=1e-12)
    assert iou == pytest.approx(1.0 / 3.0, abs=1e-12)


def test_centered_ninety_degree_rotation_has_known_overlap():
    # A 2x4x2 box and its 90-degree z rotation overlap in a 2x2x2 cube.
    size = np.array([2.0, 4.0, 2.0])
    quarter_turn = _rotation_xyz(0.0, 0.0, np.pi / 2.0)

    intersection = oriented_box_intersection_volume_3d(
        np.zeros(3), size, IDENTITY, np.zeros(3), size, quarter_turn
    )
    iou = oriented_box_iou_3d(
        np.zeros(3), size, IDENTITY, np.zeros(3), size, quarter_turn
    )

    assert intersection == pytest.approx(8.0, abs=1e-11)
    assert iou == pytest.approx(1.0 / 3.0, abs=1e-12)


def test_rotated_inner_box_uses_true_volume_not_aabb_envelope():
    outer_size = np.full(3, 4.0)
    inner_size = np.array([1.0, 2.0, 3.0])
    inner_rotation = _rotation_xyz(0.37, -0.24, 0.51)

    iou = oriented_box_iou_3d(
        np.zeros(3),
        outer_size,
        IDENTITY,
        np.zeros(3),
        inner_size,
        inner_rotation,
    )

    assert iou == pytest.approx(6.0 / 64.0, abs=1e-12)


def test_face_contact_has_zero_volume():
    assert (
        oriented_box_intersection_volume_3d(
            np.zeros(3),
            np.full(3, 2.0),
            IDENTITY,
            np.array([2.0, 0.0, 0.0]),
            np.full(3, 2.0),
            IDENTITY,
        )
        == 0.0
    )


def test_iou_is_symmetric_for_general_so3_boxes():
    first = (
        np.array([0.2, -0.1, 0.4]),
        np.array([1.2, 2.1, 0.9]),
        _rotation_xyz(0.2, -0.4, 0.8),
    )
    second = (
        np.array([-0.15, 0.35, 0.1]),
        np.array([1.7, 1.0, 1.3]),
        _rotation_xyz(-0.5, 0.3, -0.2),
    )

    forward = oriented_box_iou_3d(*first, *second)
    reverse = oriented_box_iou_3d(*second, *first)

    assert 0.0 < forward < 1.0
    assert forward == pytest.approx(reverse, rel=1e-12, abs=1e-12)


def test_large_world_translation_does_not_change_iou():
    offset = np.array([1e8, -2e8, 3e8])
    size = np.array([0.8, 1.2, 1.6])
    rotation = _rotation_xyz(0.1, -0.2, 0.3)
    relative_center = np.array([0.2, 0.0, 0.0])

    near_origin = oriented_box_iou_3d(
        np.zeros(3), size, rotation, relative_center, size, rotation
    )
    translated = oriented_box_iou_3d(
        offset, size, rotation, offset + relative_center, size, rotation
    )

    assert translated == pytest.approx(near_origin, rel=1e-7, abs=1e-9)


def test_tiny_contained_boxes_are_not_rounded_to_identity():
    outer_size = np.full(3, 2e-5)
    inner_size = np.full(3, 1e-5)

    iou = oriented_box_iou_3d(
        np.zeros(3),
        outer_size,
        IDENTITY,
        np.zeros(3),
        inner_size,
        IDENTITY,
    )

    assert iou == pytest.approx(1.0 / 8.0, rel=1e-12, abs=1e-15)


@pytest.mark.parametrize("scale", [1e-7, 1.0, 1e9])
def test_iou_is_scale_invariant_across_representable_ranges(scale):
    size = np.array([2.0, 3.0, 4.0]) * scale
    rotation = _rotation_xyz(0.2, -0.3, 0.4)
    local_offset = np.array([0.5, 0.0, 0.0]) * scale
    world_offset = rotation @ local_offset

    iou = oriented_box_iou_3d(np.zeros(3), size, rotation, world_offset, size, rotation)

    assert iou == pytest.approx(0.6, rel=1e-8, abs=1e-10)


def test_pairwise_api_supports_multiple_and_empty_batches():
    centers = np.array([[0.0, 0.0, 0.0], [4.0, 0.0, 0.0]])
    sizes = np.full((2, 3), 2.0)
    rotations = np.repeat(IDENTITY[None], 2, axis=0)

    overlaps = pairwise_oriented_box_iou_3d(
        centers, sizes, rotations, centers[:1], sizes[:1], rotations[:1]
    )
    empty = pairwise_oriented_box_iou_3d(
        centers[:0],
        sizes[:0],
        rotations[:0],
        centers,
        sizes,
        rotations,
    )

    assert overlaps.shape == (2, 1)
    assert overlaps[:, 0] == pytest.approx([1.0, 0.0])
    assert empty.shape == (0, 2)


def test_projection_prism_upper_bound_is_safe_and_tighter_than_world_aabb():
    rng = np.random.default_rng(190803851)
    centers_1 = rng.uniform(-1.0, 1.0, (12, 3))
    centers_2 = rng.uniform(-1.0, 1.0, (13, 3))
    sizes_1 = rng.uniform(0.2, 1.5, (12, 3))
    sizes_2 = rng.uniform(0.2, 1.5, (13, 3))
    rotations_1 = Rotation.random(12, random_state=rng).as_matrix()
    rotations_2 = Rotation.random(13, random_state=rng).as_matrix()

    exact = pairwise_oriented_box_iou_3d(
        centers_1,
        sizes_1,
        rotations_1,
        centers_2,
        sizes_2,
        rotations_2,
    )
    strong_upper = pairwise_oriented_box_iou_upper_bound_from_boxes_3d(
        centers_1,
        sizes_1,
        rotations_1,
        centers_2,
        sizes_2,
        rotations_2,
    )
    minimum_1, maximum_1, volumes_1 = oriented_box_aabbs_3d(
        centers_1, sizes_1, rotations_1
    )
    minimum_2, maximum_2, volumes_2 = oriented_box_aabbs_3d(
        centers_2, sizes_2, rotations_2
    )
    world_upper = pairwise_oriented_box_iou_upper_bound_3d(
        minimum_1,
        maximum_1,
        volumes_1,
        minimum_2,
        maximum_2,
        volumes_2,
    )

    assert np.all(exact <= strong_upper + 1e-14)
    assert np.all(strong_upper <= world_upper + 1e-14)
    assert np.count_nonzero(strong_upper < world_upper - 1e-12) > 0


def test_upper_bound_empty_batch_shape():
    empty_centers = np.empty((0, 3))
    empty_sizes = np.empty((0, 3))
    empty_rotations = np.empty((0, 3, 3))

    upper = pairwise_oriented_box_iou_upper_bound_from_boxes_3d(
        empty_centers,
        empty_sizes,
        empty_rotations,
        np.zeros((2, 3)),
        np.ones((2, 3)),
        np.repeat(IDENTITY[None], 2, axis=0),
    )

    assert upper.shape == (0, 2)


@pytest.mark.parametrize(
    ("size", "rotation", "message"),
    [
        (np.array([1.0, 0.0, 1.0]), IDENTITY, "strictly positive"),
        (np.array([1.0, -1.0, 1.0]), IDENTITY, "strictly positive"),
        (np.ones(3), np.diag([1.0, 1.0, -1.0]), "SO\\(3\\)"),
        (np.ones(3), np.diag([1.0, 1.0, 2.0]), "SO\\(3\\)"),
        (np.ones(3), np.full((3, 3), np.nan), "finite"),
    ],
)
def test_invalid_or_degenerate_boxes_fail_fast(size, rotation, message):
    with pytest.raises(ValueError, match=message):
        oriented_box_iou_3d(
            np.zeros(3),
            size,
            rotation,
            np.zeros(3),
            np.ones(3),
            IDENTITY,
        )
