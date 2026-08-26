"""Exact IoU for arbitrarily oriented 3D cuboids.

This module is intentionally independent from the metric classes.  It provides
an evaluation-only NumPy/SciPy implementation for boxes represented by a world
space center, full side lengths, and a rotation from box-local coordinates to
world coordinates.

The intersection of two convex cuboids is another convex polyhedron.  Its
vertices can only be one of the following:

* a corner of either cuboid contained in the other cuboid; or
* an intersection between an edge of either cuboid and a face of the other.

After collecting those vertices, :class:`scipy.spatial.ConvexHull` gives the
exact intersection volume up to floating-point arithmetic.  This is the 3D
counterpart of the intersection-vertex/convex-hull construction commonly used
for rotated 2D boxes.  It does not replace the intersection by an
axis-aligned bounding-box envelope.

The implementation is non-differentiable and is intended for evaluation and
diagnostics, not as a training loss.
"""

from __future__ import annotations

from dataclasses import dataclass
from itertools import product

import numpy as np
from scipy.spatial import ConvexHull, QhullError


_CORNER_SIGNS = np.asarray(tuple(product((-1.0, 1.0), repeat=3)), dtype=np.float64)
_EDGE_VERTEX_PAIRS = tuple(
    (first, second)
    for first in range(len(_CORNER_SIGNS))
    for second in range(first + 1, len(_CORNER_SIGNS))
    if np.count_nonzero(_CORNER_SIGNS[first] != _CORNER_SIGNS[second]) == 1
)


@dataclass(frozen=True)
class _OrientedBox:
    center: np.ndarray
    size: np.ndarray
    rotation: np.ndarray
    half_size: np.ndarray
    local_corners: np.ndarray
    volume: float


@dataclass(frozen=True)
class _PlacedBox:
    """A validated box translated into a pair-local coordinate frame."""

    center: np.ndarray
    half_size: np.ndarray
    rotation: np.ndarray
    corners: np.ndarray


def _as_vector(value: np.ndarray, *, name: str) -> np.ndarray:
    array = np.asarray(value, dtype=np.float64)
    if array.shape != (3,):
        raise ValueError(f"{name} must have shape (3,), got {array.shape}")
    if not np.isfinite(array).all():
        raise ValueError(f"{name} must contain only finite values")
    return array


def _as_box(
    center: np.ndarray,
    size: np.ndarray,
    rotation: np.ndarray,
    *,
    rotation_tolerance: float,
) -> _OrientedBox:
    center_array = _as_vector(center, name="center")
    size_array = _as_vector(size, name="size")
    if np.any(size_array <= 0.0):
        raise ValueError(
            "size must contain strictly positive full side lengths; "
            f"got {size_array.tolist()}"
        )

    rotation_array = np.asarray(rotation, dtype=np.float64)
    if rotation_array.shape != (3, 3):
        raise ValueError(f"rotation must have shape (3, 3), got {rotation_array.shape}")
    if not np.isfinite(rotation_array).all():
        raise ValueError("rotation must contain only finite values")

    gram = rotation_array.T @ rotation_array
    determinant = float(np.linalg.det(rotation_array))
    if not np.allclose(
        gram,
        np.eye(3, dtype=np.float64),
        rtol=rotation_tolerance,
        atol=rotation_tolerance,
    ) or not np.isclose(
        determinant,
        1.0,
        rtol=rotation_tolerance,
        atol=rotation_tolerance,
    ):
        raise ValueError(
            "rotation must be a proper orthonormal SO(3) matrix; "
            f"det(rotation)={determinant:.12g}, "
            f"max|R^T R-I|={np.max(np.abs(gram - np.eye(3))):.12g}"
        )

    half_size = size_array / 2.0
    volume = float(np.prod(size_array))
    if not np.isfinite(volume) or volume <= 0.0:
        raise ValueError(
            "size produces a non-representable float64 volume; "
            f"got side lengths {size_array.tolist()}"
        )
    return _OrientedBox(
        center=center_array,
        size=size_array,
        rotation=rotation_array,
        half_size=half_size,
        local_corners=_CORNER_SIGNS * half_size,
        volume=volume,
    )


def _place_box(box: _OrientedBox, center: np.ndarray) -> _PlacedBox:
    # Row-vector equivalent of ``world = center + R @ local``.
    corners = center + box.local_corners @ box.rotation.T
    return _PlacedBox(
        center=center,
        half_size=box.half_size,
        rotation=box.rotation,
        corners=corners,
    )


def _points_inside(
    points: np.ndarray,
    box: _PlacedBox,
    *,
    tolerance: float,
) -> np.ndarray:
    local = (points - box.center) @ box.rotation
    return np.all(np.abs(local) <= box.half_size + tolerance, axis=1)


def _edge_face_intersections(
    source: _PlacedBox,
    target: _PlacedBox,
    *,
    tolerance: float,
) -> list[np.ndarray]:
    """Return all source-edge/target-face intersection points."""
    intersections: list[np.ndarray] = []
    for first, second in _EDGE_VERTEX_PAIRS:
        point_0 = source.corners[first]
        point_1 = source.corners[second]
        point_0_local = (point_0 - target.center) @ target.rotation
        point_1_local = (point_1 - target.center) @ target.rotation
        local_direction = point_1_local - point_0_local
        direction_length = float(np.linalg.norm(local_direction))
        parameter_tolerance = tolerance / max(
            direction_length, np.finfo(np.float64).tiny
        )

        for axis in range(3):
            denominator = local_direction[axis]
            if abs(denominator) <= tolerance:
                # A parallel/coplanar edge has no unique edge-plane crossing.
                # Relevant endpoints and crossings with the other faces are
                # collected by the remaining cases.
                continue
            for sign in (-1.0, 1.0):
                face_coordinate = sign * target.half_size[axis]
                parameter = (face_coordinate - point_0_local[axis]) / denominator
                if (
                    parameter < -parameter_tolerance
                    or parameter > 1.0 + parameter_tolerance
                ):
                    continue

                parameter = float(np.clip(parameter, 0.0, 1.0))
                candidate_local = point_0_local + parameter * local_direction
                if np.all(np.abs(candidate_local) <= target.half_size + tolerance):
                    intersections.append(point_0 + parameter * (point_1 - point_0))
    return intersections


def _deduplicate_points(points: list[np.ndarray], *, tolerance: float) -> np.ndarray:
    if not points:
        return np.empty((0, 3), dtype=np.float64)

    ordered = np.asarray(points, dtype=np.float64)
    order = np.lexsort((ordered[:, 2], ordered[:, 1], ordered[:, 0]))
    unique: list[np.ndarray] = []
    for point in ordered[order]:
        if not any(
            np.linalg.norm(point - kept, ord=np.inf) <= tolerance for kept in unique
        ):
            unique.append(point)
    return np.asarray(unique, dtype=np.float64)


def _intersection_vertices(
    first: _PlacedBox,
    second: _PlacedBox,
    *,
    tolerance: float,
) -> np.ndarray:
    vertices: list[np.ndarray] = []
    vertices.extend(
        first.corners[_points_inside(first.corners, second, tolerance=tolerance)]
    )
    vertices.extend(
        second.corners[_points_inside(second.corners, first, tolerance=tolerance)]
    )
    vertices.extend(_edge_face_intersections(first, second, tolerance=tolerance))
    vertices.extend(_edge_face_intersections(second, first, tolerance=tolerance))
    return _deduplicate_points(vertices, tolerance=tolerance)


def _boxes_are_separated(
    first: _PlacedBox,
    second: _PlacedBox,
    *,
    tolerance: float,
) -> bool:
    """Apply the exact 15-axis separating-axis test for two OBBs."""
    center_delta = second.center - first.center
    axes = [first.rotation[:, axis] for axis in range(3)]
    axes.extend(second.rotation[:, axis] for axis in range(3))
    axes.extend(
        np.cross(first.rotation[:, first_axis], second.rotation[:, second_axis])
        for first_axis in range(3)
        for second_axis in range(3)
    )

    for axis in axes:
        axis_norm = float(np.linalg.norm(axis))
        if axis_norm <= np.finfo(np.float64).eps:
            continue
        center_distance = abs(float(center_delta @ axis))
        first_radius = float(first.half_size @ np.abs(first.rotation.T @ axis))
        second_radius = float(second.half_size @ np.abs(second.rotation.T @ axis))
        if center_distance > first_radius + second_radius + tolerance * axis_norm:
            return True
    return False


def _intersection_volume(
    first: _OrientedBox,
    second: _OrientedBox,
    *,
    absolute_tolerance: float,
    relative_tolerance: float,
) -> float:
    # Recenter each pair before computing vertices and the hull.  This avoids
    # losing small box offsets when world coordinates are large.
    relative_center = second.center - first.center
    placed_first = _place_box(first, np.zeros(3, dtype=np.float64))
    placed_second = _place_box(second, relative_center)

    geometry_scale = max(
        float(np.max(first.size)),
        float(np.max(second.size)),
    )
    tolerance = absolute_tolerance + relative_tolerance * geometry_scale
    if _boxes_are_separated(placed_first, placed_second, tolerance=tolerance):
        return 0.0
    vertices = _intersection_vertices(placed_first, placed_second, tolerance=tolerance)
    if len(vertices) < 4:
        return 0.0

    centered_vertices = vertices - np.mean(vertices, axis=0, keepdims=True)
    hull_scale = float(np.max(np.abs(centered_vertices)))
    if hull_scale <= tolerance:
        return 0.0
    normalized_vertices = centered_vertices / hull_scale
    singular_values = np.linalg.svd(
        normalized_vertices, compute_uv=False, full_matrices=False
    )
    rank_tolerance = max(
        tolerance / hull_scale,
        np.finfo(np.float64).eps
        * max(normalized_vertices.shape)
        * float(singular_values[0]),
    )
    if np.count_nonzero(singular_values > rank_tolerance) < 3:
        # Empty intersections and face/edge/point contact all have zero volume.
        return 0.0

    try:
        volume = float(ConvexHull(normalized_vertices).volume) * hull_scale**3
    except QhullError as error:
        raise RuntimeError(
            "failed to construct the full-dimensional cuboid intersection hull"
        ) from error

    if not np.isfinite(volume):
        raise RuntimeError("cuboid intersection volume is non-finite")
    return float(np.clip(volume, 0.0, min(first.volume, second.volume)))


def _volume_tolerance(
    first: _OrientedBox,
    second: _OrientedBox,
    *,
    absolute_tolerance: float,
    relative_tolerance: float,
) -> float:
    """Convert length predicates to a dimensionally valid volume tolerance."""
    volume_scale = max(first.volume, second.volume)
    return max(
        absolute_tolerance**3,
        relative_tolerance * volume_scale,
        np.finfo(np.float64).eps * volume_scale,
    )


def oriented_box_intersection_volume_3d(
    center_1: np.ndarray,
    size_1: np.ndarray,
    rotation_1: np.ndarray,
    center_2: np.ndarray,
    size_2: np.ndarray,
    rotation_2: np.ndarray,
    *,
    absolute_tolerance: float = 1e-12,
    relative_tolerance: float = 1e-9,
    rotation_tolerance: float = 1e-6,
) -> float:
    """Compute the intersection volume of two arbitrary SO(3) cuboids.

    Args:
        center_1, center_2: World-space box centers with shape ``(3,)``.
        size_1, size_2: Strictly positive *full* side lengths in local x/y/z
            order, each with shape ``(3,)``.
        rotation_1, rotation_2: Proper orthonormal matrices with shape
            ``(3, 3)`` mapping box-local coordinates to world coordinates.
        absolute_tolerance: Absolute tolerance used for geometric predicates.
        relative_tolerance: Predicate tolerance relative to pair scale.
        rotation_tolerance: Validation tolerance for ``R.T @ R == I`` and
            ``det(R) == 1``.

    Returns:
        Intersection volume as a finite non-negative Python ``float``.

    Raises:
        ValueError: If shapes are wrong, data are non-finite, side lengths are
            non-positive, tolerances are invalid, or a rotation is not SO(3).
        RuntimeError: If a numerically full-dimensional intersection cannot be
            converted into a convex hull.

    Degenerate boxes are rejected rather than silently interpreted as empty.
    Boundary-only contact between valid boxes has zero intersection volume.
    """
    for name, value in (
        ("absolute_tolerance", absolute_tolerance),
        ("relative_tolerance", relative_tolerance),
        ("rotation_tolerance", rotation_tolerance),
    ):
        if not np.isfinite(value) or value < 0.0:
            raise ValueError(f"{name} must be finite and non-negative")

    first = _as_box(
        center_1,
        size_1,
        rotation_1,
        rotation_tolerance=rotation_tolerance,
    )
    second = _as_box(
        center_2,
        size_2,
        rotation_2,
        rotation_tolerance=rotation_tolerance,
    )
    return _intersection_volume(
        first,
        second,
        absolute_tolerance=absolute_tolerance,
        relative_tolerance=relative_tolerance,
    )


def oriented_box_iou_3d(
    center_1: np.ndarray,
    size_1: np.ndarray,
    rotation_1: np.ndarray,
    center_2: np.ndarray,
    size_2: np.ndarray,
    rotation_2: np.ndarray,
    *,
    absolute_tolerance: float = 1e-12,
    relative_tolerance: float = 1e-9,
    rotation_tolerance: float = 1e-6,
) -> float:
    """Compute exact oriented cuboid IoU for arbitrary SO(3) rotations.

    Input conventions and invalid-input behavior are described by
    :func:`oriented_box_intersection_volume_3d`.
    """
    for name, value in (
        ("absolute_tolerance", absolute_tolerance),
        ("relative_tolerance", relative_tolerance),
        ("rotation_tolerance", rotation_tolerance),
    ):
        if not np.isfinite(value) or value < 0.0:
            raise ValueError(f"{name} must be finite and non-negative")

    first = _as_box(
        center_1,
        size_1,
        rotation_1,
        rotation_tolerance=rotation_tolerance,
    )
    second = _as_box(
        center_2,
        size_2,
        rotation_2,
        rotation_tolerance=rotation_tolerance,
    )
    intersection = _intersection_volume(
        first,
        second,
        absolute_tolerance=absolute_tolerance,
        relative_tolerance=relative_tolerance,
    )
    union = first.volume + second.volume - intersection
    iou = intersection / union

    # Make exact identities stable without masking material geometry errors.
    volume_tolerance = _volume_tolerance(
        first,
        second,
        absolute_tolerance=absolute_tolerance,
        relative_tolerance=relative_tolerance,
    )
    if (
        abs(intersection - first.volume) <= volume_tolerance
        and abs(intersection - second.volume) <= volume_tolerance
    ):
        return 1.0
    return float(np.clip(iou, 0.0, 1.0))


def pairwise_oriented_box_iou_3d(
    centers_1: np.ndarray,
    sizes_1: np.ndarray,
    rotations_1: np.ndarray,
    centers_2: np.ndarray,
    sizes_2: np.ndarray,
    rotations_2: np.ndarray,
    *,
    absolute_tolerance: float = 1e-12,
    relative_tolerance: float = 1e-9,
    rotation_tolerance: float = 1e-6,
) -> np.ndarray:
    """Compute all ``N x M`` oriented cuboid IoUs.

    The inputs have shapes ``(N, 3)``, ``(N, 3)``, ``(N, 3, 3)`` and the
    corresponding ``M`` shapes.  Empty batches are supported.  Validation is
    performed once per box rather than once per pair.
    """
    for name, value in (
        ("absolute_tolerance", absolute_tolerance),
        ("relative_tolerance", relative_tolerance),
        ("rotation_tolerance", rotation_tolerance),
    ):
        if not np.isfinite(value) or value < 0.0:
            raise ValueError(f"{name} must be finite and non-negative")

    def as_batch(array: np.ndarray, *, name: str, tail: tuple[int, ...]):
        converted = np.asarray(array, dtype=np.float64)
        if converted.ndim != len(tail) + 1 or converted.shape[1:] != tail:
            raise ValueError(
                f"{name} must have shape (N, {', '.join(map(str, tail))}), "
                f"got {converted.shape}"
            )
        return converted

    center_batch_1 = as_batch(centers_1, name="centers_1", tail=(3,))
    size_batch_1 = as_batch(sizes_1, name="sizes_1", tail=(3,))
    rotation_batch_1 = as_batch(rotations_1, name="rotations_1", tail=(3, 3))
    center_batch_2 = as_batch(centers_2, name="centers_2", tail=(3,))
    size_batch_2 = as_batch(sizes_2, name="sizes_2", tail=(3,))
    rotation_batch_2 = as_batch(rotations_2, name="rotations_2", tail=(3, 3))

    if not (len(center_batch_1) == len(size_batch_1) == len(rotation_batch_1)):
        raise ValueError("the first box batch lengths must match")
    if not (len(center_batch_2) == len(size_batch_2) == len(rotation_batch_2)):
        raise ValueError("the second box batch lengths must match")

    first_boxes = [
        _as_box(center, size, rotation, rotation_tolerance=rotation_tolerance)
        for center, size, rotation in zip(
            center_batch_1, size_batch_1, rotation_batch_1
        )
    ]
    second_boxes = [
        _as_box(center, size, rotation, rotation_tolerance=rotation_tolerance)
        for center, size, rotation in zip(
            center_batch_2, size_batch_2, rotation_batch_2
        )
    ]

    overlaps = np.empty((len(first_boxes), len(second_boxes)), dtype=np.float64)
    for first_index, first in enumerate(first_boxes):
        for second_index, second in enumerate(second_boxes):
            intersection = _intersection_volume(
                first,
                second,
                absolute_tolerance=absolute_tolerance,
                relative_tolerance=relative_tolerance,
            )
            union = first.volume + second.volume - intersection
            value = intersection / union
            volume_tolerance = _volume_tolerance(
                first,
                second,
                absolute_tolerance=absolute_tolerance,
                relative_tolerance=relative_tolerance,
            )
            if (
                abs(intersection - first.volume) <= volume_tolerance
                and abs(intersection - second.volume) <= volume_tolerance
            ):
                value = 1.0
            overlaps[first_index, second_index] = np.clip(value, 0.0, 1.0)
    return overlaps


def oriented_box_aabbs_3d(
    centers: np.ndarray,
    sizes: np.ndarray,
    rotations: np.ndarray,
    *,
    rotation_tolerance: float = 1e-6,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Cache world-AABB bounds and true volumes for an SO(3) box batch.

    The returned AABBs are only broad-phase envelopes.  Their volumes are not
    used as substitutes for oriented-box volumes or IoU denominators.
    """
    center_batch = np.asarray(centers, dtype=np.float64)
    size_batch = np.asarray(sizes, dtype=np.float64)
    rotation_batch = np.asarray(rotations, dtype=np.float64)
    if center_batch.ndim != 2 or center_batch.shape[1:] != (3,):
        raise ValueError(f"centers must have shape (N, 3), got {center_batch.shape}")
    if size_batch.shape != center_batch.shape:
        raise ValueError(
            "sizes must have the same (N, 3) shape as centers, "
            f"got {size_batch.shape} and {center_batch.shape}"
        )
    if rotation_batch.shape != (len(center_batch), 3, 3):
        raise ValueError(
            f"rotations must have shape (N, 3, 3), got {rotation_batch.shape}"
        )

    boxes = [
        _as_box(center, size, rotation, rotation_tolerance=rotation_tolerance)
        for center, size, rotation in zip(center_batch, size_batch, rotation_batch)
    ]
    if not boxes:
        empty_bounds = np.empty((0, 3), dtype=np.float64)
        return empty_bounds.copy(), empty_bounds.copy(), np.empty(0, dtype=np.float64)

    corners = np.stack(
        [box.center + box.local_corners @ box.rotation.T for box in boxes]
    )
    return (
        np.min(corners, axis=1),
        np.max(corners, axis=1),
        np.asarray([box.volume for box in boxes], dtype=np.float64),
    )


def pairwise_oriented_box_iou_upper_bound_3d(
    aabb_min_1: np.ndarray,
    aabb_max_1: np.ndarray,
    volumes_1: np.ndarray,
    aabb_min_2: np.ndarray,
    aabb_max_2: np.ndarray,
    volumes_2: np.ndarray,
) -> np.ndarray:
    """Return a safe ``N x M`` upper bound on true oriented-box IoU.

    Let ``U`` be the intersection volume of the two world-AABB envelopes,
    capped by each true OBB volume.  Since the true intersection ``I <= U``
    and ``I / (V1 + V2 - I)`` is monotone in ``I``, the safe bound is
    ``U / (V1 + V2 - U)``.  This is deliberately **not** raw AABB IoU, whose
    AABB-volume denominator is not a valid pruning bound.
    """

    def bounds(array, *, name):
        value = np.asarray(array, dtype=np.float64)
        if value.ndim != 2 or value.shape[1:] != (3,):
            raise ValueError(f"{name} must have shape (N, 3), got {value.shape}")
        if not np.isfinite(value).all():
            raise ValueError(f"{name} must contain only finite values")
        return value

    def volumes(array, *, name, expected):
        value = np.asarray(array, dtype=np.float64)
        if value.shape != (expected,):
            raise ValueError(f"{name} must have shape ({expected},), got {value.shape}")
        if not np.isfinite(value).all() or np.any(value <= 0.0):
            raise ValueError(f"{name} must contain positive finite volumes")
        return value

    minimum_1 = bounds(aabb_min_1, name="aabb_min_1")
    maximum_1 = bounds(aabb_max_1, name="aabb_max_1")
    minimum_2 = bounds(aabb_min_2, name="aabb_min_2")
    maximum_2 = bounds(aabb_max_2, name="aabb_max_2")
    if maximum_1.shape != minimum_1.shape or np.any(maximum_1 < minimum_1):
        raise ValueError("first AABB maxima must match and exceed minima")
    if maximum_2.shape != minimum_2.shape or np.any(maximum_2 < minimum_2):
        raise ValueError("second AABB maxima must match and exceed minima")
    volume_1 = volumes(volumes_1, name="volumes_1", expected=len(minimum_1))
    volume_2 = volumes(volumes_2, name="volumes_2", expected=len(minimum_2))

    intersection_lengths = np.minimum(
        maximum_1[:, None, :], maximum_2[None, :, :]
    ) - np.maximum(minimum_1[:, None, :], minimum_2[None, :, :])
    intersection_lengths = np.maximum(intersection_lengths, 0.0)
    # One outward ULP prevents a downward-rounded envelope from becoming an
    # unsafe threshold decision.  Zero-length separated axes remain zero.
    positive = intersection_lengths > 0.0
    intersection_lengths[positive] = np.nextafter(
        intersection_lengths[positive], np.inf
    )
    intersection_upper = np.prod(intersection_lengths, axis=-1)
    intersection_upper = np.minimum(
        intersection_upper,
        np.minimum(volume_1[:, None], volume_2[None, :]),
    )
    union_lower = volume_1[:, None] + volume_2[None, :] - intersection_upper
    upper_bound = intersection_upper / union_lower
    upper_bound = np.clip(upper_bound, 0.0, 1.0)
    below_one = upper_bound < 1.0
    upper_bound[below_one] = np.nextafter(upper_bound[below_one], np.inf)
    return upper_bound


def pairwise_oriented_box_iou_upper_bound_from_boxes_3d(
    centers_1: np.ndarray,
    sizes_1: np.ndarray,
    rotations_1: np.ndarray,
    centers_2: np.ndarray,
    sizes_2: np.ndarray,
    rotations_2: np.ndarray,
    *,
    rotation_tolerance: float = 1e-6,
    absolute_tolerance: float = 1e-12,
    relative_tolerance: float = 1e-9,
) -> np.ndarray:
    """Compute a strong, safe IoU upper bound for every box pair.

    For each pair, the true intersection is bounded by its projection-prism
    intersection volume in three orthonormal frames: world, box 1, and box 2.
    The minimum of those bounds and both true box volumes is converted to IoU
    with the true-volume denominator.  A circumsphere test cheaply identifies
    pairs with exactly zero intersection.
    """
    center_1 = np.asarray(centers_1, dtype=np.float64)
    size_1 = np.asarray(sizes_1, dtype=np.float64)
    rotation_1 = np.asarray(rotations_1, dtype=np.float64)
    center_2 = np.asarray(centers_2, dtype=np.float64)
    size_2 = np.asarray(sizes_2, dtype=np.float64)
    rotation_2 = np.asarray(rotations_2, dtype=np.float64)
    for name, value in (
        ("absolute_tolerance", absolute_tolerance),
        ("relative_tolerance", relative_tolerance),
    ):
        if not np.isfinite(value) or value < 0.0:
            raise ValueError(f"{name} must be finite and non-negative")
    minimum_1, maximum_1, volume_1 = oriented_box_aabbs_3d(
        center_1, size_1, rotation_1, rotation_tolerance=rotation_tolerance
    )
    minimum_2, maximum_2, volume_2 = oriented_box_aabbs_3d(
        center_2, size_2, rotation_2, rotation_tolerance=rotation_tolerance
    )
    if not len(center_1) or not len(center_2):
        return np.empty((len(center_1), len(center_2)), dtype=np.float64)

    size_scale = np.maximum(
        np.max(size_1, axis=1)[:, None], np.max(size_2, axis=1)[None, :]
    )
    predicate_tolerance = absolute_tolerance + relative_tolerance * size_scale
    floating_gamma = 64.0 * np.finfo(np.float64).eps

    def outward_product(
        lengths: np.ndarray, coordinate_scale: np.ndarray
    ) -> np.ndarray:
        # The exact kernel admits its configured geometric predicate tolerance.
        # Inflate by that tolerance plus a gamma-based dot/sum error guard so
        # the computed prism cannot round inward and become an unsafe bound.
        inflation = (
            2.0 * predicate_tolerance[..., None] + floating_gamma * coordinate_scale
        )
        lengths = np.maximum(lengths + inflation, 0.0)
        positive = lengths > 0.0
        lengths[positive] = np.nextafter(lengths[positive], np.inf)
        product = np.prod(lengths, axis=-1)
        positive_product = product > 0.0
        product[positive_product] *= 1.0 + 16.0 * np.finfo(np.float64).eps
        product[positive_product] = np.nextafter(product[positive_product], np.inf)
        return product

    world_lengths = np.minimum(
        maximum_1[:, None, :], maximum_2[None, :, :]
    ) - np.maximum(minimum_1[:, None, :], minimum_2[None, :, :])
    world_coordinate_scale = np.maximum(
        np.maximum(np.abs(maximum_1[:, None, :]), np.abs(minimum_1[:, None, :])),
        np.maximum(np.abs(maximum_2[None, :, :]), np.abs(minimum_2[None, :, :])),
    )
    world_bound = outward_product(world_lengths, world_coordinate_scale)

    half_1 = size_1 / 2.0
    half_2 = size_2 / 2.0
    center_delta = center_2[None, :, :] - center_1[:, None, :]
    # C[a, b] is the cosine between box-1 axis a and box-2 axis b.
    relative_rotation = np.einsum("nka,mkb->nmab", rotation_1, rotation_2)

    delta_in_1 = np.einsum("nmk,nka->nma", center_delta, rotation_1)
    half_2_in_1 = np.einsum("nmab,mb->nma", np.abs(relative_rotation), half_2)
    lengths_in_1 = np.minimum(
        half_1[:, None, :], delta_in_1 + half_2_in_1
    ) - np.maximum(-half_1[:, None, :], delta_in_1 - half_2_in_1)
    box_1_coordinate_scale = np.abs(delta_in_1) + half_1[:, None, :] + half_2_in_1
    box_1_frame_bound = outward_product(lengths_in_1, box_1_coordinate_scale)

    delta_in_2 = np.einsum("nmk,mka->nma", center_delta, rotation_2)
    half_1_in_2 = np.einsum("nmab,na->nmb", np.abs(relative_rotation), half_1)
    lengths_in_2 = np.minimum(
        half_1_in_2, delta_in_2 + half_2[None, :, :]
    ) - np.maximum(-half_1_in_2, delta_in_2 - half_2[None, :, :])
    box_2_coordinate_scale = np.abs(delta_in_2) + half_1_in_2 + half_2[None, :, :]
    box_2_frame_bound = outward_product(lengths_in_2, box_2_coordinate_scale)

    intersection_upper = np.minimum.reduce(
        (
            world_bound,
            box_1_frame_bound,
            box_2_frame_bound,
            np.broadcast_to(volume_1[:, None], world_bound.shape),
            np.broadcast_to(volume_2[None, :], world_bound.shape),
        )
    )
    radius_sum = (
        np.linalg.norm(half_1, axis=1)[:, None]
        + np.linalg.norm(half_2, axis=1)[None, :]
    )
    center_distance = np.linalg.norm(center_delta, axis=-1)
    sphere_guard = predicate_tolerance + floating_gamma * (center_distance + radius_sum)
    sphere_disjoint = center_distance > radius_sum + sphere_guard
    intersection_upper[sphere_disjoint] = 0.0

    union_lower = volume_1[:, None] + volume_2[None, :] - intersection_upper
    upper_bound = np.clip(intersection_upper / union_lower, 0.0, 1.0)
    below_one = upper_bound < 1.0
    upper_bound[below_one] *= 1.0 + 16.0 * np.finfo(np.float64).eps
    upper_bound[below_one] = np.nextafter(upper_bound[below_one], np.inf)
    return np.minimum(upper_bound, 1.0)


__all__ = [
    "oriented_box_aabbs_3d",
    "oriented_box_intersection_volume_3d",
    "oriented_box_iou_3d",
    "pairwise_oriented_box_iou_upper_bound_3d",
    "pairwise_oriented_box_iou_upper_bound_from_boxes_3d",
    "pairwise_oriented_box_iou_3d",
]
