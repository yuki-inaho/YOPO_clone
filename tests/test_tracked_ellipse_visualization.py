"""Contracts for deterministic projected-ellipse tracking overlays."""

from __future__ import annotations

import cv2
import numpy as np
import pytest
import torch
from mmengine.structures import InstanceData

from tools.visualize_tracked_ellipses import (
    _prediction_ellipses,
    prepare_output_directory,
)
from yopo.models.tracking.ellipse_overlay import (
    draw_tracked_ellipse,
    ellipse_to_cv2,
    track_color_bgr,
)


def test_ellipse_layout_converts_semiaxes_and_radians_for_opencv() -> None:
    converted = ellipse_to_cv2(
        np.array([12.5, 6.25, 80.4, 40.6, np.pi / 4]),
        image_size=(100, 160),
    )

    assert converted is not None
    assert converted.center == (80, 41)
    assert converted.axes == (12, 6)
    assert converted.angle_degrees == pytest.approx(45.0)


@pytest.mark.parametrize(
    "ellipse",
    [
        [0.0, 5.0, 20.0, 20.0, 0.0],
        [5.0, -1.0, 20.0, 20.0, 0.0],
        [5.0, 2.0, float("nan"), 20.0, 0.0],
        [10000.0, 2.0, 20.0, 20.0, 0.0],
    ],
)
def test_invalid_or_unsafe_ellipse_is_not_drawable(ellipse: list[float]) -> None:
    assert ellipse_to_cv2(np.asarray(ellipse), image_size=(100, 160)) is None


def test_ellipse_shape_violation_is_fail_closed() -> None:
    with pytest.raises(ValueError, match="shape"):
        ellipse_to_cv2(np.ones(4), image_size=(100, 160))


def test_track_color_is_stable_distinct_and_valid_bgr() -> None:
    assert track_color_bgr(17) == track_color_bgr(17)
    assert track_color_bgr(17) != track_color_bgr(18)
    assert all(0 <= channel <= 255 for channel in track_color_bgr(17))
    with pytest.raises(ValueError, match="positive"):
        track_color_bgr(0)


def test_draw_changes_pixels_and_honours_projection_validity() -> None:
    image = np.zeros((100, 160, 3), dtype=np.uint8)
    untouched = image.copy()

    assert draw_tracked_ellipse(
        image,
        np.array([12.0, 6.0, 80.0, 40.0, 0.25]),
        track_id=3,
        score=0.9,
        projected_valid=True,
    )
    assert np.count_nonzero(image != untouched) > 0

    invalid_canvas = np.zeros_like(image)
    assert not draw_tracked_ellipse(
        invalid_canvas,
        np.array([12.0, 6.0, 80.0, 40.0, 0.25]),
        track_id=3,
        score=0.9,
        projected_valid=False,
    )
    assert np.array_equal(invalid_canvas, np.zeros_like(image))


def test_output_directory_rejects_existing_content(tmp_path) -> None:
    destination = tmp_path / "render"
    prepare_output_directory(destination)
    assert destination.is_dir()
    assert cv2.imwrite(str(destination / "existing.png"), np.zeros((2, 2, 3)))

    with pytest.raises(FileExistsError, match="not empty"):
        prepare_output_directory(destination)


def test_prediction_ellipse_fields_are_required_and_shape_checked() -> None:
    with pytest.raises(ValueError, match="projected_ellipses"):
        _prediction_ellipses(InstanceData(scores=torch.ones(1)))

    malformed = InstanceData(
        projected_ellipses=torch.ones(2, 4),
        projected_valid=torch.ones(2, dtype=torch.bool),
    )
    with pytest.raises(ValueError, match="shape"):
        _prediction_ellipses(malformed)
