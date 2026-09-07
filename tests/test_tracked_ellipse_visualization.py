"""Contracts for deterministic projected-ellipse tracking overlays."""

from __future__ import annotations

import cv2
import numpy as np
import pytest
import torch
from mmengine.structures import InstanceData

from tools.visualize_tracked_ellipses import (
    _contact_sheet,
    _prediction_ellipses,
    load_raw_source_records,
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


def _write_raw_source(tmp_path, *, frame_count: int = 5) -> tuple:
    root = tmp_path / "raw"
    scene = root / "scenes" / "scene_a"
    rgb_dir = scene / "rgb"
    depth_dir = scene / "depth"
    rgb_dir.mkdir(parents=True)
    depth_dir.mkdir()
    for frame_id in range(frame_count):
        assert cv2.imwrite(
            str(rgb_dir / f"frame_{frame_id:06d}.png"),
            np.full((6, 8, 3), frame_id, dtype=np.uint8),
        )
        assert cv2.imwrite(
            str(depth_dir / f"frame_{frame_id:06d}.png"),
            np.full((6, 8), 1000 + frame_id, dtype=np.uint16),
        )
    document = {
        "format": "colmap_rgbd_v1",
        "schema_version": 1,
        "frame_count": frame_count,
        "scene_count": 1,
        "image": {"width": 8, "height": 6, "channels": 3, "dtype": "uint8"},
        "depth": {
            "width": 8,
            "height": 6,
            "dtype": "uint16",
            "unit": "mm",
            "invalid_value": 0,
        },
    }
    (root / "dataset.json").write_text(__import__("json").dumps(document))
    np.savez(
        scene / "cameras.npz",
        frame_ids=np.arange(frame_count, dtype=np.int64),
        intrinsics=np.repeat(np.eye(3, dtype=np.float32)[None], frame_count, axis=0),
        extrinsics_w2c=np.repeat(
            np.eye(3, 4, dtype=np.float32)[None], frame_count, axis=0
        ),
        quality_flags=np.ones(frame_count, dtype=bool),
    )
    return root, scene


def test_raw_source_records_preserve_contiguous_range_without_labels(tmp_path) -> None:
    root, _ = _write_raw_source(tmp_path)

    selection = load_raw_source_records(
        root, scene="scene_a", start_frame=1, frame_count=3
    )

    assert selection.frame_contract["width"] == 8
    assert selection.frame_contract["height"] == 6
    assert [record.frame_id for record in selection.records] == [1, 2, 3]
    assert all(not record.label_path.exists() for record in selection.records)
    assert selection.metadata["label_artifacts_opened"] is False


def test_raw_source_records_reject_bad_quality_and_out_of_range(tmp_path) -> None:
    root, scene = _write_raw_source(tmp_path)
    cameras = np.load(scene / "cameras.npz")
    quality = cameras["quality_flags"].copy()
    quality[2] = False
    np.savez(
        scene / "cameras.npz",
        frame_ids=cameras["frame_ids"],
        intrinsics=cameras["intrinsics"],
        extrinsics_w2c=cameras["extrinsics_w2c"],
        quality_flags=quality,
    )

    with pytest.raises(ValueError, match="quality"):
        load_raw_source_records(root, scene="scene_a", start_frame=1, frame_count=3)
    with pytest.raises(ValueError, match="outside"):
        load_raw_source_records(root, scene="scene_a", start_frame=4, frame_count=2)


def test_contact_sheet_wraps_long_clip_into_bounded_columns() -> None:
    frames = [np.full((12, 16, 3), index, np.uint8) for index in range(5)]

    sheet = _contact_sheet([frames], tile_size=(16, 12), max_columns=4)

    assert sheet.shape == (24, 64, 3)
    assert np.array_equal(sheet[:12, :16], frames[0])
    assert np.array_equal(sheet[12:, :16], frames[4])
