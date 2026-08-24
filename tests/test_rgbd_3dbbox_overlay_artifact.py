"""Contract checks for the checked-in evidence from the best 3D checkpoint."""

import json
from pathlib import Path

import cv2
import numpy as np
import pytest


REPO_ROOT = Path(__file__).resolve().parents[1]
BASELINE_OVERLAY_DIR = (
    REPO_ROOT / "work_dirs/rgbd3d_amp26_lr2e4_20ep_eval/epoch14_overlays"
)
LONG_OVERLAY_DIR = REPO_ROOT / "work_dirs/rgbd3d_long_ft80_lr1e6/epoch10_overlays"


@pytest.mark.parametrize(
    ("overlay_dir", "checkpoint_name"),
    [
        (BASELINE_OVERLAY_DIR, "best_3d_iou_0.50_epoch_14.pth"),
        (LONG_OVERLAY_DIR, "best_3d_iou_0.50_epoch_10.pth"),
    ],
)
def test_rgbd_3dbbox_overlay_artifact(overlay_dir, checkpoint_name):
    """Three decoded RGB projections retain their model and camera evidence."""
    manifest = json.loads((overlay_dir / "manifest.json").read_text())
    assert Path(manifest["checkpoint"]).name == checkpoint_name
    assert manifest["selection_rule"] == "top-1 query per image, no confidence threshold"
    assert len(manifest["images"]) == 3

    for item in manifest["images"]:
        image = cv2.imread(item["overlay_image"], cv2.IMREAD_COLOR)
        assert image is not None
        expected_width, expected_height = item["image_size_wh"]
        assert image.shape[:2] == (expected_height, expected_width)
        # Yellow BGR edges are rendered after the original RGB photograph.
        assert np.count_nonzero(np.all(image == (0, 255, 255), axis=2)) > 0
        assert item["query_index"] >= 0
        assert np.isfinite(item["score"])
        assert np.asarray(item["intrinsic_3x3"]).shape == (3, 3)
        assert np.asarray(item["object_to_camera_T_4x4"]).shape == (4, 4)
        assert np.asarray(item["size_m"]).shape == (3, )
        corners_camera = np.asarray(item["corners_camera_m"])
        projected_corners = np.asarray(item["projected_corners_xy_px"])
        assert corners_camera.shape == (8, 3)
        assert projected_corners.shape == (8, 2)
        assert np.isfinite(corners_camera).all()
        assert np.isfinite(projected_corners).all()
        assert item["visible_projected_corners"] == 8


def test_long_run_overlay_comparison_manifest():
    """The visual report preserves the baseline and marginal metric delta."""
    comparison = json.loads(
        (LONG_OVERLAY_DIR / "comparison_manifest.json").read_text()
    )
    assert comparison["selection_rule"] == (
        "top-1 query per image, no confidence threshold"
    )
    assert comparison["metric"] == "3d_iou_0.50"
    assert comparison["baseline"]["epoch"] == 14
    assert comparison["long_run"]["epoch"] == 10
    assert comparison["baseline"]["value"] == pytest.approx(0.169004)
    assert comparison["long_run"]["value"] == pytest.approx(0.1694)
    assert comparison["delta"] == pytest.approx(0.000396)
    assert len(comparison["baseline"]["overlays"]) == 3
    assert len(comparison["long_run"]["overlays"]) == 3
    assert "not practically accurate" in comparison["visual_review"]
