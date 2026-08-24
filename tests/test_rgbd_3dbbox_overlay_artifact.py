"""Contract checks for the checked-in evidence from the best 3D checkpoint."""

import json
from pathlib import Path

import cv2
import numpy as np


REPO_ROOT = Path(__file__).resolve().parents[1]
OVERLAY_DIR = REPO_ROOT / "work_dirs/rgbd3d_amp26_lr2e4_20ep_eval/epoch14_overlays"


def test_rgbd_3dbbox_overlay_artifact():
    """Three decoded RGB projections retain their model and camera evidence."""
    manifest = json.loads((OVERLAY_DIR / "manifest.json").read_text())
    assert Path(manifest["checkpoint"]).name == "best_3d_iou_0.50_epoch_14.pth"
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
        assert np.asarray(item["projected_corners_xy_px"]).shape == (8, 2)
        assert item["visible_projected_corners"] == 8
