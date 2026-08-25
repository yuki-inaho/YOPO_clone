from __future__ import annotations

import pytest
import torch

from tools.analysis_tools.render_rgbd_2d_bbox_overlays import (
    summarize_detection_geometry,
)


def test_detection_geometry_reports_oracle_recall_and_center_distance():
    report = summarize_detection_geometry(
        [torch.tensor([[0.0, 0.0, 10.0, 10.0]])],
        [torch.tensor([[0.0, 0.0, 10.0, 10.0], [20.0, 20.0, 30.0, 30.0]])],
    )
    assert report["image_count"] == 1
    assert report["gt_count"] == 1
    assert report["prediction_count"] == 2
    assert report["oracle_recall"]["iou_0.50"] == 1.0
    assert report["nearest_prediction_center_px"]["median"] == 0.0


def test_detection_geometry_rejects_mismatched_image_counts():
    with pytest.raises(ValueError, match="image counts differ"):
        summarize_detection_geometry([torch.zeros(0, 4)], [])
