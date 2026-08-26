import math

import numpy as np
import pytest
import torch
from mmcv.ops import box_iou_rotated
from shapely.geometry import Polygon

from yopo.evaluation.metrics.rotated_iou_metric import RotatedIoUMetric


def _boxes(values=()) -> torch.Tensor:
    return torch.as_tensor(values, dtype=torch.float32).reshape(-1, 5)


def _result(pred_boxes, scores, gt_boxes, ignore_boxes=()):
    return dict(
        pred_bboxes=_boxes(pred_boxes),
        pred_scores=torch.as_tensor(scores, dtype=torch.float32),
        pred_labels=torch.zeros(len(scores), dtype=torch.long),
        gt_bboxes=_boxes(gt_boxes),
        gt_labels=torch.zeros(len(gt_boxes), dtype=torch.long),
        ignore_bboxes=_boxes(ignore_boxes),
        ignore_labels=torch.zeros(len(ignore_boxes), dtype=torch.long),
    )


def _polygon(box) -> Polygon:
    cx, cy, width, height, angle = box
    corners = np.asarray([
        [-width / 2, -height / 2],
        [width / 2, -height / 2],
        [width / 2, height / 2],
        [-width / 2, height / 2],
    ])
    rotation = np.asarray([
        [math.cos(angle), -math.sin(angle)],
        [math.sin(angle), math.cos(angle)],
    ])
    return Polygon(corners @ rotation.T + np.asarray([cx, cy]))


def test_ranked_map_uses_predictions_below_operating_threshold() -> None:
    metric = RotatedIoUMetric(iou_thr=0.5, score_thr=0.05, num_classes=1)
    result = _result(
        pred_boxes=[
            [100, 100, 4, 4, 0],
            [0, 0, 4, 4, 0],
            [20, 0, 4, 4, 0],
        ],
        scores=[0.9, 0.8, 0.01],
        gt_boxes=[[0, 0, 4, 4, 0], [20, 0, 4, 4, 0]],
    )

    metrics = metric.compute_metrics([result])

    assert metrics['rbbox_mAP_50'] == pytest.approx(2 / 3)
    assert metrics['rbbox_recall_50'] == pytest.approx(0.5)
    assert metrics['rbbox_precision_50'] == pytest.approx(0.5)
    assert metrics['rbbox_num_pred'] == 2
    assert metrics['rbbox_score_thr'] == pytest.approx(0.05)


def test_duplicate_detection_is_false_positive_but_does_not_reduce_ap() -> None:
    metric = RotatedIoUMetric(iou_thr=0.5, score_thr=0.0, num_classes=1)
    box = [8, 7, 6, 4, 0.2]
    result = _result([box, box], [0.9, 0.8], [box])

    metrics = metric.compute_metrics([result])

    assert metrics['rbbox_mAP_50'] == pytest.approx(1.0)
    assert metrics['rbbox_recall_50'] == pytest.approx(1.0)
    assert metrics['rbbox_precision_50'] == pytest.approx(0.5)
    assert metrics['rbbox_mean_matched_rIoU'] == pytest.approx(1.0)


def test_highest_overlap_ignored_gt_takes_precedence() -> None:
    metric = RotatedIoUMetric(iou_thr=0.5, score_thr=0.0, num_classes=1)
    result = _result(
        pred_boxes=[[0.5, 0, 4, 4, 0], [0, 0, 4, 4, 0]],
        scores=[0.9, 0.8],
        gt_boxes=[[0, 0, 4, 4, 0]],
        ignore_boxes=[[0.5, 0, 4, 4, 0]],
    )

    metrics = metric.compute_metrics([result])

    assert metrics['rbbox_mAP_50'] == pytest.approx(1.0)
    assert metrics['rbbox_precision_50'] == pytest.approx(1.0)
    assert metrics['rbbox_num_pred'] == 1
    assert metrics['rbbox_num_ignored_pred'] == 1


def test_mmcv_rotated_iou_matches_independent_polygon_area() -> None:
    first = [0, 0, 8, 3, 0.37]
    second = [1.25, 0, 6, 4, -0.21]
    actual = float(box_iou_rotated(
        _boxes([first]), _boxes([second]), clockwise=True)[0, 0])
    first_polygon = _polygon(first)
    second_polygon = _polygon(second)
    intersection = first_polygon.intersection(second_polygon).area
    expected = intersection / (
        first_polygon.area + second_polygon.area - intersection)

    assert actual == pytest.approx(expected, abs=1e-5)


def test_metric_rejects_non_finite_geometry() -> None:
    metric = RotatedIoUMetric(iou_thr=0.5, score_thr=0.0, num_classes=1)
    result = _result([[0, 0, float('nan'), 4, 0]], [0.9],
                     [[0, 0, 4, 4, 0]])

    with pytest.raises(ValueError, match='NaN or Inf'):
        metric.compute_metrics([result])
