from __future__ import annotations

import numpy as np

from tools.analysis_tools.compare_rgbd_rotation_predictions import (
    match_image,
    pairwise_iou_xyxy,
    rotation_error_degrees,
    summarize_paired_by_anisotropy,
)


def test_pairwise_iou_and_rotation_geodesic_contracts():
    boxes = np.array([[0.0, 0.0, 10.0, 10.0]])
    np.testing.assert_allclose(pairwise_iou_xyxy(boxes, boxes), [[1.0]])
    identity = np.eye(3)[None]
    quarter_turn = np.array(
        [[[0.0, -1.0, 0.0], [1.0, 0.0, 0.0], [0.0, 0.0, 1.0]]])
    np.testing.assert_allclose(rotation_error_degrees(identity, identity), [0.0])
    np.testing.assert_allclose(
        rotation_error_degrees(identity, quarter_turn), [90.0])


def test_match_image_converts_nocs_yxyx_and_keeps_gt_identity():
    prediction = {
        "pred_instances": {
            "bboxes": np.array([[20.0, 10.0, 40.0, 30.0]]),
            "T": np.eye(4)[None],
            "scores": np.array([0.8]),
            "translations": np.array([[0.0, 0.0, 1.0]]),
        }
    }
    target = {
        "bboxes": np.array([[10.0, 20.0, 30.0, 40.0]]),
        "obb_cxcywha_rad": np.array([[30.0, 20.0, 4.0, 2.0, 0.0]]),
        "rotations": np.eye(3)[None],
        "translations": np.array([[0.0, 0.0, 1.0]]),
    }
    matched = match_image(prediction, target, iou_threshold=0.5)
    assert set(matched) == {0}
    assert matched[0]["iou"] == 1.0
    assert matched[0]["rotation_degrees"] == 0.0
    assert matched[0]["target_obb_anisotropy"] == 0.6


def test_paired_anisotropy_summary_exposes_where_second_run_improves():
    first = np.array([10.0, 20.0, 30.0, 40.0])
    second = np.array([11.0, 21.0, 28.0, 36.0])
    anisotropy = np.array([0.05, 0.15, 0.55, 0.75])

    bins = summarize_paired_by_anisotropy(
        first, second, anisotropy, bin_edges=(0.0, 0.2, 0.6, 1.0))

    assert [item["count"] for item in bins] == [2, 1, 1]
    assert bins[0]["mean_rotation_delta_degrees"] == 1.0
    assert bins[1]["mean_rotation_delta_degrees"] == -2.0
    assert bins[2]["second_better_fraction"] == 1.0
