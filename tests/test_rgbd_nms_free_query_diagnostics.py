import numpy as np
import pytest

from tools.analysis_tools.diagnose_rgbd_nms_free_queries import (
    binary_roc_auc,
    diagnose_dump,
    nms_xyxy,
    summarize_selected_predictions,
    threshold_key,
)


def test_nms_preserves_original_query_identity_and_score_order():
    boxes = np.array([
        [0.0, 0.0, 10.0, 10.0],
        [1.0, 1.0, 11.0, 11.0],
        [20.0, 20.0, 30.0, 30.0],
    ])
    scores = np.array([0.8, 0.9, 0.7])

    kept = nms_xyxy(boxes, scores, iou_threshold=0.5)

    assert kept.tolist() == [1, 2]


def test_binary_roc_auc_handles_ties_by_average_rank():
    labels = np.array([0, 0, 1, 1])
    scores = np.array([0.1, 0.5, 0.5, 0.9])

    assert np.isclose(binary_roc_auc(labels, scores), 0.875)


def test_threshold_keys_do_not_merge_adjacent_sweep_values():
    keys = {threshold_key(value) for value in (0.05, 0.1, 0.2, 0.25, 0.3, 0.35)}

    assert keys == {'0.05', '0.1', '0.2', '0.25', '0.3', '0.35'}


def test_selected_summary_counts_one_to_one_true_positives():
    predictions = [{
        "boxes": np.array([
            [0.0, 0.0, 10.0, 10.0],
            [1.0, 1.0, 11.0, 11.0],
            [20.0, 20.0, 30.0, 30.0],
        ]),
        "scores": np.array([0.9, 0.8, 0.7]),
        "gt_boxes": np.array([
            [0.0, 0.0, 10.0, 10.0],
            [20.0, 20.0, 30.0, 30.0],
        ]),
    }]

    raw = summarize_selected_predictions(
        predictions, score_threshold=0.0, nms_iou_threshold=None,
        match_iou_threshold=0.5)
    suppressed = summarize_selected_predictions(
        predictions, score_threshold=0.0, nms_iou_threshold=0.5,
        match_iou_threshold=0.5)

    assert raw == {
        "kept": 3,
        "mean_kept_per_image": 3.0,
        "true_positives": 2,
        "ground_truths": 2,
        "precision": 2 / 3,
        "recall": 1.0,
        "f1": 0.8,
        "mean_matched_iou": 1.0,
    }
    assert suppressed["kept"] == 2
    assert suppressed["precision"] == 1.0
    assert suppressed["recall"] == 1.0
    assert suppressed["f1"] == 1.0


def test_selected_summary_uses_zero_f1_for_an_empty_selection():
    predictions = [{
        "boxes": np.array([[0.0, 0.0, 10.0, 10.0]]),
        "scores": np.array([0.5]),
        "gt_boxes": np.array([[0.0, 0.0, 10.0, 10.0]]),
    }]

    summary = summarize_selected_predictions(
        predictions, score_threshold=0.9, nms_iou_threshold=None,
        match_iou_threshold=0.5)

    assert summary["kept"] == 0
    assert summary["precision"] == 0.0
    assert summary["recall"] == 0.0
    assert summary["f1"] == 0.0


def test_best_f1_selection_schema_and_deterministic_tie_break():
    predictions = [{
        "boxes": np.array([
            [0.0, 0.0, 10.0, 10.0],
            [20.0, 20.0, 30.0, 30.0],
        ]),
        "scores": np.array([0.9, 0.1]),
        "gt_boxes": np.array([[0.0, 0.0, 10.0, 10.0]]),
    }]

    result = diagnose_dump(
        predictions,
        score_thresholds=(0.5, 0.2),
        nms_iou_thresholds=(0.7, 0.3),
        match_iou_threshold=0.5,
    )

    assert result["best_f1_selection"] == {
        "score_threshold": 0.2,
        "nms_iou_threshold": None,
        "precision": 1.0,
        "recall": 1.0,
        "f1": 1.0,
        "kept": 1,
        "true_positives": 1,
    }
    assert result["selection_counterfactuals"]["0.2"]["none"]["f1"] == 1.0


@pytest.mark.parametrize("threshold", [np.nan, np.inf, -np.inf])
def test_diagnose_dump_rejects_non_finite_thresholds(threshold):
    with pytest.raises(ValueError, match="finite values"):
        diagnose_dump(
            [],
            score_thresholds=(threshold,),
            nms_iou_thresholds=(0.5,),
            match_iou_threshold=0.5,
        )
