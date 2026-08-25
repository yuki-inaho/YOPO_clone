import numpy as np

from tools.analysis_tools.diagnose_rgbd_nms_free_queries import (
    binary_roc_auc,
    nms_xyxy,
    summarize_selected_predictions,
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
        "mean_matched_iou": 1.0,
    }
    assert suppressed["kept"] == 2
    assert suppressed["precision"] == 1.0
    assert suppressed["recall"] == 1.0
