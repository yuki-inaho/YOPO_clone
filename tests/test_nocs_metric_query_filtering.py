import torch
import numpy as np
import pytest

from yopo.evaluation.metrics import nocs_metric as nocs_metric_module
from yopo.evaluation.metrics.nocs_metric import (
    NOCSMetric,
    rescale_ground_truth_bboxes,
    select_aligned_prediction_indices,
)


def test_query_filtering_applies_score_then_nms_and_preserves_indices():
    pred = {
        'bboxes': torch.tensor([
            [0.0, 0.0, 10.0, 10.0],
            [1.0, 1.0, 11.0, 11.0],
            [20.0, 20.0, 30.0, 30.0],
            [40.0, 40.0, 50.0, 50.0],
        ]),
        'scores': torch.tensor([0.9, 0.8, 0.7, 0.1]),
        'labels': torch.tensor([0, 0, 0, 0]),
    }

    keep = select_aligned_prediction_indices(
        pred,
        score_thr=0.2,
        nms_cfg={'type': 'nms', 'iou_threshold': 0.5},
    )

    assert keep.tolist() == [0, 2]


def test_query_filtering_can_keep_all_queries():
    pred = {
        'bboxes': torch.tensor([
            [0.0, 0.0, 10.0, 10.0],
            [1.0, 1.0, 11.0, 11.0],
        ]),
        'scores': torch.tensor([0.9, 0.1]),
        'labels': torch.tensor([0, 0]),
    }

    keep = select_aligned_prediction_indices(
        pred, score_thr=0.0, nms_cfg=None)

    assert keep.tolist() == [0, 1]


def test_ground_truth_boxes_are_restored_to_prediction_coordinates():
    resized = np.array([[145.2174, 340.1682, 162.6087, 356.8825]], dtype=np.float32)

    restored = rescale_ground_truth_bboxes(
        resized, (0.8695652173913043, 0.869140625))

    np.testing.assert_allclose(
        restored, [[167.0, 391.38458, 187.0, 410.61536]], rtol=1e-5)


def test_ground_truth_rescale_rejects_invalid_scale_factor():
    with pytest.raises(ValueError, match="scale_factor"):
        rescale_ground_truth_bboxes(
            np.array([[0.0, 0.0, 1.0, 1.0]], dtype=np.float32), (1.0, 0.0))


def test_metric_process_accepts_dict_data_samples_with_scale_factor(monkeypatch):
    metric = NOCSMetric(nms_cfg=None, score_thr=0.0)
    metric.dataset_meta = {'classes': ('fruit',)}
    data_sample = {
        'scale_factor': (0.5, 0.5),
        'gt_instances': {
            'labels': torch.tensor([0]),
            'bboxes': torch.tensor([[5.0, 10.0, 15.0, 20.0]]),
            'translations': torch.zeros((1, 3)),
            'rotations': torch.eye(3).reshape(1, 3, 3),
            'sizes': torch.ones((1, 3)),
            'T': torch.eye(4).reshape(1, 4, 4),
        },
        'ignored_instances': {
            'labels': torch.empty(0, dtype=torch.long),
            'bboxes': torch.empty((0, 4)),
        },
        'pred_instances': {
            'labels': torch.tensor([0]),
            'bboxes': torch.tensor([[10.0, 20.0, 30.0, 40.0]]),
            'scores': torch.tensor([0.9]),
            'translations': torch.zeros((1, 3)),
            'rotations': torch.eye(3).reshape(1, 3, 3),
            'sizes': torch.ones((1, 3)),
            'T': torch.eye(4).reshape(1, 4, 4),
        },
    }

    metric.process({}, [data_sample])

    gt, _ = metric.results[0]
    np.testing.assert_allclose(gt['bboxes'], [[10.0, 20.0, 30.0, 40.0]])


def test_nocs_ap50_keeps_full_precision_for_plateau_monitoring(monkeypatch):
    monkeypatch.setattr(
        nocs_metric_module,
        'eval_map',
        lambda *args, **kwargs: (0.123456, []),
    )
    metric = NOCSMetric(nms_cfg=None, score_thr=0.0)
    metric.dataset_meta = {'classes': ('fruit',)}
    monkeypatch.setattr(
        metric,
        'compute_independent_mAP',
        lambda *args, **kwargs: {},
    )
    gt = {
        'labels': np.array([0]),
        'bboxes': np.array([[0.0, 0.0, 10.0, 10.0]]),
    }
    pred = {
        'labels': np.array([0]),
        'bboxes': np.array([[0.0, 0.0, 10.0, 10.0]]),
        'scores': np.array([0.9]),
    }

    result = metric.compute_metrics([(gt, pred)])

    assert result['AP50'] == pytest.approx(0.123456)
