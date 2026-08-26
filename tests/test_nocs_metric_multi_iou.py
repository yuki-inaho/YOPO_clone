from collections import OrderedDict

import numpy as np
import pytest
import torch

from yopo.evaluation.metrics import nocs_metric as nocs_metric_module
from yopo.evaluation.metrics.nocs_metric import (
    NOCSMetric,
    normalize_2d_iou_thresholds,
)


COCO_IOU_THRS = tuple(np.arange(0.50, 0.96, 0.05).tolist())


def _sample(*, with_prediction: bool = True):
    gt = {
        "labels": np.array([0], dtype=np.int64),
        "bboxes": np.array([[0.0, 0.0, 10.0, 10.0]], dtype=np.float32),
        "labels_ignore": np.empty(0, dtype=np.int64),
        "bboxes_ignore": np.empty((0, 4), dtype=np.float32),
    }
    pred = {
        "labels": np.array([0], dtype=np.int64) if with_prediction else np.empty(0, dtype=np.int64),
        "bboxes": (
            np.array([[0.0, 0.0, 10.0, 10.0]], dtype=np.float32)
            if with_prediction
            else np.empty((0, 4), dtype=np.float32)
        ),
        "scores": np.array([0.9], dtype=np.float32) if with_prediction else np.empty(0, dtype=np.float32),
    }
    return gt, pred


def _metric(iou_thrs=0.5, **kwargs):
    metric = NOCSMetric(
        nms_cfg=None, score_thr=0.0, iou_thrs=iou_thrs, **kwargs)
    metric.dataset_meta = {"classes": ("fruit",)}
    metric.compute_independent_mAP = lambda *args, **kwargs: OrderedDict()
    return metric


def test_default_remains_single_full_precision_ap50(monkeypatch):
    calls = []

    def fake_eval_map(*args, **kwargs):
        calls.append(kwargs)
        return 0.123456, []

    monkeypatch.setattr(nocs_metric_module, "eval_map", fake_eval_map)

    result = _metric().compute_metrics([_sample()])

    assert result == {"AP50": pytest.approx(0.123456)}
    assert [call["iou_thr"] for call in calls] == [0.5]
    assert calls[0]["eval_mode"] == "area"
    assert calls[0]["use_legacy_coordinate"] is True


def test_coco_range_reports_per_threshold_and_ap50_95(monkeypatch):
    calls = []

    def fake_eval_map(class_preds, gts, **kwargs):
        calls.append((class_preds, gts, kwargs))
        return kwargs["iou_thr"], []

    monkeypatch.setattr(nocs_metric_module, "eval_map", fake_eval_map)

    result = _metric(COCO_IOU_THRS).compute_metrics([_sample()])

    assert [call[2]["iou_thr"] for call in calls] == pytest.approx(COCO_IOU_THRS)
    assert result["AP50"] == pytest.approx(0.50)
    assert result["AP75"] == pytest.approx(0.75)
    assert result["AP95"] == pytest.approx(0.95)
    assert result["AP50_95"] == pytest.approx(np.mean(COCO_IOU_THRS))
    assert len([key for key in result if key.startswith("AP")]) == 11

    # eval_map gets MMDetection's image -> class -> (bbox, score) layout.
    class_dets = calls[0][0]
    assert len(class_dets) == 1
    assert len(class_dets[0]) == 1
    assert class_dets[0][0].shape == (1, 5)


def test_custom_thresholds_report_each_ap_without_mislabelled_range(monkeypatch):
    monkeypatch.setattr(
        nocs_metric_module,
        "eval_map",
        lambda *args, **kwargs: (kwargs["iou_thr"] / 2.0, []),
    )

    result = _metric([0.5, 0.75]).compute_metrics([_sample()])

    assert result == {"AP50": pytest.approx(0.25), "AP75": pytest.approx(0.375)}
    assert "AP50_95" not in result


def test_real_eval_map_handles_empty_detections():
    result = _metric([0.5, 0.75]).compute_metrics(
        [_sample(with_prediction=False)])

    assert result["AP50"] == pytest.approx(0.0)
    assert result["AP75"] == pytest.approx(0.0)


def test_real_eval_map_ap_changes_at_configured_iou_boundary():
    gt, pred = _sample()
    gt["bboxes"] = np.array([[0.0, 0.0, 9.0, 9.0]], dtype=np.float32)
    # With legacy inclusive coordinates this detection has IoU = 0.60.
    pred["bboxes"] = np.array([[0.0, 0.0, 5.0, 9.0]], dtype=np.float32)

    result = _metric([0.5, 0.75]).compute_metrics([(gt, pred)])

    assert result["AP50"] == pytest.approx(1.0)
    assert result["AP75"] == pytest.approx(0.0)


@pytest.mark.parametrize(
    "iou_thrs, error_type",
    [
        ([], ValueError),
        ([0.5, np.nan], ValueError),
        ([np.inf], ValueError),
        ([0.0], ValueError),
        ([1.01], ValueError),
        ([0.5, 0.5], ValueError),
        ("0.5", TypeError),
        (True, TypeError),
    ],
)
def test_invalid_iou_thresholds_fail_fast(iou_thrs, error_type):
    with pytest.raises(error_type, match="iou_thrs"):
        normalize_2d_iou_thresholds(iou_thrs)


def test_nonfinite_eval_map_result_fails_fast(monkeypatch):
    monkeypatch.setattr(
        nocs_metric_module, "eval_map", lambda *args, **kwargs: (np.nan, []))

    with pytest.raises(ValueError, match="non-finite AP"):
        _metric().compute_metrics([_sample()])


def test_empty_processed_results_fail_with_clear_error():
    with pytest.raises(ValueError, match="at least one processed sample"):
        _metric().compute_metrics([])


def _process_data_sample():
    return {
        "gt_instances": {
            "labels": torch.tensor([0]),
            "bboxes": torch.tensor([[0.0, 0.0, 10.0, 10.0]]),
            "translations": torch.zeros((1, 3)),
            "rotations": torch.eye(3).reshape(1, 3, 3),
            "sizes": torch.ones((1, 3)),
            "T": torch.eye(4).reshape(1, 4, 4),
        },
        "ignored_instances": {
            "labels": torch.empty(0, dtype=torch.long),
            "bboxes": torch.empty((0, 4)),
        },
        "pred_instances": {
            "labels": torch.tensor([0, 0]),
            "bboxes": torch.tensor(
                [[0.0, 0.0, 10.0, 10.0], [1.0, 1.0, 11.0, 11.0]]),
            "scores": torch.tensor([0.9, 0.1]),
            "translations": torch.zeros((2, 3)),
            "rotations": torch.eye(3).repeat(2, 1, 1),
            "sizes": torch.ones((2, 3)),
            "T": torch.eye(4).repeat(2, 1, 1),
        },
    }


def test_hbb_selection_keeps_a_separate_2d_view_from_pose_predictions():
    metric = NOCSMetric(
        score_thr=0.2,
        nms_cfg={"type": "nms", "iou_threshold": 0.5},
        hbb_selection={"score_thr": 0.0, "nms_cfg": None},
    )
    metric.dataset_meta = {"classes": ("fruit",)}

    metric.process({}, [_process_data_sample()])

    _, pred = metric.results[0]
    assert pred["scores"].tolist() == pytest.approx([0.9])
    assert pred["hbb_eval"]["scores"].tolist() == pytest.approx([0.9, 0.1])
    assert pred["hbb_eval"]["bboxes"].shape == (2, 4)
    assert pred["hbb_eval"]["labels"].shape == (2,)


def test_2d_eval_uses_hbb_view_while_pose_uses_post_nms_view(monkeypatch):
    metric = _metric(
        hbb_selection={"score_thr": 0.0, "nms_cfg": None})
    gt, pred = _sample()
    pred["hbb_eval"] = {
        "labels": np.array([0, 0], dtype=np.int64),
        "bboxes": np.array(
            [[0.0, 0.0, 10.0, 10.0], [1.0, 1.0, 11.0, 11.0]],
            dtype=np.float32,
        ),
        "scores": np.array([0.9, 0.1], dtype=np.float32),
    }
    observed = {}

    def fake_eval_map(class_preds, *args, **kwargs):
        observed["hbb_count"] = len(class_preds[0][0])
        return 0.5, []

    def fake_pose(preds, *args, **kwargs):
        observed["pose_count"] = len(preds[0]["bboxes"])
        return {}

    monkeypatch.setattr(nocs_metric_module, "eval_map", fake_eval_map)
    metric.compute_independent_mAP = fake_pose

    metric.compute_metrics([(gt, pred)])

    assert observed == {"hbb_count": 2, "pose_count": 1}


def test_omitted_hbb_selection_preserves_single_view():
    metric = NOCSMetric(nms_cfg=None, score_thr=0.0)
    metric.dataset_meta = {"classes": ("fruit",)}

    metric.process({}, [_process_data_sample()])

    assert "hbb_eval" not in metric.results[0][1]


@pytest.mark.parametrize(
    "hbb_selection, error_type",
    [
        ([], TypeError),
        ({"unknown": 1}, ValueError),
        ({"score_thr": np.nan}, ValueError),
        ({"score_thr": "bad"}, TypeError),
    ],
)
def test_invalid_hbb_selection_fails_fast(hbb_selection, error_type):
    with pytest.raises(error_type, match="hbb_selection"):
        NOCSMetric(hbb_selection=hbb_selection)


def test_detection_only_metrics_never_call_pose_evaluation(monkeypatch):
    metric = NOCSMetric(
        nms_cfg=None,
        score_thr=0.0,
        iou_thrs=[0.5, 0.75],
        compute_pose_metrics=False,
    )
    metric.dataset_meta = {"classes": ("fruit",)}
    metric.compute_independent_mAP = lambda *args, **kwargs: pytest.fail(
        "detection-only validation must not run 3D pose evaluation"
    )
    monkeypatch.setattr(
        nocs_metric_module,
        "eval_map",
        lambda *args, **kwargs: (kwargs["iou_thr"], []),
    )

    result = metric.compute_metrics([_sample()])

    assert result == {"AP50": pytest.approx(0.5), "AP75": pytest.approx(0.75)}


def test_detection_only_process_does_not_require_pose_predictions():
    sample = _process_data_sample()
    sample["pred_instances"] = {
        key: sample["pred_instances"][key]
        for key in ("labels", "bboxes", "scores")
    }
    metric = NOCSMetric(
        nms_cfg=None, score_thr=0.0, compute_pose_metrics=False)
    metric.dataset_meta = {"classes": ("fruit",)}

    metric.process({}, [sample])

    _, pred = metric.results[0]
    assert set(pred) == {"labels", "bboxes", "scores"}


@pytest.mark.parametrize("value", [None, 0, "false"])
def test_invalid_compute_pose_metrics_fails_fast(value):
    with pytest.raises(TypeError, match="compute_pose_metrics"):
        NOCSMetric(compute_pose_metrics=value)


def test_detection_only_metric_rejects_pose_dump_contract(tmp_path):
    with pytest.raises(ValueError, match="dump_results_path"):
        NOCSMetric(
            compute_pose_metrics=False,
            dump_results_path=str(tmp_path),
        )
