# Copyright (c) OpenMMLab. All rights reserved.
"""Metric for evaluating 2D rotated detection boxes without MMRotate."""
from __future__ import annotations

from collections import OrderedDict
from typing import Any, Sequence

import numpy as np
import torch
from mmcv.ops import box_iou_rotated
from mmengine.evaluator import BaseMetric

from yopo.registry import METRICS


def _as_cpu_tensor(value: Any, dtype: torch.dtype) -> torch.Tensor:
    """Return a detached CPU tensor from a Tensor or YOPO box container."""
    if hasattr(value, 'tensor'):
        value = value.tensor
    return torch.as_tensor(value).detach().cpu().to(dtype=dtype)


def _field(value: Any, name: str) -> Any:
    """Read a field from either InstanceData or its evaluator dict form."""
    return value[name] if isinstance(value, dict) else getattr(value, name)


def _average_precision(recalls: np.ndarray,
                       precisions: np.ndarray) -> float:
    """Compute all-points interpolated average precision."""
    recalls = np.concatenate(([0.0], recalls, [1.0]))
    precisions = np.concatenate(([0.0], precisions, [0.0]))
    precisions = np.maximum.accumulate(precisions[::-1])[::-1]
    change_points = np.where(recalls[1:] != recalls[:-1])[0]
    return float(np.sum((recalls[change_points + 1] - recalls[change_points])
                        * precisions[change_points + 1]))


@METRICS.register_module()
class RotatedIoUMetric(BaseMetric):
    """Evaluate rotated detections using the same 5D boxes as the model.

    The metric reports rbox mAP at ``iou_thr`` and the mean IoU of true
    positives.  It is deliberately self-contained: YOPO does not depend on
    MMRotate, while :func:`mmcv.ops.box_iou_rotated` is already used by the
    rotated-box model stack.
    """

    default_prefix = None

    def __init__(self,
                 iou_thr: float = 0.5,
                 score_thr: float = 0.05,
                 num_classes: int = 1,
                 collect_device: str = 'cpu',
                 prefix: str | None = None) -> None:
        if not 0.0 < iou_thr <= 1.0:
            raise ValueError(f'iou_thr must be in (0, 1], got {iou_thr}')
        if not 0.0 <= score_thr <= 1.0:
            raise ValueError(
                f'score_thr must be in [0, 1], got {score_thr}')
        if num_classes < 1:
            raise ValueError('num_classes must be positive')
        super().__init__(collect_device=collect_device, prefix=prefix)
        self.iou_thr = iou_thr
        self.score_thr = score_thr
        self.num_classes = num_classes

    def process(self, data_batch: dict,
                data_samples: Sequence[Any]) -> None:
        """Collect detached predictions and ground truth for each image."""
        for data_sample in data_samples:
            # Evaluator passes ``BaseDataElement`` instances as dict-like
            # objects after ``BaseMetric`` conversion, matching CocoMetric.
            pred = data_sample['pred_instances']
            pred_scores = _field(pred, 'scores')
            keep = _as_cpu_tensor(pred_scores >= self.score_thr, torch.bool)
            gt = data_sample['gt_instances']
            ignored = data_sample.get('ignored_instances', None)
            result = dict(
                pred_bboxes=_as_cpu_tensor(
                    _field(pred, 'bboxes'), torch.float32)[keep],
                pred_scores=_as_cpu_tensor(pred_scores, torch.float32)[keep],
                pred_labels=_as_cpu_tensor(
                    _field(pred, 'labels'), torch.long)[keep],
                gt_bboxes=_as_cpu_tensor(_field(gt, 'bboxes'), torch.float32),
                gt_labels=_as_cpu_tensor(_field(gt, 'labels'), torch.long),
                ignore_bboxes=(
                    _as_cpu_tensor(_field(ignored, 'bboxes'), torch.float32)
                    if ignored is not None else torch.empty((0, 5))),
                ignore_labels=(
                    _as_cpu_tensor(_field(ignored, 'labels'), torch.long)
                    if ignored is not None else torch.empty((0, ),
                                                               dtype=torch.long)))
            self.results.append(result)

    @staticmethod
    def _max_iou(box: torch.Tensor, candidates: torch.Tensor) -> tuple[float,
                                                                         int]:
        if candidates.numel() == 0:
            return 0.0, -1
        ious = box_iou_rotated(
            box.unsqueeze(0), candidates, mode='iou', aligned=False,
            clockwise=True).squeeze(0)
        max_iou, index = ious.max(dim=0)
        return float(max_iou), int(index)

    def compute_metrics(self, results: list[dict[str, torch.Tensor]]) -> dict:
        """Calculate rbox AP/recall/precision and matched IoU by class."""
        class_aps: list[float] = []
        class_recalls: list[float] = []
        class_precisions: list[float] = []
        matched_ious: list[float] = []
        num_ground_truth = 0

        for class_id in range(self.num_classes):
            gt_by_image: dict[int, torch.Tensor] = {}
            ignored_by_image: dict[int, torch.Tensor] = {}
            detections: list[tuple[float, int, torch.Tensor]] = []

            for image_index, result in enumerate(results):
                gt_mask = result['gt_labels'] == class_id
                gt_by_image[image_index] = result['gt_bboxes'][gt_mask]
                ignore_mask = result['ignore_labels'] == class_id
                ignored_by_image[image_index] = result['ignore_bboxes'][
                    ignore_mask]
                pred_mask = result['pred_labels'] == class_id
                for box, score in zip(result['pred_bboxes'][pred_mask],
                                      result['pred_scores'][pred_mask]):
                    detections.append((float(score), image_index, box))

            positives = sum(len(boxes) for boxes in gt_by_image.values())
            num_ground_truth += positives
            if positives == 0:
                continue

            detections.sort(key=lambda item: item[0], reverse=True)
            matched = {
                image_index: torch.zeros(len(boxes), dtype=torch.bool)
                for image_index, boxes in gt_by_image.items()
            }
            true_positives: list[float] = []
            false_positives: list[float] = []

            for _, image_index, box in detections:
                max_iou, gt_index = self._max_iou(box,
                                                   gt_by_image[image_index])
                if (gt_index >= 0 and max_iou >= self.iou_thr
                        and not matched[image_index][gt_index]):
                    matched[image_index][gt_index] = True
                    true_positives.append(1.0)
                    false_positives.append(0.0)
                    matched_ious.append(max_iou)
                    continue

                ignore_iou, _ = self._max_iou(box,
                                               ignored_by_image[image_index])
                if ignore_iou >= self.iou_thr:
                    continue
                true_positives.append(0.0)
                false_positives.append(1.0)

            if true_positives:
                tp = np.cumsum(np.asarray(true_positives))
                fp = np.cumsum(np.asarray(false_positives))
                recalls = tp / positives
                precisions = tp / np.maximum(tp + fp, np.finfo(float).eps)
                class_aps.append(_average_precision(recalls, precisions))
                class_recalls.append(float(recalls[-1]))
                class_precisions.append(float(precisions[-1]))
            else:
                class_aps.append(0.0)
                class_recalls.append(0.0)
                class_precisions.append(0.0)

        suffix = f'{round(self.iou_thr * 100):02d}'
        return OrderedDict([
            (f'rbbox_mAP_{suffix}',
             float(np.mean(class_aps)) if class_aps else 0.0),
            (f'rbbox_recall_{suffix}',
             float(np.mean(class_recalls)) if class_recalls else 0.0),
            (f'rbbox_precision_{suffix}',
             float(np.mean(class_precisions)) if class_precisions else 0.0),
            ('rbbox_mean_matched_rIoU',
             float(np.mean(matched_ious)) if matched_ious else 0.0),
            ('rbbox_num_gt', float(num_ground_truth)),
        ])
