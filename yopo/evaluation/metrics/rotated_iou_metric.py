# Copyright (c) OpenMMLab. All rights reserved.
"""Metric for evaluating 2D rotated detection boxes without MMRotate."""
from __future__ import annotations

from collections import OrderedDict
from typing import Any, Sequence

import numpy as np
import torch
from mmcv.ops import box_iou_rotated
from mmengine.evaluator import BaseMetric

from yopo.evaluation.functional import average_precision
from yopo.registry import METRICS


def _as_cpu_tensor(value: Any, dtype: torch.dtype) -> torch.Tensor:
    """Return a detached CPU tensor from a Tensor or YOPO box container."""
    if hasattr(value, 'tensor'):
        value = value.tensor
    return torch.as_tensor(value).detach().cpu().to(dtype=dtype)


def _field(value: Any, name: str) -> Any:
    """Read a field from either InstanceData or its evaluator dict form."""
    return value[name] if isinstance(value, dict) else getattr(value, name)


@METRICS.register_module()
class RotatedIoUMetric(BaseMetric):
    """Evaluate rotated detections using the same 5D boxes as the model.

    AP is calculated from every emitted detection, as required by a ranked
    precision-recall metric. ``score_thr`` applies only to the reported
    operating-point precision, recall, and matched IoU. The metric reports
    both all-points AP (the checkpoint-selection key) and VOC07 11-point AP.

    It is deliberately self-contained: YOPO does not depend on MMRotate,
    while :func:`mmcv.ops.box_iou_rotated` is already used by the rotated-box
    model stack.
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
            gt = data_sample['gt_instances']
            ignored = data_sample.get('ignored_instances', None)
            result = dict(
                pred_bboxes=_as_cpu_tensor(
                    _field(pred, 'bboxes'), torch.float32),
                pred_scores=_as_cpu_tensor(
                    _field(pred, 'scores'), torch.float32),
                pred_labels=_as_cpu_tensor(
                    _field(pred, 'labels'), torch.long),
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

    def _validate_result(self, result: dict[str, torch.Tensor]) -> None:
        """Reject malformed evaluator inputs instead of silently biasing AP."""
        for prefix in ('pred', 'gt', 'ignore'):
            boxes = result[f'{prefix}_bboxes']
            labels = result[f'{prefix}_labels']
            if boxes.ndim != 2 or boxes.shape[1] != 5:
                raise ValueError(
                    f'{prefix}_bboxes must have shape (N, 5), got '
                    f'{tuple(boxes.shape)}')
            if labels.ndim != 1 or len(labels) != len(boxes):
                raise ValueError(
                    f'{prefix}_labels must have shape ({len(boxes)},), got '
                    f'{tuple(labels.shape)}')
            if not torch.isfinite(boxes).all():
                raise ValueError(f'{prefix}_bboxes contain NaN or Inf')
            if len(labels) and ((labels < 0).any()
                                or (labels >= self.num_classes).any()):
                raise ValueError(
                    f'{prefix}_labels must be in [0, {self.num_classes})')
        scores = result['pred_scores']
        if scores.ndim != 1 or len(scores) != len(result['pred_bboxes']):
            raise ValueError(
                'pred_scores must have the same length as pred_bboxes')
        if not torch.isfinite(scores).all():
            raise ValueError('pred_scores contain NaN or Inf')

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
        """Calculate ranked AP and thresholded operating metrics by class."""
        class_aps: list[float] = []
        class_voc07_aps: list[float] = []
        class_recalls: list[float] = []
        class_precisions: list[float] = []
        matched_ious: list[float] = []
        num_ground_truth = 0
        num_predictions = 0
        num_ignored_predictions = 0

        for result in results:
            self._validate_result(result)

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
            evaluated_scores: list[float] = []
            matched_iou_by_score: list[tuple[float, float]] = []

            for score, image_index, box in detections:
                regular = gt_by_image[image_index]
                ignored = ignored_by_image[image_index]
                candidates = torch.cat((regular, ignored), dim=0)
                max_iou, gt_index = self._max_iou(box, candidates)
                if (gt_index >= len(regular) and gt_index >= 0
                        and max_iou >= self.iou_thr):
                    if score >= self.score_thr:
                        num_ignored_predictions += 1
                    continue

                is_true_positive = (
                    gt_index >= 0 and max_iou >= self.iou_thr
                    and not matched[image_index][gt_index])
                if is_true_positive:
                    matched[image_index][gt_index] = True
                    true_positives.append(1.0)
                    false_positives.append(0.0)
                    matched_iou_by_score.append((score, max_iou))
                else:
                    true_positives.append(0.0)
                    false_positives.append(1.0)
                evaluated_scores.append(score)

            if true_positives:
                tp_flags = np.asarray(true_positives)
                fp_flags = np.asarray(false_positives)
                tp = np.cumsum(tp_flags)
                fp = np.cumsum(fp_flags)
                recalls = tp / positives
                precisions = tp / np.maximum(tp + fp, np.finfo(float).eps)
                class_aps.append(
                    float(average_precision(recalls, precisions, mode='area')))
                class_voc07_aps.append(
                    float(average_precision(
                        recalls, precisions, mode='11points')))

                operating = np.asarray(evaluated_scores) >= self.score_thr
                operating_tp = float(tp_flags[operating].sum())
                operating_fp = float(fp_flags[operating].sum())
                num_predictions += int(operating.sum())
                class_recalls.append(operating_tp / positives)
                class_precisions.append(
                    operating_tp / max(operating_tp + operating_fp,
                                       np.finfo(float).eps))
                matched_ious.extend(
                    iou for score, iou in matched_iou_by_score
                    if score >= self.score_thr)
            else:
                class_aps.append(0.0)
                class_voc07_aps.append(0.0)
                class_recalls.append(0.0)
                class_precisions.append(0.0)

        suffix = f'{round(self.iou_thr * 100):02d}'
        return OrderedDict([
            (f'rbbox_mAP_{suffix}',
             float(np.mean(class_aps)) if class_aps else 0.0),
            (f'rbbox_mAP_{suffix}_voc07',
             float(np.mean(class_voc07_aps)) if class_voc07_aps else 0.0),
            (f'rbbox_recall_{suffix}',
             float(np.mean(class_recalls)) if class_recalls else 0.0),
            (f'rbbox_precision_{suffix}',
             float(np.mean(class_precisions)) if class_precisions else 0.0),
            ('rbbox_mean_matched_rIoU',
             float(np.mean(matched_ious)) if matched_ious else 0.0),
            ('rbbox_num_gt', float(num_ground_truth)),
            ('rbbox_num_pred', float(num_predictions)),
            ('rbbox_num_ignored_pred', float(num_ignored_predictions)),
            ('rbbox_score_thr', float(self.score_thr)),
        ])
