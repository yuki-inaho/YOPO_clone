#!/usr/bin/env python3
"""Find the exact F1-optimal score threshold for DOTA OBB predictions."""

from __future__ import annotations

import argparse
import json
import pickle
from pathlib import Path

import numpy as np
import torch
from mmcv.ops import box_iou_rotated

from yopo.datasets.dota_tomato import qbox2rbox_np


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('prediction', type=Path)
    parser.add_argument('label_dir', type=Path)
    parser.add_argument('output', type=Path)
    parser.add_argument('--iou-thr', type=float, default=0.5)
    parser.add_argument('--difficulty-thr', type=int, default=100)
    parser.add_argument('--also-evaluate', type=float, nargs='*', default=(0.2,))
    return parser.parse_args()


def _tensor(value: object, dtype: torch.dtype) -> torch.Tensor:
    if hasattr(value, 'tensor'):
        value = value.tensor
    return torch.as_tensor(value, dtype=dtype).detach().cpu()


def _load_gt(path: Path, difficulty_thr: int) -> tuple[torch.Tensor, torch.Tensor]:
    regular: list[np.ndarray] = []
    ignored: list[np.ndarray] = []
    for line_number, line in enumerate(path.read_text().splitlines(), start=1):
        if not line.strip():
            continue
        parts = line.split()
        if len(parts) not in (9, 10):
            raise ValueError(f'{path}:{line_number}: expected 9 or 10 columns')
        box = qbox2rbox_np(np.asarray(parts[:8], dtype=np.float32))
        difficulty = int(parts[9]) if len(parts) == 10 else 0
        (ignored if difficulty > difficulty_thr else regular).append(box)
    empty = torch.empty((0, 5), dtype=torch.float32)
    return (
        torch.as_tensor(np.asarray(regular), dtype=torch.float32)
        if regular else empty,
        torch.as_tensor(np.asarray(ignored), dtype=torch.float32)
        if ignored else empty,
    )


def _operating_point(tp: int, fp: int, positives: int) -> dict[str, float | int]:
    precision = tp / (tp + fp) if tp + fp else 0.0
    recall = tp / positives if positives else 0.0
    f1 = 2 * precision * recall / (precision + recall) \
        if precision + recall else 0.0
    return {
        'true_positives': tp,
        'false_positives': fp,
        'false_negatives': positives - tp,
        'precision': precision,
        'recall': recall,
        'f1': f1,
    }


def evaluate(predictions: list[dict], label_dir: Path, iou_thr: float,
             difficulty_thr: int, fixed_thresholds: tuple[float, ...]) -> dict:
    gt_by_image: dict[int, torch.Tensor] = {}
    ignored_by_image: dict[int, torch.Tensor] = {}
    detections: list[tuple[float, int, torch.Tensor]] = []
    for image_index, prediction in enumerate(predictions):
        stem = Path(prediction['img_path']).stem
        gt_by_image[image_index], ignored_by_image[image_index] = _load_gt(
            label_dir / f'{stem}.txt', difficulty_thr)
        instances = prediction['pred_instances']
        boxes = _tensor(instances['bboxes'], torch.float32)
        scores = _tensor(instances['scores'], torch.float32)
        labels = _tensor(instances['labels'], torch.long)
        for box, score in zip(boxes[labels == 0], scores[labels == 0]):
            detections.append((float(score), image_index, box))

    detections.sort(key=lambda item: item[0], reverse=True)
    matched = {
        index: torch.zeros(len(boxes), dtype=torch.bool)
        for index, boxes in gt_by_image.items()
    }
    evaluated: list[tuple[float, int, int]] = []
    ignored_predictions = 0
    for score, image_index, box in detections:
        regular = gt_by_image[image_index]
        ignored = ignored_by_image[image_index]
        candidates = torch.cat((regular, ignored), dim=0)
        if len(candidates):
            ious = box_iou_rotated(
                box[None], candidates, mode='iou', aligned=False,
                clockwise=True).squeeze(0)
            max_iou, gt_index_tensor = ious.max(dim=0)
            max_iou = float(max_iou)
            gt_index = int(gt_index_tensor)
        else:
            max_iou, gt_index = 0.0, -1
        if (gt_index >= len(regular) and gt_index >= 0
                and max_iou >= iou_thr):
            ignored_predictions += 1
            continue
        is_tp = (gt_index >= 0 and max_iou >= iou_thr
                 and not matched[image_index][gt_index])
        if is_tp:
            matched[image_index][gt_index] = True
        evaluated.append((score, int(is_tp), int(not is_tp)))

    positives = sum(len(boxes) for boxes in gt_by_image.values())
    best: dict | None = None
    tp = fp = 0
    for index, (score, is_tp, is_fp) in enumerate(evaluated):
        tp += is_tp
        fp += is_fp
        if index + 1 < len(evaluated) and evaluated[index + 1][0] == score:
            continue
        point = {'score_threshold': score, 'num_predictions': tp + fp,
                 **_operating_point(tp, fp, positives)}
        if best is None or point['f1'] > best['f1']:
            best = point

    fixed = {}
    scores = np.asarray([item[0] for item in evaluated], dtype=np.float64)
    tp_flags = np.asarray([item[1] for item in evaluated], dtype=np.int64)
    fp_flags = np.asarray([item[2] for item in evaluated], dtype=np.int64)
    for threshold in fixed_thresholds:
        keep = scores >= threshold
        fixed[f'{threshold:g}'] = {
            'score_threshold': threshold,
            'num_predictions': int(keep.sum()),
            **_operating_point(
                int(tp_flags[keep].sum()), int(fp_flags[keep].sum()), positives),
        }
    return {
        'matching': 'score-ranked greedy one-to-one rotated IoU',
        'iou_threshold': iou_thr,
        'images': len(predictions),
        'ground_truths': positives,
        'ignored_predictions_all_scores': ignored_predictions,
        'best_f1': best,
        'fixed_thresholds': fixed,
    }


def main() -> None:
    args = parse_args()
    if not 0 < args.iou_thr <= 1:
        raise ValueError('--iou-thr must be in (0, 1]')
    if any(not 0 <= value <= 1 for value in args.also_evaluate):
        raise ValueError('--also-evaluate values must be in [0, 1]')
    with args.prediction.open('rb') as stream:
        predictions = pickle.load(stream)
    report = {
        'prediction': str(args.prediction.resolve()),
        'label_dir': str(args.label_dir.resolve()),
        **evaluate(
            predictions, args.label_dir, args.iou_thr,
            args.difficulty_thr, tuple(args.also_evaluate)),
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(report, indent=2) + '\n')
    print(json.dumps(report, indent=2))


if __name__ == '__main__':
    main()
