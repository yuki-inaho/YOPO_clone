#!/usr/bin/env python3
"""Render GT/predicted 2D boxes and quantify DETR query localization."""

from __future__ import annotations

import argparse
import json
import pickle
from pathlib import Path

import cv2
import numpy as np
import torch
from mmengine.config import Config

from yopo.registry import DATASETS
from yopo.structures.bbox import bbox_overlaps
from yopo.utils import register_all_modules


def _tensor(value) -> torch.Tensor:
    if hasattr(value, "tensor"):
        value = value.tensor
    return torch.as_tensor(value).detach().cpu().float()


def summarize_detection_geometry(
    gt_boxes: list[torch.Tensor], pred_boxes: list[torch.Tensor]
) -> dict:
    if len(gt_boxes) != len(pred_boxes):
        raise ValueError("GT and prediction image counts differ")
    max_ious = []
    nearest_centers = []
    gt_width_height = []
    pred_width_height = []
    for gt, pred in zip(gt_boxes, pred_boxes):
        gt = _tensor(gt)
        pred = _tensor(pred)
        if gt.ndim != 2 or gt.shape[1] != 4:
            raise ValueError(f"invalid GT box shape {tuple(gt.shape)}")
        if pred.ndim != 2 or pred.shape[1] != 4:
            raise ValueError(f"invalid prediction box shape {tuple(pred.shape)}")
        if len(gt) and len(pred):
            max_ious.append(bbox_overlaps(gt, pred).max(dim=1).values)
            gt_centers = (gt[:, :2] + gt[:, 2:]) / 2
            pred_centers = (pred[:, :2] + pred[:, 2:]) / 2
            nearest_centers.append(torch.cdist(gt_centers, pred_centers).min(dim=1).values)
        elif len(gt):
            max_ious.append(torch.zeros(len(gt)))
            nearest_centers.append(torch.full((len(gt),), float("inf")))
        gt_width_height.append(gt[:, 2:] - gt[:, :2])
        pred_width_height.append(pred[:, 2:] - pred[:, :2])

    max_iou = torch.cat(max_ious)
    center_distance = torch.cat(nearest_centers)
    gt_wh = torch.cat(gt_width_height)
    pred_wh = torch.cat(pred_width_height)

    def quantiles(values: torch.Tensor) -> dict:
        return {
            "q10": float(torch.quantile(values, 0.1)),
            "median": float(torch.quantile(values, 0.5)),
            "q90": float(torch.quantile(values, 0.9)),
        }

    return {
        "image_count": len(gt_boxes),
        "gt_count": int(len(max_iou)),
        "prediction_count": int(sum(len(boxes) for boxes in pred_boxes)),
        "oracle_recall": {
            f"iou_{threshold:.2f}": float((max_iou >= threshold).float().mean())
            for threshold in (0.1, 0.25, 0.5, 0.75)
        },
        "gt_max_iou": quantiles(max_iou),
        "nearest_prediction_center_px": quantiles(center_distance),
        "gt_width_px": quantiles(gt_wh[:, 0]),
        "gt_height_px": quantiles(gt_wh[:, 1]),
        "prediction_width_px": quantiles(pred_wh[:, 0]),
        "prediction_height_px": quantiles(pred_wh[:, 1]),
    }


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("predictions", type=Path)
    parser.add_argument("config", type=Path)
    parser.add_argument("output_dir", type=Path)
    parser.add_argument("--num-images", type=int, default=5)
    parser.add_argument("--score-threshold", type=float, default=0.2)
    parser.add_argument("--max-predictions", type=int, default=150)
    return parser.parse_args()


def _draw_box(image: np.ndarray, box: torch.Tensor, color: tuple[int, int, int], width: int) -> None:
    x1, y1, x2, y2 = box.round().int().tolist()
    cv2.rectangle(image, (x1, y1), (x2, y2), color, width, cv2.LINE_AA)


def main() -> None:
    args = parse_args()
    if args.num_images <= 0 or args.max_predictions <= 0:
        raise ValueError("num-images and max-predictions must be positive")
    if not 0 <= args.score_threshold <= 1:
        raise ValueError("score-threshold must be in [0, 1]")

    register_all_modules()
    config = Config.fromfile(args.config)
    dataset = DATASETS.build(config.val_dataloader.dataset)
    with args.predictions.open("rb") as stream:
        predictions = pickle.load(stream)
    if len(predictions) != len(dataset):
        raise ValueError(
            f"prediction count {len(predictions)} != dataset count {len(dataset)}"
        )

    gt_boxes = []
    pred_boxes = []
    samples = []
    for index, prediction in enumerate(predictions):
        sample = dataset[index]["data_samples"]
        if Path(sample.img_path).name != Path(prediction["img_path"]).name:
            raise ValueError(
                f"image order mismatch at {index}: {sample.img_path} vs "
                f"{prediction['img_path']}"
            )
        gt = _tensor(sample.gt_instances.bboxes)
        pred = prediction["pred_instances"]
        boxes = _tensor(pred["bboxes"])
        scores = _tensor(pred["scores"])
        gt_boxes.append(gt)
        pred_boxes.append(boxes)
        samples.append((sample, gt, boxes, scores))

    report = summarize_detection_geometry(gt_boxes, pred_boxes)
    report["config"] = str(args.config.resolve())
    report["predictions"] = str(args.predictions.resolve())
    report["score_threshold"] = args.score_threshold
    report["overlays"] = []

    args.output_dir.mkdir(parents=True, exist_ok=True)
    selected = np.linspace(0, len(samples) - 1, min(args.num_images, len(samples)), dtype=int)
    for ordinal, index in enumerate(selected, start=1):
        sample, gt, boxes, scores = samples[int(index)]
        image = cv2.imread(sample.img_path, cv2.IMREAD_COLOR)
        if image is None:
            raise FileNotFoundError(sample.img_path)
        keep = torch.where(scores >= args.score_threshold)[0]
        keep = keep[torch.argsort(scores[keep], descending=True)][: args.max_predictions]

        for box in gt:
            _draw_box(image, box, (0, 220, 0), 1)
        for rank, query_index in enumerate(keep.tolist()):
            _draw_box(image, boxes[query_index], (0, 220, 255), 1)
            if rank < 12:
                x1, y1 = boxes[query_index, :2].round().int().tolist()
                cv2.putText(
                    image,
                    f"{float(scores[query_index]):.2f}",
                    (x1, max(10, y1 - 2)),
                    cv2.FONT_HERSHEY_SIMPLEX,
                    0.32,
                    (0, 220, 255),
                    1,
                    cv2.LINE_AA,
                )
        cv2.putText(
            image,
            f"GT green={len(gt)}  pred yellow={len(keep)}  thr={args.score_threshold:.2f}",
            (8, 18),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.45,
            (255, 255, 255),
            2,
            cv2.LINE_AA,
        )
        cv2.putText(
            image,
            f"GT green={len(gt)}  pred yellow={len(keep)}  thr={args.score_threshold:.2f}",
            (8, 18),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.45,
            (0, 0, 0),
            1,
            cv2.LINE_AA,
        )
        output = args.output_dir / f"{ordinal:02d}_{Path(sample.img_path).stem}_2d_boxes.png"
        if not cv2.imwrite(str(output), image):
            raise OSError(f"failed to write {output}")
        report["overlays"].append(
            {
                "image_index": int(index),
                "source": str(Path(sample.img_path).resolve()),
                "output": str(output.resolve()),
                "gt_count": len(gt),
                "drawn_prediction_count": len(keep),
                "drawn_score_min": float(scores[keep].min()) if len(keep) else None,
                "drawn_score_max": float(scores[keep].max()) if len(keep) else None,
            }
        )
        print(f"overlay: {output}")

    report_path = args.output_dir / "report.json"
    report_path.write_text(json.dumps(report, indent=2) + "\n", encoding="utf-8")
    print(f"report: {report_path}")


if __name__ == "__main__":
    main()
