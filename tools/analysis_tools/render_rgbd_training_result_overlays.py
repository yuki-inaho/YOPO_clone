#!/usr/bin/env python3
"""Render matched 2D BBOX, predicted OBB, and GT/predicted 3D cuboids."""

from __future__ import annotations

import argparse
import json
import pickle
from pathlib import Path

import cv2
import numpy as np
import torch
from mmengine.config import Config
from scipy.optimize import linear_sum_assignment

from tools.analysis_tools.render_rgbd_3dbbox_overlays import (
    EDGES,
    cuboid_corners,
    intrinsic_matrix,
    project_corners,
)
from tools.analysis_tools.diagnose_rgbd_nms_free_queries import nms_xyxy
from yopo.registry import DATASETS
from yopo.utils import register_all_modules


GT_COLOR = (0, 210, 0)
PRED_COLOR = (0, 220, 255)
OBB_COLOR = (255, 0, 255)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("predictions", type=Path)
    parser.add_argument("config", type=Path)
    parser.add_argument("output_dir", type=Path)
    parser.add_argument("--checkpoint", type=Path)
    parser.add_argument("--num-images", type=int, default=6)
    parser.add_argument("--iou-threshold", type=float, default=0.5)
    parser.add_argument("--max-matches", type=int, default=10)
    parser.add_argument(
        "--all-predictions",
        action="store_true",
        help="draw every prediction above the confidence threshold without matching",
    )
    parser.add_argument("--score-threshold", type=float, default=0.2)
    parser.add_argument("--max-predictions", type=int, default=150)
    parser.add_argument(
        "--nms-iou-threshold",
        type=float,
        help=(
            "class-agnostic 2D NMS deployment post-processing; retained query "
            "indices select the corresponding aligned OBB/3D predictions"
        ),
    )
    return parser.parse_args()


def _array(value: object) -> np.ndarray:
    if hasattr(value, "tensor"):
        value = value.tensor
    if isinstance(value, torch.Tensor):
        value = value.detach().cpu().numpy()
    return np.asarray(value)


def pairwise_iou(first: np.ndarray, second: np.ndarray) -> np.ndarray:
    first = np.asarray(first, dtype=np.float64)
    second = np.asarray(second, dtype=np.float64)
    top_left = np.maximum(first[:, None, :2], second[None, :, :2])
    bottom_right = np.minimum(first[:, None, 2:], second[None, :, 2:])
    intersection = np.prod(np.clip(bottom_right - top_left, 0.0, None), axis=-1)
    first_area = np.prod(np.clip(first[:, 2:] - first[:, :2], 0.0, None), axis=-1)
    second_area = np.prod(np.clip(second[:, 2:] - second[:, :2], 0.0, None), axis=-1)
    union = first_area[:, None] + second_area[None, :] - intersection
    return np.divide(intersection, union, out=np.zeros_like(intersection), where=union > 0)


def matched_pairs(
    pred_boxes: np.ndarray,
    gt_boxes: np.ndarray,
    threshold: float,
) -> list[tuple[int, int, float]]:
    if not len(pred_boxes) or not len(gt_boxes):
        return []
    ious = pairwise_iou(pred_boxes, gt_boxes)
    pred_indices, gt_indices = linear_sum_assignment(-ious)
    pairs = [
        (int(pred), int(gt), float(ious[pred, gt]))
        for pred, gt in zip(pred_indices, gt_indices)
        if ious[pred, gt] >= threshold
    ]
    return sorted(pairs, key=lambda item: item[2], reverse=True)


def draw_box(
    image: np.ndarray,
    box: np.ndarray,
    color: tuple[int, int, int],
    width: int,
) -> None:
    x1, y1, x2, y2 = np.rint(box).astype(np.int32).tolist()
    cv2.rectangle(image, (x1, y1), (x2, y2), color, width, cv2.LINE_AA)


def draw_obb(
    image: np.ndarray,
    compact: np.ndarray,
    center_xy: np.ndarray,
    width: int = 2,
) -> None:
    height, width = image.shape[:2]
    covariance = np.array(
        [[compact[2], compact[3]], [compact[3], compact[4]]],
        dtype=np.float64,
    )
    scale = np.diag([width, height])
    covariance_px = scale @ covariance @ scale
    eigenvalues, eigenvectors = np.linalg.eigh(covariance_px)
    order = np.argsort(eigenvalues)[::-1]
    eigenvalues = eigenvalues[order]
    eigenvectors = eigenvectors[:, order]
    if not np.isfinite(eigenvalues).all() or eigenvalues[-1] <= 0:
        raise FloatingPointError("predicted OBB covariance is not finite SPD")
    axes = np.maximum(np.rint(np.sqrt(eigenvalues)), 1).astype(np.int32)
    major_axis = eigenvectors[:, 0]
    angle = float(np.degrees(np.arctan2(major_axis[1], major_axis[0])))
    center = tuple(np.rint(center_xy).astype(np.int32).tolist())
    cv2.ellipse(
        image,
        center,
        tuple(axes.tolist()),
        angle,
        0,
        360,
        OBB_COLOR,
        width,
        cv2.LINE_AA,
    )


def draw_cuboid(
    image: np.ndarray,
    size: np.ndarray,
    transform: np.ndarray,
    intrinsic: np.ndarray,
    color: tuple[int, int, int],
    width: int,
) -> None:
    projected = project_corners(cuboid_corners(size, transform), intrinsic)
    points = np.rint(projected).astype(np.int32)
    for start, end in EDGES:
        cv2.line(
            image,
            tuple(points[start]),
            tuple(points[end]),
            color,
            width,
            cv2.LINE_AA,
        )


def label_panel(image: np.ndarray, text: str) -> None:
    cv2.putText(image, text, (8, 20), cv2.FONT_HERSHEY_SIMPLEX, 0.48,
                (0, 0, 0), 3, cv2.LINE_AA)
    cv2.putText(image, text, (8, 20), cv2.FONT_HERSHEY_SIMPLEX, 0.48,
                (255, 255, 255), 1, cv2.LINE_AA)


def main() -> None:
    args = parse_args()
    if args.num_images <= 0 or args.max_matches <= 0:
        raise ValueError("num-images and max-matches must be positive")
    if args.max_predictions <= 0:
        raise ValueError("max-predictions must be positive")
    if not 0.0 <= args.iou_threshold <= 1.0:
        raise ValueError("iou-threshold must be in [0, 1]")
    if not 0.0 <= args.score_threshold <= 1.0:
        raise ValueError("score-threshold must be in [0, 1]")
    if (args.nms_iou_threshold is not None and
            not 0.0 <= args.nms_iou_threshold <= 1.0):
        raise ValueError("nms-iou-threshold must be in [0, 1]")
    if args.nms_iou_threshold is not None and not args.all_predictions:
        raise ValueError("nms-iou-threshold requires --all-predictions")
    if not args.predictions.is_file():
        raise FileNotFoundError(args.predictions)

    register_all_modules()
    config = Config.fromfile(args.config)
    dataset = DATASETS.build(config.val_dataloader.dataset)
    with args.predictions.open("rb") as stream:
        predictions = pickle.load(stream)
    if len(predictions) != len(dataset):
        raise ValueError(
            f"prediction count {len(predictions)} != dataset count {len(dataset)}"
        )

    selected = np.linspace(
        0, len(dataset) - 1, min(args.num_images, len(dataset)), dtype=int
    )
    args.output_dir.mkdir(parents=True, exist_ok=True)
    manifest = {
        "predictions": str(args.predictions.resolve()),
        "config": str(args.config.resolve()),
        "checkpoint": str(args.checkpoint.resolve()) if args.checkpoint else None,
        "matching": "one-to-one Hungarian on 2D xyxy IoU",
        "iou_threshold": args.iou_threshold,
        "draw_mode": "all predictions" if args.all_predictions else "matched",
        "score_threshold": args.score_threshold if args.all_predictions else None,
        "nms_iou_threshold": (
            args.nms_iou_threshold if args.all_predictions else None),
        "nms_role": (
            "deployment post-processing; retained 2D indices select the same "
            "query-aligned OBB and 3D predictions"
            if args.nms_iou_threshold is not None else None),
        "max_predictions": args.max_predictions if args.all_predictions else None,
        "legend": {
            "green": "GT 2D BBOX / GT 3D cuboid",
            "yellow": "matched predicted 2D BBOX / predicted 3D cuboid",
            "magenta": (
                "not drawn in all-predictions mode because background-query OBBs "
                "are not supervised"
                if args.all_predictions else "predicted 2D OBB ellipse"
            ),
        },
        "images": [],
    }

    for ordinal, index in enumerate(selected, start=1):
        prediction = predictions[int(index)]
        sample = dataset[int(index)]["data_samples"]
        if Path(sample.img_path).name != Path(prediction["img_path"]).name:
            raise ValueError(f"image order mismatch at index {index}")
        image = cv2.imread(sample.img_path, cv2.IMREAD_COLOR)
        if image is None:
            raise FileNotFoundError(sample.img_path)

        gt = sample.gt_instances
        pred = prediction["pred_instances"]
        gt_boxes = _array(gt.bboxes).astype(np.float64)
        pred_boxes = _array(pred["bboxes"]).astype(np.float64)
        pairs = matched_pairs(pred_boxes, gt_boxes, args.iou_threshold)
        accepted_pair_count = len(pairs)
        pairs = pairs[:args.max_matches]
        if not pairs and not args.all_predictions:
            raise ValueError(f"no accepted match for image index {index}")

        panel_2d = image.copy()
        panel_3d = image.copy()
        intrinsic = intrinsic_matrix(prediction["intrinsic"])
        # All-predictions mode intentionally omits query OBB ellipses because
        # unmatched/background OBBs are not supervised.  Keep ordinary
        # teacher-free dumps usable in that mode; matched rendering still
        # requires the explicit diagnostic OBB output.
        pred_obbs = (
            None if args.all_predictions else _array(pred["obb_gaussians"])
        )
        pred_scores = _array(pred["scores"])
        pred_sizes = _array(pred["sizes"])
        pred_transforms = _array(pred["T"])
        gt_sizes = _array(gt.sizes)
        gt_transforms = _array(gt.T)
        pair_records = []
        if args.all_predictions:
            pred_indices = np.flatnonzero(pred_scores >= args.score_threshold)
            if args.nms_iou_threshold is not None and len(pred_indices):
                local_kept = nms_xyxy(
                    pred_boxes[pred_indices], pred_scores[pred_indices],
                    args.nms_iou_threshold)
                pred_indices = pred_indices[local_kept]
            else:
                order = np.argsort(pred_scores[pred_indices])[::-1]
                pred_indices = pred_indices[order]
            pred_indices = pred_indices[:args.max_predictions]
            selection_text = f"conf>={args.score_threshold:.2f}"
            if args.nms_iou_threshold is not None:
                selection_text += f" | 2D NMS={args.nms_iou_threshold:.2f}"
            for pred_index in pred_indices.tolist():
                pred_box = pred_boxes[pred_index]
                draw_box(panel_2d, pred_box, PRED_COLOR, 1)
                draw_cuboid(
                    panel_3d,
                    pred_sizes[pred_index],
                    pred_transforms[pred_index],
                    intrinsic,
                    PRED_COLOR,
                    1,
                )
            label_panel(
                panel_2d,
                f"ALL 2D pred: bbox yellow | {selection_text} | n={len(pred_indices)}",
            )
            label_panel(
                panel_3d,
                f"ALL 3D pred: yellow | same query IDs | {selection_text} | n={len(pred_indices)}",
            )
        else:
            pred_indices = np.empty(0, dtype=np.int64)
            for box in gt_boxes:
                draw_box(panel_2d, box, GT_COLOR, 1)
            for rank, (pred_index, gt_index, iou) in enumerate(pairs):
                pred_box = pred_boxes[pred_index]
                draw_box(panel_2d, pred_box, PRED_COLOR, 2)
                draw_obb(panel_2d, pred_obbs[pred_index],
                         (pred_box[:2] + pred_box[2:]) * 0.5)
                draw_cuboid(panel_3d, gt_sizes[gt_index], gt_transforms[gt_index],
                             intrinsic, GT_COLOR, 1)
                draw_cuboid(panel_3d, pred_sizes[pred_index],
                             pred_transforms[pred_index], intrinsic, PRED_COLOR, 2)
                if rank < 5:
                    anchor = np.rint(pred_box[:2]).astype(np.int32)
                    cv2.putText(
                        panel_2d,
                        f"{float(pred_scores[pred_index]):.2f}/{iou:.2f}",
                        (int(anchor[0]), max(10, int(anchor[1]) - 2)),
                        cv2.FONT_HERSHEY_SIMPLEX,
                        0.32,
                        PRED_COLOR,
                        1,
                        cv2.LINE_AA,
                    )
                pair_records.append({
                    "prediction_index": pred_index,
                    "gt_index": gt_index,
                    "score": float(pred_scores[pred_index]),
                    "bbox_iou": iou,
                })
            label_panel(
                panel_2d,
                f"2D: GT green | pred yellow | OBB magenta | matches={len(pairs)}",
            )
            label_panel(
                panel_3d,
                f"3D: GT green | pred yellow | IoU2D>={args.iou_threshold:.2f}",
        )
        combined = np.concatenate([panel_2d, panel_3d], axis=1)
        suffix = (
            f"all_conf{args.score_threshold:.2f}"
            if args.all_predictions else "matched"
        )
        output = args.output_dir / (
            f"{ordinal:02d}_{Path(sample.img_path).stem}_{suffix}_result.png"
        )
        if not cv2.imwrite(str(output), combined):
            raise OSError(f"failed to write {output}")
        manifest["images"].append({
            "dataset_index": int(index),
            "source": str(Path(sample.img_path).resolve()),
            "output": str(output.resolve()),
            "gt_count": int(len(gt_boxes)),
            "accepted_match_count": accepted_pair_count,
            "drawn_prediction_count": int(len(pred_indices)) if args.all_predictions else len(pairs),
            "drawn_score_min": (
                float(pred_scores[pred_indices].min())
                if args.all_predictions and len(pred_indices) else None
            ),
            "drawn_score_max": (
                float(pred_scores[pred_indices].max())
                if args.all_predictions and len(pred_indices) else None
            ),
            "matches": pair_records,
        })
        print(f"overlay: {output}")

    manifest_path = args.output_dir / "manifest.json"
    manifest_path.write_text(json.dumps(manifest, indent=2) + "\n")
    print(f"manifest: {manifest_path}")


if __name__ == "__main__":
    main()
