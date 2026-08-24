#!/usr/bin/env python3
"""Project top-1 RGB-D 3D BBOX predictions onto their original RGB images."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import cv2
import numpy as np
from mmengine.fileio import load


EDGES = ((0, 1), (1, 2), (2, 3), (3, 0), (4, 5), (5, 6), (6, 7), (7, 4),
         (0, 4), (1, 5), (2, 6), (3, 7))


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("predictions", help="val-loop prediction .pkl dump")
    parser.add_argument("output_dir", help="directory for PNG overlays and manifest")
    parser.add_argument("--config", required=True, help="source RGB-D model config")
    parser.add_argument("--checkpoint", required=True, help="checkpoint used for dump")
    parser.add_argument("--num-images", type=int, default=3)
    return parser.parse_args()


def intrinsic_matrix(intrinsic: object) -> np.ndarray:
    values = np.asarray(intrinsic, dtype=np.float64)
    if values.shape == (4, ):
        fx, fy, cx, cy = values
        return np.array([[fx, 0.0, cx], [0.0, fy, cy], [0.0, 0.0, 1.0]])
    if values.shape == (3, 3):
        return values
    raise ValueError(f"intrinsic must be [fx, fy, cx, cy] or 3x3, got {values.shape}")


def cuboid_corners(size: np.ndarray, transform: np.ndarray) -> np.ndarray:
    if size.shape != (3, ) or transform.shape != (4, 4):
        raise ValueError(f"expected size (3,) and T (4,4), got {size.shape}, {transform.shape}")
    if not np.isfinite(size).all() or not np.isfinite(transform).all():
        raise FloatingPointError("non-finite predicted size or object-to-camera transform")
    half = size / 2.0
    corners = np.array(
        [[-half[0], -half[1], -half[2]], [half[0], -half[1], -half[2]],
         [half[0], half[1], -half[2]], [-half[0], half[1], -half[2]],
         [-half[0], -half[1], half[2]], [half[0], -half[1], half[2]],
         [half[0], half[1], half[2]], [-half[0], half[1], half[2]]],
        dtype=np.float64)
    return corners @ transform[:3, :3].T + transform[:3, 3]


def project_corners(corners_camera: np.ndarray, intrinsic: np.ndarray) -> np.ndarray:
    if not np.isfinite(corners_camera).all():
        raise FloatingPointError("non-finite camera-space cuboid corner")
    depths = corners_camera[:, 2]
    if np.any(depths <= 1e-6):
        raise ValueError(f"cuboid has non-positive camera depth: {depths.tolist()}")
    homogeneous = (intrinsic @ corners_camera.T).T
    projected = homogeneous[:, :2] / homogeneous[:, 2:3]
    if not np.isfinite(projected).all():
        raise FloatingPointError("non-finite projected cuboid corner")
    return projected


def render(image: np.ndarray, projected: np.ndarray, score: float) -> np.ndarray:
    canvas = image.copy()
    points = np.rint(projected).astype(np.int32)
    for start, end in EDGES:
        cv2.line(canvas, tuple(points[start]), tuple(points[end]), (0, 255, 255), 2,
                 lineType=cv2.LINE_AA)
    text = f"fruit top-1 score={score:.4f}"
    text_size, baseline = cv2.getTextSize(text, cv2.FONT_HERSHEY_SIMPLEX, 0.45, 1)
    raw_anchor = points[np.argmin(points[:, 1])]
    anchor = (int(np.clip(raw_anchor[0], 0, image.shape[1] - text_size[0] - 1)),
              int(np.clip(raw_anchor[1], text_size[1] + baseline, image.shape[0] - 1)))
    cv2.putText(canvas, text, anchor, cv2.FONT_HERSHEY_SIMPLEX, 0.45,
                (0, 0, 0), 3, cv2.LINE_AA)
    cv2.putText(canvas, text, anchor, cv2.FONT_HERSHEY_SIMPLEX, 0.45,
                (0, 255, 255), 1, cv2.LINE_AA)
    return canvas


def candidate(record: dict, index: int) -> dict:
    pred = record["pred_instances"]
    scores = np.asarray(pred["scores"], dtype=np.float64)
    if scores.ndim != 1 or len(scores) == 0 or not np.isfinite(scores).all():
        raise FloatingPointError(f"invalid scores for validation record {index}")
    top_index = int(np.argmax(scores))
    transform = np.asarray(pred["T"], dtype=np.float64)[top_index]
    size = np.asarray(pred["sizes"], dtype=np.float64)[top_index]
    intrinsic = intrinsic_matrix(record["intrinsic"])
    corners_camera = cuboid_corners(size, transform)
    projected = project_corners(corners_camera, intrinsic)
    image_path = Path(record["img_path"])
    image = cv2.imread(str(image_path), cv2.IMREAD_COLOR)
    if image is None:
        raise FileNotFoundError(f"cannot decode RGB image: {image_path}")
    height, width = image.shape[:2]
    visible_corners = int(
        ((projected[:, 0] >= 0) & (projected[:, 0] < width)
         & (projected[:, 1] >= 0) & (projected[:, 1] < height)).sum())
    return {
        "record_index": index,
        "top_index": top_index,
        "score": float(scores[top_index]),
        "image_path": image_path,
        "image": image,
        "intrinsic": intrinsic,
        "transform": transform,
        "size": size,
        "corners_camera": corners_camera,
        "projected": projected,
        "visible_corners": visible_corners,
    }


def main() -> None:
    args = parse_args()
    if args.num_images < 1:
        raise ValueError("--num-images must be positive")
    predictions_path = Path(args.predictions)
    checkpoint_path = Path(args.checkpoint)
    if not predictions_path.is_file() or not checkpoint_path.is_file():
        raise FileNotFoundError("prediction dump and checkpoint must both exist")

    records = load(str(predictions_path))
    candidates = [candidate(record, index) for index, record in enumerate(records)]
    if len(candidates) < args.num_images:
        raise ValueError(f"need {args.num_images} predictions, got {len(candidates)}")
    # Selection has no confidence threshold: rank every image's top-1 query by
    # score, preferring a cuboid with at least one projected corner on-screen.
    candidates.sort(key=lambda item: (item["visible_corners"] > 0, item["score"]), reverse=True)
    selected = candidates[:args.num_images]

    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    manifest_images = []
    for ordinal, item in enumerate(selected, start=1):
        output_path = output_dir / f"{ordinal:02d}_{item['image_path'].stem}_3dbbox_overlay.png"
        if not cv2.imwrite(str(output_path), render(item["image"], item["projected"], item["score"])):
            raise OSError(f"failed to write overlay: {output_path}")
        manifest_images.append({
            "input_image": str(item["image_path"].resolve()),
            "overlay_image": str(output_path.resolve()),
            "selection": "top-1 query for this image; ranked globally without confidence threshold",
            "record_index": item["record_index"],
            "query_index": item["top_index"],
            "score": item["score"],
            "intrinsic_3x3": item["intrinsic"].tolist(),
            "object_to_camera_T_4x4": item["transform"].tolist(),
            "size_m": item["size"].tolist(),
            "corners_camera_m": item["corners_camera"].tolist(),
            "projected_corners_xy_px": item["projected"].tolist(),
            "visible_projected_corners": item["visible_corners"],
            "image_size_wh": [int(item["image"].shape[1]), int(item["image"].shape[0])],
        })
        print(f"overlay: {output_path}")

    manifest_path = output_dir / "manifest.json"
    manifest_path.write_text(json.dumps({
        "config": str(Path(args.config).resolve()),
        "checkpoint": str(checkpoint_path.resolve()),
        "prediction_dump": str(predictions_path.resolve()),
        "camera_convention": "T maps object-frame metres to camera-frame metres; project with K and z>0",
        "selection_rule": "top-1 query per image, no confidence threshold",
        "images": manifest_images,
    }, indent=2) + "\n")
    print(f"manifest: {manifest_path}")


if __name__ == "__main__":
    main()
