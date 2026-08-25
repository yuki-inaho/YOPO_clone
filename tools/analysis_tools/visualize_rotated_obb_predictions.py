#!/usr/bin/env python3
"""Overlay 2D rotated-object predictions on RGB images.

The YOPO rotated DETR head returns boxes as ``(cx, cy, w, h, angle_rad)`` in
the original image coordinate system.  This script draws those exact rotated
corners on RGB validation images and writes a compact JSON manifest alongside
the PNGs so the visual evidence remains tied to its checkpoint.
"""

from __future__ import annotations

import argparse
import json
import math
from pathlib import Path

import cv2
import numpy as np

from yopo.apis import inference_detector, init_detector
from yopo.utils import register_mmengine_checkpoint_safe_globals


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('config', help='YOPO model config')
    parser.add_argument('checkpoint', help='trained checkpoint')
    parser.add_argument('image_dir', help='directory containing RGB images')
    parser.add_argument('output_dir', help='directory for overlay PNG files')
    parser.add_argument('--num-images', type=int, default=3)
    parser.add_argument('--score-thr', type=float, default=0.05)
    parser.add_argument('--max-dets', type=int, default=15)
    parser.add_argument('--device', default='cuda:0')
    return parser.parse_args()


def rotated_corners(box: np.ndarray) -> np.ndarray:
    """Return image-space corners for an ``xywha`` box with radians angle."""
    cx, cy, width, height, angle = (float(value) for value in box)
    local = np.array(
        [[-width / 2, -height / 2], [width / 2, -height / 2],
         [width / 2, height / 2], [-width / 2, height / 2]],
        dtype=np.float32)
    cos_angle, sin_angle = math.cos(angle), math.sin(angle)
    rotation = np.array(
        [[cos_angle, -sin_angle], [sin_angle, cos_angle]], dtype=np.float32)
    return local @ rotation.T + np.array([cx, cy], dtype=np.float32)


def draw_predictions(image: np.ndarray, boxes: np.ndarray,
                     scores: np.ndarray) -> np.ndarray:
    canvas = image.copy()
    for index, (box, score) in enumerate(zip(boxes, scores), start=1):
        corners = np.rint(rotated_corners(box)).astype(np.int32)
        cv2.polylines(canvas, [corners], True, (0, 255, 64), 2,
                      lineType=cv2.LINE_AA)
        anchor = tuple(corners[np.argmin(corners[:, 1])])
        cv2.putText(canvas, f'tomato {score:.2f}', anchor,
                    cv2.FONT_HERSHEY_SIMPLEX, 0.46, (0, 20, 0), 3,
                    cv2.LINE_AA)
        cv2.putText(canvas, f'tomato {score:.2f}', anchor,
                    cv2.FONT_HERSHEY_SIMPLEX, 0.46, (0, 255, 64), 1,
                    cv2.LINE_AA)
    return canvas


def main() -> None:
    args = parse_args()
    if args.num_images < 1 or args.max_dets < 1:
        raise ValueError('--num-images and --max-dets must both be positive')
    register_mmengine_checkpoint_safe_globals()

    image_dir = Path(args.image_dir)
    output_dir = Path(args.output_dir)
    image_paths = sorted(
        path for path in image_dir.iterdir()
        if path.suffix.lower() in {'.jpg', '.jpeg', '.png', '.bmp'})
    if len(image_paths) < args.num_images:
        raise FileNotFoundError(
            f'need {args.num_images} images, found {len(image_paths)} in '
            f'{image_dir}')

    output_dir.mkdir(parents=True, exist_ok=True)
    model = init_detector(args.config, args.checkpoint, device=args.device)
    manifest: list[dict] = []
    for image_path in image_paths[:args.num_images]:
        result = inference_detector(model, str(image_path))
        instances = result.pred_instances
        boxes = instances.bboxes.detach().cpu().numpy()
        scores = instances.scores.detach().cpu().numpy()
        order = np.argsort(-scores)
        keep = order[scores[order] >= args.score_thr][:args.max_dets]
        boxes, scores = boxes[keep], scores[keep]

        image = cv2.imread(str(image_path), cv2.IMREAD_COLOR)
        if image is None:
            raise FileNotFoundError(f'cannot decode {image_path}')
        rendered = draw_predictions(image, boxes, scores)
        output_path = output_dir / f'{image_path.stem}_obb_overlay.png'
        if not cv2.imwrite(str(output_path), rendered):
            raise OSError(f'failed to write {output_path}')
        manifest.append({
            'input_image': str(image_path),
            'overlay_image': str(output_path),
            'num_drawn': int(len(scores)),
            'score_threshold': args.score_thr,
            'scores': [round(float(score), 6) for score in scores],
            'boxes_xywha_rad': [
                [round(float(value), 4) for value in box] for box in boxes
            ],
        })
        print(f'{output_path}: {len(scores)} predictions')

    manifest_path = output_dir / 'manifest.json'
    manifest_path.write_text(
        json.dumps({
            'config': str(Path(args.config).resolve()),
            'checkpoint': str(Path(args.checkpoint).resolve()),
            'image_dir': str(image_dir.resolve()),
            'score_threshold': args.score_thr,
            'max_dets_per_image': args.max_dets,
            'images': manifest,
        }, indent=2) + '\n')
    print(f'manifest: {manifest_path}')


if __name__ == '__main__':
    main()
