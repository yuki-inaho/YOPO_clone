#!/usr/bin/env python3
"""Export YOPO GauCho-3D predictions into a portable Open3D viewer bundle.

Run this script with the YOPO repository's Python environment, not the viewer's
lightweight Open3D environment.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import shutil
from pathlib import Path
from typing import Any

import cv2
import numpy as np
import torch
from mmcv.ops import nms_rotated
from mmengine.config import Config
from mmengine.dataset import pseudo_collate
from mmengine.runner import Runner

from yopo.utils import (register_all_modules,
                        register_mmengine_checkpoint_safe_globals)


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _intrinsic_matrix(value: Any) -> np.ndarray:
    array = np.asarray(value, dtype=np.float64)
    if array.shape == (4,):
        fx, fy, cx, cy = array.tolist()
        return np.array([[fx, 0.0, cx], [0.0, fy, cy], [0.0, 0.0, 1.0]])
    return array.reshape(3, 3)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("config", type=Path)
    parser.add_argument("checkpoint", type=Path)
    parser.add_argument("bundle_dir", type=Path)
    parser.add_argument("--num-frames", type=int, default=10)
    parser.add_argument("--nms-iou-threshold", type=float, default=0.2)
    parser.add_argument("--max-predictions", type=int, default=128)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    if args.num_frames < 1 or args.max_predictions < 1:
        raise ValueError("num-frames and max-predictions must be positive")
    if not 0.0 < args.nms_iou_threshold <= 1.0:
        raise ValueError("nms-iou-threshold must be in (0, 1]")
    if not args.config.is_file() or not args.checkpoint.is_file():
        raise FileNotFoundError("config and checkpoint must both exist")

    frames_dir = args.bundle_dir / "frames"
    frames_dir.mkdir(parents=True, exist_ok=True)
    register_all_modules()
    register_mmengine_checkpoint_safe_globals()
    cfg = Config.fromfile(str(args.config))
    cfg.load_from = None
    cfg.resume = False
    cfg.work_dir = str(args.bundle_dir / "export_runner")
    cfg.default_hooks.pop("checkpoint", None)
    cfg.custom_hooks = [
        hook for hook in cfg.get("custom_hooks", [])
        if hook.get("type") not in {
            "EarlyStoppingHook", "ScheduleFreeOptimizerModeHook"}
    ]
    runner = Runner.from_cfg(cfg)
    runner.load_checkpoint(str(args.checkpoint))
    model = runner.model.eval()
    dataset = runner.val_dataloader.dataset
    indices = np.linspace(
        0, len(dataset) - 1, min(args.num_frames, len(dataset)),
        dtype=np.int64).tolist()

    manifest: dict[str, Any] = {
        "format": "gaucho3d-open3d-bundle-v1",
        "description": "RGB-D validation frames and GauCho-3D predictions",
        "checkpoint": str(args.checkpoint.resolve()),
        "checkpoint_sha256": _sha256(args.checkpoint),
        "config": str(args.config.resolve()),
        "dataset_length": len(dataset),
        "selected_indices": indices,
        "depth_units": "uint16 millimetres",
        "depth_scale": 1000.0,
        "coordinate_system": "OpenCV camera: +x right, +y down, +z forward",
        "ellipsoid_state": "center t [m], covariance Sigma [m^2]",
        "selection": {
            "projected_valid": True,
            "finite": True,
            "rotated_nms_iou_threshold": args.nms_iou_threshold,
            "max_predictions": args.max_predictions,
            "score_threshold": None,
        },
        "frames": [],
    }

    with torch.no_grad():
        for ordinal, dataset_index in enumerate(indices):
            result = model.test_step(pseudo_collate([dataset[dataset_index]]))[0]
            pred = result.pred_instances
            required = (
                "scores", "labels", "ellipsoid_centers", "ellipsoid_shapes",
                "projected_ellipses", "projected_valid")
            missing = [name for name in required if not hasattr(pred, name)]
            if missing:
                raise AttributeError(f"missing prediction fields: {missing}")

            scores = pred.scores.detach().cpu().float()
            labels = pred.labels.detach().cpu().long()
            centers = pred.ellipsoid_centers.detach().cpu().float()
            sigmas = pred.ellipsoid_shapes.detach().cpu().float()
            projected = pred.projected_ellipses.detach().cpu().float()
            valid = pred.projected_valid.detach().cpu().bool()
            finite = (
                torch.isfinite(centers).all(-1)
                & torch.isfinite(sigmas).all(-1).all(-1)
                & torch.isfinite(projected).all(-1))
            positive = (projected[:, 0] > 0) & (projected[:, 1] > 0)
            selection = valid & finite & positive
            scores = scores[selection]
            labels = labels[selection]
            centers = centers[selection]
            sigmas = sigmas[selection]
            projected = projected[selection]
            envelopes = torch.stack(
                (projected[:, 2], projected[:, 3], 2.0 * projected[:, 0],
                 2.0 * projected[:, 1], projected[:, 4]), dim=-1)
            if len(envelopes):
                _, keep = nms_rotated(
                    envelopes, scores, args.nms_iou_threshold, labels=labels)
                keep = keep[:args.max_predictions]
                scores = scores[keep]
                labels = labels[keep]
                centers = centers[keep]
                sigmas = sigmas[keep]
                projected = projected[keep]

            color_source = Path(result.img_path).resolve()
            depth_source = Path(
                str(color_source).replace("_color.png", "_depth.png"))
            if not color_source.is_file() or not depth_source.is_file():
                raise FileNotFoundError(
                    f"RGB-D pair missing: {color_source}, {depth_source}")
            color = cv2.imread(str(color_source), cv2.IMREAD_COLOR)
            depth = cv2.imread(str(depth_source), cv2.IMREAD_UNCHANGED)
            if color is None or depth is None or depth.dtype != np.uint16:
                raise ValueError(f"invalid RGB-D pair for {color_source}")
            if color.shape[:2] != depth.shape:
                raise ValueError(f"RGB-D dimensions differ for {color_source}")

            prefix = f"{ordinal:02d}_val{dataset_index:03d}"
            color_name = f"{prefix}_color.png"
            depth_name = f"{prefix}_depth.png"
            prediction_name = f"{prefix}_predictions.npz"
            shutil.copy2(color_source, frames_dir / color_name)
            shutil.copy2(depth_source, frames_dir / depth_name)
            np.savez_compressed(
                frames_dir / prediction_name,
                intrinsic=_intrinsic_matrix(result.intrinsic),
                scores=scores.numpy(),
                labels=labels.numpy(),
                centers=centers.numpy(),
                sigmas=sigmas.numpy(),
                projected_ellipses=projected.numpy(),
                image_size=np.asarray([color.shape[1], color.shape[0]], np.int32),
            )
            frame_id = f"{color_source.parent.name}/{color_source.stem.removesuffix('_color')}"
            manifest["frames"].append({
                "dataset_index": dataset_index,
                "frame_id": frame_id,
                "color": f"frames/{color_name}",
                "depth": f"frames/{depth_name}",
                "predictions": f"frames/{prediction_name}",
                "prediction_count": int(len(scores)),
                "source_color": str(color_source),
                "source_depth": str(depth_source),
            })
            print(f"exported val={dataset_index:03d} predictions={len(scores):3d} "
                  f"{frame_id}")

    manifest_path = args.bundle_dir / "manifest.json"
    manifest_path.write_text(json.dumps(manifest, indent=2) + "\n")
    print(f"bundle: {manifest_path.resolve()}")


if __name__ == "__main__":
    main()
