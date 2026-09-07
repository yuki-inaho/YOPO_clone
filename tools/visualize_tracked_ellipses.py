#!/usr/bin/env python3
"""Render projected 3D ellipses and causal track IDs for two RGB-D clips."""

from __future__ import annotations

import argparse
import time
from pathlib import Path
from typing import Any

import cv2
import numpy as np
import torch
from mmengine.config import Config
from mmengine.runner.checkpoint import load_checkpoint

from yopo.datasets.pose_estimation.yopo_sequence import RGBDFrameReader, SequenceIndex
from yopo.datasets.transforms.raw_depth import compose_metric_rgbd_input
from yopo.models.tracking.causal_inference import (
    build_detection_observations,
    load_sequence_descriptor,
    make_detector_data_sample,
)
from yopo.models.tracking.ellipse_overlay import draw_tracked_ellipse
from yopo.models.tracking.online_tracker import OnlineGeometryTracker, TrackerConfig
from yopo.models.tracking.sequence_training import atomic_write_json, sha256_file
from yopo.registry import MODELS
from yopo.utils import register_all_modules, register_mmengine_checkpoint_safe_globals


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", type=Path, required=True)
    parser.add_argument("--manifest", type=Path, required=True)
    parser.add_argument("--detector-checkpoint", type=Path, required=True)
    parser.add_argument("--descriptor-checkpoint", type=Path, required=True)
    parser.add_argument(
        "--descriptor-mode", choices=("B0", "B2", "C0", "C1"), default="C1"
    )
    parser.add_argument("--split", choices=("train", "val", "smoke"), default="smoke")
    parser.add_argument("--window-indices", type=int, nargs=2, default=(0, 34))
    parser.add_argument("--max-frames", type=int, default=4)
    parser.add_argument("--fps", type=float, default=2.0)
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--output-dir", type=Path, required=True)
    return parser.parse_args()


def prepare_output_directory(path: Path) -> Path:
    destination = path.expanduser().resolve()
    if destination.is_file() or (destination.is_dir() and any(destination.iterdir())):
        raise FileExistsError(f"output directory is not empty: {destination}")
    destination.mkdir(parents=True, exist_ok=True)
    return destination


def _prediction_ellipses(predictions: object) -> tuple[torch.Tensor, torch.Tensor]:
    ellipses = getattr(predictions, "projected_ellipses", None)
    valid = getattr(predictions, "projected_valid", None)
    if (
        not isinstance(ellipses, torch.Tensor)
        or ellipses.ndim != 2
        or ellipses.shape[1] != 5
    ):
        raise ValueError("detector requires projected_ellipses tensor with shape [N,5]")
    if not isinstance(valid, torch.Tensor) or valid.shape != (len(ellipses),):
        raise ValueError("detector requires projected_valid tensor with shape [N]")
    return ellipses, valid.bool()


def _frame_sample(record: Any, image_size: tuple[int, int]):
    return make_detector_data_sample(
        frame_id=record.frame_id,
        image_path=str(record.color_path),
        intrinsic=record.intrinsic,
        image_size=image_size,
    )


def _banner(image: np.ndarray, text: str) -> None:
    overlay = image.copy()
    cv2.rectangle(overlay, (0, 0), (image.shape[1], 30), (0, 0, 0), -1)
    cv2.addWeighted(overlay, 0.65, image, 0.35, 0.0, dst=image)
    cv2.putText(
        image,
        text,
        (8, 20),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.50,
        (255, 255, 255),
        1,
        cv2.LINE_AA,
    )


def _contact_sheet(
    clips: list[list[np.ndarray]], *, tile_size=(320, 240)
) -> np.ndarray:
    columns = max(len(clip) for clip in clips)
    rows = []
    for clip in clips:
        tiles = [
            cv2.resize(frame, tile_size, interpolation=cv2.INTER_AREA) for frame in clip
        ]
        tiles.extend(np.zeros_like(tiles[0]) for _ in range(columns - len(tiles)))
        rows.append(np.concatenate(tiles, axis=1))
    return np.concatenate(rows, axis=0)


def main() -> None:
    args = parse_args()
    if args.max_frames <= 0 or args.fps <= 0:
        raise ValueError("--max-frames and --fps must be positive")
    if len(set(args.window_indices)) != 2:
        raise ValueError("--window-indices must select two distinct clips")
    paths = (
        args.config,
        args.manifest,
        args.detector_checkpoint,
        args.descriptor_checkpoint,
    )
    for path in paths:
        if not path.expanduser().is_file():
            raise FileNotFoundError(f"required input not found: {path}")
    output_dir = prepare_output_directory(args.output_dir)
    device = torch.device(args.device)
    if device.type == "cuda" and not torch.cuda.is_available():
        raise RuntimeError("CUDA was requested but is unavailable")

    config_path, manifest_path, detector_path, descriptor_path = (
        path.expanduser().resolve() for path in paths
    )
    config = Config.fromfile(config_path)
    settings = config.sequence_context
    inference = settings.inference
    if inference.detector_precision != "float32":
        raise ValueError("G10 GauCho postprocessing currently requires float32")
    register_all_modules(init_default_scope=True)
    register_mmengine_checkpoint_safe_globals()
    detector = MODELS.build(config.model)
    load_checkpoint(detector, str(detector_path), map_location="cpu", strict=True)
    detector.to(device).eval()
    detector_sha = sha256_file(detector_path)

    index = SequenceIndex(manifest_path, split=args.split)
    if any(not 0 <= value < len(index.windows) for value in args.window_indices):
        raise IndexError("a window index is outside the selected split")
    windows = [index.windows[value] for value in args.window_indices]
    if set(windows[0].frame_ids) & set(windows[1].frame_ids):
        raise ValueError("selected clips must not share frames")
    image_size = (
        int(index.frame_contract["height"]),
        int(index.frame_contract["width"]),
    )
    reader = RGBDFrameReader(index.frame_contract)
    descriptor = None
    descriptor_metadata = None
    clip_reports = []
    rendered_clips: list[list[np.ndarray]] = []
    artifact_paths: list[Path] = []
    if device.type == "cuda":
        torch.cuda.empty_cache()
        torch.cuda.reset_peak_memory_stats(device)

    with torch.inference_mode():
        for clip_index, (window_index, window) in enumerate(
            zip(args.window_indices, windows)
        ):
            tracker = OnlineGeometryTracker(TrackerConfig(**dict(inference.tracker)))
            clip_dir = output_dir / f"clip_{clip_index:02d}"
            clip_dir.mkdir()
            video_path = output_dir / f"clip_{clip_index:02d}.mp4"
            writer = cv2.VideoWriter(
                str(video_path),
                cv2.VideoWriter_fourcc(*"mp4v"),
                args.fps,
                (image_size[1], image_size[0]),
            )
            if not writer.isOpened():
                raise RuntimeError(f"cannot open video writer: {video_path}")
            frame_reports = []
            rendered_frames = []
            try:
                for ordinal, frame_id in enumerate(window.frame_ids[: args.max_frames]):
                    record = index.record(window.scene, frame_id)
                    frame = reader.read(record)
                    rgbd = compose_metric_rgbd_input(
                        frame["image_bgr"], frame["depth_mm"]
                    )
                    inputs = (
                        torch.from_numpy(rgbd)
                        .permute(2, 0, 1)
                        .unsqueeze(0)
                        .to(device=device, dtype=torch.float32)
                    )
                    started = time.perf_counter()
                    samples, backbone_features = (
                        detector.predict_with_backbone_features(
                            inputs, [_frame_sample(record, image_size)], rescale=True
                        )
                    )
                    predictions = samples[0].pred_instances
                    projected, projected_valid = _prediction_ellipses(predictions)
                    appearance_dim = sum(
                        int(level.shape[1]) for level in backbone_features
                    )
                    if descriptor is None:
                        descriptor, descriptor_metadata = load_sequence_descriptor(
                            descriptor_path,
                            requested_mode=args.descriptor_mode,
                            allow_untrained=False,
                            appearance_dim=appearance_dim,
                            head_config=settings.head,
                            detector_sha256=detector_sha,
                            seed=int(settings.seed),
                            device=device,
                        )
                    observations = build_detection_observations(
                        predictions,
                        backbone_features,
                        descriptor,
                        image_size=image_size,
                        extrinsic_w2c=torch.from_numpy(frame["extrinsic_w2c"]).to(
                            device
                        ),
                        score_threshold=float(inference.score_threshold),
                        max_detections=int(inference.max_detections),
                    )
                    retained_projected_valid_count = sum(
                        bool(projected_valid[item.prediction_index])
                        for item in observations
                    )
                    update = tracker.update(
                        [item.detection for item in observations],
                        frame_index=record.frame_id,
                    )
                    if len(update.detection_track_ids) != len(observations):
                        raise RuntimeError("tracker IDs do not align with observations")
                    rendered = frame["image_bgr"].copy()
                    drawn_ids = []
                    invalid_indices = []
                    for observation, track_id in zip(
                        observations, update.detection_track_ids
                    ):
                        prediction_index = observation.prediction_index
                        if prediction_index >= len(projected):
                            raise RuntimeError(
                                "observation prediction index is out of range"
                            )
                        drawn = draw_tracked_ellipse(
                            rendered,
                            projected[prediction_index].detach().cpu().numpy(),
                            track_id=track_id,
                            score=observation.detection.confidence,
                            projected_valid=bool(projected_valid[prediction_index]),
                        )
                        (drawn_ids if drawn else invalid_indices).append(
                            track_id if drawn else prediction_index
                        )
                    _banner(
                        rendered,
                        f"clip {clip_index} | frame {record.frame_id} | "
                        f"tracks {len(observations)} | ellipses {len(drawn_ids)}",
                    )
                    if device.type == "cuda":
                        torch.cuda.synchronize(device)
                    output_path = (
                        clip_dir / f"{ordinal:02d}_frame_{record.frame_id:06d}.png"
                    )
                    if not cv2.imwrite(str(output_path), rendered):
                        raise OSError(f"failed to write image: {output_path}")
                    writer.write(rendered)
                    artifact_paths.append(output_path)
                    rendered_frames.append(rendered)
                    frame_reports.append(
                        {
                            "frame_id": record.frame_id,
                            "prediction_count": len(predictions),
                            "retained_detection_count": len(observations),
                            "projected_valid_count": int(projected_valid.sum().item()),
                            "retained_projected_valid_count": (
                                retained_projected_valid_count
                            ),
                            "drawn_ellipse_count": len(drawn_ids),
                            "invalid_retained_prediction_indices": invalid_indices,
                            "detection_track_ids": list(update.detection_track_ids),
                            "created_track_ids": list(update.created_track_ids),
                            "matched_track_detection_pairs": [
                                list(pair)
                                for pair in update.matched_track_detection_pairs
                            ],
                            "latency_ms": (time.perf_counter() - started) * 1000.0,
                            "image": str(output_path),
                        }
                    )
            finally:
                writer.release()
            if not rendered_frames:
                raise RuntimeError("selected clip produced no frame")
            artifact_paths.append(video_path)
            rendered_clips.append(rendered_frames)
            clip_reports.append(
                {
                    "clip_index": clip_index,
                    "window_index": window_index,
                    "sequence_id": window.sequence_id,
                    "physical_scene": window.scene,
                    "source_row": window.source_row,
                    "tracker_reset": True,
                    "track_id_namespace": f"clip_{clip_index:02d}",
                    "video": str(video_path),
                    "frames": frame_reports,
                }
            )

    contact_sheet = _contact_sheet(rendered_clips)
    contact_path = output_dir / "two_clip_contact_sheet.png"
    if not cv2.imwrite(str(contact_path), contact_sheet):
        raise OSError(f"failed to write contact sheet: {contact_path}")
    artifact_paths.append(contact_path)
    report = {
        "status": "completed",
        "schema": "yopo_tracked_projected_ellipse_visualization_v1",
        "interpretation": "two_sequential_clips_from_one_physical_scene",
        "physical_scene_count": len({window.scene for window in windows}),
        "rendered_clip_count": len(windows),
        "tracker_reset_per_clip": True,
        "input_contract": "current_rgbd_and_manifest_camera_only_no_label_v1",
        "causal": True,
        "future_frame_access": False,
        "label_artifacts_opened": False,
        "ellipse_contract": "projected_ellipses_semiaxes_first_radians_v1",
        "split": args.split,
        "descriptor": descriptor_metadata,
        "detector": {
            "checkpoint_kind": "raw",
            "checkpoint_sha256": detector_sha,
            "precision": "float32",
        },
        "clips": clip_reports,
        "contact_sheet": str(contact_path),
        "cuda_peak_vram_bytes": (
            int(torch.cuda.max_memory_allocated(device)) if device.type == "cuda" else 0
        ),
        "artifacts": [
            {
                "path": str(path.resolve()),
                "sha256": sha256_file(path),
                "bytes": path.stat().st_size,
            }
            for path in artifact_paths
        ],
        "provenance": {
            "config_sha256": sha256_file(config_path),
            "manifest_sha256": sha256_file(manifest_path),
        },
    }
    report_path = output_dir / "visualization_report.json"
    atomic_write_json(report_path, report)
    print(report_path)


if __name__ == "__main__":
    main()
