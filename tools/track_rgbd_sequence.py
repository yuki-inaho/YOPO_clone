#!/usr/bin/env python3
"""Run label-free G10 detector, descriptor, and online tracker inference."""

from __future__ import annotations

import argparse
import time
from pathlib import Path
from typing import Any

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
from yopo.models.tracking.online_tracker import OnlineGeometryTracker, TrackerConfig
from yopo.models.tracking.sequence_training import atomic_write_json, sha256_file
from yopo.registry import MODELS
from yopo.structures import DetDataSample
from yopo.utils import register_all_modules, register_mmengine_checkpoint_safe_globals


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", type=Path, required=True)
    parser.add_argument("--manifest", type=Path, required=True)
    parser.add_argument("--detector-checkpoint", type=Path, required=True)
    parser.add_argument("--descriptor-checkpoint", type=Path)
    parser.add_argument("--allow-untrained-descriptor", action="store_true")
    parser.add_argument("--descriptor-mode", choices=("B0", "B2", "C0", "C1"))
    parser.add_argument("--split", choices=("train", "val", "smoke"), default="smoke")
    parser.add_argument("--window-index", type=int, default=0)
    parser.add_argument("--max-frames", type=int)
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--output", type=Path, required=True)
    return parser.parse_args()


def _data_sample(record: Any, image_size: tuple[int, int]) -> DetDataSample:
    return make_detector_data_sample(
        frame_id=record.frame_id,
        image_path=str(record.color_path),
        intrinsic=record.intrinsic,
        image_size=image_size,
    )


def main() -> None:
    args = parse_args()
    for path in (args.config, args.manifest, args.detector_checkpoint):
        if not path.expanduser().is_file():
            raise FileNotFoundError(f"required input not found: {path}")
    if args.max_frames is not None and args.max_frames <= 0:
        raise ValueError("--max-frames must be positive")
    if args.allow_untrained_descriptor and (
        args.max_frames is None or args.max_frames > 8
    ):
        raise ValueError(
            "untrained descriptor is limited to an explicit <=8-frame smoke"
        )
    device = torch.device(args.device)
    if device.type == "cuda" and not torch.cuda.is_available():
        raise RuntimeError("CUDA was requested but is unavailable")

    config_path = args.config.expanduser().resolve()
    manifest_path = args.manifest.expanduser().resolve()
    detector_path = args.detector_checkpoint.expanduser().resolve()
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
    detector_sha256 = sha256_file(detector_path)

    index = SequenceIndex(manifest_path, split=args.split)
    if not 0 <= args.window_index < len(index.windows):
        raise IndexError("--window-index is outside the selected split")
    window = index.windows[args.window_index]
    frame_ids = window.frame_ids[: args.max_frames]
    reader = RGBDFrameReader(index.frame_contract)
    image_size = (
        int(index.frame_contract["height"]),
        int(index.frame_contract["width"]),
    )
    tracker = OnlineGeometryTracker(TrackerConfig(**dict(inference.tracker)))
    descriptor = None
    descriptor_metadata = None
    frame_reports = []
    if device.type == "cuda":
        torch.cuda.empty_cache()
        torch.cuda.reset_peak_memory_stats(device)
    started = time.perf_counter()
    with torch.inference_mode():
        for frame_id in frame_ids:
            record = index.record(window.scene, frame_id)
            frame = reader.read(record)
            rgbd = compose_metric_rgbd_input(frame["image_bgr"], frame["depth_mm"])
            inputs = (
                torch.from_numpy(rgbd)
                .permute(2, 0, 1)
                .unsqueeze(0)
                .to(device=device, dtype=torch.float32)
            )
            frame_started = time.perf_counter()
            samples, backbone_features = detector.predict_with_backbone_features(
                inputs, [_data_sample(record, image_size)], rescale=True
            )
            predictions = samples[0].pred_instances
            appearance_dim = sum(int(level.shape[1]) for level in backbone_features)
            if descriptor is None:
                descriptor, descriptor_metadata = load_sequence_descriptor(
                    args.descriptor_checkpoint,
                    requested_mode=args.descriptor_mode,
                    allow_untrained=args.allow_untrained_descriptor,
                    appearance_dim=appearance_dim,
                    head_config=settings.head,
                    detector_sha256=detector_sha256,
                    seed=int(settings.seed),
                    device=device,
                )
            observations = build_detection_observations(
                predictions,
                backbone_features,
                descriptor,
                image_size=image_size,
                extrinsic_w2c=torch.from_numpy(frame["extrinsic_w2c"]).to(device),
                score_threshold=float(inference.score_threshold),
                max_detections=int(inference.max_detections),
            )
            update = tracker.update(
                [observation.detection for observation in observations],
                frame_index=record.frame_id,
            )
            if device.type == "cuda":
                torch.cuda.synchronize(device)
            frame_reports.append(
                {
                    "frame_id": record.frame_id,
                    "detector_prediction_count": len(predictions),
                    "retained_detection_count": len(observations),
                    "geometry_valid_count": sum(
                        observation.detection.center_world is not None
                        for observation in observations
                    ),
                    "detection_track_ids": list(update.detection_track_ids),
                    "created_track_ids": list(update.created_track_ids),
                    "matched_track_detection_pairs": [
                        list(pair) for pair in update.matched_track_detection_pairs
                    ],
                    "latency_ms": (time.perf_counter() - frame_started) * 1000.0,
                }
            )
    elapsed = time.perf_counter() - started
    report = {
        "status": "completed",
        "run_kind": (
            "plumbing_smoke_untrained"
            if descriptor_metadata["plumbing_only"]
            else "causal_tracking_inference"
        ),
        "input_contract": "current_rgbd_and_manifest_camera_only_no_label_v1",
        "causal": True,
        "future_frame_access": False,
        "label_artifacts_opened": False,
        "split": args.split,
        "sequence_id": window.sequence_id,
        "source_row": window.source_row,
        "frame_ids": list(frame_ids),
        "detector": {
            "checkpoint_kind": "raw",
            "checkpoint_sha256": detector_sha256,
            "precision": "float32",
        },
        "descriptor": descriptor_metadata,
        "frames": frame_reports,
        "latency_ms_per_frame": elapsed * 1000.0 / len(frame_ids),
        "cuda_peak_vram_bytes": (
            int(torch.cuda.max_memory_allocated(device)) if device.type == "cuda" else 0
        ),
        "provenance": {
            "config_sha256": sha256_file(config_path),
            "manifest_sha256": sha256_file(manifest_path),
        },
    }
    atomic_write_json(args.output, report)
    print(args.output)


if __name__ == "__main__":
    main()
