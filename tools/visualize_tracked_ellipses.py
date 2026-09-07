#!/usr/bin/env python3
"""Render projected 3D ellipses and causal track IDs for two RGB-D clips."""

from __future__ import annotations

import argparse
import json
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Mapping

import cv2
import numpy as np
import torch
from mmengine.config import Config
from mmengine.runner.checkpoint import load_checkpoint

from yopo.datasets.pose_estimation.yopo_sequence import (
    FrameRecord,
    RGBDFrameReader,
    SequenceIndex,
)
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


@dataclass(frozen=True)
class RawSourceSelection:
    """Validated annotation-free frame range from a COLMAP RGB-D source."""

    frame_contract: dict[str, Any]
    records: tuple[FrameRecord, ...]
    metadata: dict[str, Any]


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
    parser.add_argument("--window-indices", type=int, nargs=2)
    parser.add_argument("--max-frames", type=int, default=4)
    parser.add_argument("--raw-source-root", type=Path)
    parser.add_argument("--scene", default="scene_000000")
    parser.add_argument("--start-frame", type=int, default=0)
    parser.add_argument("--frame-count", type=int, default=300)
    parser.add_argument("--contact-samples", type=int, default=12)
    parser.add_argument("--progress-interval", type=int, default=25)
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


def _positive_int(value: Any, field: str) -> int:
    if not isinstance(value, int) or isinstance(value, bool) or value <= 0:
        raise ValueError(f"{field} must be a positive integer")
    return value


def _mapping(value: Any, field: str) -> Mapping[str, Any]:
    if not isinstance(value, Mapping):
        raise ValueError(f"raw dataset {field} must be an object")
    return value


def load_raw_source_records(
    root: Path, *, scene: str, start_frame: int, frame_count: int
) -> RawSourceSelection:
    """Load one exact contiguous raw RGB-D range without opening annotations."""

    source_root = root.expanduser().resolve()
    dataset_path = source_root / "dataset.json"
    scene_root = (source_root / "scenes" / scene).resolve()
    scenes_root = (source_root / "scenes").resolve()
    cameras_path = scene_root / "cameras.npz"
    if not dataset_path.is_file() or not cameras_path.is_file():
        raise FileNotFoundError(
            "raw source requires dataset.json and scene cameras.npz"
        )
    if not scene or not scene_root.is_relative_to(scenes_root):
        raise ValueError("raw source scene escapes the scenes directory")
    try:
        document = json.loads(dataset_path.read_text(encoding="utf-8"))
    except (OSError, UnicodeError, json.JSONDecodeError) as error:
        raise ValueError(f"cannot read raw dataset.json: {error}") from error
    if (
        not isinstance(document, Mapping)
        or document.get("format") != "colmap_rgbd_v1"
        or document.get("schema_version") != 1
    ):
        raise ValueError("unsupported raw RGB-D dataset contract")
    total_frames = _positive_int(document.get("frame_count"), "dataset frame_count")
    requested_count = _positive_int(frame_count, "frame_count")
    if (
        not isinstance(start_frame, int)
        or isinstance(start_frame, bool)
        or start_frame < 0
    ):
        raise ValueError("start_frame must be a non-negative integer")
    stop_frame = start_frame + requested_count
    if stop_frame > total_frames:
        raise ValueError(
            f"requested frame range [{start_frame},{stop_frame}) is outside "
            f"dataset frame_count={total_frames}"
        )

    image = _mapping(document.get("image"), "image")
    depth = _mapping(document.get("depth"), "depth")
    width = _positive_int(image.get("width"), "image width")
    height = _positive_int(image.get("height"), "image height")
    expected = {
        "image channels": (image.get("channels"), 3),
        "image dtype": (image.get("dtype"), "uint8"),
        "depth width": (depth.get("width"), width),
        "depth height": (depth.get("height"), height),
        "depth dtype": (depth.get("dtype"), "uint16"),
        "depth invalid_value": (depth.get("invalid_value"), 0),
    }
    for field, (actual, required) in expected.items():
        if actual != required:
            raise ValueError(f"raw dataset {field}={actual!r}, expected {required!r}")
    depth_unit = depth.get("unit")
    if depth_unit not in {"mm", "millimeters"}:
        raise ValueError(
            f"raw dataset depth unit={depth_unit!r}, expected 'mm' or 'millimeters'"
        )

    with np.load(cameras_path, allow_pickle=False) as cameras:
        required_keys = {
            "frame_ids",
            "intrinsics",
            "extrinsics_w2c",
            "quality_flags",
        }
        if not required_keys.issubset(cameras.files):
            raise ValueError("raw camera archive is missing required arrays")
        frame_ids = np.asarray(cameras["frame_ids"], dtype=np.int64)
        intrinsics = np.asarray(cameras["intrinsics"], dtype=np.float32)
        extrinsics = np.asarray(cameras["extrinsics_w2c"], dtype=np.float32)
        quality = np.asarray(cameras["quality_flags"], dtype=bool)
    if (
        frame_ids.shape != (total_frames,)
        or intrinsics.shape != (total_frames, 3, 3)
        or extrinsics.shape != (total_frames, 3, 4)
        or quality.shape != (total_frames,)
        or len(np.unique(frame_ids)) != total_frames
    ):
        raise ValueError("raw camera arrays do not align with dataset frame_count")
    row_by_id = {int(frame_id): row for row, frame_id in enumerate(frame_ids)}
    requested_ids = tuple(range(start_frame, stop_frame))
    if any(frame_id not in row_by_id for frame_id in requested_ids):
        raise ValueError("raw camera frame IDs do not cover the requested range")

    records = []
    for frame_id in requested_ids:
        row = row_by_id[frame_id]
        if not quality[row]:
            raise ValueError(f"raw camera quality is false for frame {frame_id}")
        intrinsic = intrinsics[row]
        extrinsic = extrinsics[row]
        if not np.isfinite(intrinsic).all() or not np.isfinite(extrinsic).all():
            raise ValueError(f"raw camera is non-finite for frame {frame_id}")
        stem = f"frame_{frame_id:06d}"
        color_path = scene_root / "rgb" / f"{stem}.png"
        depth_path = scene_root / "depth" / f"{stem}.png"
        if not color_path.is_file() or not depth_path.is_file():
            raise FileNotFoundError(
                f"raw RGB-D artifact is missing for frame {frame_id}"
            )
        records.append(
            FrameRecord(
                scene=scene,
                frame_id=frame_id,
                source_stem=stem,
                split="raw",
                color_path=color_path,
                depth_path=depth_path,
                label_path=source_root / ".annotations_are_not_opened" / f"{stem}.pkl",
                annotation_kind="unlabeled",
                intrinsic=intrinsic.copy(),
                extrinsic_w2c=extrinsic.copy(),
            )
        )
    frame_contract = {
        "width": width,
        "height": height,
        "rgb_dtype": "uint8",
        "rgb_channels": 3,
        "depth_dtype": "uint16",
        "depth_unit": "mm",
        "depth_invalid_value": 0,
    }
    metadata = {
        "kind": "raw_colmap_rgbd_v1",
        "scene": scene,
        "start_frame": start_frame,
        "stop_frame_exclusive": stop_frame,
        "frame_count": requested_count,
        "contiguous_frame_ids": True,
        "camera_quality_all_true": True,
        "label_artifacts_opened": False,
        "source_timestamps_available": False,
        "source_depth_unit": depth_unit,
        "normalized_depth_unit": "mm",
        "dataset_sha256": sha256_file(dataset_path),
        "cameras_sha256": sha256_file(cameras_path),
    }
    return RawSourceSelection(frame_contract, tuple(records), metadata)


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
    clips: list[list[np.ndarray]], *, tile_size=(320, 240), max_columns: int = 4
) -> np.ndarray:
    if not clips or any(not clip for clip in clips) or max_columns <= 0:
        raise ValueError("contact sheet requires non-empty clips and positive columns")
    rows = []
    for clip in clips:
        tiles = [
            cv2.resize(frame, tile_size, interpolation=cv2.INTER_AREA) for frame in clip
        ]
        for offset in range(0, len(tiles), max_columns):
            row = tiles[offset : offset + max_columns]
            row.extend(np.zeros_like(row[0]) for _ in range(max_columns - len(row)))
            rows.append(np.concatenate(row, axis=1))
    return np.concatenate(rows, axis=0)


def main() -> None:
    args = parse_args()
    if (
        args.max_frames <= 0
        or args.fps <= 0
        or args.contact_samples <= 0
        or args.progress_interval <= 0
    ):
        raise ValueError(
            "--max-frames, --fps, --contact-samples, and --progress-interval "
            "must be positive"
        )
    window_indices = tuple(args.window_indices or (0, 34))
    if args.raw_source_root is not None and args.window_indices is not None:
        raise ValueError(
            "--raw-source-root and --window-indices are mutually exclusive"
        )
    if args.raw_source_root is None and len(set(window_indices)) != 2:
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
    if args.raw_source_root is not None:
        raw_selection = load_raw_source_records(
            args.raw_source_root,
            scene=args.scene,
            start_frame=args.start_frame,
            frame_count=args.frame_count,
        )
        frame_contract = raw_selection.frame_contract
        clip_specs = [
            {
                "selection_kind": "continuous_raw_source_range",
                "window_index": None,
                "sequence_id": (
                    f"{args.scene}/raw/{args.start_frame:06d}-"
                    f"{args.start_frame + args.frame_count - 1:06d}"
                ),
                "physical_scene": args.scene,
                "source_row": None,
                "records": raw_selection.records,
            }
        ]
        interpretation = "one_continuous_raw_source_clip"
        source_metadata = raw_selection.metadata
        contact_name = "continuous_contact_sheet.png"
        report_split = "raw"
    else:
        if any(not 0 <= value < len(index.windows) for value in window_indices):
            raise IndexError("a window index is outside the selected split")
        windows = [index.windows[value] for value in window_indices]
        if set(windows[0].frame_ids) & set(windows[1].frame_ids):
            raise ValueError("selected clips must not share frames")
        frame_contract = index.frame_contract
        clip_specs = [
            {
                "selection_kind": "manifest_window",
                "window_index": window_index,
                "sequence_id": window.sequence_id,
                "physical_scene": window.scene,
                "source_row": window.source_row,
                "records": tuple(
                    index.record(window.scene, frame_id)
                    for frame_id in window.frame_ids[: args.max_frames]
                ),
            }
            for window_index, window in zip(window_indices, windows)
        ]
        interpretation = "two_sequential_clips_from_one_physical_scene"
        source_metadata = {
            "kind": "yopo_sequence_manifest_windows",
            "window_indices": list(window_indices),
            "label_artifacts_opened": False,
        }
        contact_name = "two_clip_contact_sheet.png"
        report_split = args.split
    image_size = (
        int(frame_contract["height"]),
        int(frame_contract["width"]),
    )
    reader = RGBDFrameReader(frame_contract)
    descriptor = None
    descriptor_metadata = None
    clip_reports = []
    rendered_clips: list[list[np.ndarray]] = []
    artifact_paths: list[Path] = []
    if device.type == "cuda":
        torch.cuda.empty_cache()
        torch.cuda.reset_peak_memory_stats(device)

    with torch.inference_mode():
        for clip_index, clip_spec in enumerate(clip_specs):
            tracker = OnlineGeometryTracker(TrackerConfig(**dict(inference.tracker)))
            records = clip_spec["records"]
            sample_count = min(args.contact_samples, len(records))
            sample_ordinals = set(
                np.rint(np.linspace(0, len(records) - 1, sample_count))
                .astype(np.int64)
                .tolist()
            )
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
                for ordinal, record in enumerate(records):
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
                    if ordinal in sample_ordinals:
                        rendered_frames.append(rendered.copy())
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
                    if (
                        ordinal + 1
                    ) % args.progress_interval == 0 or ordinal + 1 == len(records):
                        print(
                            f"clip={clip_index} frames={ordinal + 1}/{len(records)} "
                            f"frame_id={record.frame_id} "
                            f"tracks={len(observations)} drawn={len(drawn_ids)}",
                            flush=True,
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
                    "selection_kind": clip_spec["selection_kind"],
                    "window_index": clip_spec["window_index"],
                    "sequence_id": clip_spec["sequence_id"],
                    "physical_scene": clip_spec["physical_scene"],
                    "source_row": clip_spec["source_row"],
                    "frame_count": len(records),
                    "tracker_reset": True,
                    "track_id_namespace": f"clip_{clip_index:02d}",
                    "video": str(video_path),
                    "frames": frame_reports,
                }
            )

    contact_sheet = _contact_sheet(rendered_clips)
    contact_path = output_dir / contact_name
    if not cv2.imwrite(str(contact_path), contact_sheet):
        raise OSError(f"failed to write contact sheet: {contact_path}")
    artifact_paths.append(contact_path)
    report = {
        "status": "completed",
        "schema": "yopo_tracked_projected_ellipse_visualization_v2",
        "interpretation": interpretation,
        "physical_scene_count": len(
            {clip_spec["physical_scene"] for clip_spec in clip_specs}
        ),
        "rendered_clip_count": len(clip_specs),
        "rendered_frame_count": sum(
            len(clip_spec["records"]) for clip_spec in clip_specs
        ),
        "playback_fps": args.fps,
        "tracker_reset_per_clip": True,
        "tracker_config": dict(inference.tracker),
        "input_contract": "current_rgbd_and_manifest_camera_only_no_label_v1",
        "causal": True,
        "future_frame_access": False,
        "label_artifacts_opened": False,
        "ellipse_contract": "projected_ellipses_semiaxes_first_radians_v1",
        "split": report_split,
        "source": source_metadata,
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
