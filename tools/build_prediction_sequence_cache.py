#!/usr/bin/env python3
"""Build prediction-derived YOPO sequence features with pseudo supervision."""

from __future__ import annotations

import argparse
import shutil
from pathlib import Path
from typing import Any, Iterable

import numpy as np
import torch
from mmengine.config import Config
from mmengine.runner.checkpoint import load_checkpoint

from tools.train_sequence_context import _feature_contract, _model_input
from yopo.datasets.pose_estimation.yopo_sequence import FrameRecordReader, SequenceIndex
from yopo.models.tracking.causal_inference import (
    build_prediction_observation_batch,
    make_detector_data_sample,
)
from yopo.models.tracking.geometry_context import camera_to_world_points
from yopo.models.tracking.prediction_cache import (
    PREDICTION_FEATURE_CACHE_SCHEMA,
    assign_predictions_to_teachers,
)
from yopo.models.tracking.sequence_training import (
    atomic_torch_save,
    atomic_write_json,
    build_pair_examples,
    frame_key,
    sha256_file,
)
from yopo.registry import MODELS
from yopo.utils import register_all_modules, register_mmengine_checkpoint_safe_globals


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", type=Path, required=True)
    parser.add_argument("--manifest", type=Path, required=True)
    parser.add_argument("--checkpoint", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--report", type=Path, required=True)
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--batch-size", type=int)
    parser.add_argument("--max-frames", type=int)
    return parser.parse_args()


def _chunks(items: list[Any], size: int) -> Iterable[list[Any]]:
    for offset in range(0, len(items), size):
        yield items[offset : offset + size]


def _load_detector(
    config: Config, checkpoint_path: Path, device: torch.device
) -> torch.nn.Module:
    register_all_modules(init_default_scope=True)
    register_mmengine_checkpoint_safe_globals()
    detector = MODELS.build(config.model)
    load_checkpoint(detector, str(checkpoint_path), map_location="cpu", strict=True)
    return detector.requires_grad_(False).eval().to(device)


def _records_by_split(
    manifest_path: Path,
) -> tuple[dict[str, list[Any]], dict[str, SequenceIndex]]:
    indexes = {
        split: SequenceIndex(manifest_path, split=split)
        for split in ("train", "val", "smoke")
    }
    result = {}
    for split, index in indexes.items():
        keys = sorted(
            {
                (window.scene, value)
                for window in index.windows
                for value in window.frame_ids
            }
        )
        result[split] = [index.record(*key) for key in keys]
    return result, indexes


def _predict_batch(
    detector: torch.nn.Module,
    records: list[Any],
    decoded: list[dict[str, Any]],
    *,
    image_size: tuple[int, int],
    device: torch.device,
) -> tuple[list[Any], tuple[torch.Tensor, ...]]:
    inputs = torch.stack([_model_input(frame) for frame in decoded]).to(
        device=device, dtype=torch.float32
    )
    samples = [
        make_detector_data_sample(
            frame_id=record.frame_id,
            image_path=str(record.color_path),
            intrinsic=record.intrinsic,
            image_size=image_size,
        )
        for record in records
    ]
    outputs, backbone_features = detector.predict_with_backbone_features(
        inputs, samples, rescale=True
    )
    return outputs, backbone_features


def _probe_batch_size(
    detector: torch.nn.Module,
    record: Any,
    frame: dict[str, Any],
    *,
    image_size: tuple[int, int],
    device: torch.device,
    candidates: list[int],
) -> tuple[int, list[dict[str, int | str]]]:
    attempts = []
    for size in candidates:
        if size <= 0:
            raise ValueError("prediction batch probe sizes must be positive")
        try:
            torch.cuda.empty_cache()
            torch.cuda.reset_peak_memory_stats(device)
            outputs, features = _predict_batch(
                detector,
                [record] * size,
                [frame] * size,
                image_size=image_size,
                device=device,
            )
            if len(outputs) != size:
                raise RuntimeError("detector returned a different batch length")
            peak = int(torch.cuda.max_memory_allocated(device))
            attempts.append(
                {"batch_size": size, "status": "success", "peak_bytes": peak}
            )
            del outputs, features
            torch.cuda.empty_cache()
            return size, attempts
        except torch.OutOfMemoryError:
            attempts.append({"batch_size": size, "status": "oom"})
            torch.cuda.empty_cache()
    raise RuntimeError("no prediction batch size fits the selected CUDA device")


def _teacher_world_centers(
    label: dict[str, Any], extrinsic_w2c: np.ndarray, device: torch.device
) -> torch.Tensor:
    valid = np.asarray(label["ellipsoid_valid"], dtype=bool)
    centers_camera = torch.from_numpy(
        np.asarray(label["ellipsoid_centers_cfov_m"], dtype=np.float32)[valid]
    ).to(device)
    extrinsic = torch.eye(4, dtype=torch.float32, device=device)
    extrinsic[:3] = torch.from_numpy(extrinsic_w2c).to(device)
    return camera_to_world_points(centers_camera, extrinsic)


def _quality_summary(values: torch.Tensor) -> dict[str, float | int | None]:
    values = values.detach().float().cpu()
    values = values[torch.isfinite(values)]
    if not len(values):
        return {"count": 0, "median": None, "p10": None, "p90": None}
    return {
        "count": len(values),
        "median": float(torch.quantile(values, 0.50)),
        "p10": float(torch.quantile(values, 0.10)),
        "p90": float(torch.quantile(values, 0.90)),
    }


def main() -> None:
    args = parse_args()
    paths = [args.config, args.manifest, args.checkpoint]
    if any(not path.expanduser().is_file() for path in paths):
        raise FileNotFoundError("config, manifest and checkpoint must exist")
    if args.max_frames is not None and args.max_frames <= 0:
        raise ValueError("--max-frames must be positive")
    output = args.output.expanduser().resolve()
    report_path = args.report.expanduser().resolve()
    if output.exists() or report_path.exists():
        raise FileExistsError("output and report paths must be unused")
    config_path = args.config.expanduser().resolve()
    manifest_path = args.manifest.expanduser().resolve()
    checkpoint_path = args.checkpoint.expanduser().resolve()
    config = Config.fromfile(config_path)
    feature_contract = _feature_contract(config)
    if feature_contract["schema"] != PREDICTION_FEATURE_CACHE_SCHEMA:
        raise ValueError("config does not select prediction-derived observations")
    settings = config.sequence_context
    matching = settings.prediction_matching
    device = torch.device(args.device)
    if device.type != "cuda" or not torch.cuda.is_available():
        raise RuntimeError("prediction cache extraction requires explicit CUDA")
    records_by_split, indexes = _records_by_split(manifest_path)
    ordered = [
        record
        for split in ("train", "val", "smoke")
        for record in records_by_split[split]
    ]
    if args.max_frames is not None:
        ordered = ordered[: args.max_frames]
    index = indexes["train"]
    image_size = (
        int(index.frame_contract["height"]),
        int(index.frame_contract["width"]),
    )
    reader = FrameRecordReader(index.frame_contract)
    detector = _load_detector(config, checkpoint_path, device)
    first_frame = reader.read(ordered[0])
    if args.batch_size is None:
        batch_size, probe_attempts = _probe_batch_size(
            detector,
            ordered[0],
            first_frame,
            image_size=image_size,
            device=device,
            candidates=list(settings.feature_extraction.prediction_batch_probe_sizes),
        )
    else:
        if args.batch_size <= 0:
            raise ValueError("--batch-size must be positive")
        batch_size = args.batch_size
        probe_attempts = [{"batch_size": batch_size, "status": "configured"}]
    torch.cuda.empty_cache()
    torch.cuda.reset_peak_memory_stats(device)
    frames = {}
    split_counts = {
        split: {
            "frames": 0,
            "detector_predictions": 0,
            "retained_predictions": 0,
            "matched": 0,
            "unknown": 0,
            "geometry_invalid": 0,
        }
        for split in ("train", "val", "smoke")
    }
    with torch.inference_mode():
        for batch_records in _chunks(ordered, batch_size):
            decoded = [reader.read(record) for record in batch_records]
            samples, backbone_features = _predict_batch(
                detector,
                batch_records,
                decoded,
                image_size=image_size,
                device=device,
            )
            for batch_index, (record, frame, sample) in enumerate(
                zip(batch_records, decoded, samples)
            ):
                raw = build_prediction_observation_batch(
                    sample.pred_instances,
                    backbone_features,
                    image_size=image_size,
                    extrinsic_w2c=torch.from_numpy(frame["extrinsic_w2c"]).to(device),
                    score_threshold=float(matching.score_threshold),
                    max_detections=int(matching.max_detections),
                    batch_index=batch_index,
                )
                label = frame["label"]
                teacher_valid = np.asarray(label["ellipsoid_valid"], dtype=bool)
                assignment = assign_predictions_to_teachers(
                    prediction_boxes_xyxy=raw.bboxes_xyxy,
                    prediction_labels=raw.labels,
                    teacher_obbs=np.asarray(label["obb_cxcywha_rad"])[teacher_valid],
                    teacher_class_ids=np.asarray(label["class_ids"])[teacher_valid],
                    teacher_coordinate_scale=float(
                        label.get("obb_coordinate_scale", 1.0)
                    ),
                    class_id_mapping=dict(matching.class_id_mapping),
                    min_iou=float(matching.min_iou),
                    max_center_distance_px=float(matching.max_center_distance_px),
                )
                valid_label_rows = np.flatnonzero(teacher_valid)
                original_teacher_indices = torch.full_like(
                    assignment.teacher_indices, -1
                )
                teacher_centers = torch.full(
                    (len(raw.prediction_indices), 3),
                    float("nan"),
                    dtype=torch.float32,
                    device=device,
                )
                matched_predictions = assignment.matched_prediction_indices
                matched_teachers = assignment.matched_teacher_indices
                if len(matched_predictions):
                    original_teacher_indices[matched_predictions] = torch.from_numpy(
                        valid_label_rows[matched_teachers.cpu().numpy()]
                    ).to(device)
                    teacher_world = _teacher_world_centers(
                        label, frame["extrinsic_w2c"], device
                    )
                    teacher_centers[matched_predictions] = teacher_world[
                        matched_teachers
                    ]
                key = frame_key(record.scene, record.frame_id)
                frames[key] = {
                    "scene": record.scene,
                    "frame_id": record.frame_id,
                    "split": record.split,
                    "appearance": raw.appearance.to(dtype=torch.float16).cpu(),
                    "centers_world": raw.centers_world.float().cpu(),
                    "prediction_indices": raw.prediction_indices.long().cpu(),
                    "labels": raw.labels.long().cpu(),
                    "bboxes_xyxy": raw.bboxes_xyxy.float().cpu(),
                    "scores": raw.scores.float().cpu(),
                    "geometry_valid": raw.geometry_valid.bool().cpu(),
                    "teacher_indices": original_teacher_indices.long().cpu(),
                    "teacher_centers_world": teacher_centers.cpu(),
                    "teacher_ious": assignment.ious.float().cpu(),
                    "teacher_center_distances_px": assignment.center_distances_px.float().cpu(),
                    "annotation_kind": record.annotation_kind,
                }
                counts = split_counts[record.split]
                counts["frames"] += 1
                counts["detector_predictions"] += len(sample.pred_instances)
                counts["retained_predictions"] += len(raw.prediction_indices)
                counts["matched"] += len(matched_predictions)
                counts["unknown"] += len(raw.prediction_indices) - len(
                    matched_predictions
                )
                counts["geometry_invalid"] += int((~raw.geometry_valid).sum())
            del samples, backbone_features
    peak = int(torch.cuda.max_memory_allocated(device))
    nonempty = [value for value in frames.values() if len(value["appearance"])]
    if not nonempty:
        raise ValueError("prediction cache contains no retained detection")
    appearance_dim = int(nonempty[0]["appearance"].shape[1])
    cache = {
        "schema": PREDICTION_FEATURE_CACHE_SCHEMA,
        "manifest_sha256": sha256_file(manifest_path),
        "g10_checkpoint_sha256": sha256_file(checkpoint_path),
        "config_sha256": sha256_file(config_path),
        "appearance_dim": appearance_dim,
        "image_size": list(image_size),
        "splits": sorted({value["split"] for value in frames.values()}),
        "annotation_kind": "pseudo",
        "feature_contract": feature_contract,
        "prediction_matching": dict(matching),
        "frames": frames,
    }
    quality_by_split = {}
    pair_counts = {}
    for split in ("train", "val", "smoke"):
        split_frames = [value for value in frames.values() if value["split"] == split]
        if not split_frames:
            empty_summary = _quality_summary(torch.empty(0))
            quality_by_split[split] = {
                "retained_score": empty_summary,
                "matched_iou": empty_summary,
                "matched_center_distance_px": empty_summary,
                "matched_translation_error_m": empty_summary,
            }
            pair_counts[split] = 0
            continue
        matched_masks = [value["teacher_indices"] >= 0 for value in split_frames]
        scores = torch.cat([value["scores"] for value in split_frames])
        ious = torch.cat(
            [
                value["teacher_ious"][mask]
                for value, mask in zip(split_frames, matched_masks)
            ]
        )
        center_distances = torch.cat(
            [
                value["teacher_center_distances_px"][mask]
                for value, mask in zip(split_frames, matched_masks)
            ]
        )
        translation_errors = torch.cat(
            [
                torch.linalg.vector_norm(
                    value["centers_world"][mask] - value["teacher_centers_world"][mask],
                    dim=1,
                )
                for value, mask in zip(split_frames, matched_masks)
            ]
        )
        quality_by_split[split] = {
            "retained_score": _quality_summary(scores),
            "matched_iou": _quality_summary(ious),
            "matched_center_distance_px": _quality_summary(center_distances),
            "matched_translation_error_m": _quality_summary(translation_errors),
        }
        pair_counts[split] = len(
            build_pair_examples(
                manifest_path,
                split=split,
                cache=cache,
                max_distance_m=float(settings.association.max_distance_m),
                ambiguity_margin_m=float(settings.association.ambiguity_margin_m),
                min_matches=int(
                    settings.pairs.min_train_matches
                    if split == "train"
                    else settings.pairs.min_eval_matches
                ),
            )
        )
    output.parent.mkdir(parents=True, exist_ok=True)
    report_path.parent.mkdir(parents=True, exist_ok=True)
    atomic_torch_save(output, cache)
    disk = shutil.disk_usage(output.parent)
    report = {
        "status": "completed",
        "schema": PREDICTION_FEATURE_CACHE_SCHEMA,
        "feature_cache": str(output),
        "feature_cache_sha256": sha256_file(output),
        "manifest_sha256": cache["manifest_sha256"],
        "g10_checkpoint_sha256": cache["g10_checkpoint_sha256"],
        "config_sha256": cache["config_sha256"],
        "frame_count": len(frames),
        "appearance_dim": appearance_dim,
        "batch_size": batch_size,
        "probe_attempts": probe_attempts,
        "cuda_peak_vram_bytes": peak,
        "split_counts": split_counts,
        "quality_by_split": quality_by_split,
        "pair_counts": pair_counts,
        "storage_free_bytes_after": int(disk.free),
        "matching": dict(matching),
    }
    atomic_write_json(report_path, report)
    print(report_path)


if __name__ == "__main__":
    main()
