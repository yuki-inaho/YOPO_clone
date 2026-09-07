#!/usr/bin/env python3
"""Extract frozen G10 features and train sequence identity/context heads."""

from __future__ import annotations

import argparse
import hashlib
import json
import math
import shutil
from pathlib import Path
from typing import Any, Iterable

import numpy as np
import torch
from mmengine.config import Config

from yopo.datasets.pose_estimation.yopo_sequence import FrameRecordReader, SequenceIndex
from yopo.datasets.transforms.raw_depth import compose_metric_rgbd_input
from yopo.models.tracking.geometry_context import camera_to_world_points
from yopo.models.tracking.prediction_cache import PREDICTION_FEATURE_CACHE_SCHEMA
from yopo.models.tracking.sequence_training import (
    EXPERIMENT_MODES,
    FEATURE_CACHE_SCHEMA,
    atomic_torch_save,
    atomic_write_json,
    build_pair_examples,
    frame_key,
    load_feature_cache,
    sample_pyramid_at_centers,
    sha256_file,
    train_modes,
)
from yopo.registry import MODELS
from yopo.utils import register_all_modules


def _chunks(items: list[Any], size: int) -> Iterable[list[Any]]:
    for offset in range(0, len(items), size):
        yield items[offset : offset + size]


def _model_input(frame: dict[str, Any]) -> torch.Tensor:
    rgbd = compose_metric_rgbd_input(
        np.asarray(frame["image_bgr"]), np.asarray(frame["depth_mm"])
    )
    return torch.from_numpy(rgbd).permute(2, 0, 1)


def _resolved_config_sha256(config: Config) -> str:
    return hashlib.sha256(config.pretty_text.encode("utf-8")).hexdigest()


def _canonical_config_value(value: Any) -> Any:
    if isinstance(value, dict):
        return {
            str(key): _canonical_config_value(item)
            for key, item in sorted(value.items(), key=lambda pair: str(pair[0]))
        }
    if isinstance(value, (list, tuple)):
        return [_canonical_config_value(item) for item in value]
    if isinstance(value, Path):
        return str(value)
    if value is None or isinstance(value, (bool, int, float, str)):
        return value
    raise TypeError(f"unsupported feature producer config value: {type(value)!r}")


def _feature_producer_config_sha256(config: Config) -> str:
    sequence_context = config.get("sequence_context", {})
    producer_config = {
        "model": config.get("model", {}),
        "feature_extraction": sequence_context.get("feature_extraction", {}),
        "prediction_matching": sequence_context.get("prediction_matching", {}),
    }
    payload = json.dumps(
        _canonical_config_value(producer_config),
        sort_keys=True,
        separators=(",", ":"),
        ensure_ascii=True,
    )
    return hashlib.sha256(payload.encode("utf-8")).hexdigest()


def _feature_contract(config: Config) -> dict[str, Any]:
    sequence_context = config.get("sequence_context", {})
    feature_settings = sequence_context.get("feature_extraction", {})
    source = feature_settings.get("source", "frozen_g10_backbone_pyramid_obb_center")
    if source == "frozen_g10_backbone_pyramid_obb_center":
        sampler = "three_level_bilinear_obb_center_align_corners_false_v1"
        schema = FEATURE_CACHE_SCHEMA
    elif source == "frozen_g10_detector_prediction_hbb_center":
        sampler = "three_level_bilinear_prediction_hbb_center_align_corners_false_v1"
        schema = PREDICTION_FEATURE_CACHE_SCHEMA
    else:
        raise ValueError(f"unsupported sequence feature source: {source!r}")
    return {
        "schema": schema,
        "observation_source": source,
        "input": {
            "channels": "RGB_plus_depth_m",
            "rgb_scale": 255.0,
            "depth_source_unit": "millimetres_uint16",
            "depth_norm_scale": 1000.0,
            "color_order": "RGB",
        },
        "sampler": sampler,
        "geometry": "stable_direct_pairwise_distance_v1",
        "producer_config_sha256": _feature_producer_config_sha256(config),
    }


def load_frozen_g10_backbone(
    config: Config, checkpoint_path: Path, device: torch.device
) -> torch.nn.Module:
    register_all_modules(init_default_scope=True)
    backbone = MODELS.build(config.model.backbone)
    checkpoint = torch.load(checkpoint_path, map_location="cpu", weights_only=False)
    state = {
        key.removeprefix("backbone."): value
        for key, value in checkpoint["state_dict"].items()
        if key.startswith("backbone.")
    }
    backbone.load_state_dict(state, strict=True)
    backbone.requires_grad_(False).eval().to(device)
    return backbone


def probe_feature_batch_size(
    backbone: torch.nn.Module,
    *,
    device: torch.device,
    image_size: tuple[int, int],
    candidates: list[int],
) -> tuple[int, list[dict[str, Any]], int]:
    """Try only decreasing batches; OOM changes no architecture or data."""

    attempts = []
    height, width = image_size
    for batch_size in candidates:
        try:
            if device.type == "cuda":
                torch.cuda.empty_cache()
                torch.cuda.reset_peak_memory_stats(device)
            inputs = torch.zeros(batch_size, 4, height, width, device=device)
            with (
                torch.inference_mode(),
                torch.autocast(
                    device_type=device.type,
                    dtype=torch.bfloat16,
                    enabled=device.type == "cuda",
                ),
            ):
                outputs = backbone(inputs)
                _ = sum(level.square().mean() for level in outputs)
            del inputs, outputs
            peak = (
                int(torch.cuda.max_memory_allocated(device))
                if device.type == "cuda"
                else 0
            )
            attempts.append(
                {"batch_size": batch_size, "status": "success", "peak_bytes": peak}
            )
            return batch_size, attempts, peak
        except torch.OutOfMemoryError:
            attempts.append({"batch_size": batch_size, "status": "oom"})
            if device.type == "cuda":
                torch.cuda.empty_cache()
    raise RuntimeError("no feature extraction batch size fits the selected device")


def extract_feature_cache(
    *,
    config: Config,
    manifest_path: Path,
    checkpoint_path: Path,
    cache_path: Path,
    splits: tuple[str, ...],
    device: torch.device,
    batch_size: int | None,
    probe_candidates: list[int],
) -> tuple[dict[str, Any], dict[str, Any]]:
    backbone = load_frozen_g10_backbone(config, checkpoint_path, device)
    indexes = {split: SequenceIndex(manifest_path, split=split) for split in splits}
    records = {}
    for index in indexes.values():
        for window in index.windows:
            for frame_id in window.frame_ids:
                record = index.record(window.scene, frame_id)
                records[frame_key(window.scene, frame_id)] = record
    ordered = [records[key] for key in sorted(records)]
    first_index = next(iter(indexes.values()))
    height = int(first_index.frame_contract["height"])
    width = int(first_index.frame_contract["width"])
    if batch_size is None:
        batch_size, attempts, probe_peak = probe_feature_batch_size(
            backbone,
            device=device,
            image_size=(height, width),
            candidates=probe_candidates,
        )
    else:
        attempts = [{"batch_size": batch_size, "status": "configured"}]
        probe_peak = 0
    reader = FrameRecordReader(first_index.frame_contract)
    frames = {}
    if device.type == "cuda":
        torch.cuda.empty_cache()
        torch.cuda.reset_peak_memory_stats(device)
    for batch_records in _chunks(ordered, batch_size):
        decoded = [reader.read(record) for record in batch_records]
        inputs = torch.stack([_model_input(frame) for frame in decoded]).to(device)
        with (
            torch.inference_mode(),
            torch.autocast(
                device_type=device.type,
                dtype=torch.bfloat16,
                enabled=device.type == "cuda",
            ),
        ):
            pyramid = backbone(inputs)
            for batch_index, (record, frame) in enumerate(zip(batch_records, decoded)):
                label = frame["label"]
                valid = torch.from_numpy(
                    np.asarray(label["ellipsoid_valid"], dtype=bool)
                )
                centers_px = torch.from_numpy(
                    np.asarray(label["obb_cxcywha_rad"], dtype=np.float32)[valid, :2]
                ).to(device)
                appearance = sample_pyramid_at_centers(
                    pyramid,
                    batch_index=batch_index,
                    centers_px=centers_px,
                    image_size=(height, width),
                )
                centers_camera = torch.from_numpy(
                    np.asarray(label["ellipsoid_centers_cfov_m"], dtype=np.float32)[
                        valid
                    ]
                ).to(device)
                extrinsic = torch.eye(4, dtype=torch.float32, device=device)
                extrinsic[:3] = torch.from_numpy(frame["extrinsic_w2c"]).to(device)
                centers_world = camera_to_world_points(centers_camera, extrinsic)
                key = frame_key(record.scene, record.frame_id)
                frames[key] = {
                    "scene": record.scene,
                    "frame_id": record.frame_id,
                    "appearance": appearance.to(dtype=torch.float16).cpu(),
                    "centers_world": centers_world.float().cpu(),
                    "annotation_kind": record.annotation_kind,
                }
        del inputs, pyramid
    peak = int(torch.cuda.max_memory_allocated(device)) if device.type == "cuda" else 0
    appearance_dim = next(iter(frames.values()))["appearance"].shape[1]
    cache = {
        "schema": FEATURE_CACHE_SCHEMA,
        "manifest_sha256": sha256_file(manifest_path),
        "g10_checkpoint_sha256": sha256_file(checkpoint_path),
        "appearance_dim": int(appearance_dim),
        "image_size": [height, width],
        "splits": list(splits),
        "annotation_kind": "pseudo",
        "feature_contract": _feature_contract(config),
        "frames": frames,
    }
    atomic_torch_save(cache_path, cache)
    report = {
        "feature_cache": str(cache_path),
        "feature_cache_sha256": sha256_file(cache_path),
        "frame_count": len(frames),
        "instance_count": sum(len(frame["appearance"]) for frame in frames.values()),
        "appearance_dim": int(appearance_dim),
        "batch_size": batch_size,
        "probe_attempts": attempts,
        "probe_peak_vram_bytes": probe_peak,
        "extraction_peak_vram_bytes": peak,
    }
    del backbone
    if device.type == "cuda":
        torch.cuda.empty_cache()
    return cache, report


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", type=Path, required=True)
    parser.add_argument("--manifest", type=Path)
    parser.add_argument("--checkpoint", type=Path)
    parser.add_argument("--work-dir", type=Path)
    parser.add_argument(
        "--feature-cache",
        type=Path,
        help="Validated read-only feature cache to reuse instead of extracting one.",
    )
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--feature-batch-size", type=int)
    parser.add_argument("--epochs", type=int)
    parser.add_argument("--patience", type=int)
    parser.add_argument("--max-pairs", type=int)
    parser.add_argument(
        "--negative-scope",
        choices=("matched_only", "all_detections"),
        help="Explicit identity-loss negative set; recorded in run provenance.",
    )
    parser.add_argument("--modes", nargs="+", choices=EXPERIMENT_MODES)
    run_group = parser.add_mutually_exclusive_group()
    run_group.add_argument("--smoke", action="store_true")
    run_group.add_argument("--pilot", action="store_true")
    run_group.add_argument("--full", action="store_true")
    return parser.parse_args()


def _resolve(value: Path | None, fallback: str) -> Path:
    return (value or Path(fallback)).expanduser().resolve()


def _run_kind(args: argparse.Namespace) -> str:
    if args.smoke:
        return "smoke"
    if args.pilot:
        if args.max_pairs is None or args.epochs is None:
            raise ValueError("--pilot requires explicit --max-pairs and --epochs")
        return "pilot"
    if getattr(args, "full", False):
        if args.max_pairs is not None:
            raise ValueError("--full rejects --max-pairs")
        return "full"
    if args.max_pairs is not None:
        raise ValueError("--max-pairs requires --pilot or --smoke")
    raise ValueError("select exactly one run mode: --smoke, --pilot, or --full")


def _validate_feature_cache(
    cache: dict[str, Any],
    *,
    cache_path: Path,
    manifest_path: Path,
    checkpoint_path: Path,
    config: Config,
    required_splits: tuple[str, ...],
) -> dict[str, Any]:
    expected_contract = _feature_contract(config)
    if cache.get("schema") != expected_contract["schema"]:
        raise ValueError(
            "feature cache schema differs from configured observation source"
        )
    expected = (sha256_file(manifest_path), sha256_file(checkpoint_path))
    actual = (cache.get("manifest_sha256"), cache.get("g10_checkpoint_sha256"))
    if actual != expected:
        raise ValueError("feature cache provenance differs from inputs")
    if cache.get("feature_contract") != expected_contract:
        raise ValueError("feature cache contract differs from this run")
    if cache.get("annotation_kind") != "pseudo":
        raise ValueError("feature cache must declare pseudo annotations")
    available_splits = set(cache.get("splits", ()))
    missing_splits = set(required_splits) - available_splits
    if missing_splits:
        raise ValueError(
            f"feature cache misses required splits: {sorted(missing_splits)}"
        )
    frames = cache["frames"]
    instance_count = 0
    for key, frame in frames.items():
        appearance = frame.get("appearance")
        centers = frame.get("centers_world")
        if not isinstance(appearance, torch.Tensor) or appearance.ndim != 2:
            raise ValueError(f"feature cache frame {key!r} has invalid appearance")
        if not isinstance(centers, torch.Tensor) or centers.shape != (
            len(appearance),
            3,
        ):
            raise ValueError(f"feature cache frame {key!r} has invalid centers")
        if appearance.shape[1] != int(cache["appearance_dim"]):
            raise ValueError(f"feature cache frame {key!r} has inconsistent dimension")
        if not torch.isfinite(appearance).all():
            raise FloatingPointError(f"feature cache frame {key!r} is nonfinite")
        if cache.get("schema") == PREDICTION_FEATURE_CACHE_SCHEMA:
            required = {
                "prediction_indices": (len(appearance),),
                "labels": (len(appearance),),
                "bboxes_xyxy": (len(appearance), 4),
                "scores": (len(appearance),),
                "geometry_valid": (len(appearance),),
                "teacher_indices": (len(appearance),),
                "teacher_centers_world": (len(appearance), 3),
                "teacher_ious": (len(appearance),),
                "teacher_center_distances_px": (len(appearance),),
            }
            for field, shape in required.items():
                value = frame.get(field)
                if not isinstance(value, torch.Tensor) or value.shape != shape:
                    raise ValueError(f"feature cache frame {key!r} has invalid {field}")
            geometry_valid = frame["geometry_valid"].bool()
            if (
                not torch.isfinite(centers[geometry_valid]).all()
                or not torch.isnan(centers[~geometry_valid]).all()
            ):
                raise FloatingPointError(
                    f"feature cache frame {key!r} violates geometry validity"
                )
            teacher_indices = frame["teacher_indices"].long()
            teacher_matched = teacher_indices >= 0
            teacher_centers = frame["teacher_centers_world"]
            if (
                not torch.isfinite(teacher_centers[teacher_matched]).all()
                or not torch.isnan(teacher_centers[~teacher_matched]).all()
            ):
                raise FloatingPointError(
                    f"feature cache frame {key!r} violates teacher unknown contract"
                )
            if (
                not torch.isfinite(frame["teacher_ious"][teacher_matched]).all()
                or not torch.isnan(frame["teacher_ious"][~teacher_matched]).all()
            ):
                raise FloatingPointError(
                    f"feature cache frame {key!r} has invalid teacher IoU"
                )
        elif not torch.isfinite(centers).all():
            raise FloatingPointError(f"feature cache frame {key!r} is nonfinite")
        instance_count += len(appearance)
    return {
        "feature_cache": str(cache_path),
        "feature_cache_sha256": sha256_file(cache_path),
        "schema": cache["schema"],
        "observation_source": expected_contract["observation_source"],
        "frame_count": len(frames),
        "instance_count": instance_count,
        "appearance_dim": int(cache["appearance_dim"]),
        "validated": True,
        "reused": True,
    }


def _update_budget(
    *,
    pair_count: int,
    pair_batch_size: int,
    epochs: int,
    modes: list[str],
    warmup_steps: int,
    training_strategy: str = "independent_v1",
) -> dict[str, Any]:
    if pair_batch_size <= 0 or epochs <= 0:
        raise ValueError("pair batch size and epochs must be positive")
    updates_per_epoch = math.ceil(pair_count / pair_batch_size)
    if training_strategy == "independent_v1":
        trajectory_ids = [mode for mode in modes if mode != "B1"]
    elif training_strategy == "residual_context_curriculum_v1":
        if set(modes) != set(EXPERIMENT_MODES):
            raise ValueError(
                f"residual curriculum requires exactly these modes: {EXPERIMENT_MODES}"
            )
        trajectory_ids = ["appearance", "context_adapter"]
    else:
        raise ValueError(
            f"unsupported sequence training strategy: {training_strategy!r}"
        )
    planned_per_mode = updates_per_epoch * epochs
    return {
        "pair_count": pair_count,
        "pair_batch_size": pair_batch_size,
        "epochs": epochs,
        "updates_per_epoch_per_mode": updates_per_epoch,
        "planned_updates_per_trainable_mode": planned_per_mode,
        "training_strategy": training_strategy,
        "trajectory_ids": trajectory_ids,
        "trainable_trajectory_count": len(trajectory_ids),
        "trainable_mode_count": len(trajectory_ids),
        "planned_updates_all_trainable_modes": planned_per_mode * len(trajectory_ids),
        "warmup_steps_per_mode": warmup_steps,
        "warmup_can_complete": planned_per_mode >= warmup_steps,
    }


def main() -> None:
    args = parse_args()
    run_kind = _run_kind(args)
    config = Config.fromfile(args.config)
    settings = config.sequence_context
    manifest_path = _resolve(args.manifest, settings.manifest_path)
    checkpoint_path = _resolve(args.checkpoint, settings.initial_checkpoint)
    work_dir = _resolve(args.work_dir, settings.work_dir)
    if not manifest_path.is_file() or not checkpoint_path.is_file():
        raise FileNotFoundError("manifest and G10 initial checkpoint must exist")
    device = torch.device(args.device)
    if device.type == "cuda" and not torch.cuda.is_available():
        raise RuntimeError("CUDA was requested but is unavailable")
    modes = list(args.modes or settings.modes)
    epochs = int(args.epochs or settings.training.epochs)
    patience = int(args.patience or settings.training.patience)
    max_pairs = args.max_pairs
    splits = ("smoke",) if run_kind == "smoke" else ("train", "val", "smoke")
    if args.smoke:
        epochs = 1
        patience = 1
        max_pairs = max_pairs or 2
    work_dir.mkdir(parents=True, exist_ok=True)
    local_cache_path = work_dir / (
        "feature_cache_smoke.pth" if args.smoke else "feature_cache.pth"
    )
    cache_path = (
        args.feature_cache.expanduser().resolve()
        if args.feature_cache is not None
        else local_cache_path
    )
    if cache_path.is_file():
        cache = load_feature_cache(cache_path)
        feature_report = _validate_feature_cache(
            cache,
            cache_path=cache_path,
            manifest_path=manifest_path,
            checkpoint_path=checkpoint_path,
            config=config,
            required_splits=splits,
        )
    else:
        if args.feature_cache is not None:
            raise FileNotFoundError(
                f"explicit feature cache does not exist: {cache_path}"
            )
        cache, feature_report = extract_feature_cache(
            config=config,
            manifest_path=manifest_path,
            checkpoint_path=checkpoint_path,
            cache_path=cache_path,
            splits=splits,
            device=device,
            batch_size=args.feature_batch_size,
            probe_candidates=list(settings.feature_extraction.probe_batch_sizes),
        )
    training_split = "smoke" if run_kind == "smoke" else "train"
    validation_split = "smoke" if run_kind == "smoke" else "val"
    association = settings.association
    train_examples = build_pair_examples(
        manifest_path,
        split=training_split,
        cache=cache,
        max_distance_m=float(association.max_distance_m),
        ambiguity_margin_m=float(association.ambiguity_margin_m),
        min_matches=int(settings.pairs.min_train_matches),
        max_pairs=max_pairs,
    )
    validation_examples = build_pair_examples(
        manifest_path,
        split=validation_split,
        cache=cache,
        max_distance_m=float(association.max_distance_m),
        ambiguity_margin_m=float(association.ambiguity_margin_m),
        min_matches=int(settings.pairs.min_eval_matches),
        max_pairs=max_pairs,
    )
    training_config = {
        "seed": int(settings.seed),
        "head": dict(settings.head),
        "training_strategy": str(settings.training.get("strategy", "independent_v1")),
        "optimizer": dict(settings.optimizer),
        "pair_batch_size": int(settings.training.pair_batch_size),
        "embedding_gate": float(settings.evaluation.embedding_gate),
        "association": dict(settings.association),
        "loss": {
            "temperature": 0.07,
            "negative_scope": args.negative_scope or "matched_only",
        },
    }
    budget = _update_budget(
        pair_count=len(train_examples),
        pair_batch_size=int(settings.training.pair_batch_size),
        epochs=epochs,
        modes=modes,
        warmup_steps=int(settings.optimizer.warmup_steps),
        training_strategy=training_config["training_strategy"],
    )
    if run_kind != "smoke" and not budget["warmup_can_complete"]:
        raise ValueError(
            f"{run_kind} update budget cannot complete optimizer warmup: "
            f"{budget['planned_updates_per_trainable_mode']} < "
            f"{budget['warmup_steps_per_mode']}"
        )
    provenance = {
        "manifest_sha256": sha256_file(manifest_path),
        "config_sha256": sha256_file(args.config),
        "resolved_config_sha256": _resolved_config_sha256(config),
        "feature_contract": _feature_contract(config),
        "g10_checkpoint_sha256": sha256_file(checkpoint_path),
        "g10_checkpoint_kind": "raw",
        "feature_cache_sha256": sha256_file(cache_path),
        "annotation_kind": "pseudo",
        "metric_prefix": "proxy_",
        "optimizer": "AmuseOptimizer",
        "training_strategy": training_config["training_strategy"],
        "loss_contract": dict(training_config["loss"]),
    }
    run_dir = work_dir / run_kind
    if run_dir.is_dir() and any(run_dir.iterdir()):
        raise FileExistsError(
            f"run directory is not empty; use a new work directory: {run_dir}"
        )
    disk = shutil.disk_usage(work_dir)
    gpu = {"device": str(device), "available": device.type == "cuda"}
    if device.type == "cuda":
        free_bytes, total_bytes = torch.cuda.mem_get_info(device)
        gpu.update(
            {
                "name": torch.cuda.get_device_name(device),
                "free_memory_bytes": int(free_bytes),
                "total_memory_bytes": int(total_bytes),
            }
        )
    preflight = {
        "status": "passed",
        "run_kind": run_kind,
        "inputs": {
            "config": str(args.config.expanduser().resolve()),
            "manifest": str(manifest_path),
            "g10_checkpoint": str(checkpoint_path),
            "feature_cache": str(cache_path),
        },
        "cli_overrides": {"negative_scope": args.negative_scope},
        "outputs": {"work_dir": str(work_dir), "run_dir": str(run_dir)},
        "provenance": provenance,
        "feature_cache_validation": feature_report,
        "train_pair_count": len(train_examples),
        "validation_pair_count": len(validation_examples),
        "update_budget": budget,
        "gpu": gpu,
        "storage": {
            "free_bytes": int(disk.free),
            "total_bytes": int(disk.total),
        },
        "resolved_config_pretty_text": config.pretty_text,
    }
    preflight_path = run_dir / "preflight.json"
    atomic_write_json(preflight_path, preflight)
    provenance["preflight_sha256"] = sha256_file(preflight_path)
    if device.type == "cuda":
        torch.cuda.empty_cache()
        torch.cuda.reset_peak_memory_stats(device)
    report = train_modes(
        modes=modes,
        train_examples=train_examples,
        validation_examples=validation_examples,
        cache=cache,
        config=training_config,
        provenance=provenance,
        work_dir=run_dir,
        device=device,
        epochs=epochs,
        patience=patience,
    )
    sequence_head_peak = (
        int(torch.cuda.max_memory_allocated(device)) if device.type == "cuda" else 0
    )
    feature_peak = int(feature_report.get("extraction_peak_vram_bytes", 0))
    report.update(
        {
            "status": "completed",
            "run_kind": run_kind,
            "feature_extraction": feature_report,
            "provenance": provenance,
            "train_pair_count": len(train_examples),
            "validation_pair_count": len(validation_examples),
            "device": str(device),
            "feature_extraction_peak_vram_bytes": feature_peak,
            "sequence_head_peak_vram_bytes": sequence_head_peak,
            "cuda_peak_vram_bytes": max(feature_peak, sequence_head_peak),
            "preflight": {
                "path": str(preflight_path),
                "sha256": sha256_file(preflight_path),
            },
            "update_budget": {
                **budget,
                "actual_updates_by_mode": {
                    mode: int(item.get("actual_updates", 0))
                    for mode, item in report["modes"].items()
                    if item.get("trajectory_id") is not None
                },
                "actual_updates_by_trajectory": {
                    trajectory_id: int(item["actual_updates"])
                    for trajectory_id, item in report.get("trajectories", {}).items()
                },
            },
        }
    )
    output_name = {
        "smoke": "smoke_training_report.json",
        "pilot": "pilot_training_report.json",
        "full": "full_training_report.json",
    }[run_kind]
    output = work_dir / output_name
    atomic_write_json(output, report)
    print(output)


if __name__ == "__main__":
    main()
