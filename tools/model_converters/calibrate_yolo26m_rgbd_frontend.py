#!/usr/bin/env python
"""Calibrate the seven fresh YOLO26-to-YOPO RGB-D boundary leaves."""

from __future__ import annotations

import argparse
import copy
import hashlib
import json
import math
from pathlib import Path
from typing import Any

from mmengine.config import Config
from mmengine.runner import Runner
import torch
import torch.nn.functional as F
from torch import Tensor, nn

from yopo.registry import MODELS
from yopo.utils import register_all_modules
from yopo.utils.yolo26_frontend_calibration import (
    factor_depth_adapter,
    fit_ridge_projection_from_moments,
)


BOUNDARY_KEYS = {
    "backbone.depth_beta",
    *(f"backbone.depth_adapters.{level}.weight" for level in range(3)),
    *(f"neck.projections.{level}.weight" for level in range(3)),
}


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _load_checkpoint(path: Path) -> dict[str, Any]:
    if not path.is_file():
        raise FileNotFoundError(path)
    checkpoint = torch.load(path, map_location="cpu", weights_only=False)
    if not isinstance(checkpoint, dict) or not isinstance(
        checkpoint.get("state_dict"), dict
    ):
        raise ValueError(f"checkpoint has no state_dict: {path.name}")
    return checkpoint


def _load_component(
    module: nn.Module,
    state: dict[str, Tensor],
    prefix: str,
) -> None:
    component_state = {
        key.removeprefix(prefix): value
        for key, value in state.items()
        if key.startswith(prefix)
    }
    incompatible = module.load_state_dict(component_state, strict=True)
    if incompatible.missing_keys or incompatible.unexpected_keys:
        raise RuntimeError(f"strict component load failed for {prefix}: {incompatible}")


def _deterministic_train_loader(
    config: Config,
    *,
    batch_size: int,
    num_workers: int,
    seed: int,
):
    loader_config = copy.deepcopy(config.train_dataloader)
    loader_config.batch_size = batch_size
    loader_config.num_workers = num_workers
    loader_config.persistent_workers = num_workers > 0
    split = str(loader_config.dataset.get("split", "")).lower()
    if "train" not in split:
        raise ValueError(
            f"calibration loader must use a named train split, got {split!r}"
        )
    loader_config.dataset.pipeline = [
        transform
        for transform in loader_config.dataset.pipeline
        if not str(transform.get("type", "")).startswith("Random")
    ]
    return Runner.build_dataloader(loader_config, seed=seed)


def _sample_aligned(
    *features: Tensor,
    maximum: int,
    generator: torch.Generator,
) -> tuple[Tensor, ...]:
    if len(features) < 2 or any(feature.ndim != 4 for feature in features):
        raise ValueError("calibration features must be NCHW")
    reference = features[0]
    if any(
        feature.shape[0] != reference.shape[0]
        or feature.shape[-2:] != reference.shape[-2:]
        for feature in features[1:]
    ):
        raise ValueError(
            "calibration feature grids differ: "
            + ", ".join(str(tuple(feature.shape)) for feature in features)
        )
    flattened = [
        feature.permute(0, 2, 3, 1).reshape(-1, feature.shape[1])
        for feature in features
    ]
    count = min(maximum, flattened[0].shape[0])
    indices = torch.randperm(flattened[0].shape[0], generator=generator)[:count]
    indices = indices.to(device=reference.device)
    return tuple(feature[indices].float() for feature in flattened)


def _relative_rmse(
    prediction: Tensor,
    target: Tensor,
) -> tuple[float, float, int]:
    difference = prediction.float() - target.float()
    squared_error = float(difference.square().sum().cpu())
    squared_target = float(target.float().square().sum().cpu())
    return squared_error, squared_target, target.numel()


def _fit_error(
    gram: Tensor,
    cross: Tensor,
    target_square: float,
    projection: Tensor,
) -> float:
    coefficients = projection.T
    error_square = (
        target_square
        - 2.0 * float((coefficients * cross).sum())
        + float((coefficients * (gram @ coefficients)).sum())
    )
    return math.sqrt(max(error_square, 0.0) / max(target_square, 1e-30))


def calibrate_checkpoint(
    student_config_path: Path,
    student_checkpoint_path: Path,
    teacher_config_path: Path,
    teacher_checkpoint_path: Path,
    output_path: Path,
    *,
    device: str = "cuda",
    calibration_batches: int = 16,
    holdout_batches: int = 4,
    batch_size: int = 4,
    num_workers: int = 4,
    samples_per_level: int = 2048,
    ridge: float = 1e-4,
    depth_ridge: float = 1e-6,
    seed: int = 20260906,
    scale: str | None = None,
) -> tuple[Path, Path]:
    """Calibrate and save a complete weight-only student checkpoint."""

    positive = {
        "calibration_batches": calibration_batches,
        "holdout_batches": holdout_batches,
        "batch_size": batch_size,
        "samples_per_level": samples_per_level,
    }
    if any(not isinstance(value, int) or value <= 0 for value in positive.values()):
        raise ValueError(f"positive integer arguments required: {positive}")
    if num_workers < 0:
        raise ValueError("num_workers must be non-negative")

    register_all_modules()
    torch.manual_seed(seed)
    generator = torch.Generator().manual_seed(seed)
    torch_device = torch.device(device)

    student_config = Config.fromfile(str(student_config_path))
    teacher_config = Config.fromfile(str(teacher_config_path))
    student_checkpoint = _load_checkpoint(student_checkpoint_path)
    teacher_checkpoint = _load_checkpoint(teacher_checkpoint_path)
    student_state = student_checkpoint["state_dict"]
    teacher_state = teacher_checkpoint["state_dict"]

    student_backbone = (
        MODELS.build(student_config.model.backbone).to(torch_device).eval()
    )
    student_neck = MODELS.build(student_config.model.neck).to(torch_device).eval()
    teacher_backbone = (
        MODELS.build(teacher_config.model.backbone).to(torch_device).eval()
    )
    teacher_neck = MODELS.build(teacher_config.model.neck).to(torch_device).eval()
    preprocessor = MODELS.build(student_config.model.data_preprocessor).to(torch_device)
    preprocessor.eval()
    _load_component(student_backbone, student_state, "backbone.")
    _load_component(student_neck, student_state, "neck.")
    _load_component(teacher_backbone, teacher_state, "backbone.")
    _load_component(teacher_neck, teacher_state, "neck.")

    if student_backbone.num_scales != 3 or teacher_backbone.num_scales != 3:
        raise ValueError("calibration requires exactly three feature levels")
    student_rgb = student_backbone.rgb_backbone
    student_scale = getattr(student_rgb, "scale", None)
    if student_scale not in {"n", "s", "m"}:
        raise ValueError(f"student RGB backbone has invalid scale {student_scale!r}")
    if scale is not None and scale != student_scale:
        raise ValueError(
            f"student config scale {student_scale!r} does not match requested {scale!r}"
        )
    student_indices = sorted(student_rgb.return_idx)
    student_channels = tuple(
        student_rgb._out_channels[index] for index in student_indices
    )
    projection_channels = tuple(
        int(layer.weight.shape[0]) for layer in student_neck.projections
    )
    if tuple(int(layer.weight.shape[1]) for layer in student_neck.projections) != (
        student_channels
    ):
        raise ValueError("student neck projection input channels do not match RGB taps")
    if len(set(projection_channels)) != 1:
        raise ValueError("student neck projections must have one shared output width")
    loader = _deterministic_train_loader(
        student_config,
        batch_size=batch_size,
        num_workers=num_workers,
        seed=seed,
    )
    iterator = iter(loader)
    grams = [
        torch.zeros(channels, channels, dtype=torch.float64)
        for channels in student_channels
    ]
    crosses = [
        torch.zeros(channels, output, dtype=torch.float64)
        for channels, output in zip(student_channels, projection_channels)
    ]
    target_squares = [0.0, 0.0, 0.0]
    sample_counts = [0, 0, 0]

    with torch.inference_mode():
        for _ in range(calibration_batches):
            batch = next(iterator)
            inputs = preprocessor(batch, training=False)["inputs"]
            teacher_rgb = teacher_backbone.rgb_backbone(inputs[:, :3])
            student_rgb = student_backbone.rgb_backbone(inputs[:, :3])
            for level in range(3):
                target = teacher_neck.projections[level](teacher_rgb[level])
                source_samples, target_samples = _sample_aligned(
                    student_rgb[level],
                    target,
                    maximum=samples_per_level,
                    generator=generator,
                )
                grams[level] += (source_samples.T @ source_samples).double().cpu()
                crosses[level] += (source_samples.T @ target_samples).double().cpu()
                target_squares[level] += float(
                    target_samples.square().sum().double().cpu()
                )
                sample_counts[level] += source_samples.shape[0]

    fitted_projections: list[Tensor] = []
    fitted_adapters: list[Tensor] = []
    fit_reports: list[dict[str, Any]] = []
    for level in range(3):
        fitted = fit_ridge_projection_from_moments(
            grams[level], crosses[level], ridge=ridge
        )
        student_projection = fitted
        teacher_projection = (
            teacher_neck.projections[level]
            .weight.detach()
            .cpu()
            .squeeze(-1)
            .squeeze(-1)
        ).double()
        teacher_adapter = (
            teacher_backbone.depth_adapters[level]
            .weight.detach()
            .cpu()
            .squeeze(-1)
            .squeeze(-1)
        ).double()
        adapter = factor_depth_adapter(
            student_projection,
            teacher_projection,
            teacher_adapter,
            ridge=depth_ridge,
        )
        projected_depth_target = teacher_projection @ teacher_adapter
        projected_depth_actual = student_projection @ adapter
        depth_relative_error = float(
            (projected_depth_actual - projected_depth_target).norm()
            / projected_depth_target.norm().clamp_min(1e-30)
        )
        singular_values = torch.linalg.svdvals(student_projection)
        fitted_projections.append(fitted.float())
        fitted_adapters.append(adapter.float())
        fit_reports.append(
            {
                "level": level,
                "sample_count": sample_counts[level],
                "rgb_fit_relative_rmse": _fit_error(
                    grams[level],
                    crosses[level],
                    target_squares[level],
                    fitted,
                ),
                "depth_factor_relative_error": depth_relative_error,
                "projection_rank": int(torch.linalg.matrix_rank(fitted)),
                "projection_condition": float(
                    singular_values.max() / singular_values.min().clamp_min(1e-30)
                ),
            }
        )

    original_projections = [
        layer.weight.detach().clone() for layer in student_neck.projections
    ]
    original_adapters = [
        layer.weight.detach().clone() for layer in student_backbone.depth_adapters
    ]
    original_beta = student_backbone.depth_beta.detach().clone()
    with torch.no_grad():
        for level in range(3):
            student_neck.projections[level].weight.copy_(
                fitted_projections[level].reshape_as(
                    student_neck.projections[level].weight
                )
            )
            student_backbone.depth_adapters[level].weight.copy_(
                fitted_adapters[level].reshape_as(
                    student_backbone.depth_adapters[level].weight
                )
            )
        student_backbone.depth_beta.copy_(teacher_backbone.depth_beta)

    holdout_error = [
        {"initial_square": 0.0, "calibrated_square": 0.0, "target_square": 0.0}
        for _ in range(3)
    ]
    with torch.inference_mode():
        for _ in range(holdout_batches):
            batch = next(iterator)
            inputs = preprocessor(batch, training=False)["inputs"]
            teacher_fused, _ = teacher_backbone._forward_modalities(inputs)
            student_rgb = student_backbone.rgb_backbone(inputs[:, :3])
            student_depth = student_backbone.depth_backbone(inputs[:, 3:4])
            for level in range(3):
                target = teacher_neck.projections[level](teacher_fused[level])
                initial_fused = student_rgb[level] + original_beta[level].to(
                    student_rgb[level].dtype
                ) * F.conv2d(student_depth[level], original_adapters[level])
                initial = F.conv2d(initial_fused, original_projections[level])
                calibrated_fused = student_rgb[level] + student_backbone.depth_beta[
                    level
                ].to(student_rgb[level].dtype) * student_backbone.depth_adapters[level](
                    student_depth[level]
                )
                calibrated = student_neck.projections[level](calibrated_fused)
                initial_samples, calibrated_samples, target_samples = _sample_aligned(
                    initial,
                    calibrated,
                    target,
                    maximum=samples_per_level,
                    generator=generator,
                )
                initial_square, initial_target_square, _ = _relative_rmse(
                    initial_samples, target_samples
                )
                calibrated_square, calibrated_target_square, _ = _relative_rmse(
                    calibrated_samples, target_samples
                )
                values = holdout_error[level]
                values["initial_square"] += initial_square
                values["calibrated_square"] += calibrated_square
                values["target_square"] += 0.5 * (
                    initial_target_square + calibrated_target_square
                )

    holdout_reports = []
    for level, values in enumerate(holdout_error):
        denominator = max(values["target_square"], 1e-30)
        initial_relative = math.sqrt(values["initial_square"] / denominator)
        calibrated_relative = math.sqrt(values["calibrated_square"] / denominator)
        if not calibrated_relative < initial_relative:
            raise RuntimeError(
                f"holdout projection did not improve at level {level}: "
                f"{initial_relative} -> {calibrated_relative}"
            )
        holdout_reports.append(
            {
                "level": level,
                "initial_relative_rmse": initial_relative,
                "calibrated_relative_rmse": calibrated_relative,
                "improvement_fraction": 1.0 - calibrated_relative / initial_relative,
            }
        )

    calibrated_state = {
        key: value.detach().cpu().clone() for key, value in student_state.items()
    }
    calibrated_state["backbone.depth_beta"] = (
        student_backbone.depth_beta.detach().cpu().clone()
    )
    for level in range(3):
        calibrated_state[f"backbone.depth_adapters.{level}.weight"] = (
            student_backbone.depth_adapters[level].weight.detach().cpu().clone()
        )
        calibrated_state[f"neck.projections.{level}.weight"] = (
            student_neck.projections[level].weight.detach().cpu().clone()
        )
    changed = {
        key
        for key in calibrated_state
        if not torch.equal(calibrated_state[key], student_state[key])
    }
    if changed != BOUNDARY_KEYS:
        raise RuntimeError(
            "calibration boundary drift: "
            f"missing={sorted(BOUNDARY_KEYS - changed)}, "
            f"extra={sorted(changed - BOUNDARY_KEYS)[:3]}"
        )
    if not all(
        bool(torch.isfinite(value).all()) for value in calibrated_state.values()
    ):
        raise RuntimeError("calibrated state contains non-finite values")

    strict_model = MODELS.build(student_config.model)
    incompatible = strict_model.load_state_dict(calibrated_state, strict=True)
    if incompatible.missing_keys or incompatible.unexpected_keys:
        raise RuntimeError(f"strict calibrated model load failed: {incompatible}")

    metadata = {
        "format": "yopo_yolo26_rgbd_frontend_calibrated_v2",
        "scale": student_scale,
        "rgb_channels": list(student_channels),
        "projection_channels": list(projection_channels),
        "student_checkpoint": student_checkpoint_path.name,
        "student_checkpoint_sha256": _sha256(student_checkpoint_path),
        "teacher_checkpoint": teacher_checkpoint_path.name,
        "teacher_checkpoint_sha256": _sha256(teacher_checkpoint_path),
        "student_config": student_config_path.name,
        "student_config_sha256": _sha256(student_config_path),
        "teacher_config": teacher_config_path.name,
        "teacher_config_sha256": _sha256(teacher_config_path),
        "split": "train",
        "labels_used": False,
        "random_transforms": False,
        "seed": seed,
        "calibration_batches": calibration_batches,
        "holdout_batches": holdout_batches,
        "batch_size": batch_size,
        "samples_per_level": samples_per_level,
        "ridge": ridge,
        "depth_ridge": depth_ridge,
        "changed_keys": sorted(changed),
        "state_leaf_count": len(calibrated_state),
        "optimizer_state": "excluded",
    }
    output_path.parent.mkdir(parents=True, exist_ok=True)
    torch.save({"meta": metadata, "state_dict": calibrated_state}, output_path)
    report = {
        **metadata,
        "output": output_path.name,
        "output_sha256": _sha256(output_path),
        "fit": fit_reports,
        "holdout": holdout_reports,
    }
    report_path = output_path.with_suffix(output_path.suffix + ".json")
    report_path.write_text(
        json.dumps(report, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )
    return output_path, report_path


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--student-config", type=Path, required=True)
    parser.add_argument("--student-checkpoint", type=Path, required=True)
    parser.add_argument("--teacher-config", type=Path, required=True)
    parser.add_argument("--teacher-checkpoint", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--calibration-batches", type=int, default=16)
    parser.add_argument("--holdout-batches", type=int, default=4)
    parser.add_argument("--batch-size", type=int, default=4)
    parser.add_argument("--num-workers", type=int, default=4)
    parser.add_argument("--samples-per-level", type=int, default=2048)
    parser.add_argument("--ridge", type=float, default=1e-4)
    parser.add_argument("--depth-ridge", type=float, default=1e-6)
    parser.add_argument("--seed", type=int, default=20260906)
    parser.add_argument("--scale", choices=("n", "s", "m"))
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    output, report = calibrate_checkpoint(
        args.student_config.expanduser().resolve(),
        args.student_checkpoint.expanduser().resolve(),
        args.teacher_config.expanduser().resolve(),
        args.teacher_checkpoint.expanduser().resolve(),
        args.output.expanduser().resolve(),
        device=args.device,
        calibration_batches=args.calibration_batches,
        holdout_batches=args.holdout_batches,
        batch_size=args.batch_size,
        num_workers=args.num_workers,
        samples_per_level=args.samples_per_level,
        ridge=args.ridge,
        depth_ridge=args.depth_ridge,
        seed=args.seed,
        scale=args.scale,
    )
    print(json.dumps({"output": str(output), "report": str(report)}, indent=2))


if __name__ == "__main__":
    main()
