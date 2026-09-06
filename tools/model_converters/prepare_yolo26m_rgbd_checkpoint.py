#!/usr/bin/env python
"""Create audited YOLO26 n/s/m YOPO RGB-D initial weights."""

from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path
from typing import Any

from mmengine.config import Config
import numpy as np
import torch

from yopo.registry import MODELS
from yopo.utils import register_all_modules
from yopo.utils.jax_yolo26_transfer import convert_jax_yolo26_backbone_arrays
from yopo.utils.rotated_yolo26_transfer import (
    convert_rotated_yolo26_backbone_arrays,
)
from yopo.utils.yolo26_rgbd_initialization import select_stage8_reuse_state


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _expected_fresh_target(key: str) -> bool:
    return (
        key.startswith("backbone.depth_adapters.")
        or key == "backbone.depth_beta"
        or key.startswith("neck.projections.")
        or (
            key.startswith("backbone.rgb_backbone.")
            and key.endswith("num_batches_tracked")
        )
    )


def _nested(mapping: dict[str, Any], *path: str) -> Any:
    value: Any = mapping
    for key in path:
        if not isinstance(value, dict):
            return None
        value = value.get(key)
    return value


def _source_scale(metadata: dict[str, Any], source_format: str) -> str | None:
    if source_format == "deim":
        return _nested(metadata, "config", "model", "backbone", "variant")
    return _nested(metadata, "config", "model", "yolo26_scale") or _nested(
        metadata, "model_config", "yolo26_scale"
    )


def prepare_checkpoint(
    backbone_checkpoint: Path,
    stage8_checkpoint: Path,
    target_config: Path,
    output: Path,
    *,
    weights: str = "ema",
    scale: str = "m",
    source_format: str = "auto",
    seed: int = 3407,
) -> tuple[Path, Path]:
    """Build a complete weight-only checkpoint with every origin audited."""

    if scale not in {"n", "s", "m"}:
        raise ValueError("scale must be one of 'n', 's', or 'm'")
    if source_format not in {"auto", "deim", "rotated"}:
        raise ValueError("source_format must be 'auto', 'deim', or 'rotated'")
    if source_format == "auto":
        source_format = "deim" if backbone_checkpoint.is_dir() else "rotated"
    for required in (stage8_checkpoint, target_config):
        if not required.is_file():
            raise FileNotFoundError(required)

    torch.manual_seed(seed)
    register_all_modules()
    config = Config.fromfile(str(target_config))
    model = MODELS.build(config.model)
    target_state = model.state_dict()
    target_scale = getattr(model.backbone.rgb_backbone, "scale", None)
    if target_scale != scale:
        raise ValueError(
            f"target config scale {target_scale!r} does not match requested {scale!r}"
        )

    source_metadata: dict[str, Any]
    source_sha256: str
    source_manifest_sha256: str | None = None
    if source_format == "deim":
        arrays_path = backbone_checkpoint / "arrays.npz"
        manifest_path = backbone_checkpoint / "manifest.json"
        for required in (arrays_path, manifest_path):
            if not required.is_file():
                raise FileNotFoundError(required)
        manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
        source_sha256 = _sha256(arrays_path)
        if source_sha256 != manifest.get("array_sha256"):
            raise ValueError("DEIM arrays SHA-256 does not match its manifest")
        source_manifest_sha256 = _sha256(manifest_path)
        source_metadata = manifest.get("metadata") or {}
        with np.load(arrays_path, allow_pickle=False) as source_arrays:
            yolo_state, yolo_report = convert_jax_yolo26_backbone_arrays(
                source_arrays,
                target_state,
                weights=weights,
                strict=True,
            )
    else:
        if not backbone_checkpoint.is_file():
            raise FileNotFoundError(backbone_checkpoint)
        source_sha256 = _sha256(backbone_checkpoint)
        with np.load(backbone_checkpoint, allow_pickle=False) as source_arrays:
            if "__metadata__" not in source_arrays.files:
                raise ValueError("Rotated checkpoint metadata is missing")
            source_metadata = json.loads(
                bytes(source_arrays["__metadata__"].tobytes()).decode("utf-8")
            )
            yolo_state, yolo_report = convert_rotated_yolo26_backbone_arrays(
                source_arrays,
                target_state,
                weights=weights,
                strict=True,
            )
    actual_source_scale = _source_scale(source_metadata, source_format)
    if actual_source_scale != scale:
        raise ValueError(
            f"{source_format} source scale {actual_source_scale!r} "
            f"does not match requested {scale!r}"
        )
    expected_leaf_count = {"n": 200, "s": 200, "m": 250}[scale]
    if len(yolo_report.mapped) != expected_leaf_count:
        raise ValueError(
            f"{scale} backbone mapped {len(yolo_report.mapped)} leaves, "
            f"expected {expected_leaf_count}"
        )

    stage8 = torch.load(stage8_checkpoint, map_location="cpu", weights_only=False)
    if not isinstance(stage8, dict) or not isinstance(stage8.get("state_dict"), dict):
        raise ValueError("stage8 checkpoint must contain a state_dict mapping")
    reused_state, reuse_report = select_stage8_reuse_state(
        stage8["state_dict"],
        target_state,
        strict=True,
    )
    overlap = sorted(set(yolo_state).intersection(reused_state))
    if overlap:
        raise ValueError(f"JAX and stage8 initialization overlap: {overlap[:3]}")
    selected = {**reused_state, **yolo_state}
    expected_missing = {key for key in target_state if _expected_fresh_target(key)}
    actual_missing = set(target_state).difference(selected)
    if actual_missing != expected_missing:
        unexpected = sorted(actual_missing.difference(expected_missing))
        overselected = sorted(expected_missing.difference(actual_missing))
        raise ValueError(
            "initialization boundary drift: "
            f"unexpected_missing={unexpected[:3]}, overselected={overselected[:3]}"
        )

    incompatible = model.load_state_dict(selected, strict=False)
    # PyTorch intentionally suppresses missing ``num_batches_tracked`` for
    # backward-compatible BatchNorm loading, so audit those defaults directly
    # through ``actual_missing`` above and compare the reported subset here.
    reported_missing = {
        key for key in expected_missing if not key.endswith("num_batches_tracked")
    }
    if (
        incompatible.unexpected_keys
        or set(incompatible.missing_keys) != reported_missing
    ):
        raise RuntimeError(
            "model load audit failed: "
            f"missing={incompatible.missing_keys[:3]}, "
            f"unexpected={incompatible.unexpected_keys[:3]}"
        )
    if not torch.equal(model.backbone.depth_beta.detach().cpu(), torch.zeros(3)):
        raise RuntimeError("depth_beta must remain exactly zero at initialization")

    final_state = {
        key: value.detach().cpu().clone() for key, value in model.state_dict().items()
    }
    metadata: dict[str, Any] = {
        "format": "yopo_yolo26_rgbd_initial_v2",
        "architecture": f"YOLO26{scale} layers 0-10 RGB + HGNetV2-B0 depth",
        "scale": scale,
        "backbone_source_format": source_format,
        "backbone_weights": weights,
        "backbone_step": source_metadata.get("step"),
        "backbone_checkpoint_sha256": source_sha256,
        "backbone_manifest_sha256": source_manifest_sha256,
        "stage8_checkpoint_sha256": _sha256(stage8_checkpoint),
        "target_config": target_config.name,
        "target_config_sha256": _sha256(target_config),
        "fresh_seed": seed,
        "mapped_backbone_leaf_count": len(yolo_report.mapped),
        "mapped_stage8_leaf_count": len(reuse_report.mapped),
        "fresh_target_leaf_count": len(expected_missing),
        "state_leaf_count": len(final_state),
        "optimizer_state": "excluded",
    }
    output.parent.mkdir(parents=True, exist_ok=True)
    torch.save({"meta": metadata, "state_dict": final_state}, output)

    report = {
        **metadata,
        "output": output.name,
        "output_sha256": _sha256(output),
        "fresh_target": sorted(expected_missing),
        "backbone_transfer": yolo_report.to_dict(),
        "stage8_reuse": reuse_report.to_dict(),
    }
    report_path = output.with_suffix(output.suffix + ".json")
    report_path.write_text(
        json.dumps(report, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    return output, report_path


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--backbone-checkpoint",
        "--jax-checkpoint",
        dest="backbone_checkpoint",
        type=Path,
        required=True,
    )
    parser.add_argument("--stage8-checkpoint", type=Path, required=True)
    parser.add_argument("--target-config", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--weights", choices=("ema", "params"), default="ema")
    parser.add_argument("--scale", choices=("n", "s", "m"), default="m")
    parser.add_argument(
        "--source-format", choices=("auto", "deim", "rotated"), default="auto"
    )
    parser.add_argument("--seed", type=int, default=3407)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    output, report = prepare_checkpoint(
        args.backbone_checkpoint.expanduser().resolve(),
        args.stage8_checkpoint.expanduser().resolve(),
        args.target_config.expanduser().resolve(),
        args.output.expanduser().resolve(),
        weights=args.weights,
        scale=args.scale,
        source_format=args.source_format,
        seed=args.seed,
    )
    print(json.dumps({"output": str(output), "report": str(report)}, indent=2))


if __name__ == "__main__":
    main()
