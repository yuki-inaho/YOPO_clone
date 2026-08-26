#!/usr/bin/env python
"""Create a weight-only checkpoint for the compact B1+B0 YOPO config."""

from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path

import torch
from mmengine.config import Config

from yopo.pruning import (
    build_compact_yopo_transfer_state,
    load_pruning_choices,
)
from yopo.registry import MODELS
from yopo.utils import register_all_modules


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("source_checkpoint", type=Path)
    parser.add_argument("target_config", type=Path)
    parser.add_argument("output", type=Path)
    parser.add_argument(
        "--pruning-plan",
        type=Path,
        help="Optional JSON choices from calibrated Group Fisher/Taylor importance.",
    )
    parser.add_argument(
        "--allow-smoke-plan",
        action="store_true",
        help=(
            "Allow a purpose=smoke plan for wiring tests. Never use this flag "
            "to claim production-calibrated pruning."
        ),
    )
    parser.add_argument(
        "--report",
        type=Path,
        help="Defaults to <output>.report.json.",
    )
    return parser.parse_args()


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def main() -> None:
    args = parse_args()
    if not args.source_checkpoint.is_file():
        raise FileNotFoundError(args.source_checkpoint)
    if not args.target_config.is_file():
        raise FileNotFoundError(args.target_config)

    register_all_modules()
    cfg = Config.fromfile(args.target_config)
    model = MODELS.build(cfg.model)
    checkpoint = torch.load(
        args.source_checkpoint,
        map_location="cpu",
        weights_only=False,
    )
    if "state_dict" not in checkpoint:
        raise KeyError("source checkpoint has no state_dict")
    choices = load_pruning_choices(
        args.pruning_plan,
        allow_smoke=args.allow_smoke_plan,
    )
    result = build_compact_yopo_transfer_state(
        checkpoint["state_dict"],
        model.state_dict(),
        pruning_choices=choices,
    )
    incompatible = model.load_state_dict(result.state_dict, strict=False)
    if incompatible.unexpected_keys:
        raise RuntimeError(
            f"transplanted state has unexpected keys: {incompatible.unexpected_keys}"
        )

    args.output.parent.mkdir(parents=True, exist_ok=True)
    report_path = args.report or args.output.with_suffix(
        args.output.suffix + ".report.json"
    )
    report = dict(result.report)
    pruning_plan_sha256 = (
        sha256(args.pruning_plan) if args.pruning_plan is not None else None
    )
    pruning_plan_metadata = None
    if args.pruning_plan is not None:
        plan_payload = json.loads(args.pruning_plan.read_text(encoding="utf-8"))
        pruning_plan_metadata = plan_payload.get("metadata")
    report.update(
        source_checkpoint=str(args.source_checkpoint.resolve()),
        source_sha256=sha256(args.source_checkpoint),
        target_config=str(args.target_config.resolve()),
        target_config_sha256=sha256(args.target_config),
        target_parameter_count=sum(parameter.numel() for parameter in model.parameters()),
        missing_target_key_count=len(incompatible.missing_keys),
        pruning_plan=(
            str(args.pruning_plan.resolve()) if args.pruning_plan is not None else None
        ),
        pruning_plan_sha256=pruning_plan_sha256,
        pruning_plan_metadata=pruning_plan_metadata,
    )
    torch.save(
        {
            "state_dict": dict(result.state_dict),
            "meta": {
                "checkpoint_type": "compact_yopo_partial_weights",
                "source_checkpoint": str(args.source_checkpoint.resolve()),
                "source_sha256": report["source_sha256"],
                "target_config": str(args.target_config.resolve()),
                "target_config_sha256": report["target_config_sha256"],
                "pruning_plan": report["pruning_plan"],
                "pruning_plan_sha256": pruning_plan_sha256,
                "pruning_plan_metadata": pruning_plan_metadata,
            },
        },
        args.output,
    )
    report_path.parent.mkdir(parents=True, exist_ok=True)
    report_path.write_text(
        json.dumps(report, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    print(json.dumps({
        "output": str(args.output.resolve()),
        "report": str(report_path.resolve()),
        "loaded_key_count": report["loaded_key_count"],
        "target_parameter_count": report["target_parameter_count"],
    }, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
