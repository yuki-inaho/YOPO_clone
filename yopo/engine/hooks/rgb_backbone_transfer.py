"""Strict runtime hook for transferring a 2D checkpoint into the RGB branch."""

from __future__ import annotations

import json
from pathlib import Path

import torch
from mmengine.hooks import Hook

from yopo.registry import HOOKS
from yopo.utils.partial_checkpoint import build_rgb_backbone_transfer_state


@HOOKS.register_module()
class RGBBackboneTransferHook(Hook):
    """Load only verified RGB B2 tensors before the first training batch."""

    priority = "VERY_HIGH"

    def __init__(self, checkpoint: str, report_filename: str = "partial_transfer_report.json") -> None:
        self.checkpoint = checkpoint
        self.report_filename = report_filename
        self._loaded = False

    def before_train(self, runner) -> None:
        if self._loaded:
            return

        model = runner.model.module if hasattr(runner.model, "module") else runner.model
        if not all(
            not parameter.requires_grad for parameter in model.backbone.rgb_backbone.parameters()
        ):
            raise RuntimeError("RGB backbone must be frozen before partial transfer")

        checkpoint_path = Path(self.checkpoint)
        if not checkpoint_path.is_file():
            raise FileNotFoundError(f"RGB transfer checkpoint not found: {checkpoint_path}")
        checkpoint = torch.load(checkpoint_path, map_location="cpu", weights_only=False)
        if "state_dict" not in checkpoint:
            raise KeyError(f"RGB transfer checkpoint lacks state_dict: {checkpoint_path}")

        target_state = model.state_dict()
        selection = build_rgb_backbone_transfer_state(checkpoint["state_dict"], target_state)
        incompatible = model.load_state_dict(selection.state_dict, strict=False)
        expected_missing = set(target_state).difference(selection.state_dict)
        actual_missing = set(incompatible.missing_keys)
        if actual_missing != expected_missing:
            raise RuntimeError(
                "partial transfer missing-key contract violated: "
                f"expected={sorted(expected_missing)}, actual={sorted(actual_missing)}"
            )
        if incompatible.unexpected_keys:
            raise RuntimeError(
                "partial transfer produced unexpected keys: "
                f"{sorted(incompatible.unexpected_keys)}"
            )

        report = {
            "checkpoint": str(checkpoint_path),
            "loaded_key_count": len(selection.state_dict),
            "loaded_target_prefix": "backbone.rgb_backbone.",
            "ignored_source_keys": list(selection.ignored_source_keys),
            "missing_target_rgb_keys": list(selection.missing_target_keys),
            "missing_non_rgb_key_count": len(actual_missing),
            "unexpected_keys": list(incompatible.unexpected_keys),
        }
        report_path = Path(runner.work_dir) / self.report_filename
        report_path.write_text(json.dumps(report, indent=2, sort_keys=True) + "\n", encoding="utf-8")
        self._loaded = True
