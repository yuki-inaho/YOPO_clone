"""Auditable transfer from an RGB OBB detector into an RGB-D OBB model."""

from __future__ import annotations

import json
from collections import Counter
from pathlib import Path

import torch
from mmengine.hooks import Hook

from yopo.registry import HOOKS
from yopo.utils.partial_checkpoint import build_rgb_backbone_transfer_state


_SHARED_ROOTS = {
    "bbox_head",
    "decoder",
    "encoder",
    "level_embed",
    "query_embedding",
    "reference_points_fc",
}


def _expand_query_embedding(
    source: torch.Tensor,
    target: torch.Tensor,
    seed: int,
) -> torch.Tensor:
    """Copy old queries and initialize added queries reproducibly."""
    if source.ndim != target.ndim or source.shape[1:] != target.shape[1:]:
        raise ValueError(
            "query embedding dimensions are incompatible: "
            f"source={tuple(source.shape)}, target={tuple(target.shape)}"
        )
    if source.shape[0] > target.shape[0]:
        raise ValueError(
            "query transfer does not truncate trained queries: "
            f"source={source.shape[0]}, target={target.shape[0]}"
        )
    expanded = target.detach().clone()
    expanded[: source.shape[0]].copy_(source)
    added = target.shape[0] - source.shape[0]
    if added:
        repeats = source[torch.arange(added) % source.shape[0]]
        generator = torch.Generator(device="cpu").manual_seed(seed)
        noise = torch.randn(
            repeats.shape,
            dtype=repeats.dtype,
            device="cpu",
            generator=generator,
        ).to(repeats.device)
        scale = source.float().std().clamp_min(1e-6).to(source.dtype) * 0.01
        expanded[source.shape[0] :].copy_(repeats + noise * scale)
    return expanded


def build_rgbd_obb_transfer_state(
    source_state: dict[str, torch.Tensor],
    target_state: dict[str, torch.Tensor],
    query_seed: int = 736512,
) -> tuple[dict[str, torch.Tensor], dict]:
    """Select only RGB-backbone and shared detector tensors."""
    rgb_selection = build_rgb_backbone_transfer_state(
        source_state,
        target_state,
    )
    selected = dict(rgb_selection.state_dict)
    ignored_roots: Counter[str] = Counter()
    loaded_roots: Counter[str] = Counter(
        {"backbone.rgb_backbone": len(rgb_selection.state_dict)}
    )

    for source_key, source_value in source_state.items():
        root = source_key.split(".", 1)[0]
        if root == "backbone":
            continue
        if root not in _SHARED_ROOTS:
            ignored_roots[root] += 1
            continue
        if source_key not in target_state:
            raise KeyError(f"shared detector target key is missing: {source_key}")

        target_value = target_state[source_key]
        if source_key == "query_embedding.weight":
            selected[source_key] = _expand_query_embedding(
                source_value,
                target_value,
                query_seed,
            )
        elif source_value.shape == target_value.shape:
            selected[source_key] = source_value
        else:
            raise ValueError(
                f"shared detector shape mismatch for {source_key}: "
                f"source={tuple(source_value.shape)}, "
                f"target={tuple(target_value.shape)}"
            )
        loaded_roots[root] += 1

    report = {
        "loaded_key_count": len(selected),
        "loaded_groups": dict(sorted(loaded_roots.items())),
        "ignored_groups": dict(sorted(ignored_roots.items())),
        "ignored_rgb_source_keys": list(rgb_selection.ignored_source_keys),
        "source_queries": int(source_state["query_embedding.weight"].shape[0]),
        "target_queries": int(target_state["query_embedding.weight"].shape[0]),
        "query_expansion_seed": query_seed,
    }
    return selected, report


def build_rgbd_feature_transfer_state(
    source_state: dict[str, torch.Tensor],
    target_state: dict[str, torch.Tensor],
    compatible_detector_roots: tuple[str, ...] = (),
) -> tuple[dict[str, torch.Tensor], dict]:
    """Transfer the RGB backbone and an exactly compatible pretrained neck.

    This narrower contract is intended for detector topology changes where
    query/decoder/head tensors must not leak into the new model.
    """
    rgb_selection = build_rgb_backbone_transfer_state(source_state, target_state)
    selected = dict(rgb_selection.state_dict)
    neck_source = {
        key: value for key, value in source_state.items()
        if key.startswith('neck.')
    }
    if not neck_source:
        raise KeyError('source checkpoint has no neck tensors')

    missing = sorted(key for key in neck_source if key not in target_state)
    mismatched = sorted(
        key for key, value in neck_source.items()
        if key in target_state and value.shape != target_state[key].shape
    )
    if missing or mismatched:
        raise ValueError(
            'pretrained neck is not exactly target-compatible: '
            f'missing={missing}, shape_mismatch={mismatched}')
    selected.update(neck_source)
    detector_counts: Counter[str] = Counter()
    for key, value in source_state.items():
        root = key.split('.', 1)[0]
        if root not in compatible_detector_roots:
            continue
        if key not in target_state:
            continue
        if value.shape != target_state[key].shape:
            raise ValueError(
                f'compatible detector tensor shape mismatch for {key}: '
                f'{tuple(value.shape)} vs {tuple(target_state[key].shape)}')
        selected[key] = value
        detector_counts[root] += 1
    loaded_groups = {
        'backbone.rgb_backbone': len(rgb_selection.state_dict),
        'neck': len(neck_source),
    }
    loaded_groups.update(dict(sorted(detector_counts.items())))
    report = {
        'loaded_key_count': len(selected),
        'loaded_groups': loaded_groups,
        'ignored_rgb_source_keys': list(rgb_selection.ignored_source_keys),
    }
    return selected, report


@HOOKS.register_module()
class RGBDFeatureTransferHook(Hook):
    """Strictly load only the pretrained RGB backbone and neck."""

    priority = 'VERY_HIGH'

    def __init__(self,
                 checkpoint: str,
                 report_filename: str = 'rgbd_feature_transfer_report.json',
                 compatible_detector_roots=()):
        self.checkpoint = checkpoint
        self.report_filename = report_filename
        self.compatible_detector_roots = tuple(compatible_detector_roots)
        self._loaded = False

    def before_train(self, runner) -> None:
        if self._loaded:
            return
        checkpoint_path = Path(self.checkpoint)
        if not checkpoint_path.is_file():
            raise FileNotFoundError(
                f'RGB feature checkpoint not found: {checkpoint_path}')
        checkpoint = torch.load(
            checkpoint_path, map_location='cpu', weights_only=False)
        if 'state_dict' not in checkpoint:
            raise KeyError(
                f'RGB checkpoint lacks state_dict: {checkpoint_path}')
        model = runner.model.module if hasattr(runner.model, 'module') else runner.model
        selected, report = build_rgbd_feature_transfer_state(
            checkpoint['state_dict'], model.state_dict(),
            self.compatible_detector_roots)
        incompatible = model.load_state_dict(selected, strict=False)
        if incompatible.unexpected_keys:
            raise RuntimeError(
                f'unexpected feature-transfer keys: {incompatible.unexpected_keys}')
        report.update(
            checkpoint=str(checkpoint_path.resolve()),
            missing_target_key_count=len(incompatible.missing_keys),
            unexpected_keys=list(incompatible.unexpected_keys),
        )
        report_path = Path(runner.work_dir) / self.report_filename
        report_path.write_text(
            json.dumps(report, indent=2, sort_keys=True) + '\n',
            encoding='utf-8')
        self._loaded = True


@HOOKS.register_module()
class RGBDOBBTransferHook(Hook):
    """Load compatible RGB detector tensors before RGB-D OBB training."""

    priority = "VERY_HIGH"

    def __init__(
        self,
        checkpoint: str,
        report_filename: str = "rgbd_obb_transfer_report.json",
        query_seed: int = 736512,
    ) -> None:
        self.checkpoint = checkpoint
        self.report_filename = report_filename
        self.query_seed = int(query_seed)
        self._loaded = False

    def before_train(self, runner) -> None:
        if self._loaded:
            return
        checkpoint_path = Path(self.checkpoint)
        if not checkpoint_path.is_file():
            raise FileNotFoundError(
                f"RGB OBB transfer checkpoint not found: {checkpoint_path}"
            )
        checkpoint = torch.load(
            checkpoint_path,
            map_location="cpu",
            weights_only=False,
        )
        if "state_dict" not in checkpoint:
            raise KeyError(
                f"RGB OBB checkpoint lacks state_dict: {checkpoint_path}"
            )

        model = runner.model.module if hasattr(runner.model, "module") else runner.model
        target_state = model.state_dict()
        selected, report = build_rgbd_obb_transfer_state(
            checkpoint["state_dict"],
            target_state,
            self.query_seed,
        )
        incompatible = model.load_state_dict(selected, strict=False)
        # PyTorch intentionally omits BatchNorm counters from missing_keys.
        expected_missing = {
            key
            for key in set(target_state).difference(selected)
            if not key.endswith("num_batches_tracked")
        }
        if set(incompatible.missing_keys) != expected_missing:
            raise RuntimeError("RGB-D OBB transfer missing-key contract violated")
        if incompatible.unexpected_keys:
            raise RuntimeError(
                "RGB-D OBB transfer produced unexpected keys: "
                f"{sorted(incompatible.unexpected_keys)}"
            )

        report.update(
            checkpoint=str(checkpoint_path.resolve()),
            missing_target_key_count=len(incompatible.missing_keys),
            unexpected_keys=list(incompatible.unexpected_keys),
        )
        report_path = Path(runner.work_dir) / self.report_filename
        report_path.write_text(
            json.dumps(report, indent=2, sort_keys=True) + "\n",
            encoding="utf-8",
        )
        self._loaded = True
