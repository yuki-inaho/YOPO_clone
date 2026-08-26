"""Strict composition of a 3D pose checkpoint and a 2D RGB-D feature source."""

from __future__ import annotations

import json
from collections import Counter
from pathlib import Path

import torch
from mmengine.hooks import Hook

from yopo.registry import HOOKS

from .rgbd_obb_transfer import _expand_query_embedding


_FEATURE_PREFIXES = ('backbone.', 'neck.')
_REBUILT_POSE_PREFIXES = ('bbox_head.distillation_teacher.',)
_DETECTION_REPAIR_PREFIXES = (
    'encoder.',
    'decoder.',
    'level_embed',
    'memory_trans_fc.',
    'memory_trans_norm.',
    'bbox_head.cls_branches.',
    'bbox_head.reg_branches.',
)


def build_rgbd_pose_curriculum_transfer_state(
    pose_state: dict[str, torch.Tensor],
    feature_state: dict[str, torch.Tensor],
    target_state: dict[str, torch.Tensor],
    query_seed: int = 736512,
) -> tuple[dict[str, torch.Tensor], dict]:
    """Compose exact feature tensors with compatible 3D pose tensors.

    Backbone and neck tensors come exclusively from the final 2D RGB-D model.
    All remaining tensors come from the mature 3D CoP model. The sole allowed
    shape adaptation is a deterministic expansion of the DINO content-query
    embedding; every other source pose mismatch fails before training.
    """
    target_feature_keys = {
        key for key in target_state if key.startswith(_FEATURE_PREFIXES)
    }
    missing_features = sorted(target_feature_keys.difference(feature_state))
    if missing_features:
        raise KeyError(
            'feature checkpoint misses target keys: '
            + ', '.join(missing_features))

    selected: dict[str, torch.Tensor] = {}
    loaded_groups: Counter[str] = Counter()
    for key in sorted(target_feature_keys):
        source_value = feature_state[key]
        if source_value.shape != target_state[key].shape:
            raise ValueError(
                f'feature tensor shape mismatch for {key}: '
                f'{tuple(source_value.shape)} vs '
                f'{tuple(target_state[key].shape)}')
        selected[key] = source_value
        loaded_groups[key.split('.', 1)[0]] += 1

    ignored_pose_features = []
    rebuilt_pose_keys = []
    for key, source_value in pose_state.items():
        if key.startswith(_FEATURE_PREFIXES):
            ignored_pose_features.append(key)
            continue
        if (key.startswith(_REBUILT_POSE_PREFIXES)
                and key not in target_state):
            rebuilt_pose_keys.append(key)
            continue
        if key not in target_state:
            raise KeyError(f'pose target key is missing: {key}')
        target_value = target_state[key]
        if key == 'query_embedding.weight':
            selected[key] = _expand_query_embedding(
                source_value, target_value, query_seed)
        elif source_value.shape == target_value.shape:
            selected[key] = source_value
        else:
            raise ValueError(
                f'pose tensor shape mismatch for {key}: '
                f'{tuple(source_value.shape)} vs '
                f'{tuple(target_value.shape)}')
        loaded_groups[key.split('.', 1)[0]] += 1

    source_queries = pose_state.get('query_embedding.weight')
    target_queries = target_state.get('query_embedding.weight')
    target_only = sorted(set(target_state).difference(selected))
    report = {
        'loaded_key_count': len(selected),
        'loaded_groups': dict(sorted(loaded_groups.items())),
        'ignored_pose_feature_key_count': len(ignored_pose_features),
        'rebuilt_pose_key_count': len(rebuilt_pose_keys),
        'target_only_keys': target_only,
        'source_queries': (
            int(source_queries.shape[0]) if source_queries is not None else None),
        'target_queries': (
            int(target_queries.shape[0]) if target_queries is not None else None),
        'query_expansion_seed': int(query_seed),
    }
    return selected, report


def build_detection_repair_transfer_state(
    source_state: dict[str, torch.Tensor],
    target_state: dict[str, torch.Tensor],
    query_seed: int = 736512,
) -> tuple[dict[str, torch.Tensor], dict]:
    """Select a complete pretrained DINO 2D path without touching 3D pose.

    The target keeps its portable RGB-D backbone/neck and all pose-specific
    branches.  Encoder, decoder, content queries, two-stage proposal transform,
    classification branches, and 2D box branches are transferred as one
    coherent unit.  Omitting any target tensor from that unit is an error; the
    only permitted shape adaptation is deterministic content-query expansion.
    """
    target_2d_keys = {
        key for key in target_state
        if key == 'query_embedding.weight'
        or key.startswith(_DETECTION_REPAIR_PREFIXES)
    }
    missing = sorted(target_2d_keys.difference(source_state))
    if missing:
        raise KeyError(
            'detection checkpoint misses target 2D keys: '
            + ', '.join(missing))

    selected: dict[str, torch.Tensor] = {}
    loaded_groups: Counter[str] = Counter()
    for key in sorted(target_2d_keys):
        source_value = source_state[key]
        target_value = target_state[key]
        if key == 'query_embedding.weight':
            selected[key] = _expand_query_embedding(
                source_value, target_value, query_seed)
        elif source_value.shape == target_value.shape:
            selected[key] = source_value
        else:
            raise ValueError(
                f'2D tensor shape mismatch for {key}: '
                f'{tuple(source_value.shape)} vs '
                f'{tuple(target_value.shape)}')
        loaded_groups[key.split('.', 1)[0]] += 1

    source_queries = source_state.get('query_embedding.weight')
    target_queries = target_state.get('query_embedding.weight')
    report = {
        'loaded_key_count': len(selected),
        'loaded_groups': dict(sorted(loaded_groups.items())),
        'source_queries': (
            int(source_queries.shape[0]) if source_queries is not None else None),
        'target_queries': (
            int(target_queries.shape[0]) if target_queries is not None else None),
        'query_expansion_seed': int(query_seed),
        'preserved_prefixes': ['backbone.', 'neck.', 'bbox_head.cop_'],
    }
    return selected, report


@HOOKS.register_module()
class RGBDPoseCurriculumTransferHook(Hook):
    """Load mature pose tensors and the final 2D RGB-D feature pyramid."""

    priority = 'VERY_HIGH'

    def __init__(self,
                 pose_checkpoint: str,
                 feature_checkpoint: str,
                 report_filename: str = 'rgbd_pose_transfer_report.json',
                 query_seed: int = 736512) -> None:
        self.pose_checkpoint = pose_checkpoint
        self.feature_checkpoint = feature_checkpoint
        self.report_filename = report_filename
        self.query_seed = int(query_seed)
        self._loaded = False

    @staticmethod
    def _load_state(path: Path, label: str) -> dict[str, torch.Tensor]:
        if not path.is_file():
            raise FileNotFoundError(f'{label} checkpoint not found: {path}')
        checkpoint = torch.load(path, map_location='cpu', weights_only=False)
        if 'state_dict' not in checkpoint:
            raise KeyError(f'{label} checkpoint lacks state_dict: {path}')
        return checkpoint['state_dict']

    def before_train(self, runner) -> None:
        if self._loaded:
            return
        pose_path = Path(self.pose_checkpoint)
        feature_path = Path(self.feature_checkpoint)
        pose_state = self._load_state(pose_path, 'pose')
        feature_state = self._load_state(feature_path, 'feature')
        model = runner.model.module if hasattr(runner.model, 'module') \
            else runner.model
        selected, report = build_rgbd_pose_curriculum_transfer_state(
            pose_state,
            feature_state,
            model.state_dict(),
            self.query_seed,
        )
        incompatible = model.load_state_dict(selected, strict=False)
        if incompatible.unexpected_keys:
            raise RuntimeError(
                'composite transfer produced unexpected keys: '
                f'{sorted(incompatible.unexpected_keys)}')
        report.update(
            pose_checkpoint=str(pose_path.resolve()),
            feature_checkpoint=str(feature_path.resolve()),
            missing_target_keys=sorted(incompatible.missing_keys),
            unexpected_keys=sorted(incompatible.unexpected_keys),
        )
        report_path = Path(runner.work_dir) / self.report_filename
        report_path.write_text(
            json.dumps(report, indent=2, sort_keys=True) + '\n',
            encoding='utf-8')
        self._loaded = True


@HOOKS.register_module()
class DetectionRepairTransferHook(Hook):
    """Overlay a proven DINO 2D path onto an already-loaded 3D model."""

    priority = 'VERY_HIGH'

    def __init__(self,
                 checkpoint: str,
                 report_filename: str = 'detection_repair_transfer_report.json',
                 query_seed: int = 736512) -> None:
        self.checkpoint = checkpoint
        self.report_filename = report_filename
        self.query_seed = int(query_seed)
        self._loaded = False

    def before_train(self, runner) -> None:
        if self._loaded:
            return
        source_path = Path(self.checkpoint)
        source_state = RGBDPoseCurriculumTransferHook._load_state(
            source_path, 'detection')
        model = runner.model.module if hasattr(runner.model, 'module') \
            else runner.model
        selected, report = build_detection_repair_transfer_state(
            source_state,
            model.state_dict(),
            self.query_seed,
        )
        incompatible = model.load_state_dict(selected, strict=False)
        if incompatible.unexpected_keys:
            raise RuntimeError(
                'detection repair transfer produced unexpected keys: '
                f'{sorted(incompatible.unexpected_keys)}')
        report.update(
            checkpoint=str(source_path.resolve()),
            missing_target_key_count=len(incompatible.missing_keys),
            unexpected_keys=sorted(incompatible.unexpected_keys),
        )
        report_path = Path(runner.work_dir) / self.report_filename
        report_path.write_text(
            json.dumps(report, indent=2, sort_keys=True) + '\n',
            encoding='utf-8')
        self._loaded = True
