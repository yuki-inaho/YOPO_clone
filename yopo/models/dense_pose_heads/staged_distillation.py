"""Frozen head adapters for cumulative CoP distillation.

This module deliberately owns checkpoint adaptation, teacher inference, and
distillation loss reduction. Detection heads only provide decoder features and
consume the resulting losses, which keeps stage selection config-driven.
"""

import copy
from typing import Dict, Mapping, Sequence, Tuple

import torch
import torch.nn as nn
import torch.nn.functional as F
from mmcv.cnn import Linear
from torch import Tensor

from ..layers import inverse_sigmoid


class StagedDistillationTeacher(nn.Module):
    """Frozen OBB-center and prior-pose heads used as soft teachers."""

    VALID_ATTRIBUTES = frozenset({'center', 'z', 'size', 'rotation'})
    VALID_CENTER_SOURCES = ('pose_center', 'obb_adapter')
    POSE_BRANCH_NAMES = {
        'z': 'reg_z_branch',
        'size': 'reg_size_branch',
        'rotation': 'reg_rotation_branch',
    }

    def __init__(
            self,
            attributes: Sequence[str],
            embed_dims: int,
            num_reg_fcs: int,
            num_pred_layer: int,
            student_center_branches: nn.ModuleList,
            student_cls_branches: nn.ModuleList,
            student_pose_branches: Mapping[str, nn.ModuleList],
            center_uses_bbox: bool,
            pose_uses_bbox: Mapping[str, bool],
            center_teacher_source: str = 'pose_center',
            obb_checkpoint: str = None,
            pose_checkpoint: str = None,
            loss_weights: Mapping[str, float] = None,
            score_threshold: float = 0.0) -> None:
        super().__init__()
        self.attributes = self.validate_attributes(attributes)
        if not 0.0 <= score_threshold <= 1.0:
            raise ValueError('distill_score_threshold must be in [0, 1]')
        self.score_threshold = float(score_threshold)
        configured_weights = loss_weights or {}
        self.loss_weights = {
            attribute: float(configured_weights.get(attribute, 1.0))
            for attribute in self.attributes
        }
        self.pose_uses_bbox = dict(pose_uses_bbox)
        if center_teacher_source not in self.VALID_CENTER_SOURCES:
            raise ValueError(
                'center_teacher_source must be one of '
                f'{self.VALID_CENTER_SOURCES}, got '
                f'{center_teacher_source!r}')
        self.center_teacher_source = center_teacher_source
        self.center_uses_bbox = center_uses_bbox
        self.report = {}

        if 'center' in self.attributes:
            if center_teacher_source == 'pose_center':
                if not pose_checkpoint:
                    raise ValueError(
                        'pose_teacher_checkpoint is required for pose_center '
                        'distillation')
                center_state = self._load_state(pose_checkpoint)
                self.center_reg_branches = copy.deepcopy(
                    student_center_branches)
                self.center_cls_branches = copy.deepcopy(
                    student_cls_branches)
                self.report['center'] = dict(
                    checkpoint=pose_checkpoint,
                    source='frozen_pose_center_head',
                    reg=self._load_branch_list(
                        self.center_reg_branches,
                        center_state,
                        'reg_centers_2d_branch'),
                    cls=self._load_branch_list(
                        self.center_cls_branches,
                        center_state,
                        'cls_branches'))
            else:
                if not obb_checkpoint:
                    raise ValueError(
                        'obb_center_teacher_checkpoint is required for '
                        'obb_adapter distillation')
                obb_state = self._load_state(obb_checkpoint)
                self.center_reg_branches = nn.ModuleList([
                    self._make_center_branch(embed_dims, num_reg_fcs)
                    for _ in range(num_pred_layer)
                ])
                self.center_cls_branches = nn.ModuleList([
                    Linear(embed_dims, 1) for _ in range(num_pred_layer)
                ])
                self.report['center'] = dict(
                    checkpoint=obb_checkpoint,
                    source='frozen_obb_head_adapter',
                    reg=self._load_branch_list(
                        self.center_reg_branches,
                        obb_state,
                        'reg_branches',
                        take_first_rows=2),
                    cls=self._load_branch_list(
                        self.center_cls_branches,
                        obb_state,
                        'cls_branches'))

        pose_attributes = [
            attribute for attribute in self.attributes
            if attribute in self.POSE_BRANCH_NAMES
        ]
        self.pose_branches = nn.ModuleDict()
        if pose_attributes:
            if not pose_checkpoint:
                raise ValueError(
                    'pose_teacher_checkpoint is required for z/size/rotation '
                    'distillation')
            pose_state = self._load_state(pose_checkpoint)
            for attribute in pose_attributes:
                teacher_branches = copy.deepcopy(
                    student_pose_branches[attribute])
                self.pose_branches[attribute] = teacher_branches
                self.report[attribute] = dict(
                    checkpoint=pose_checkpoint,
                    source='frozen_parallel_pose_head',
                    **self._load_branch_list(
                        teacher_branches,
                        pose_state,
                        self.POSE_BRANCH_NAMES[attribute]))

        self._freeze(self)

    @classmethod
    def validate_attributes(cls, attributes: Sequence[str]) -> Tuple[str, ...]:
        attributes = tuple(attributes)
        if (len(set(attributes)) != len(attributes) or
                not set(attributes).issubset(cls.VALID_ATTRIBUTES)):
            raise ValueError(
                'distill_attributes must contain unique values from '
                f'{sorted(cls.VALID_ATTRIBUTES)}, got {attributes!r}')
        return attributes

    @staticmethod
    def _make_center_branch(embed_dims: int,
                            num_reg_fcs: int) -> nn.Sequential:
        layers = []
        for _ in range(num_reg_fcs):
            layers.extend((Linear(embed_dims, embed_dims), nn.ReLU()))
        layers.append(Linear(embed_dims, 2))
        return nn.Sequential(*layers)

    @staticmethod
    def _load_state(checkpoint_path: str) -> Dict[str, Tensor]:
        checkpoint = torch.load(
            checkpoint_path, map_location='cpu', weights_only=False)
        state_dict = checkpoint.get('state_dict', checkpoint)
        if not isinstance(state_dict, dict):
            raise RuntimeError(
                f'Teacher checkpoint {checkpoint_path!r} has no state dict')
        return state_dict

    @staticmethod
    def _find_source_layers(state_dict: Dict[str, Tensor], base: str):
        for prefix in (f'bbox_head.{base}', base):
            marker = f'{prefix}.'
            layer_ids = {
                int(key[len(marker):].split('.', 1)[0])
                for key in state_dict
                if key.startswith(marker) and
                key[len(marker):].split('.', 1)[0].isdigit()
            }
            if layer_ids:
                return prefix, sorted(layer_ids)
        raise RuntimeError(f'Teacher checkpoint is missing {base!r}')

    @staticmethod
    def _freeze(module: nn.Module) -> None:
        module.eval()
        for parameter in module.parameters():
            parameter.requires_grad_(False)

    @classmethod
    def _load_branch_list(cls, branches: nn.ModuleList,
                          state_dict: Dict[str, Tensor], base: str,
                          take_first_rows: int = None) -> dict:
        source_prefix, source_layers = cls._find_source_layers(
            state_dict, base)
        mappings = []
        for target_layer, branch in enumerate(branches):
            source_layer = source_layers[min(target_layer,
                                             len(source_layers) - 1)]
            branch_state = {}
            final_module_index = len(branch) - 1 if isinstance(
                branch, nn.Sequential) else None
            for local_key, target_value in branch.state_dict().items():
                source_key = f'{source_prefix}.{source_layer}.{local_key}'
                if source_key not in state_dict:
                    raise RuntimeError(
                        f'Teacher checkpoint is missing {source_key!r}')
                source_value = state_dict[source_key]
                if (take_first_rows is not None and
                        final_module_index is not None and
                        local_key.startswith(f'{final_module_index}.')):
                    source_value = source_value[:take_first_rows]
                if source_value.shape != target_value.shape:
                    raise RuntimeError(
                        f'Teacher tensor shape mismatch for {source_key}: '
                        f'{tuple(source_value.shape)} != '
                        f'{tuple(target_value.shape)}')
                branch_state[local_key] = source_value
            branch.load_state_dict(branch_state, strict=True)
            mappings.append((target_layer, source_layer))
        return dict(source_prefix=source_prefix, layer_mappings=mappings)

    def train(self, mode: bool = True):
        """Keep every teacher in evaluation mode when the parent trains."""
        super().train(False)
        return self

    @torch.no_grad()
    def forward(self, hidden_states: Tensor, references: Sequence[Tensor],
                bbox_preds: Tensor):
        targets = {attribute: [] for attribute in self.attributes}
        scores = []
        for layer_id, hidden_state in enumerate(hidden_states):
            reference = inverse_sigmoid(references[layer_id])
            if 'center' in self.attributes:
                center_input = hidden_state
                if (self.center_teacher_source == 'pose_center' and
                        self.center_uses_bbox):
                    center_input = torch.cat(
                        (center_input, bbox_preds[layer_id]), dim=-1)
                center = self.center_reg_branches[layer_id](center_input)
                if self.center_teacher_source == 'pose_center':
                    center = (center.sigmoid() +
                              bbox_preds[layer_id][..., :2] - 0.5)
                else:
                    center = (center + reference[..., :2]).sigmoid()
                targets['center'].append(center)
                scores.append(
                    self.center_cls_branches[layer_id](hidden_state)
                    .sigmoid().amax(dim=-1))

            for attribute, branches in self.pose_branches.items():
                pose_input = hidden_state
                if self.pose_uses_bbox.get(attribute, False):
                    pose_input = torch.cat(
                        (pose_input, bbox_preds[layer_id]), dim=-1)
                targets[attribute].append(branches[layer_id](pose_input))

        stacked_targets = {
            attribute: torch.stack(values)
            for attribute, values in targets.items()
        }
        if scores:
            stacked_scores = torch.stack(scores)
        else:
            stacked_scores = torch.ones_like(hidden_states[..., 0])
        return stacked_targets, stacked_scores

    def loss(self, student_outputs: Mapping[str, Tensor],
             targets: Mapping[str, Tensor], scores: Tensor,
             first_matching_query: int = 0) -> Dict[str, Tensor]:
        score_mask = scores[:, :, first_matching_query:] >= \
            self.score_threshold
        losses = {}
        for attribute in self.attributes:
            student = student_outputs[attribute][
                :, :, first_matching_query:]
            teacher = targets[attribute][
                :, :, first_matching_query:].detach()
            per_query_loss = F.smooth_l1_loss(
                student, teacher, reduction='none').mean(dim=-1)
            if score_mask.any():
                loss = per_query_loss[score_mask].mean()
            else:
                loss = student.sum() * 0.0
            losses[f'loss_distill_{attribute}'] = (
                loss * self.loss_weights[attribute])
        return losses
