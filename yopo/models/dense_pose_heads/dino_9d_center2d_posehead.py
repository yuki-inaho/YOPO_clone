# Copyright (c) OpenMMLab. All rights reserved.
import copy
import math
from typing import Dict, List, Tuple

import torch
import torch.nn as nn
import torch.nn.functional as F
from mmcv.cnn import Linear
from mmengine.structures import InstanceData
from mmengine.model import BaseModule, bias_init_with_prob, constant_init
from torch import Tensor
from scipy.optimize import linear_sum_assignment

from yopo.registry import MODELS, TASK_UTILS
from yopo.structures import SampleList
from yopo.structures.bbox import (bbox_cxcywh_to_xyxy, bbox_overlaps,
                                   bbox_xyxy_to_cxcywh)
from yopo.utils import (InstanceList, OptInstanceList, reduce_mean,
    ConfigType, OptMultiConfig)
from ..layers import inverse_sigmoid
from ..losses import QualityFocalLoss
from ..losses.gaucho3d_geometry import (cholesky3d_from_raw,
                                        dual_plane_cholesky3d,
                                        clamp_sigma_max_axis,
                                        ellipsoid_from_rotation_size,
                                        front_margin,
                                        gaussian_to_ellipse2d,
                                        project_ellipsoid_dual_quadric,
                                        scale_shape_cholesky2d,
                                        scale_shape_cholesky3d,
                                        sigma_from_cholesky,
                                        sigma_from_cholesky2d,
                                        symmetry_class)
from ..losses.projected_ellipsoid_loss import _rotation_6d_to_matrix
from ..utils import multi_apply
from .depth_query_context import CoPStageFusion, MultiScaleDepthQuerySampler
from .matchability_quality import aligned_hbb_iou_quality_targets
from .simple_dino_9dposehead import SimpleDINO9DPoseHead
from .staged_distillation import StagedDistillationTeacher


def compact_gaussian_orientation_descriptor(
        compact: Tensor, eps: float = 1e-7) -> Tensor:
    """Encode a 2D Gaussian axis as an anisotropy-aware spin-2 vector.

    For covariance ``[[xx, xy], [xy, yy]]``, this returns
    ``((xx - yy) / trace, 2 * xy / trace)``.  Its angle is twice the OBB
    axis angle, so the representation identifies axes that differ by 180
    degrees.  Its magnitude is the covariance anisotropy and therefore tends
    to zero when the projected object is rotationally unobservable.
    """
    if compact.shape[-1] != 5:
        raise ValueError(
            f'compact Gaussian width must be 5, got {compact.shape}')
    sigma_xx, sigma_xy, sigma_yy = (
        compact[..., 2], compact[..., 3], compact[..., 4])
    trace = (sigma_xx + sigma_yy).clamp_min(eps)
    return torch.stack(
        ((sigma_xx - sigma_yy) / trace, 2.0 * sigma_xy / trace),
        dim=-1)


def _compact_gaussian_from_raw_cholesky(raw_obb: Tensor) -> Tensor:
    """Convert unconstrained lower-Cholesky values to a compact SPD Gaussian."""
    l11 = F.softplus(raw_obb[..., 0]) + 1e-4
    l21 = raw_obb[..., 1]
    l22 = F.softplus(raw_obb[..., 2]) + 1e-4
    return torch.stack(
        (torch.zeros_like(l11), torch.zeros_like(l11), l11.square(),
         l11 * l21, l21.square() + l22.square()),
        dim=-1)


def aligned_iou_quality_targets(
        labels: Tensor,
        bbox_predictions: Tensor,
        bbox_targets: Tensor,
        num_classes: int) -> Tensor:
    """Build detached IoU targets for one-to-one quality classification.

    Hungarian-positive queries receive their aligned 2D IoU in ``[0, 1]``;
    background and ignored queries receive zero.  Detaching both box tensors
    keeps the quality loss responsible for score ranking only, rather than
    creating a second route that changes box geometry to increase its target.
    """
    return aligned_hbb_iou_quality_targets(
        labels, bbox_predictions, bbox_targets, num_classes)


def one_to_many_bbox_targets(
        bbox_predictions: Tensor,
        gt_bboxes_xyxy: Tensor,
        gt_labels: Tensor,
        num_classes: int,
        topk: int = 2) -> dict[str, Tensor]:
    """Assign up to ``topk`` unique queries to every GT using 2D geometry.

    Hungarian seeds provide one unique query per GT whenever query capacity is
    sufficient.  Remaining queries are greedily assigned by normalized L1 +
    IoU cost until each GT reaches ``topk``.  A query is never positive for
    two GTs, so classification and regression targets cannot conflict.
    Assignment decisions are detached; gradients flow only through the
    returned auxiliary losses.
    """
    if topk < 1:
        raise ValueError(f'topk must be positive, got {topk}')
    if bbox_predictions.ndim != 2 or bbox_predictions.shape[-1] != 4:
        raise ValueError(
            'bbox_predictions must have shape (queries, 4), got '
            f'{tuple(bbox_predictions.shape)}')
    if gt_bboxes_xyxy.ndim != 2 or gt_bboxes_xyxy.shape[-1] != 4:
        raise ValueError(
            'gt_bboxes_xyxy must have shape (objects, 4), got '
            f'{tuple(gt_bboxes_xyxy.shape)}')
    if len(gt_bboxes_xyxy) != len(gt_labels):
        raise ValueError('GT boxes and labels must have the same length')

    num_queries = bbox_predictions.shape[0]
    device = bbox_predictions.device
    labels = torch.full(
        (num_queries,), num_classes, dtype=torch.long, device=device)
    bbox_targets = torch.zeros_like(bbox_predictions)
    bbox_weights = torch.zeros_like(bbox_predictions)
    assigned_gt_indices = torch.full(
        (num_queries,), -1, dtype=torch.long, device=device)
    if num_queries == 0 or len(gt_bboxes_xyxy) == 0:
        return dict(
            labels=labels,
            bbox_targets=bbox_targets,
            bbox_weights=bbox_weights,
            assigned_gt_indices=assigned_gt_indices)

    gt_bboxes_xyxy = gt_bboxes_xyxy.to(
        device=device, dtype=bbox_predictions.dtype)
    gt_labels = gt_labels.to(device=device, dtype=torch.long)
    gt_cxcywh = bbox_xyxy_to_cxcywh(gt_bboxes_xyxy)
    l1_cost = torch.cdist(bbox_predictions, gt_cxcywh, p=1)
    iou = bbox_overlaps(
        bbox_cxcywh_to_xyxy(bbox_predictions), gt_bboxes_xyxy,
        mode='iou')
    cost = l1_cost + 2.0 * (1.0 - iou)

    seed_queries, seed_gt = linear_sum_assignment(cost.detach().float().cpu())
    seed_queries = torch.as_tensor(seed_queries, dtype=torch.long, device=device)
    seed_gt = torch.as_tensor(seed_gt, dtype=torch.long, device=device)
    assigned_gt_indices[seed_queries] = seed_gt
    per_gt_count = torch.bincount(
        seed_gt, minlength=len(gt_bboxes_xyxy)).tolist()

    # Inspect only a small geometric shortlist per GT.  This preserves the
    # up-to-topk contract while avoiding a Python walk over every Q*G edge.
    candidate_count = min(num_queries, max(8, topk * 4))
    candidate_costs, candidate_queries = torch.topk(
        cost.detach(), k=candidate_count, dim=0, largest=False,
        sorted=True)
    candidate_gt = torch.arange(
        len(gt_bboxes_xyxy), device=device).unsqueeze(0).expand_as(
            candidate_queries)
    candidate_order = torch.argsort(
        candidate_costs.reshape(-1), stable=True).cpu().tolist()
    candidate_queries = candidate_queries.reshape(-1).cpu().tolist()
    candidate_gt = candidate_gt.reshape(-1).cpu().tolist()
    filled_gt_count = sum(count >= topk for count in per_gt_count)
    for candidate_index in candidate_order:
        query_index = candidate_queries[candidate_index]
        gt_index = candidate_gt[candidate_index]
        if assigned_gt_indices[query_index] >= 0:
            continue
        if per_gt_count[gt_index] >= topk:
            continue
        assigned_gt_indices[query_index] = gt_index
        per_gt_count[gt_index] += 1
        if per_gt_count[gt_index] == topk:
            filled_gt_count += 1
        if filled_gt_count == len(per_gt_count):
            break

    positive = assigned_gt_indices >= 0
    positive_gt = assigned_gt_indices[positive]
    labels[positive] = gt_labels[positive_gt]
    bbox_targets[positive] = gt_cxcywh[positive_gt]
    bbox_weights[positive] = 1.0
    return dict(
        labels=labels,
        bbox_targets=bbox_targets,
        bbox_weights=bbox_weights,
        assigned_gt_indices=assigned_gt_indices)


@MODELS.register_module()
class DINO9DCenter2DPoseHead(SimpleDINO9DPoseHead):
    def __init__(
            self,
            num_classes: int,
            embed_dims: int = 256,
            num_reg_fcs: int = 2,
            rot_dim: int = 6,
            sync_cls_avg_factor: bool = False,
            share_pred_layer: bool = False,
            num_pred_layer: int = 6,
            as_two_stage: bool = False,
            classwise_rotation: bool = False,
            classwise_sizes: bool = False,
            loss_cls: ConfigType = dict(
                type='CrossEntropyLoss',
                bg_cls_weight=0.1,
                use_sigmoid=False,
                loss_weight=1.0,
                class_weight=1.0),
            use_cuboid_conditioning: bool = False,
            use_intrinsinc_for_bbox: bool = False,
            use_bbox_for_centers_2d: bool = True,
            use_bbox_for_z : bool = False,
            use_bbox_for_rotation : bool = False,
            use_bbox_for_size : bool = False,
            use_log_z: bool = False,
            use_cop_chain: bool = False,
            cop_prediction_mode: str = None,
            cop_chain_order: Tuple[str, str, str] = ('size', 'rotation', 'z'),
            cop_aux_loss_weights: ConfigType = None,
            cop_use_bbox_conditioning: bool = False,
            cop_fusion_mode: str = 'residual',
            cop_depth_context: ConfigType = None,
            cop_encoder_pose_supervision: bool = True,
            cop_obb_rotation_conditioning: bool = False,
            cop_obb_rotation_refinement: bool = False,
            expose_obb_aux_predictions: bool = False,
            gaucho_ellipsoid: bool = False,
            gaucho_chart: str = 'scale_shape',
            gaucho_dual_plane_fix_rc_zero: bool = False,
            gaucho_size_prior: float = 0.03,
            gaucho_classwise: bool = True,
            gaucho_use_bbox_conditioning: bool = True,
            gaucho_ellipse2d: bool = False,
            gaucho_ellipse2d_dn: bool = False,
            expose_gaucho_predictions: bool = False,
            loss_ellipse2d: ConfigType = None,
            loss_ellipsoid: ConfigType = None,
            loss_ellipsoid_projection: ConfigType = None,
            loss_ellipsoid_max_axis: ConfigType = None,
            loss_ellipse2d_corner: ConfigType = None,
            loss_ellipse2d_angle: ConfigType = None,
            ellipsoid_gt_max_diameter: float = None,
            ellipsoid_max_diameter: float = None,
            sensor_depth_scale: float = None,
            sensor_depth_window: int = 2,
            sensor_depth_anchor: bool = False,
            sensor_depth_anchor_denoising: bool = True,
            distill_attributes: Tuple[str, ...] = (),
            center_teacher_source: str = 'pose_center',
            obb_center_teacher_checkpoint: str = None,
            pose_teacher_checkpoint: str = None,
            distill_loss_weights: ConfigType = None,
            distill_score_threshold: float = 0.0,
            loss_bbox: ConfigType = dict(type='L1Loss', loss_weight=5.0),
            loss_iou: ConfigType = dict(type='GIoULoss', loss_weight=2.0),
            loss_centers_2d: ConfigType = dict(type='L1PoseLoss', loss_weight=5.0),
            loss_z: ConfigType = dict(type='L2PoseLoss', loss_weight=5.0),
            loss_rotation: ConfigType = dict(type='L2PoseLoss', loss_weight=5.0),
            loss_sizes: ConfigType = dict(type='L2PoseLoss', loss_weight=5.0),
            loss_projection: ConfigType = None,
            loss_obb_aux: ConfigType = None,
            o2m_aux_topk: int = 0,
            o2m_aux_loss_weight: float = 0.5,
            o2m_aux_loss_cls: ConfigType = dict(
                type='FocalLoss', use_sigmoid=True, gamma=2.0, alpha=0.25,
                loss_weight=1.0),
            projection_geometry_source: str = 'target',
            train_intrinsic_to_image_space: bool = False,
            train_cfg: ConfigType = dict(
                assigner=dict(
                    type='HungarianAssigner',
                    match_costs=[
                        dict(type='ClassificationCost', weight=1.),
                        dict(type='BBoxL1Cost', weight=5.0, box_format='xywh'),
                        dict(type='IoUCost', iou_mode='giou', weight=2.0),
                        dict(type='PoseCost', weight=5.0)
                    ])),
            test_cfg: ConfigType = dict(max_per_img=100),
            init_cfg: OptMultiConfig = None,
            quality_target: ConfigType = None,
            loss_rotation_frame: ConfigType = None) -> None:
        BaseModule.__init__(self, init_cfg=init_cfg)
        self.bg_cls_weight = 0
        self.sync_cls_avg_factor = sync_cls_avg_factor

        if train_cfg:
            assert 'assigner' in train_cfg, 'assigner should be provided ' \
                                            'when train_cfg is set.'
            assigner = train_cfg['assigner']
            self.assigner = TASK_UTILS.build(assigner)
            if train_cfg.get('sampler', None) is not None:
                raise RuntimeError('DETR do not build sampler.')
        self._assigner_uses_ellipse = self._match_costs_use_ellipse(train_cfg)
        self.num_classes = num_classes
        self.embed_dims = embed_dims
        self.num_reg_fcs = num_reg_fcs
        self.rot_dim = rot_dim
        self.train_cfg = train_cfg
        self.test_cfg = test_cfg

        self.loss_cls = MODELS.build(loss_cls)
        self.loss_bbox = MODELS.build(loss_bbox)
        self.loss_iou = MODELS.build(loss_iou)
        self.uses_quality_target = (
            isinstance(self.loss_cls, QualityFocalLoss)
            or bool(getattr(
                self.loss_cls, 'requires_quality_target', False)))
        if quality_target is not None and not self.uses_quality_target:
            raise ValueError(
                'quality_target requires a classification loss that accepts '
                '(labels, quality) targets')
        if self.uses_quality_target:
            quality_target_cfg = copy.deepcopy(
                quality_target or dict(
                    type='MatchabilityQualityPolicy',
                    source='hbb_iou'))
            configured_num_classes = quality_target_cfg.pop(
                'num_classes', num_classes)
            if configured_num_classes != num_classes:
                raise ValueError(
                    'quality_target.num_classes must match the head: '
                    f'{configured_num_classes} != {num_classes}')
            quality_target_cfg['num_classes'] = num_classes
            self.quality_target_policy = MODELS.build(quality_target_cfg)
        else:
            self.quality_target_policy = None

        self.use_cuboid_conditioning = use_cuboid_conditioning
        self.use_intrinsinc_for_bbox = use_intrinsinc_for_bbox
        self.use_bbox_for_centers_2d = use_bbox_for_centers_2d
        self.use_bbox_for_z = use_bbox_for_z
        self.use_bbox_for_rotation = use_bbox_for_rotation
        self.use_bbox_for_size = use_bbox_for_size

        self.use_log_z = use_log_z

        if cop_prediction_mode is None:
            cop_prediction_mode = 'auxiliary' if use_cop_chain else 'parallel'
        valid_cop_modes = {'parallel', 'auxiliary', 'chain'}
        if cop_prediction_mode not in valid_cop_modes:
            raise ValueError(
                f'cop_prediction_mode must be one of {sorted(valid_cop_modes)}, '
                f'got {cop_prediction_mode!r}')
        expected_attributes = {'z', 'size', 'rotation'}
        if len(cop_chain_order) != 3 or set(cop_chain_order) != expected_attributes:
            raise ValueError(
                'cop_chain_order must contain z, size, and rotation exactly once, '
                f'got {cop_chain_order!r}')
        self.cop_prediction_mode = cop_prediction_mode
        self.cop_chain_order = tuple(cop_chain_order)
        self.cop_use_bbox_conditioning = cop_use_bbox_conditioning
        if cop_fusion_mode not in CoPStageFusion.VALID_MODES:
            raise ValueError(
                'cop_fusion_mode must be one of '
                f'{CoPStageFusion.VALID_MODES}, got {cop_fusion_mode!r}')
        self.cop_fusion_mode = cop_fusion_mode
        self.use_cop_chain = cop_prediction_mode != 'parallel'
        self._uses_auxiliary_chain = cop_prediction_mode == 'auxiliary'
        self.cop_encoder_pose_supervision = (
            cop_prediction_mode != 'chain' or
            bool(cop_encoder_pose_supervision))
        if not self.cop_encoder_pose_supervision:
            if train_cfg is None or train_cfg.get('encoder_assigner') is None:
                raise ValueError(
                    'encoder_assigner is required when chain mode disables '
                    'encoder pose supervision')
            self.encoder_assigner = TASK_UTILS.build(
                train_cfg['encoder_assigner'])
        else:
            self.encoder_assigner = getattr(self, 'assigner', None)
        self.requires_depth_features = (
            self.use_cop_chain and cop_fusion_mode == 'depth_dense')
        self.cop_depth_context_cfg = dict(cop_depth_context or {})
        self.last_depth_context_shape = None
        self.last_dense_input_shape = None
        if self.use_cop_chain:
            default_cop_loss_weights = dict(
                size=3.0, rotation=2.0, z=1.0)
            configured_cop_loss_weights = (
                default_cop_loss_weights
                if cop_aux_loss_weights is None
                else dict(cop_aux_loss_weights))
            if set(configured_cop_loss_weights) != expected_attributes:
                raise ValueError(
                    'cop_aux_loss_weights must contain z, size, and rotation '
                    'exactly once')
            if any(not math.isfinite(float(weight)) or float(weight) < 0.0
                   for weight in configured_cop_loss_weights.values()):
                raise ValueError(
                    'cop_aux_loss_weights values must be finite and '
                    'non-negative')
            self.cop_loss_weights = {
                attribute: float(configured_cop_loss_weights[attribute])
                for attribute in ('z', 'size', 'rotation')
            }

        self.distill_attributes = \
            StagedDistillationTeacher.validate_attributes(
                distill_attributes)
        if center_teacher_source not in \
                StagedDistillationTeacher.VALID_CENTER_SOURCES:
            raise ValueError(
                'center_teacher_source must be one of '
                f'{StagedDistillationTeacher.VALID_CENTER_SOURCES}, got '
                f'{center_teacher_source!r}')
        self.center_teacher_source = center_teacher_source
        configured_distill_weights = distill_loss_weights or {}
        self.distill_loss_weights = {
            attribute: float(configured_distill_weights.get(attribute, 1.0))
            for attribute in self.distill_attributes
        }
        if not 0.0 <= distill_score_threshold <= 1.0:
            raise ValueError('distill_score_threshold must be in [0, 1]')
        self.distill_score_threshold = float(distill_score_threshold)
        self.obb_center_teacher_checkpoint = obb_center_teacher_checkpoint
        self.pose_teacher_checkpoint = pose_teacher_checkpoint

        self.loss_centers_2d = MODELS.build(loss_centers_2d)
        self.loss_z = MODELS.build(loss_z)

        self.loss_rotation = MODELS.build(loss_rotation)
        self.loss_rotation_frame = (
            MODELS.build(loss_rotation_frame)
            if loss_rotation_frame is not None else None)
        if self.loss_rotation_frame is not None and rot_dim != 6:
            raise ValueError(
                'loss_rotation_frame requires rot_dim=6 raw rotation '
                f'predictions, got rot_dim={rot_dim}')
        self.loss_sizes = MODELS.build(loss_sizes)
        self.loss_projection = (
            MODELS.build(loss_projection)
            if loss_projection is not None else None)
        self.loss_obb_aux = (
            MODELS.build(loss_obb_aux)
            if loss_obb_aux is not None else None)
        if self.loss_obb_aux is not None and not self.use_cop_chain:
            raise ValueError('loss_obb_aux requires use_cop_chain=True')
        self.cop_obb_rotation_conditioning = bool(
            cop_obb_rotation_conditioning)
        self.cop_obb_rotation_refinement = bool(
            cop_obb_rotation_refinement)
        if (self.cop_obb_rotation_conditioning and
                self.cop_obb_rotation_refinement):
            raise ValueError(
                'cop_obb_rotation_conditioning and '
                'cop_obb_rotation_refinement are mutually exclusive')
        if ((self.cop_obb_rotation_conditioning or
             self.cop_obb_rotation_refinement) and
                self.loss_obb_aux is None):
            raise ValueError(
                'OBB rotation conditioning/refinement requires loss_obb_aux')
        if ((self.cop_obb_rotation_conditioning or
             self.cop_obb_rotation_refinement) and
                self.cop_chain_order[-1] != 'rotation'):
            raise ValueError(
                'OBB rotation conditioning/refinement requires rotation to '
                'be the final CoP stage')
        self.expose_obb_aux_predictions = bool(
            expose_obb_aux_predictions)
        if self.expose_obb_aux_predictions and self.loss_obb_aux is None:
            raise ValueError(
                'expose_obb_aux_predictions requires loss_obb_aux')

        # ── GauCho-3D ellipsoid branch ─────────────────────────────────
        # The canonical 3D state is ``Ellipsoid3D(t, Sigma)``.  ``Sigma`` comes
        # from a Cholesky chart, so no angle, quaternion, or axis ordering is
        # regressed and the unobservable pose of a sphere or spheroid simply
        # does not appear in the loss.  The 3D centre is *not* duplicated: it
        # reuses the existing 2D centre and depth branches, which keeps this an
        # Independent baseline over a shared centre rather than a second,
        # competing translation estimate.
        valid_charts = {'scale_shape', 'direct', 'dual_plane'}
        if gaucho_chart not in valid_charts:
            raise ValueError(
                f'gaucho_chart must be one of {sorted(valid_charts)}, '
                f'got {gaucho_chart!r}')
        if gaucho_size_prior <= 0.0 or not math.isfinite(gaucho_size_prior):
            raise ValueError('gaucho_size_prior must be positive and finite')
        self.gaucho_ellipsoid = bool(gaucho_ellipsoid)
        self.gaucho_chart = gaucho_chart
        # Ablation A4 vs A5: pinning rc = 0 reduces dual-plane to two orthogonal
        # 2D GauCho factors, which cannot represent a general triaxial
        # ellipsoid.  Measuring that shortfall is the point.
        self.gaucho_dual_plane_fix_rc_zero = bool(gaucho_dual_plane_fix_rc_zero)
        if gaucho_dual_plane_fix_rc_zero and gaucho_chart != 'dual_plane':
            raise ValueError(
                "gaucho_dual_plane_fix_rc_zero only applies to "
                f"gaucho_chart='dual_plane', got {gaucho_chart!r}")
        self.gaucho_size_prior = float(gaucho_size_prior)
        self.gaucho_classwise = bool(gaucho_classwise)
        self.gaucho_use_bbox_conditioning = bool(gaucho_use_bbox_conditioning)
        self.gaucho_eps = 1e-7
        self.gaucho_ellipse2d = bool(gaucho_ellipse2d)
        self.gaucho_ellipse2d_dn = bool(gaucho_ellipse2d_dn)
        self.expose_gaucho_predictions = bool(expose_gaucho_predictions)
        self.loss_ellipse2d = (
            MODELS.build(loss_ellipse2d)
            if loss_ellipse2d is not None else None)
        if self.gaucho_ellipse2d != (self.loss_ellipse2d is not None):
            raise ValueError(
                'gaucho_ellipse2d and loss_ellipse2d must be enabled together')
        if self.gaucho_ellipse2d_dn and not self.gaucho_ellipse2d:
            raise ValueError(
                'gaucho_ellipse2d_dn requires gaucho_ellipse2d=True')
        if self._assigner_uses_ellipse and not self.gaucho_ellipse2d:
            raise ValueError(
                'Ellipse2DKLDCost requires gaucho_ellipse2d=True')
        self.loss_ellipsoid = (
            MODELS.build(loss_ellipsoid)
            if loss_ellipsoid is not None else None)
        self.loss_ellipsoid_projection = (
            MODELS.build(loss_ellipsoid_projection)
            if loss_ellipsoid_projection is not None else None)
        self.loss_ellipsoid_max_axis = (
            MODELS.build(loss_ellipsoid_max_axis)
            if loss_ellipsoid_max_axis is not None else None)
        # The ellipse KLD reduces to the Frobenius norm of L_p^{-1} L_g, which
        # is nearly constant under rotation once the shape is near-circular --
        # and these objects have a median aspect ratio of 1.20.  Measured: the
        # angle error sits at 17 degrees and did not move when the KLD weight
        # was tripled, while the metric, which compares envelope corners, still
        # charges 0.20 of IoU for it.  This term supplies the orientation
        # gradient the KLD structurally cannot.  Auxiliary, never a replacement.
        self.loss_ellipse2d_corner = (
            MODELS.build(loss_ellipse2d_corner)
            if loss_ellipse2d_corner is not None else None)
        if self.loss_ellipse2d_corner is not None and not self.gaucho_ellipse2d:
            raise ValueError(
                'loss_ellipse2d_corner requires gaucho_ellipse2d=True')
        # Explicit, pi-periodic angle supervision, weighted by how identifiable
        # the target's orientation is.  The reference implementation this work
        # is measured against does exactly this rather than leaving orientation
        # to the KLD, which has no rotation gradient for a near-circular shape.
        self.loss_ellipse2d_angle = (
            MODELS.build(loss_ellipse2d_angle)
            if loss_ellipse2d_angle is not None else None)
        if self.loss_ellipse2d_angle is not None and not self.gaucho_ellipse2d:
            raise ValueError(
                'loss_ellipse2d_angle requires gaucho_ellipse2d=True')
        # A bound on the physical size of the object, in metres.  Two distinct
        # jobs, deliberately separate knobs: ``ellipsoid_gt_max_diameter``
        # drops annotations larger than the objects can be from the 3D shape
        # supervision, and ``ellipsoid_max_diameter`` caps the emitted
        # ellipsoid at inference.  Both default to off, so a config that does
        # not ask for them is bit-identical to before.
        for name, value in (('ellipsoid_gt_max_diameter',
                             ellipsoid_gt_max_diameter),
                            ('ellipsoid_max_diameter', ellipsoid_max_diameter)):
            if value is not None and not value > 0.0:
                raise ValueError(f'{name} must be positive, got {value}')
        self.ellipsoid_gt_max_diameter = ellipsoid_gt_max_diameter
        self.ellipsoid_max_diameter = ellipsoid_max_diameter
        # Metres per unit of the packed depth channel.  With it set, the
        # emitted ellipsoid keeps its predicted bearing and shape but takes
        # its *range* from the sensor, which measurement says is four times
        # more accurate than the regressed one.  ``None`` leaves the model
        # exactly as it was.
        if sensor_depth_scale is not None and not sensor_depth_scale > 0.0:
            raise ValueError(
                'sensor_depth_scale must be positive, got '
                f'{sensor_depth_scale}')
        self.sensor_depth_scale = sensor_depth_scale
        self.sensor_depth_window = int(sensor_depth_window)
        # Structural anchor: the forward pass adds the sensor's depth at
        # each query's predicted centre to the regressed z, so the network
        # learns the residual and every consumer of z -- losses, matching,
        # denoising, inference -- keeps seeing absolute metres unchanged.
        # Measured basis: the depth channel reads the object centre to
        # 5.3 mm where the regression is off by 21.9 mm, and range alone
        # is a 15x factor on the shared 3D AP.
        self.sensor_depth_anchor = bool(sensor_depth_anchor)
        # Denoising queries are noised copies of the annotations, so the
        # depth sampled at their displaced centre lands off the object and
        # the residual is asked to undo a displacement it cannot see.
        # ``cop_z_out`` is shared with the matching queries, so that
        # correction leaks into them.  Turning this off leaves denoising
        # regressing absolute depth exactly as before.
        self.sensor_depth_anchor_denoising = bool(
            sensor_depth_anchor_denoising)
        self._num_denoising_queries = 0
        if self.sensor_depth_anchor and self.sensor_depth_scale is None:
            raise ValueError(
                'sensor_depth_anchor requires sensor_depth_scale')
        self.sensor_depth_map = None
        if self.loss_ellipsoid_max_axis is not None and \
                not self.gaucho_ellipsoid:
            raise ValueError(
                'loss_ellipsoid_max_axis requires gaucho_ellipsoid=True')
        if not self.gaucho_ellipsoid and (
                self.loss_ellipsoid is not None or
                self.loss_ellipsoid_projection is not None):
            raise ValueError(
                'loss_ellipsoid/loss_ellipsoid_projection require '
                'gaucho_ellipsoid=True')
        if self.gaucho_ellipsoid and self.loss_ellipsoid is None and \
                self.loss_ellipsoid_projection is None:
            raise ValueError(
                'gaucho_ellipsoid=True needs at least one ellipsoid loss')
        # ``self.train_intrinsic_to_image_space`` is assigned further down, so
        # read the argument directly.
        if self.loss_ellipsoid_projection is not None and \
                not train_intrinsic_to_image_space:
            # The projection target is the OBB Gaussian *after* the resize
            # pipeline, so the dual quadric has to be projected with the
            # image-space K.  With the stored original-image K the centre
            # round-trips and the error hides entirely in the shape term,
            # silently pulling against the metric-scale direct 3D loss.
            raise ValueError(
                'loss_ellipsoid_projection requires '
                'train_intrinsic_to_image_space=True; the projection target '
                'lives in resized image pixels')
        if self.loss_ellipsoid_projection is not None and \
                self.loss_ellipsoid is None:
            # Optimizing only ``2D observation <-> projected 3D`` lets the two
            # collapse onto a common wrong ellipse.  A second, independent
            # ground of supervision is mandatory.
            raise ValueError(
                'loss_ellipsoid_projection alone is not a valid objective; '
                'pair it with loss_ellipsoid (direct 3D supervision)')
        if projection_geometry_source not in {'target', 'prediction'}:
            raise ValueError(
                "projection_geometry_source must be 'target' or 'prediction', "
                f"got {projection_geometry_source!r}")
        self.projection_geometry_source = projection_geometry_source
        self.train_intrinsic_to_image_space = bool(
            train_intrinsic_to_image_space)

        if self.loss_cls.use_sigmoid:
            self.cls_out_channels = num_classes
        else:
            self.cls_out_channels = num_classes + 1
        
        self.share_pred_layer = share_pred_layer
        self.num_pred_layer = num_pred_layer
        self.as_two_stage = as_two_stage
        self.classwise_rotation = classwise_rotation
        self.classwise_sizes = classwise_sizes
        if o2m_aux_topk < 0:
            raise ValueError('o2m_aux_topk must be non-negative')
        if o2m_aux_loss_weight < 0:
            raise ValueError('o2m_aux_loss_weight must be non-negative')
        self.o2m_aux_topk = int(o2m_aux_topk)
        self.o2m_aux_loss_weight = float(o2m_aux_loss_weight)
        self.o2m_aux_loss_cls = (
            MODELS.build(o2m_aux_loss_cls)
            if self.o2m_aux_topk else None)


        self._init_layers()
        self._init_distillation_teachers()

    def _classification_loss(
            self,
            cls_scores: Tensor,
            labels: Tensor,
            label_weights: Tensor,
            bbox_predictions: Tensor,
            bbox_targets: Tensor,
            avg_factor,
            obb_predictions: Tensor = None,
            obb_targets: Tensor = None) -> Tensor:
        """Dispatch hard-label and quality-aware classification uniformly.

        Geometry-quality construction is delegated to a configurable,
        stateless policy.  The detector head only supplies aligned tensors and
        therefore does not need a branch for each future quality source.
        """
        if not self.uses_quality_target:
            return self.loss_cls(
                cls_scores, labels, label_weights, avg_factor=avg_factor)
        quality = self.quality_target_policy(
            labels=labels,
            bbox_predictions=bbox_predictions,
            bbox_targets=bbox_targets,
            obb_predictions=obb_predictions,
            obb_targets=obb_targets,
        )
        return self.loss_cls(
            cls_scores, (labels, quality), label_weights,
            avg_factor=avg_factor)

    @staticmethod
    def _normalize_obb_gaussian_targets(
            targets: Tensor, factors: Tensor) -> Tensor:
        """Normalize compact Gaussian targets into the head output frame."""
        if targets.ndim != 2 or targets.shape[-1] != 5:
            raise ValueError(
                'compact Gaussian targets must have shape (N, 5), got '
                f'{targets.shape}')
        if factors.shape != (len(targets), 4):
            raise ValueError(
                'image factors must have shape (N, 4), got '
                f'{factors.shape}')
        normalized = targets.clone()
        image_width = factors[:, 0]
        image_height = factors[:, 1]
        normalized[:, 0] /= image_width
        normalized[:, 1] /= image_height
        normalized[:, 2] /= image_width.square()
        normalized[:, 3] /= image_width * image_height
        normalized[:, 4] /= image_height.square()
        return normalized

    def replicate(self, layer, num_layers):
        """Replicate a layer using shared instances or deep copies based on self.share_pred_layer."""
        if self.share_pred_layer:
            return nn.ModuleList([layer] * num_layers)
        else:
            return nn.ModuleList([copy.deepcopy(layer) for _ in range(num_layers)])

    def _init_layers(self) -> None:
        """Initialize classification branch and pose regression branches."""
        fc_cls = Linear(self.embed_dims, self.cls_out_channels)
        self.cls_branches = self.replicate(fc_cls, self.num_pred_layer)

        reg_branch = []
        for _ in range(self.num_reg_fcs):
            reg_branch.append(Linear(self.embed_dims, self.embed_dims))
            reg_branch.append(nn.ReLU())
        reg_branch.append(Linear(self.embed_dims, 4))
        reg_branch = nn.Sequential(*reg_branch)
        self.reg_branches = self.replicate(reg_branch, self.num_pred_layer)

        # Separate training-only one-to-many predictor.  It consumes shared
        # decoder features but is never called by ``forward``/``predict``.
        if self.o2m_aux_topk:
            self.o2m_cls_branch = Linear(
                self.embed_dims, self.cls_out_channels)
            self.o2m_reg_branch = copy.deepcopy(reg_branch)

        all_branches = []
        for i in range(4): # centers_2d, z, rotation, sizes
            reg_branch = []
            for _ in range(self.num_reg_fcs):
                embed_dims = self.embed_dims
                if i == 0 and self.use_bbox_for_centers_2d:
                    embed_dims += 4
                elif i == 1 and self.use_bbox_for_z:
                    embed_dims += 4
                elif i == 2 and self.use_bbox_for_rotation:
                    embed_dims += 4
                elif i == 3 and self.use_bbox_for_size:
                    embed_dims += 4
                reg_branch.append(Linear(embed_dims, embed_dims))
                reg_branch.append(nn.ReLU())
            if i == 0: # centers_2d
                reg_branch.append(Linear(embed_dims, 2))
            elif i == 1: # z
                reg_branch.append(Linear(embed_dims, 1))
            elif i == 2: # rotation
                if self.classwise_rotation:
                    out_dim = self.rot_dim * self.num_classes
                else:
                    out_dim = self.rot_dim
                reg_branch.append(Linear(embed_dims, out_dim))
            elif i == 3: # sizes
                out_dim = 3 * self.num_classes if self.classwise_sizes else 3
                reg_branch.append(Linear(embed_dims, out_dim))
            reg_branch = nn.Sequential(*reg_branch)
            all_branches.append(reg_branch)

        # GauCho-2D amodal ellipse branch: five unconstrained values per
        # class, ``(tx, ty, r, u, v)``, in the same scale--shape Cholesky chart
        # the reference 2D implementation uses.  No angle is predicted.
        if self.gaucho_ellipse2d:
            ellipse_dims = self.embed_dims + 4
            ellipse_branch = []
            for _ in range(self.num_reg_fcs):
                ellipse_branch.append(Linear(ellipse_dims, ellipse_dims))
                ellipse_branch.append(nn.ReLU())
            ellipse_out = 5 * self.num_classes if self.gaucho_classwise else 5
            ellipse_branch.append(Linear(ellipse_dims, ellipse_out))
            self.reg_ellipse2d_branch = self.replicate(
                nn.Sequential(*ellipse_branch), self.num_pred_layer)

        # GauCho-3D shape branch: six unconstrained values per class that a
        # Cholesky chart turns into an SPD(3) shape matrix.
        if self.gaucho_ellipsoid:
            gaucho_dims = self.embed_dims
            if self.gaucho_use_bbox_conditioning:
                gaucho_dims += 4
            gaucho_branch = []
            for _ in range(self.num_reg_fcs):
                gaucho_branch.append(Linear(gaucho_dims, gaucho_dims))
                gaucho_branch.append(nn.ReLU())
            gaucho_out = 6 * self.num_classes if self.gaucho_classwise else 6
            gaucho_branch.append(Linear(gaucho_dims, gaucho_out))
            self.reg_ellipsoid_branch = self.replicate(
                nn.Sequential(*gaucho_branch), self.num_pred_layer)

        self.reg_centers_2d_branch = self.replicate(all_branches[0], self.num_pred_layer)
        self.reg_z_branch = self.replicate(all_branches[1], self.num_pred_layer)
        self.reg_rotation_branch = self.replicate(all_branches[2], self.num_pred_layer)
        self.reg_size_branch = self.replicate(all_branches[3], self.num_pred_layer)

        # ── CoP (Chain-of-Prediction) attribute nets ────────────────────
        # AttributeNet A(·) per attribute (MonoCoP Eq.6): two Linear layers
        # with ReLU between. ``cop_chain_order`` controls the order and each
        # step uses residual aggregation. ``auxiliary`` keeps the historical
        # parallel primary path; ``chain`` routes these outputs into matching,
        # losses, denoising losses, and inference.
        if self.use_cop_chain:

            def _attr_net():
                return nn.Sequential(
                    nn.Linear(self.embed_dims, self.embed_dims), nn.ReLU(),
                    nn.Linear(self.embed_dims, self.embed_dims))

            self.cop_size_net = self.replicate(_attr_net(), self.num_pred_layer)
            self.cop_rotation_net = self.replicate(
                _attr_net(), self.num_pred_layer)
            self.cop_z_net = self.replicate(_attr_net(), self.num_pred_layer)

            # Final prediction heads on the aggregated CoP features.
            s_dim = 3 * self.num_classes if self.classwise_sizes else 3
            r_dim = self.rot_dim * self.num_classes if self.classwise_rotation else self.rot_dim
            self.cop_size_out = self.replicate(
                nn.Linear(self.embed_dims, s_dim), self.num_pred_layer)
            self.cop_rotation_out = self.replicate(
                nn.Linear(self.embed_dims, r_dim), self.num_pred_layer)
            self.cop_z_out = self.replicate(
                nn.Linear(self.embed_dims, 1), self.num_pred_layer)
            if self.loss_obb_aux is not None:
                self.cop_obb_out = self.replicate(
                    nn.Linear(self.embed_dims, 3), self.num_pred_layer)
                if (self.cop_obb_rotation_conditioning or
                        self.cop_obb_rotation_refinement):
                    self.cop_obb_rotation_embed = self.replicate(
                        nn.Linear(2, self.embed_dims), self.num_pred_layer)
            if self.cop_use_bbox_conditioning:
                self.cop_bbox_embed = self.replicate(
                    nn.Linear(4, self.embed_dims), self.num_pred_layer)
            self.cop_stage_fusions = nn.ModuleDict({
                attribute: self.replicate(
                    CoPStageFusion(
                        embed_dims=self.embed_dims,
                        mode=self.cop_fusion_mode,
                    ),
                    self.num_pred_layer,
                )
                for attribute in ('z', 'size', 'rotation')
            })
            if self.requires_depth_features:
                self.depth_query_sampler = MultiScaleDepthQuerySampler(
                    embed_dims=self.embed_dims,
                    **self.cop_depth_context_cfg,
                )


    @torch.no_grad()
    def _sample_depth_anchor(self, centres_norm: Tensor) -> Tensor:
        """Metric depth at each query's predicted centre, as a constant.

        ``centres_norm`` is ``(bs, num_queries, 2)`` in normalized image
        coordinates and arrives detached: giving the sampling location a
        gradient would let the centre go shopping for convenient depths.

        A 5x5 median around the centre, valid pixels only -- the same window
        the offline measurement used when it found 5.3 mm.  A centre with no
        valid return in its window falls back to the image's own median valid
        depth, so the residual the network learns keeps meaning "correction to
        an approximately right range" instead of flipping between residual and
        absolute regression on the 3.6% of centres the sensor misses.
        """
        depth_map = self.sensor_depth_map
        batch, queries = centres_norm.shape[:2]
        if depth_map is None:
            return centres_norm.new_zeros(batch, queries, 1)
        depth = depth_map[:, 0].float()
        height, width = depth.shape[-2:]
        x = (centres_norm[..., 0].float() * width).round().long()
        y = (centres_norm[..., 1].float() * height).round().long()
        x = x.clamp(0, width - 1)
        y = y.clamp(0, height - 1)

        window = self.sensor_depth_window
        samples = []
        for dy in range(-window, window + 1):
            for dx in range(-window, window + 1):
                yy = (y + dy).clamp(0, height - 1)
                xx = (x + dx).clamp(0, width - 1)
                flat = (yy * width + xx).reshape(batch, queries)
                samples.append(torch.gather(
                    depth.reshape(batch, -1), 1, flat))
        stacked = torch.stack(samples, dim=-1)
        stacked = torch.where(stacked > 0, stacked,
                              torch.full_like(stacked, float("nan")))
        measured = stacked.nanmedian(dim=-1).values

        fallback = []
        for index in range(batch):
            valid = depth[index][depth[index] > 0]
            fallback.append(valid.median() if valid.numel()
                            else depth.new_tensor(0.0))
        fallback = torch.stack(fallback).unsqueeze(-1).expand(batch, queries)
        anchored = torch.where(torch.isfinite(measured), measured, fallback)
        return (anchored * self.sensor_depth_scale).unsqueeze(-1).to(
            centres_norm.dtype)

    def _init_distillation_teachers(self) -> None:
        """Delegate frozen teacher adaptation to the distillation module."""
        self._last_distillation_targets = {}
        self._last_distillation_scores = None
        self.distillation_teacher_report = {}
        self.distillation_teacher = None
        if not self.distill_attributes:
            return
        self.distillation_teacher = StagedDistillationTeacher(
            attributes=self.distill_attributes,
            embed_dims=self.embed_dims,
            num_reg_fcs=self.num_reg_fcs,
            num_pred_layer=self.num_pred_layer,
            student_center_branches=self.reg_centers_2d_branch,
            student_cls_branches=self.cls_branches,
            student_pose_branches={
                'z': self.reg_z_branch,
                'size': self.reg_size_branch,
                'rotation': self.reg_rotation_branch,
            },
            center_uses_bbox=self.use_bbox_for_centers_2d,
            pose_uses_bbox={
                'z': self.use_bbox_for_z,
                'size': self.use_bbox_for_size,
                'rotation': self.use_bbox_for_rotation,
            },
            center_teacher_source=self.center_teacher_source,
            obb_checkpoint=self.obb_center_teacher_checkpoint,
            pose_checkpoint=self.pose_teacher_checkpoint,
            loss_weights=self.distill_loss_weights,
            score_threshold=self.distill_score_threshold)
        self.distillation_teacher_report = self.distillation_teacher.report

    def init_weights(self) -> None:
        """Initialize weights of the DINO 9D pose head."""
        if self.loss_cls.use_sigmoid:
            bias_init = bias_init_with_prob(0.01)
            for m in self.cls_branches:
                if hasattr(m, 'bias') and m.bias is not None:
                    nn.init.constant_(m.bias, bias_init)
            if self.o2m_aux_topk:
                nn.init.constant_(self.o2m_cls_branch.bias, bias_init)
        # Initialize bbox regression branches
        for m in self.reg_branches:
            constant_init(m[-1], 0, bias=0)
        if self.o2m_aux_topk:
            constant_init(self.o2m_reg_branch[-1], 0, bias=0)
        nn.init.constant_(self.reg_branches[0][-1].bias.data[2:], -2.0)
        
        # Zero-initialize the GauCho shape branch so every query starts as an
        # isotropic ellipsoid of exactly the size prior.  A random start would
        # place some queries at extreme anisotropies whose Cholesky diagonal
        # sits at the clip boundary, where the chart has no gradient.
        if self.gaucho_ellipsoid:
            for m in self.reg_ellipsoid_branch:
                constant_init(m[-1], 0, bias=0)
        if self.gaucho_ellipse2d:
            # ``r = u = v = 0`` starts every query at the circle inscribed in
            # its own predicted box, which is the correct uninformed prior.
            for m in self.reg_ellipse2d_branch:
                constant_init(m[-1], 0, bias=0)

        # Initialize centers_2d regression branches
        for m in self.reg_centers_2d_branch:
            constant_init(m[-1], 0, bias=0)
        nn.init.constant_(self.reg_centers_2d_branch[0][-1].bias.data, -2.0)
        for m in self.reg_z_branch:
            # constant_init(m[-1], 0, bias=0.0)
            constant_init(m[-1], 0, bias=0.5)
        for m in self.reg_rotation_branch:
            # constant_init(m[-1], 0, bias=0.0)
            constant_init(m[-1], 0, bias=0.5)
        for m in self.reg_size_branch:
            # constant_init(m[-1], 0, bias=0.0)
            constant_init(m[-1], 0, bias=0.5)

        if self.use_cop_chain:
            for m in self.cop_size_out:
                constant_init(m, 0, bias=0.5)
            for m in self.cop_rotation_out:
                constant_init(m, 0, bias=0.5)
            for m in self.cop_z_out:
                constant_init(m, 0, bias=0.5)
            if self.loss_obb_aux is not None:
                initial_cholesky = -3.4915204  # softplus(x) ~= 0.03
                for m in self.cop_obb_out:
                    nn.init.normal_(m.weight, mean=0.0, std=1e-3)
                    with torch.no_grad():
                        m.bias.copy_(m.bias.new_tensor(
                            [initial_cholesky, 0.0, initial_cholesky]))
                if (self.cop_obb_rotation_conditioning or
                        self.cop_obb_rotation_refinement):
                    for m in self.cop_obb_rotation_embed:
                        nn.init.normal_(m.weight, mean=0.0, std=1e-3)
                        nn.init.zeros_(m.bias)

        if self.as_two_stage:
            for m in self.reg_branches:
                nn.init.constant_(m[-1].bias.data[2:], 0.0)
            # for m in self.reg_centers_2d_branch:
                # nn.init.constant_(m[-1].bias.data, 0.0)

    def forward(self, hidden_states: Tensor,
                references: List[Tensor],
                batch_img_metas=None,
                depth_features=None,
                compute_distillation_targets: bool = False
                ) -> Tuple[Tensor, Tensor, Tensor, Tensor, Tensor, Tensor]:
        """Forward function for 9D pose estimation with separate centers_2d and z."""
        all_layers_outputs_classes = []
        all_layers_outputs_coords = []
        all_layers_outputs_centers_2d = []
        all_layers_outputs_z = []
        all_layers_outputs_rotations = []
        all_layers_outputs_sizes = []
        all_layers_outputs_size_chain = []
        all_layers_outputs_rotation_chain = []
        all_layers_outputs_z_chain = []
        all_layers_outputs_obb_aux = []
        all_layers_outputs_ellipsoid = []
        all_layers_outputs_ellipse2d = []
        depth_context_shapes = []

        if self.requires_depth_features and depth_features is None:
            raise ValueError(
                'depth_features are required when '
                'cop_fusion_mode="depth_dense"')

        if self.use_cuboid_conditioning:
            if batch_img_metas is None:
                raise ValueError('batch_img_metas should not be None when use_cuboid_conditioning is True.')
            intrinsics = [i['intrinsics'] for i in batch_img_metas]
            intrinsics = torch.tensor(intrinsics, device=hidden_states.device)


        for layer_id in range(hidden_states.shape[0]):
            reference = inverse_sigmoid(references[layer_id])
            hidden_state = hidden_states[layer_id]
            outputs_class = self.cls_branches[layer_id](hidden_state)
            tmp_reg_bbox_preds = self.reg_branches[layer_id](hidden_state)

            if reference.shape[-1] == 4:
                tmp_reg_bbox_preds += reference
            else:
                assert reference.shape[-1] == 2
                tmp_reg_bbox_preds[..., :2] += reference
            outputs_coord = tmp_reg_bbox_preds.sigmoid()
            if self.requires_depth_features:
                depth_query = self.depth_query_sampler(
                    depth_features, outputs_coord)
                depth_context_shapes.append(tuple(depth_query.shape))
            else:
                depth_query = None

            if self.use_bbox_for_centers_2d:
                tmp_centers_2d_input = torch.cat((hidden_state, outputs_coord), dim=-1)
            else:
                tmp_centers_2d_input = hidden_state

            if self.use_bbox_for_z:
                tmp_z_input = torch.cat((hidden_state, outputs_coord), dim=-1)
            else:
                tmp_z_input = hidden_state

            if self.use_bbox_for_rotation:
                tmp_rotation_input = torch.cat((hidden_state, outputs_coord), dim=-1)
            else:
                tmp_rotation_input = hidden_state
            if self.use_bbox_for_size:
                tmp_size_input = torch.cat((hidden_state, outputs_coord), dim=-1)
            else:
                tmp_size_input = hidden_state

            tmp_reg_centers_2d_preds = self.reg_centers_2d_branch[layer_id](tmp_centers_2d_input)
            tmp_reg_centers_2d_preds = tmp_reg_centers_2d_preds.sigmoid() + outputs_coord[..., :2] - 0.5
            if self.cop_prediction_mode != 'chain':
                tmp_reg_z_preds = self.reg_z_branch[layer_id](tmp_z_input)
                tmp_rotation_preds = self.reg_rotation_branch[layer_id](tmp_rotation_input)
                tmp_sizes = self.reg_size_branch[layer_id](tmp_size_input)
            else:
                tmp_reg_z_preds = tmp_rotation_preds = tmp_sizes = None

            if self.use_cop_chain:
                chain_feature = hidden_state
                if self.cop_use_bbox_conditioning:
                    chain_feature = chain_feature + self.cop_bbox_embed[layer_id](
                        outputs_coord)
                chain_outputs = {}
                chain_features = {}
                nets = {
                    'z': self.cop_z_net[layer_id],
                    'size': self.cop_size_net[layer_id],
                    'rotation': self.cop_rotation_net[layer_id],
                }
                outs = {
                    'z': self.cop_z_out[layer_id],
                    'size': self.cop_size_out[layer_id],
                    'rotation': self.cop_rotation_out[layer_id],
                }
                tmp_obb_aux = None
                for attribute in self.cop_chain_order:
                    stage_input = self.cop_stage_fusions[attribute][layer_id](
                        chain_feature, hidden_state, depth_query)
                    if (attribute == 'rotation' and
                            self.cop_obb_rotation_conditioning):
                        raw_obb = self.cop_obb_out[layer_id](stage_input)
                        tmp_obb_aux = _compact_gaussian_from_raw_cholesky(
                            raw_obb)
                        orientation_descriptor = \
                            compact_gaussian_orientation_descriptor(
                                tmp_obb_aux)
                        stage_input = stage_input + \
                            self.cop_obb_rotation_embed[layer_id](
                                orientation_descriptor)
                    chain_feature = nets[attribute](stage_input) + stage_input
                    chain_outputs[attribute] = outs[attribute](chain_feature)
                    chain_features[attribute] = chain_feature
                if self.loss_obb_aux is not None and tmp_obb_aux is None:
                    raw_obb = self.cop_obb_out[layer_id](
                        chain_features['rotation'])
                    tmp_obb_aux = _compact_gaussian_from_raw_cholesky(raw_obb)
                if self.cop_obb_rotation_refinement:
                    orientation_descriptor = \
                        compact_gaussian_orientation_descriptor(
                            tmp_obb_aux)
                    refined_rotation_feature = chain_features['rotation'] + \
                        self.cop_obb_rotation_embed[layer_id](
                            orientation_descriptor)
                    chain_outputs['rotation'] = self.cop_rotation_out[
                        layer_id](refined_rotation_feature)
                tmp_z_chain = chain_outputs['z']
                tmp_sizes_chain = chain_outputs['size']
                tmp_rotation_chain = chain_outputs['rotation']
                if self.cop_prediction_mode == 'chain':
                    tmp_reg_z_preds = tmp_z_chain
                    tmp_sizes = tmp_sizes_chain
                    tmp_rotation_preds = tmp_rotation_chain
                    tmp_sizes_chain = tmp_rotation_chain = tmp_z_chain = None
            else:
                tmp_sizes_chain = tmp_rotation_chain = tmp_z_chain = None
                tmp_obb_aux = None


            if (self.sensor_depth_anchor and tmp_reg_z_preds is not None
                    and self.sensor_depth_map is not None):
                # The network's z output becomes a residual: what leaves the
                # head is anchor + delta, still absolute metres, so no loss,
                # target, matcher or decoder changes meaning.  Zero-initialised
                # z outputs therefore start the model at exactly "trust the
                # sensor", and training can only improve on that.
                anchor = self._sample_depth_anchor(
                    tmp_reg_centers_2d_preds.detach())
                split = self._num_denoising_queries
                if split:
                    anchor = torch.cat(
                        (torch.zeros_like(anchor[:, :split]),
                         anchor[:, split:]), dim=1)
                tmp_reg_z_preds = tmp_reg_z_preds + anchor

            all_layers_outputs_classes.append(outputs_class)
            all_layers_outputs_coords.append(outputs_coord)
            all_layers_outputs_centers_2d.append(tmp_reg_centers_2d_preds)
            all_layers_outputs_z.append(tmp_reg_z_preds)
            all_layers_outputs_rotations.append(tmp_rotation_preds)
            all_layers_outputs_sizes.append(tmp_sizes)
            all_layers_outputs_size_chain.append(tmp_sizes_chain)
            all_layers_outputs_rotation_chain.append(tmp_rotation_chain)
            all_layers_outputs_z_chain.append(tmp_z_chain)
            all_layers_outputs_obb_aux.append(tmp_obb_aux)

            if self.gaucho_ellipse2d:
                all_layers_outputs_ellipse2d.append(
                    self.reg_ellipse2d_branch[layer_id](
                        torch.cat((hidden_state, outputs_coord), dim=-1)))
            else:
                all_layers_outputs_ellipse2d.append(None)

            if self.gaucho_ellipsoid:
                if self.gaucho_use_bbox_conditioning:
                    gaucho_input = torch.cat(
                        (hidden_state, outputs_coord), dim=-1)
                else:
                    gaucho_input = hidden_state
                all_layers_outputs_ellipsoid.append(
                    self.reg_ellipsoid_branch[layer_id](gaucho_input))
            else:
                all_layers_outputs_ellipsoid.append(None)

        all_layers_outputs_classes = torch.stack(all_layers_outputs_classes)
        all_layers_outputs_coords = torch.stack(all_layers_outputs_coords)
        all_layers_outputs_centers_2d = torch.stack(all_layers_outputs_centers_2d)
        all_layers_outputs_z = torch.stack(all_layers_outputs_z)
        all_layers_outputs_rotations = torch.stack(all_layers_outputs_rotations)
        all_layers_outputs_sizes = torch.stack(all_layers_outputs_sizes)
        if self._uses_auxiliary_chain:
            all_layers_outputs_size_chain = torch.stack(all_layers_outputs_size_chain)
            all_layers_outputs_rotation_chain = torch.stack(all_layers_outputs_rotation_chain)
            all_layers_outputs_z_chain = torch.stack(all_layers_outputs_z_chain)
        else:
            all_layers_outputs_size_chain = all_layers_outputs_rotation_chain = None
            all_layers_outputs_z_chain = None
        if self.loss_obb_aux is not None:
            all_layers_outputs_obb_aux = torch.stack(
                all_layers_outputs_obb_aux)
        else:
            all_layers_outputs_obb_aux = None
        if self.gaucho_ellipsoid:
            all_layers_outputs_ellipsoid = torch.stack(
                all_layers_outputs_ellipsoid)
        else:
            all_layers_outputs_ellipsoid = None
        if self.gaucho_ellipse2d:
            all_layers_outputs_ellipse2d = torch.stack(
                all_layers_outputs_ellipse2d)
        else:
            all_layers_outputs_ellipse2d = None

        if depth_context_shapes:
            first_shape = depth_context_shapes[0]
            if any(shape != first_shape for shape in depth_context_shapes):
                raise RuntimeError(
                    f'depth context shape changed by layer: '
                    f'{depth_context_shapes}')
            self.last_depth_context_shape = (
                len(depth_context_shapes), *first_shape)
            dense_shapes = [
                self.cop_stage_fusions[self.cop_chain_order[0]][layer_id]
                .last_dense_input_shape
                for layer_id in range(len(depth_context_shapes))
            ]
            if any(shape != dense_shapes[0] for shape in dense_shapes):
                raise RuntimeError(
                    f'dense CoP input shape changed by layer: {dense_shapes}')
            self.last_dense_input_shape = (
                len(dense_shapes), *dense_shapes[0])
        else:
            self.last_depth_context_shape = None
            self.last_dense_input_shape = None

        if (compute_distillation_targets and
                self.distillation_teacher is not None):
            (self._last_distillation_targets,
             self._last_distillation_scores) = self.distillation_teacher(
                 hidden_states, references, all_layers_outputs_coords)
        else:
            self._last_distillation_targets = {}
            self._last_distillation_scores = None

        # The GauCho tensor is appended last so existing positional consumers
        # (``outs[:6]`` for inference, ``outs[9]`` for OBB diagnostics) are
        # untouched and disabled configs keep byte-identical behaviour.
        return (all_layers_outputs_classes, all_layers_outputs_coords,
                all_layers_outputs_centers_2d, all_layers_outputs_z,
                all_layers_outputs_rotations, all_layers_outputs_sizes,
                all_layers_outputs_size_chain, all_layers_outputs_rotation_chain,
                all_layers_outputs_z_chain, all_layers_outputs_obb_aux,
                all_layers_outputs_ellipsoid, all_layers_outputs_ellipse2d)

    def predict(self,
                hidden_states: Tensor,
                references: List[Tensor],
                batch_data_samples: SampleList,
                rescale: bool = True,
                depth_features=None) -> InstanceList:
        """Predict poses and optionally expose query OBBs for diagnostics."""
        batch_img_metas = [
            data_sample.metainfo for data_sample in batch_data_samples
        ]
        outs = self(
            hidden_states, references, depth_features=depth_features)
        predictions = self.predict_by_feat(
            *outs[:6], batch_img_metas=batch_img_metas, rescale=rescale)
        expose_gaucho = self.expose_gaucho_predictions and (
            self.gaucho_ellipse2d or self.gaucho_ellipsoid)
        if not self.expose_obb_aux_predictions and not expose_gaucho:
            return predictions

        cls_scores = outs[0][-1]
        obb_predictions = outs[9][-1] if self.expose_obb_aux_predictions \
            else None
        for image_index, result in enumerate(predictions):
            cls_score = cls_scores[image_index]
            max_per_img = self.test_cfg.get(
                'max_per_img', len(cls_score))
            if self.loss_cls.use_sigmoid:
                _, indexes = cls_score.sigmoid().reshape(-1).topk(
                    max_per_img)
                bbox_index = indexes // self.num_classes
                query_labels = indexes % self.num_classes
            else:
                scores, query_labels = F.softmax(
                    cls_score, dim=-1)[..., :-1].max(-1)
                _, bbox_index = scores.topk(max_per_img)
                query_labels = query_labels[bbox_index]
            if obb_predictions is not None:
                result.obb_gaussians = obb_predictions[image_index][bbox_index]
            if expose_gaucho:
                self._attach_gaucho_predictions(
                    result, outs, image_index, bbox_index, query_labels,
                    batch_img_metas[image_index], rescale=rescale)
        return predictions


    def _attach_gaucho_predictions(self, result, outs, image_index,
                                   bbox_index, query_labels, img_meta,
                                   rescale: bool) -> None:
        """Decode the GauCho outputs for the surviving queries.

        The 2D ellipse is emitted both as ``(mu, Sigma)`` and as the oriented
        rectangle that circumscribes it.  That envelope is what a rotated-box
        metric can score against an OBB annotation, and it is the same quantity
        the 2D reference implementation reports -- it is *not* an independently
        regressed box.
        """
        box_preds = outs[1][-1][image_index][bbox_index]
        img_h, img_w = img_meta['img_shape']
        scale = box_preds.new_tensor([img_w, img_h, img_w, img_h])
        box_px = box_preds * scale
        rows = torch.arange(len(bbox_index), device=box_preds.device)

        if self.gaucho_ellipse2d:
            raw = outs[11][-1][image_index][bbox_index]
            if self.gaucho_classwise:
                raw = raw.reshape(-1, self.num_classes, 5)[rows, query_labels]
            mean, _, sigma = self._decode_gaucho_ellipse2d(raw, box_px)
            ellipse = gaussian_to_ellipse2d(mean, sigma)
            if rescale and 'scale_factor' in img_meta:
                factor = ellipse.new_tensor(img_meta['scale_factor'])
                # Axes and centre share the image scaling; the angle does not.
                ellipse = torch.cat(
                    (ellipse[:, 0:1] / factor[0], ellipse[:, 1:2] / factor[1],
                     ellipse[:, 2:3] / factor[0], ellipse[:, 3:4] / factor[1],
                     ellipse[:, 4:5]), dim=-1)
            result.ellipses = ellipse
            result.ellipse_gaussians = torch.cat(
                (ellipse[:, 2:4], sigma[:, 0, 0:1], sigma[:, 0, 1:2],
                 sigma[:, 1, 1:2]), dim=-1)
            # (cx, cy, w, h, theta): the oriented envelope of the ellipse.
            result.ellipse_obb = torch.stack(
                (ellipse[:, 2], ellipse[:, 3], 2.0 * ellipse[:, 0],
                 2.0 * ellipse[:, 1], ellipse[:, 4]), dim=-1)

        if self.gaucho_ellipsoid:
            raw3d = outs[10][-1][image_index][bbox_index]
            if self.gaucho_classwise:
                raw3d = raw3d.reshape(-1, self.num_classes, 6)[
                    rows, query_labels]
            cholesky3d = self._gaucho_cholesky(raw3d)
            sigma3d = sigma_from_cholesky(cholesky3d)
            if self.ellipsoid_max_diameter is not None:
                # The soft training term makes an oversized ellipsoid
                # expensive; only this makes it impossible.  Both the corrected
                # shape and whether it was corrected are exposed, so the clamp
                # rate is auditable instead of invisible.
                sigma3d, clamped = clamp_sigma_max_axis(
                    sigma3d, self.ellipsoid_max_diameter)
                result.ellipsoid_clamped = clamped
            result.ellipsoid_shapes = sigma3d
            result.ellipsoid_symmetry = symmetry_class(sigma3d)
            if hasattr(result, 'translations'):
                # ``T`` is a homogeneous 4x4; ``translations`` is the same
                # camera-frame centre already in (N, 3), so take it directly
                # rather than slicing a column whose length is 4.
                center = result.translations
                result.ellipsoid_centers = center
                result.ellipsoid_front_margin = front_margin(center, sigma3d)
                intrinsic = self._intrinsic_matrix(
                    img_meta['intrinsic'], sigma3d).unsqueeze(0).expand(
                        len(sigma3d), 3, 3)
                mean, shape, valid = project_ellipsoid_dual_quadric(
                    center, sigma3d, intrinsic)
                result.projected_ellipses = gaussian_to_ellipse2d(mean, shape)
                result.projected_valid = valid

    def _distillation_losses(self, outs: tuple,
                             dn_meta: Dict[str, int]) -> Dict[str, Tensor]:
        if self.distillation_teacher is None:
            return {}
        if (not self._last_distillation_targets or
                self._last_distillation_scores is None):
            raise RuntimeError(
                'distillation targets are unavailable; generate them with '
                'compute_distillation_targets=True through the loss path')
        student_outputs = {
            'center': outs[2],
            'z': outs[3],
            'rotation': outs[4],
            'size': outs[5],
        }
        first_matching_query = 0
        if dn_meta is not None:
            first_matching_query = dn_meta['num_denoising_queries']
        return self.distillation_teacher.loss(
            student_outputs,
            self._last_distillation_targets,
            self._last_distillation_scores,
            first_matching_query=first_matching_query)

    def loss(self, hidden_states: Tensor, references: List[Tensor],
             enc_outputs_class: Tensor, enc_outputs_coord: Tensor,
             enc_outputs_centers_2d: Tensor, enc_outputs_z: Tensor,
             enc_outputs_rotation: Tensor, enc_outputs_size: Tensor,
             batch_data_samples: SampleList, dn_meta: Dict[str, int],
             depth_features=None) -> dict:
        """Perform forward propagation and loss calculation."""
        batch_gt_instances = []
        batch_img_metas = []
        for data_sample in batch_data_samples:
            batch_img_metas.append(data_sample.metainfo)
            batch_gt_instances.append(data_sample.gt_instances)

        # ``forward`` cannot see ``dn_meta``; the split is needed there to
        # keep the depth anchor off the denoising queries.
        self._num_denoising_queries = (
            int(dn_meta['num_denoising_queries'])
            if dn_meta and not self.sensor_depth_anchor_denoising else 0)
        outs = self(
            hidden_states,
            references,
            depth_features=depth_features,
            compute_distillation_targets=True)
        # ``outs`` carries the GauCho tensor last; keep the historical ten
        # positional entries and hand the new one over by keyword.
        loss_inputs = outs[:10] + (enc_outputs_class, enc_outputs_coord,
                                   enc_outputs_centers_2d, enc_outputs_z,
                                   enc_outputs_rotation, enc_outputs_size,
                                   batch_gt_instances, batch_img_metas, dn_meta)
        losses = self.loss_by_feat(
            *loss_inputs,
            all_layers_ellipsoid_preds=outs[10],
            all_layers_ellipse2d_preds=outs[11])
        if self.o2m_aux_topk:
            losses.update(self._loss_o2m_auxiliary(
                hidden_states, references, batch_gt_instances,
                batch_img_metas, dn_meta))
        losses.update(self._distillation_losses(outs, dn_meta))
        return losses

    def _loss_o2m_auxiliary(
            self, hidden_states: Tensor, references: List[Tensor],
            batch_gt_instances: InstanceList,
            batch_img_metas: List[dict],
            dn_meta: Dict[str, int] | None) -> Dict[str, Tensor]:
        """Compute last-layer 2D one-to-many losses for training only."""
        layer_id = hidden_states.shape[0] - 1
        hidden = hidden_states[layer_id]
        reference = references[layer_id]
        if dn_meta is not None:
            num_dn = dn_meta['num_denoising_queries']
            hidden = hidden[:, num_dn:]
            reference = reference[:, num_dn:]
        reference_unact = inverse_sigmoid(reference)
        cls_scores = self.o2m_cls_branch(hidden)
        bbox_unact = self.o2m_reg_branch(hidden)
        if reference_unact.shape[-1] == 4:
            bbox_unact = bbox_unact + reference_unact
        else:
            bbox_unact[..., :2] = (
                bbox_unact[..., :2] + reference_unact)
        bbox_predictions = bbox_unact.sigmoid()

        targets = []
        factors = []
        for prediction, gt_instances, img_meta in zip(
                bbox_predictions, batch_gt_instances, batch_img_metas):
            img_h, img_w = img_meta['img_shape']
            factor = prediction.new_tensor([img_w, img_h, img_w, img_h])
            gt_boxes = gt_instances.bboxes
            if hasattr(gt_boxes, 'tensor'):
                gt_boxes = gt_boxes.tensor
            gt_boxes = gt_boxes.to(
                device=prediction.device, dtype=prediction.dtype) / factor
            targets.append(one_to_many_bbox_targets(
                prediction, gt_boxes, gt_instances.labels,
                num_classes=self.num_classes, topk=self.o2m_aux_topk))
            factors.append(factor.unsqueeze(0).repeat(len(prediction), 1))

        labels = torch.cat([target['labels'] for target in targets])
        bbox_targets = torch.cat([
            target['bbox_targets'] for target in targets])
        bbox_weights = torch.cat([
            target['bbox_weights'] for target in targets])
        flat_cls_scores = cls_scores.reshape(-1, self.cls_out_channels)
        flat_bbox_predictions = bbox_predictions.reshape(-1, 4)
        num_positive = int((labels < self.num_classes).sum())
        cls_avg_factor = max(num_positive, 1)
        loss_cls = self.o2m_aux_loss_cls(
            flat_cls_scores, labels, torch.ones_like(labels),
            avg_factor=cls_avg_factor)

        factors = torch.cat(factors)
        pred_xyxy = bbox_cxcywh_to_xyxy(flat_bbox_predictions) * factors
        target_xyxy = bbox_cxcywh_to_xyxy(bbox_targets) * factors
        loss_bbox = self.loss_bbox(
            flat_bbox_predictions, bbox_targets, bbox_weights,
            avg_factor=cls_avg_factor)
        loss_iou = self.loss_iou(
            pred_xyxy, target_xyxy, bbox_weights,
            avg_factor=cls_avg_factor)
        weight = self.o2m_aux_loss_weight
        return {
            'loss_o2m_cls': loss_cls * weight,
            'loss_o2m_bbox': loss_bbox * weight,
            'loss_o2m_iou': loss_iou * weight,
        }
    def loss_by_feat(self, all_layers_cls_scores: Tensor, all_layers_bbox_preds: Tensor,
                     all_layers_centers_2d_preds: Tensor, all_layers_z_preds: Tensor,
                     all_layers_rotation_preds: Tensor, all_layers_sizes_preds: Tensor,
                     all_layers_sizes_chain_preds: Tensor,
                     all_layers_rotation_chain_preds: Tensor,
                     all_layers_z_chain_preds: Tensor,
                     all_layers_obb_aux_preds: Tensor,
                     enc_cls_scores: Tensor, enc_bbox_preds: Tensor,
                     enc_outputs_centers_2d: Tensor, enc_outputs_z: Tensor,
                     enc_outputs_rotation: Tensor, enc_outputs_size: Tensor,
                     batch_gt_instances: InstanceList, batch_img_metas: List[dict],
                     dn_meta: Dict[str, int],
                     batch_gt_instances_ignore: OptInstanceList = None,
                     all_layers_ellipsoid_preds: Tensor = None,
                     all_layers_ellipse2d_preds: Tensor = None
                     ) -> Dict[str, Tensor]:
        """Loss function.

        ``all_layers_ellipsoid_preds`` is keyword-only-by-position at the end so
        every existing positional caller keeps working unchanged.
        """
        # extract denoising and matching part of outputs
        (all_layers_matching_cls_scores, all_layers_matching_bbox_preds,
         all_layers_matching_centers_2d_preds, all_layers_matching_z_preds,
         all_layers_matching_rotation_preds, all_layers_matching_sizes_preds,
         all_layers_denoising_cls_scores, all_layers_denoising_bbox_preds,
         all_layers_denoising_centers_2d_preds, all_layers_denoising_z_preds,
         all_layers_denoising_rotation_preds, all_layers_denoising_sizes_preds) = \
            self.split_outputs(all_layers_cls_scores, all_layers_bbox_preds, 
                              all_layers_centers_2d_preds, all_layers_z_preds,
                              all_layers_rotation_preds, all_layers_sizes_preds, dn_meta)

        # CoP chain outputs: only keep the matching (non-denoising) slice for
        # the auxiliary chain losses (the denoising branch keeps the plain
        # parallel predictions).
        if self._uses_auxiliary_chain and dn_meta is not None:
            n_dn = dn_meta['num_denoising_queries']
            all_m_sizes_chain = all_layers_sizes_chain_preds[:, :, n_dn:, :]
            all_m_rotation_chain = all_layers_rotation_chain_preds[:, :, n_dn:, :]
            all_m_z_chain = all_layers_z_chain_preds[:, :, n_dn:, :]
        else:
            all_m_sizes_chain = all_layers_sizes_chain_preds
            all_m_rotation_chain = all_layers_rotation_chain_preds
            all_m_z_chain = all_layers_z_chain_preds
        if all_layers_obb_aux_preds is not None and dn_meta is not None:
            n_dn = dn_meta['num_denoising_queries']
            all_m_obb_aux = all_layers_obb_aux_preds[:, :, n_dn:, :]
        else:
            all_m_obb_aux = all_layers_obb_aux_preds
        if all_layers_ellipsoid_preds is not None and dn_meta is not None:
            n_dn = dn_meta['num_denoising_queries']
            all_m_ellipsoid = all_layers_ellipsoid_preds[:, :, n_dn:, :]
        else:
            all_m_ellipsoid = all_layers_ellipsoid_preds
        if all_layers_ellipse2d_preds is not None and dn_meta is not None:
            n_dn = dn_meta['num_denoising_queries']
            all_dn_ellipse2d = all_layers_ellipse2d_preds[:, :, :n_dn, :]
            all_m_ellipse2d = all_layers_ellipse2d_preds[:, :, n_dn:, :]
        else:
            all_dn_ellipse2d = None
            all_m_ellipse2d = all_layers_ellipse2d_preds

        loss_dict = self.loss_by_feat_simple(
            all_layers_matching_cls_scores, all_layers_matching_bbox_preds,
            all_layers_matching_centers_2d_preds, all_layers_matching_z_preds,
            all_layers_matching_rotation_preds, all_layers_matching_sizes_preds,
            all_m_sizes_chain, all_m_rotation_chain, all_m_z_chain,
            all_m_obb_aux, all_m_ellipsoid, all_m_ellipse2d,
            batch_gt_instances=batch_gt_instances,
            batch_img_metas=batch_img_metas,
            batch_gt_instances_ignore=batch_gt_instances_ignore)

        # loss of proposal generated from encode feature map.
        if enc_cls_scores is not None:
            encoder_pose_supervision = self.cop_encoder_pose_supervision
            # The encoder proposal has no GauCho branch, so only the first
            # twelve entries of the per-layer tuple are meaningful here.
            (enc_loss_cls, enc_losses_bbox, enc_losses_iou,
             enc_loss_centers_2d, enc_loss_z, enc_loss_rotation, enc_loss_size,
             _enc_projection, _enc_obb_aux, _enc_size_chain, _enc_rotation_chain,
             _enc_z_chain) = \
                self.loss_by_feat_single(enc_cls_scores, enc_bbox_preds,
                                       enc_outputs_centers_2d, enc_outputs_z,
                                       enc_outputs_rotation, enc_outputs_size,
                                       None, None, None, None,
                                       batch_gt_instances=batch_gt_instances,
                                       batch_img_metas=batch_img_metas,
                                       pose_supervision=(
                                           encoder_pose_supervision),
                                       obb_aux_supervision=False,
                                       assigner=(
                                           self.encoder_assigner))[:12]
            loss_dict['enc_loss_cls'] = enc_loss_cls
            loss_dict['enc_loss_bbox'] = enc_losses_bbox
            loss_dict['enc_loss_iou'] = enc_losses_iou
            loss_dict['enc_loss_centers_2d'] = enc_loss_centers_2d
            if encoder_pose_supervision:
                loss_dict['enc_loss_z'] = enc_loss_z
                loss_dict['enc_loss_rotation'] = enc_loss_rotation
                loss_dict['enc_loss_size'] = enc_loss_size

        if all_layers_denoising_cls_scores is not None:
            # calculate denoising loss from all decoder layers
            dn_losses = self.loss_dn(all_layers_denoising_cls_scores, all_layers_denoising_bbox_preds,
                        all_layers_denoising_centers_2d_preds, all_layers_denoising_z_preds,
                        all_layers_denoising_rotation_preds, all_layers_denoising_sizes_preds,
                        batch_gt_instances=batch_gt_instances, batch_img_metas=batch_img_metas,
                        dn_meta=dn_meta,
                        all_layers_denoising_ellipse2d_preds=all_dn_ellipse2d)
            # A subclass or test double may still return the historical
            # 7-tuple, without the denoising ellipse term.  Tolerated the same
            # way ``loss_by_feat_single``'s own arity growth is a few lines
            # below, so that adding a loss never silently becomes a breaking
            # change for an override.
            (dn_losses_cls, dn_losses_bbox, dn_losses_iou,
             dn_losses_centers_2d, dn_losses_z, dn_losses_rotation,
             dn_losses_sizes) = dn_losses[:7]
            dn_losses_ellipse2d = (
                dn_losses[7] if len(dn_losses) > 7
                else [None] * len(dn_losses_cls))
            # collate denoising loss
            loss_dict['dn_loss_cls'] = dn_losses_cls[-1]
            loss_dict['dn_loss_bbox'] = dn_losses_bbox[-1]
            loss_dict['dn_loss_iou'] = dn_losses_iou[-1]
            loss_dict['dn_loss_centers_2d'] = dn_losses_centers_2d[-1]
            loss_dict['dn_loss_z'] = dn_losses_z[-1]
            loss_dict['dn_loss_rotation'] = dn_losses_rotation[-1]
            loss_dict['dn_loss_size'] = dn_losses_sizes[-1]
            if self.gaucho_ellipse2d_dn:
                loss_dict['dn_loss_ellipse2d'] = dn_losses_ellipse2d[-1]

            for num_dec_layer, (loss_cls_i, loss_bbox_i, loss_iou_i, 
                               loss_centers_2d_i, loss_z_i, loss_rotation_i,
                               loss_sizes_i, loss_ellipse2d_i) in \
                    enumerate(zip(dn_losses_cls[:-1], dn_losses_bbox[:-1], dn_losses_iou[:-1],
                                 dn_losses_centers_2d[:-1], dn_losses_z[:-1],
                                 dn_losses_rotation[:-1], dn_losses_sizes[:-1],
                                 dn_losses_ellipse2d[:-1])):
                loss_dict[f'd{num_dec_layer}.dn_loss_cls'] = loss_cls_i
                loss_dict[f'd{num_dec_layer}.dn_loss_bbox'] = loss_bbox_i
                loss_dict[f'd{num_dec_layer}.dn_loss_iou'] = loss_iou_i
                loss_dict[f'd{num_dec_layer}.dn_loss_centers_2d'] = loss_centers_2d_i
                loss_dict[f'd{num_dec_layer}.dn_loss_z'] = loss_z_i
                loss_dict[f'd{num_dec_layer}.dn_loss_rotation'] = loss_rotation_i
                loss_dict[f'd{num_dec_layer}.dn_loss_size'] = loss_sizes_i
                if self.gaucho_ellipse2d_dn:
                    loss_dict[f'd{num_dec_layer}.dn_loss_ellipse2d'] = \
                        loss_ellipse2d_i

        return loss_dict

    def loss_by_feat_simple(self, all_layers_cls_scores: Tensor, all_layers_bbox_preds: Tensor,
                           all_layers_centers_2d_preds: Tensor, all_layers_z_preds: Tensor,
                           all_layers_rotation_preds: Tensor, all_layers_sizes_preds: Tensor,
                           all_layers_sizes_chain_preds: Tensor,
                           all_layers_rotation_chain_preds: Tensor,
                           all_layers_z_chain_preds: Tensor,
                           all_layers_obb_aux_preds: Tensor,
                           all_layers_ellipsoid_preds: Tensor = None,
                           all_layers_ellipse2d_preds: Tensor = None,
                           batch_gt_instances: InstanceList = None,
                           batch_img_metas: List[dict] = None,
                           batch_gt_instances_ignore: OptInstanceList = None
                           ) -> Dict[str, Tensor]:
        """Loss function.

        ``all_layers_ellipsoid_preds`` sits in the eleventh positional slot so
        ``loss_by_feat_simple(*head(...))`` keeps working now that ``forward``
        emits the GauCho tensor.  Callers that pass only the historical ten
        tensors and supply the batch arguments by keyword are unaffected.
        """
        if batch_gt_instances is None or batch_img_metas is None:
            raise ValueError(
                'batch_gt_instances and batch_img_metas are required')
        assert batch_gt_instances_ignore is None, \
            f'{self.__class__.__name__} only supports ' \
            'for batch_gt_instances_ignore setting to None.'

        # Chain-only mode does not instantiate the legacy auxiliary prediction
        # path. ``multi_apply`` nevertheless needs one item per decoder layer.
        num_decoder_layers = all_layers_cls_scores.shape[0]
        if self._uses_auxiliary_chain:
            size_chain_inputs = all_layers_sizes_chain_preds
            rotation_chain_inputs = all_layers_rotation_chain_preds
            z_chain_inputs = all_layers_z_chain_preds
        else:
            size_chain_inputs = [None] * num_decoder_layers
            rotation_chain_inputs = [None] * num_decoder_layers
            z_chain_inputs = [None] * num_decoder_layers
        if all_layers_obb_aux_preds is not None:
            obb_aux_inputs = all_layers_obb_aux_preds
        else:
            obb_aux_inputs = [None] * num_decoder_layers
        if all_layers_ellipsoid_preds is not None:
            ellipsoid_inputs = all_layers_ellipsoid_preds
        else:
            ellipsoid_inputs = [None] * num_decoder_layers
        if all_layers_ellipse2d_preds is not None:
            ellipse2d_inputs = all_layers_ellipse2d_preds
        else:
            ellipse2d_inputs = [None] * num_decoder_layers

        # ``multi_apply`` cannot vary a keyword argument per layer, and
        # ``ellipsoid_preds`` has to stay a trailing default so existing
        # positional callers and test doubles keep working.  Expanding the map
        # explicitly keeps both properties.
        per_layer_losses = [
            self.loss_by_feat_single(
                cls_scores, bbox_preds, centers_2d_preds, z_preds,
                rotation_preds, sizes_preds, size_chain, rotation_chain,
                z_chain, obb_aux,
                batch_gt_instances=batch_gt_instances,
                batch_img_metas=batch_img_metas,
                ellipsoid_preds=ellipsoid,
                ellipse2d_preds=ellipse2d)
            for (cls_scores, bbox_preds, centers_2d_preds, z_preds,
                 rotation_preds, sizes_preds, size_chain, rotation_chain,
                 z_chain, obb_aux, ellipsoid, ellipse2d) in zip(
                     all_layers_cls_scores, all_layers_bbox_preds,
                     all_layers_centers_2d_preds, all_layers_z_preds,
                     all_layers_rotation_preds, all_layers_sizes_preds,
                     size_chain_inputs, rotation_chain_inputs, z_chain_inputs,
                     obb_aux_inputs, ellipsoid_inputs, ellipse2d_inputs)
        ]
        transposed = tuple(map(list, zip(*per_layer_losses)))
        (losses_cls, losses_bbox, losses_iou, losses_centers_2d,
         losses_z, losses_rotation, losses_sizes, losses_projection,
         losses_obb_aux, losses_size_chain, losses_rotation_chain,
         losses_z_chain) = transposed[:12]
        # A subclass or test double may still return the historical 12-tuple.
        empty = [None] * num_decoder_layers
        losses_ellipsoid = transposed[12] if len(transposed) > 12 else empty
        losses_ellipsoid_projection = (
            transposed[13] if len(transposed) > 13 else empty)
        losses_ellipse2d = transposed[14] if len(transposed) > 14 else empty
        losses_ellipsoid_max_axis = (
            transposed[15] if len(transposed) > 15 else empty)
        losses_ellipse2d_corner = (
            transposed[16] if len(transposed) > 16 else empty)
        losses_ellipse2d_angle = (
            transposed[17] if len(transposed) > 17 else empty)

        losses_rotation_frame = None
        if self.loss_rotation_frame is not None:
            # ``all_layers_rotation_preds`` is the matching-query slice of the
            # primary rotation output also consumed by inference.  ``forward``
            # routes the parallel branch here in parallel/auxiliary mode and
            # the CoP branch here in chain mode.  Applying the target-free
            # constraint at this boundary therefore avoids supervising the DN,
            # encoder, and auxiliary-chain streams by accident.
            rotation_frame_inputs = all_layers_rotation_preds
            if self.classwise_rotation:
                expected_width = self.num_classes * self.rot_dim
                if rotation_frame_inputs.shape[-1] != expected_width:
                    raise ValueError(
                        'classwise raw rotation predictions must have width '
                        f'num_classes * rot_dim = {expected_width}, got '
                        f'{rotation_frame_inputs.shape[-1]}')
                # Every class frame can be selected during inference.  Keep
                # all candidates valid instead of routing the regularizer
                # through a non-differentiable predicted-class argmax.
                rotation_frame_inputs = rotation_frame_inputs.unflatten(
                    -1, (self.num_classes, self.rot_dim))
            losses_rotation_frame = [
                self.loss_rotation_frame(layer_predictions)
                for layer_predictions in rotation_frame_inputs
            ]

        loss_dict = dict()
        # loss from the last decoder layer
        loss_dict['loss_cls'] = losses_cls[-1]
        loss_dict['loss_bbox'] = losses_bbox[-1]
        loss_dict['loss_iou'] = losses_iou[-1]
        loss_dict['loss_centers_2d'] = losses_centers_2d[-1]
        loss_dict['loss_z'] = losses_z[-1]
        loss_dict['loss_rotation'] = losses_rotation[-1]
        if losses_rotation_frame is not None:
            loss_dict['loss_rotation_frame'] = losses_rotation_frame[-1]
        loss_dict['loss_size'] = losses_sizes[-1]
        if self.loss_projection is not None:
            loss_dict['loss_projection'] = losses_projection[-1]
        if self.loss_obb_aux is not None:
            loss_dict['loss_obb_aux'] = losses_obb_aux[-1]
        if self.loss_ellipse2d is not None:
            loss_dict['loss_ellipse2d'] = losses_ellipse2d[-1]
        if self.loss_ellipsoid is not None:
            loss_dict['loss_ellipsoid'] = losses_ellipsoid[-1]
        if self.loss_ellipsoid_projection is not None:
            loss_dict['loss_ellipsoid_projection'] = \
                losses_ellipsoid_projection[-1]
        if self.loss_ellipsoid_max_axis is not None:
            loss_dict['loss_ellipsoid_max_axis'] = \
                losses_ellipsoid_max_axis[-1]
        if self.loss_ellipse2d_corner is not None:
            loss_dict['loss_ellipse2d_corner'] = losses_ellipse2d_corner[-1]
        if self.loss_ellipse2d_angle is not None:
            loss_dict['loss_ellipse2d_angle'] = losses_ellipse2d_angle[-1]
        if self._uses_auxiliary_chain:
            loss_dict['loss_size_chain'] = losses_size_chain[-1]
            loss_dict['loss_rotation_chain'] = losses_rotation_chain[-1]
            loss_dict['loss_z_chain'] = losses_z_chain[-1]

        # loss from other decoder layers
        num_dec_layer = 0
        for i, (loss_cls_i, loss_bbox_i, loss_iou_i, loss_centers_2d_i,
                loss_z_i, loss_rotation_i, loss_sizes_i, loss_projection_i,
                loss_obb_aux_i) in \
                enumerate(zip(losses_cls[:-1], losses_bbox[:-1], losses_iou[:-1],
                   losses_centers_2d[:-1], losses_z[:-1], losses_rotation[:-1],
                   losses_sizes[:-1], losses_projection[:-1],
                   losses_obb_aux[:-1])):
            loss_dict[f'd{num_dec_layer}.loss_cls'] = loss_cls_i
            loss_dict[f'd{num_dec_layer}.loss_bbox'] = loss_bbox_i
            loss_dict[f'd{num_dec_layer}.loss_iou'] = loss_iou_i
            loss_dict[f'd{num_dec_layer}.loss_centers_2d'] = loss_centers_2d_i
            loss_dict[f'd{num_dec_layer}.loss_z'] = loss_z_i
            loss_dict[f'd{num_dec_layer}.loss_rotation'] = loss_rotation_i
            if losses_rotation_frame is not None:
                loss_dict[f'd{num_dec_layer}.loss_rotation_frame'] = \
                    losses_rotation_frame[i]
            loss_dict[f'd{num_dec_layer}.loss_size'] = loss_sizes_i
            if self.loss_projection is not None:
                loss_dict[f'd{num_dec_layer}.loss_projection'] = \
                    loss_projection_i
            if self.loss_obb_aux is not None:
                loss_dict[f'd{num_dec_layer}.loss_obb_aux'] = loss_obb_aux_i
            if self.loss_ellipse2d is not None:
                loss_dict[f'd{num_dec_layer}.loss_ellipse2d'] = \
                    losses_ellipse2d[i]
            if self.loss_ellipsoid is not None:
                loss_dict[f'd{num_dec_layer}.loss_ellipsoid'] = \
                    losses_ellipsoid[i]
            if self.loss_ellipsoid_projection is not None:
                loss_dict[f'd{num_dec_layer}.loss_ellipsoid_projection'] = \
                    losses_ellipsoid_projection[i]
            if self.loss_ellipsoid_max_axis is not None:
                loss_dict[f'd{num_dec_layer}.loss_ellipsoid_max_axis'] = \
                    losses_ellipsoid_max_axis[i]
            if self.loss_ellipse2d_corner is not None:
                loss_dict[f'd{num_dec_layer}.loss_ellipse2d_corner'] = \
                    losses_ellipse2d_corner[i]
            if self.loss_ellipse2d_angle is not None:
                loss_dict[f'd{num_dec_layer}.loss_ellipse2d_angle'] = \
                    losses_ellipse2d_angle[i]
            if self._uses_auxiliary_chain:
                loss_dict[f'd{num_dec_layer}.loss_size_chain'] = losses_size_chain[i]
                loss_dict[f'd{num_dec_layer}.loss_rotation_chain'] = losses_rotation_chain[i]
                loss_dict[f'd{num_dec_layer}.loss_z_chain'] = losses_z_chain[i]
            num_dec_layer += 1
        return loss_dict

    def loss_dn(self, all_layers_denoising_cls_scores: Tensor, all_layers_denoising_bbox_preds: Tensor,
                all_layers_denoising_centers_2d_preds: Tensor, all_layers_denoising_z_preds: Tensor,
                all_layers_denoising_rotation_preds: Tensor, all_layers_denoising_sizes_preds: Tensor,
                batch_gt_instances: InstanceList, batch_img_metas: List[dict],
                dn_meta: Dict[str, int],
                all_layers_denoising_ellipse2d_preds: Tensor = None
                ) -> Tuple[List[Tensor]]:
        """Calculate denoising loss."""
        ellipse_inputs = (
            all_layers_denoising_ellipse2d_preds
            if all_layers_denoising_ellipse2d_preds is not None else
            [None] * len(all_layers_denoising_cls_scores))
        return multi_apply(self._loss_dn_single, all_layers_denoising_cls_scores,
                          all_layers_denoising_bbox_preds, all_layers_denoising_centers_2d_preds,
                          all_layers_denoising_z_preds, all_layers_denoising_rotation_preds,
                          all_layers_denoising_sizes_preds, ellipse_inputs,
                          batch_gt_instances=batch_gt_instances,
                          batch_img_metas=batch_img_metas, dn_meta=dn_meta)

    def _loss_dn_single(self, dn_cls_scores: Tensor, dn_bbox_preds: Tensor,
                        dn_centers_2d_preds: Tensor, dn_z_preds: Tensor,
                        dn_rotation_preds: Tensor, dn_sizes_preds: Tensor,
                        dn_ellipse2d_preds: Tensor,
                        batch_gt_instances: InstanceList, batch_img_metas: List[dict],
                        dn_meta: Dict[str, int]) -> Tuple[Tensor]:
        """Denoising loss for outputs from a single decoder layer."""
        cls_reg_targets = self.get_dn_targets(batch_gt_instances, batch_img_metas, dn_meta)
        (labels_list, label_weights_list, bbox_targets_list, bbox_weights_list,
         dn_centers_2d_targets_list, dn_centers_2d_weights_list,
         dn_z_targets_list, dn_z_weights_list,
         dn_rotation_targets_list, dn_rotation_weights_list,
         dn_sizes_targets_list, dn_sizes_weights_list,
         dn_obb_gaussian_targets_list, dn_obb_gaussian_weights_list,
         num_total_pos, num_total_neg) = cls_reg_targets
        
        labels = torch.cat(labels_list, 0)
        label_weights = torch.cat(label_weights_list, 0)
        bbox_targets = torch.cat(bbox_targets_list, 0)
        bbox_weights = torch.cat(bbox_weights_list, 0)
        centers_2d_targets = torch.cat(dn_centers_2d_targets_list, 0)
        centers_2d_weights = torch.cat(dn_centers_2d_weights_list, 0)
        z_targets = torch.cat(dn_z_targets_list, 0)
        z_weights = torch.cat(dn_z_weights_list, 0)
        rotation_targets = torch.cat(dn_rotation_targets_list, 0)
        rotation_weights = torch.cat(dn_rotation_weights_list, 0)
        sizes_targets = torch.cat(dn_sizes_targets_list, 0)
        sizes_weights = torch.cat(dn_sizes_weights_list, 0)
        obb_gaussian_targets = torch.cat(
            dn_obb_gaussian_targets_list, 0)
        obb_gaussian_weights = torch.cat(
            dn_obb_gaussian_weights_list, 0)

        # classification loss
        cls_scores = dn_cls_scores.reshape(-1, self.cls_out_channels)
        cls_avg_factor = num_total_pos * 1.0 + num_total_neg * self.bg_cls_weight
        if self.sync_cls_avg_factor:
            cls_avg_factor = reduce_mean(cls_scores.new_tensor([cls_avg_factor]))
        cls_avg_factor = max(cls_avg_factor, 1)

        if len(cls_scores) > 0:
            loss_cls = self._classification_loss(
                cls_scores=cls_scores,
                labels=labels,
                label_weights=label_weights,
                bbox_predictions=dn_bbox_preds.reshape(-1, 4),
                bbox_targets=bbox_targets,
                avg_factor=cls_avg_factor,
            )
        else:
            loss_cls = torch.zeros(1, dtype=cls_scores.dtype, device=cls_scores.device)

        num_total_pos = loss_cls.new_tensor([num_total_pos])
        num_total_pos = torch.clamp(reduce_mean(num_total_pos), min=1).item()

        # construct factors used for rescale bboxes
        factors = []
        for img_meta, bbox_pred in zip(batch_img_metas, dn_bbox_preds):
            img_h, img_w = img_meta['img_shape']
            factor = bbox_pred.new_tensor([img_w, img_h, img_w, img_h]).unsqueeze(0).repeat(bbox_pred.size(0), 1)
            factors.append(factor)
        factors = torch.cat(factors)

        bbox_preds = dn_bbox_preds.reshape(-1, 4)
        bboxes = bbox_cxcywh_to_xyxy(bbox_preds) * factors
        bboxes_gt = bbox_cxcywh_to_xyxy(bbox_targets) * factors

        loss_iou = self.loss_iou(bboxes, bboxes_gt, bbox_weights, avg_factor=num_total_pos)
        loss_bbox = self.loss_bbox(bbox_preds, bbox_targets, bbox_weights, avg_factor=num_total_pos)
        
        # centers_2d loss
        centers_2d_preds = dn_centers_2d_preds.reshape(-1, 2)
        loss_centers_2d = self.loss_centers_2d(centers_2d_preds, centers_2d_targets, centers_2d_weights, avg_factor=num_total_pos)

        # z loss
        z_preds = dn_z_preds.reshape(-1, 1)
        loss_z = self.loss_z(z_preds, z_targets, z_weights, avg_factor=num_total_pos)

        indexing_labels = labels.clone()
        indexing_labels[indexing_labels == self.num_classes] = 0

        # rotation loss
        # rotation_preds = dn_rotation_preds.reshape(-1, 6)
        if self.classwise_rotation:
            rotation_preds = dn_rotation_preds.reshape(-1, self.num_classes, self.rot_dim)
            rotation_preds = rotation_preds[
                torch.arange(rotation_preds.size(0),
                             device=rotation_preds.device), indexing_labels]
        else:
            rotation_preds = dn_rotation_preds.reshape(-1, self.rot_dim)
        loss_rotation = self.loss_rotation(rotation_preds, rotation_targets, rotation_weights, labels=labels, avg_factor=num_total_pos)

        # sizes loss
        # sizes_preds = dn_sizes_preds.reshape(-1, 3)
        if self.classwise_sizes:
            sizes_preds = dn_sizes_preds.reshape(-1, self.num_classes, 3)
            sizes_preds = sizes_preds[
                torch.arange(sizes_preds.size(0), device=sizes_preds.device),
                indexing_labels]
        else:
            sizes_preds = dn_sizes_preds.reshape(-1, 3)
        loss_sizes = self.loss_sizes(sizes_preds, sizes_targets, sizes_weights, avg_factor=num_total_pos)

        loss_ellipse2d = dn_bbox_preds.sum() * 0.0
        if self.gaucho_ellipse2d_dn and dn_ellipse2d_preds is not None:
            if self.gaucho_classwise:
                raw2d = dn_ellipse2d_preds.reshape(
                    -1, self.num_classes, 5)
                raw2d = raw2d[
                    torch.arange(raw2d.size(0), device=raw2d.device),
                    indexing_labels]
            else:
                raw2d = dn_ellipse2d_preds.reshape(-1, 5)
            box_px = (bbox_preds * factors).detach()
            mean, cholesky, _ = self._decode_gaucho_ellipse2d(
                raw2d, box_px)
            loss_ellipse2d = self.loss_ellipse2d(
                mean, cholesky, obb_gaussian_targets,
                weight=obb_gaussian_weights, avg_factor=num_total_pos)

        return (loss_cls, loss_bbox, loss_iou, loss_centers_2d, loss_z,
                loss_rotation, loss_sizes, loss_ellipse2d)

    def get_dn_targets(self, batch_gt_instances: InstanceList, batch_img_metas: List[dict],
                       dn_meta: Dict[str, int]) -> tuple:
        """Get targets in denoising part for a batch of images."""
        (labels_list, label_weights_list, bbox_targets_list, bbox_weights_list,
         centers_2d_targets_list, centers_2d_weights_list,
         z_targets_list, z_weights_list,
         rotation_targets_list, rotation_weights_list,
         sizes_targets_list, sizes_weights_list,
         obb_gaussian_targets_list, obb_gaussian_weights_list,
         pos_inds_list, neg_inds_list) = multi_apply(
             self._get_dn_targets_single, batch_gt_instances, batch_img_metas, dn_meta=dn_meta)
        num_total_pos = sum((inds.numel() for inds in pos_inds_list))
        num_total_neg = sum((inds.numel() for inds in neg_inds_list))
        return (labels_list, label_weights_list, bbox_targets_list, bbox_weights_list,
                centers_2d_targets_list, centers_2d_weights_list,
                z_targets_list, z_weights_list,
                rotation_targets_list, rotation_weights_list,
                sizes_targets_list, sizes_weights_list,
                obb_gaussian_targets_list, obb_gaussian_weights_list,
                num_total_pos, num_total_neg)

    def _get_dn_targets_single(self, gt_instances: InstanceData, img_meta: dict,
                               dn_meta: Dict[str, int]) -> tuple:
        """Get targets in denoising part for one image."""
        gt_bboxes = gt_instances.bboxes
        gt_labels = gt_instances.labels
        gt_centers_2d = gt_instances.centers_2d
        gt_z = gt_instances.z
        gt_rotations = gt_instances.rotations
        gt_sizes = gt_instances.sizes

        img_h, img_w = img_meta['img_shape']
        factor = gt_bboxes.new_tensor([img_w, img_h, img_w, img_h]).unsqueeze(0)

        num_groups = dn_meta['num_denoising_groups']
        num_denoising_queries = dn_meta['num_denoising_queries']
        num_queries_each_group = int(num_denoising_queries / num_groups)
        device = gt_bboxes.device

        if len(gt_labels) > 0:
            t = torch.arange(len(gt_labels), dtype=torch.long, device=device)
            t = t.unsqueeze(0).repeat(num_groups, 1)
            pos_assigned_gt_inds = t.flatten()
            pos_inds = torch.arange(num_groups, dtype=torch.long, device=device)
            pos_inds = pos_inds.unsqueeze(1) * num_queries_each_group + t
            pos_inds = pos_inds.flatten()
        else:
            pos_inds = pos_assigned_gt_inds = gt_bboxes.new_tensor([], dtype=torch.long)

        neg_inds = pos_inds + num_queries_each_group // 2

        # Initialize all targets
        labels = gt_bboxes.new_full((num_denoising_queries, ), self.num_classes, dtype=torch.long)
        labels[pos_inds] = gt_labels[pos_assigned_gt_inds]
        label_weights = gt_bboxes.new_ones(num_denoising_queries)

        bbox_targets = torch.zeros(num_denoising_queries, 4, device=device)
        bbox_weights = torch.zeros(num_denoising_queries, 4, device=device)
        bbox_weights[pos_inds] = 1.0

        centers_2d_targets = torch.zeros(num_denoising_queries, 2, device=device)
        centers_2d_weights = torch.zeros(num_denoising_queries, 2, device=device)
        centers_2d_weights[pos_inds] = 1.0

        z_targets = torch.zeros(num_denoising_queries, 1, device=device)
        z_weights = torch.zeros(num_denoising_queries, 1, device=device)
        z_weights[pos_inds] = 1.0

        rotation_targets = torch.zeros(num_denoising_queries, 6, device=device)
        rotation_weights = torch.zeros(num_denoising_queries, 6, device=device)
        rotation_weights[pos_inds] = 1.0

        sizes_targets = torch.zeros(num_denoising_queries, 3, device=device)
        sizes_weights = torch.zeros(num_denoising_queries, 3, device=device)
        sizes_weights[pos_inds] = 1.0

        obb_gaussian_targets = torch.zeros(
            num_denoising_queries, 5, device=device)
        obb_gaussian_weights = torch.zeros(
            num_denoising_queries, device=device)
        if self.gaucho_ellipse2d_dn:
            if 'obb_gaussians' not in gt_instances:
                raise KeyError(
                    'DN ellipse supervision requires '
                    'gt_instances.obb_gaussians')
            obb_gaussian_weights[pos_inds] = 1.0

        # Set targets for positive samples
        gt_bboxes_normalized = gt_bboxes / factor
        gt_bboxes_targets = bbox_xyxy_to_cxcywh(gt_bboxes_normalized)
        bbox_targets[pos_inds] = gt_bboxes_targets.repeat([num_groups, 1])

        gt_centers_2d_normalized = gt_centers_2d / factor[:, :2]
        centers_2d_targets[pos_inds] = gt_centers_2d_normalized.repeat([num_groups, 1])

        z_targets[pos_inds] = gt_z.repeat([num_groups, 1])
        rotation_targets[pos_inds] = gt_rotations.repeat([num_groups, 1])
        sizes_targets[pos_inds] = gt_sizes.repeat([num_groups, 1])
        if self.gaucho_ellipse2d_dn:
            obb_gaussian_targets[pos_inds] = \
                gt_instances.obb_gaussians.repeat([num_groups, 1])

        return (labels, label_weights, bbox_targets, bbox_weights,
                centers_2d_targets, centers_2d_weights, z_targets, z_weights,
                rotation_targets, rotation_weights, sizes_targets, sizes_weights,
                obb_gaussian_targets, obb_gaussian_weights,
                pos_inds, neg_inds)

    @property
    def _requires_obb_gaussians(self) -> bool:
        """Whether any active objective consumes the annotated 2D OBB Gaussian.

        The GauCho-2D ellipse and the dual-quadric projection term read the
        same annotation as the older projection and OBB auxiliary losses, so
        target construction has to be gated on all four together.  Leaving the
        new losses out here would silently hand them all-zero targets.
        """
        return (self.loss_projection is not None
                or self.loss_obb_aux is not None
                or self.loss_ellipse2d is not None
                or self.loss_ellipsoid_projection is not None)

    def _gaucho_cholesky(self, raw: Tensor) -> Tensor:
        """Turn raw head outputs into an SPD(3) Cholesky factor.

        Every chart is offset by ``log(gaucho_size_prior)`` on its logarithmic
        diagonal so a zero-initialized head starts at an isotropic ellipsoid of
        the prior radius.  Without the offset the ``direct`` and ``dual_plane``
        charts would begin at a one-metre sphere, which for fruit is several
        orders of magnitude off and puts the first updates in the clipped
        region of the chart.
        """
        if self.gaucho_chart == 'scale_shape':
            # ``scale_shape_cholesky3d`` already multiplies by the prior, after
            # its own clipping.
            return scale_shape_cholesky3d(raw, self.gaucho_size_prior)
        # Scale the finished factor rather than biasing the raw logs.  Biasing
        # first would push the prior through the chart's own +/-log_clip, which
        # makes the usable range depend on the prior (asymmetric shrink/grow
        # headroom, and a small enough prior starts fully clipped with no
        # gradient), and would leave the shear entries at unit scale while the
        # diagonal sits at the prior.  ``L -> rho0 L`` gives
        # ``Sigma -> rho0^2 Sigma`` and scales every entry alike.
        if self.gaucho_chart == 'direct':
            return self.gaucho_size_prior * cholesky3d_from_raw(raw)
        return self.gaucho_size_prior * dual_plane_cholesky3d(
            raw, fix_rc_zero=self.gaucho_dual_plane_fix_rc_zero)

    @staticmethod
    def _match_costs_use_ellipse(train_cfg) -> bool:
        """Does any configured match cost consume the predicted ellipse?"""
        if not train_cfg:
            return False
        for key in ('assigner', 'encoder_assigner'):
            assigner = train_cfg.get(key) if hasattr(train_cfg, 'get') else None
            if not assigner:
                continue
            for cost in assigner.get('match_costs', ()) or ():
                if isinstance(cost, dict) and \
                        cost.get('type') == 'Ellipse2DKLDCost':
                    return True
        return False

    def _decode_gaucho_ellipse2d(
            self, raw: Tensor, box_px: Tensor
    ) -> tuple[Tensor, Tensor, Tensor]:
        """Decode ``(tx, ty, r, u, v)`` against a reference box, once.

        The query's own predicted box plays the role the FPN cell plays in the
        dense reference implementation: it fixes the reference centre and
        radius of the chart.  Training and inference must decode identically or
        the model optimizes one ellipse and evaluation scores another, so this
        lives in one place; whether ``box_px`` is detached is the caller's
        decision.

        ``box_px`` is pixel ``(cx, cy, w, h)``.  Returns ``(mean, cholesky,
        sigma)``.
        """
        reference_center = box_px[..., :2]
        reference_radius = 0.5 * (
            box_px[..., 2] * box_px[..., 3]).clamp_min(self.gaucho_eps).sqrt()
        cholesky = scale_shape_cholesky2d(raw[..., 2:5], reference_radius)
        sigma = sigma_from_cholesky2d(cholesky)
        extent = 2.0 * sigma.diagonal(dim1=-2, dim2=-1).sum(
            dim=-1).clamp_min(self.gaucho_eps).sqrt()
        mean = reference_center + raw[..., :2] * extent[..., None]
        return mean, cholesky, sigma

    def _gaucho_query_intrinsics(self, batch_img_metas, bbox_preds_list,
                                 reference: Tensor) -> Tensor:
        """Per-query camera matrices, one row per flattened query."""
        intrinsics = []
        for img_meta, per_image_bbox_preds in zip(
                batch_img_metas, bbox_preds_list):
            intrinsic = self._training_intrinsic_matrix(img_meta, reference)
            intrinsics.append(
                intrinsic.unsqueeze(0).repeat(
                    per_image_bbox_preds.size(0), 1, 1))
        return torch.cat(intrinsics, dim=0)

    @staticmethod
    def _backproject(pixels: Tensor, depth: Tensor,
                     intrinsics: Tensor) -> Tensor:
        """Camera-frame points from pixel centres and metric depth.

        Solved in an autocast-disabled float32 island for the same reason
        :meth:`_recover_translation` is: CUDA has no bfloat16 kernel for
        ``torch.linalg.solve``, and the compact configs train under bfloat16
        AMP.  Gradients to the centre and depth branches are preserved.
        """
        with torch.autocast(device_type=pixels.device.type, enabled=False):
            pixels = pixels.float()
            depth = depth.float()
            intrinsics = intrinsics.float()
            homogeneous = torch.cat(
                (pixels, torch.ones_like(pixels[:, :1])), dim=-1)
            rays = torch.linalg.solve(
                intrinsics, homogeneous.unsqueeze(-1)).squeeze(-1)
            return depth * rays

    def loss_by_feat_single(self, cls_scores: Tensor, bbox_preds: Tensor,
                           centers_2d_preds: Tensor, z_preds: Tensor,
                           rotation_preds: Tensor, sizes_preds: Tensor,
                           sizes_chain_preds: Tensor,
                           rotation_chain_preds: Tensor, z_chain_preds: Tensor,
                           obb_aux_preds: Tensor,
                           batch_gt_instances: InstanceList,
                           batch_img_metas: List[dict],
                           pose_supervision: bool = True,
                           obb_aux_supervision: bool = True,
                           assigner=None,
                           ellipsoid_preds: Tensor = None,
                           ellipse2d_preds: Tensor = None) -> Tuple[Tensor]:
        """Loss function for outputs from a single decoder layer."""
        if not pose_supervision:
            query_shape = centers_2d_preds.shape[:-1]
            z_preds = centers_2d_preds.new_zeros((*query_shape, 1))
            rotation_preds = centers_2d_preds.new_zeros(
                (*query_shape, self.rot_dim))
            sizes_preds = centers_2d_preds.new_zeros((*query_shape, 3))
        num_imgs = cls_scores.size(0)
        cls_scores_list = [cls_scores[i] for i in range(num_imgs)]
        bbox_preds_list = [bbox_preds[i] for i in range(num_imgs)]
        centers_2d_preds_list = [centers_2d_preds[i] for i in range(num_imgs)]
        z_preds_list = [z_preds[i] for i in range(num_imgs)]
        rotation_preds_list = [rotation_preds[i] for i in range(num_imgs)]
        sizes_preds_list = [sizes_preds[i] for i in range(num_imgs)]
        ellipse2d_preds_list = (
            [ellipse2d_preds[i] for i in range(num_imgs)]
            if self._assigner_uses_ellipse and ellipse2d_preds is not None
            else None)

        cls_reg_targets = self.get_targets(cls_scores_list, bbox_preds_list,
                                          centers_2d_preds_list, z_preds_list,
                                          rotation_preds_list, sizes_preds_list,  
                                          batch_gt_instances, batch_img_metas,
                                          ellipse2d_preds_list,
                                          assigner=assigner,
                                          pose_for_matching=pose_supervision)
        (labels_list, label_weights_list, bbox_targets_list, bbox_weights_list,
         centers_2d_targets_list, centers_2d_weights_list,
         z_targets_list, z_weights_list,
         rotation_targets_list, rotation_weights_list,
         sizes_targets_list, sizes_weights_list,
         obb_gaussian_targets_list, obb_gaussian_weights_list,
         num_total_pos, num_total_neg) = cls_reg_targets

        labels = torch.cat(labels_list, 0)
        label_weights = torch.cat(label_weights_list, 0)
        bbox_targets = torch.cat(bbox_targets_list, 0)
        bbox_weights = torch.cat(bbox_weights_list, 0)
        centers_2d_targets = torch.cat(centers_2d_targets_list, 0)
        centers_2d_weights = torch.cat(centers_2d_weights_list, 0)
        z_targets = torch.cat(z_targets_list, 0)
        z_weights = torch.cat(z_weights_list, 0)
        rotation_targets = torch.cat(rotation_targets_list, 0)
        rotation_weights = torch.cat(rotation_weights_list, 0)
        sizes_targets = torch.cat(sizes_targets_list, 0)
        sizes_weights = torch.cat(sizes_weights_list, 0)
        obb_gaussian_targets = torch.cat(obb_gaussian_targets_list, 0)
        obb_gaussian_weights = torch.cat(obb_gaussian_weights_list, 0)

        # Image factors are also the compact-Gaussian coordinate transform.
        # Build them once and reuse them for quality, HBB, projection and OBB
        # objectives rather than maintaining separate normalization paths.
        factors = []
        for img_meta, bbox_pred in zip(batch_img_metas, bbox_preds):
            img_h, img_w = img_meta['img_shape']
            factor = bbox_pred.new_tensor(
                [img_w, img_h, img_w, img_h]).unsqueeze(0).repeat(
                    bbox_pred.size(0), 1)
            factors.append(factor)
        factors = torch.cat(factors, 0)
        flat_bbox_preds = bbox_preds.reshape(-1, 4)
        indexing_labels = labels.clone()
        indexing_labels[indexing_labels == self.num_classes] = 0

        # Decode the matched class channel once.  The same ellipse then drives
        # both the KLD regression objective and, when configured, the
        # matchability score target.  This prevents classification confidence
        # from continuing to rank only HBB quality while evaluation ranks the
        # ellipse envelope.
        ellipse_mean = ellipse_cholesky = ellipse_sigma = None
        if self.loss_ellipse2d is not None and ellipse2d_preds is not None:
            if self.gaucho_classwise:
                raw2d = ellipse2d_preds.reshape(
                    -1, self.num_classes, 5)
                raw2d = raw2d[
                    torch.arange(raw2d.size(0), device=raw2d.device),
                    indexing_labels]
            else:
                raw2d = ellipse2d_preds.reshape(-1, 5)
            box_px = (flat_bbox_preds * factors).detach()
            ellipse_mean, ellipse_cholesky, ellipse_sigma = \
                self._decode_gaucho_ellipse2d(raw2d, box_px)

        flat_obb_aux_preds = None
        normalized_obb_targets = None
        if obb_aux_preds is not None:
            flat_obb_aux_preds = obb_aux_preds.reshape(-1, 5)
            normalized_obb_targets = self._normalize_obb_gaussian_targets(
                obb_gaussian_targets, factors)

        # classification loss
        cls_scores = cls_scores.reshape(-1, self.cls_out_channels)
        cls_avg_factor = num_total_pos * 1.0 + num_total_neg * self.bg_cls_weight
        if self.sync_cls_avg_factor:
            cls_avg_factor = reduce_mean(cls_scores.new_tensor([cls_avg_factor]))
        cls_avg_factor = max(cls_avg_factor, 1)

        quality_geometry_predictions = flat_obb_aux_preds
        quality_geometry_targets = normalized_obb_targets
        if self.uses_quality_target and \
                self.quality_target_policy.source in {
                    'ellipse_kld', 'ellipse_kld_blend'}:
            if ellipse_sigma is not None:
                quality_geometry_predictions = torch.stack(
                    (ellipse_mean[:, 0], ellipse_mean[:, 1],
                     ellipse_sigma[:, 0, 0], ellipse_sigma[:, 0, 1],
                     ellipse_sigma[:, 1, 1]), dim=-1)
                quality_geometry_targets = obb_gaussian_targets
            else:
                # Encoder proposals have no ellipse branch.  A config can make
                # this explicit with missing_obb='hbb_iou'.
                quality_geometry_predictions = None
                quality_geometry_targets = None

        loss_cls = self._classification_loss(
            cls_scores=cls_scores,
            labels=labels,
            label_weights=label_weights,
            bbox_predictions=flat_bbox_preds,
            bbox_targets=bbox_targets,
            avg_factor=cls_avg_factor,
            obb_predictions=quality_geometry_predictions,
            obb_targets=quality_geometry_targets,
        )

        num_total_pos = loss_cls.new_tensor([num_total_pos])
        num_total_pos = torch.clamp(reduce_mean(num_total_pos), min=1).item()

        bbox_preds = flat_bbox_preds
        bboxes = bbox_cxcywh_to_xyxy(bbox_preds) * factors
        bboxes_gt = bbox_cxcywh_to_xyxy(bbox_targets) * factors

        loss_iou = self.loss_iou(bboxes, bboxes_gt, bbox_weights, avg_factor=num_total_pos)
        loss_bbox = self.loss_bbox(bbox_preds, bbox_targets, bbox_weights, avg_factor=num_total_pos)
        
        # centers_2d loss
        centers_2d_preds = centers_2d_preds.reshape(-1, 2)
        loss_centers_2d = self.loss_centers_2d(centers_2d_preds, centers_2d_targets, centers_2d_weights, avg_factor=num_total_pos)

        if not pose_supervision:
            zero = loss_centers_2d * 0.0
            return (loss_cls, loss_bbox, loss_iou, loss_centers_2d,
                    zero, zero, zero, zero, zero, zero, zero, zero)

        # z loss
        z_preds = z_preds.reshape(-1, 1)
        loss_z = self.loss_z(z_preds, z_targets, z_weights, avg_factor=num_total_pos)

        # rotation loss
        # rotation_preds = rotation_preds.reshape(-1, 6)
        if self.classwise_rotation:
            rotation_preds = rotation_preds.reshape(-1, self.num_classes, self.rot_dim)
            rotation_preds = rotation_preds[torch.arange(rotation_preds.size(0), device=rotation_preds.device), indexing_labels]
        else:
            rotation_preds = rotation_preds.reshape(-1, self.rot_dim)
        loss_rotation = self.loss_rotation(rotation_preds, rotation_targets, rotation_weights, labels=labels, avg_factor=num_total_pos)

        # sizes loss
        # sizes_preds = sizes_preds.reshape(-1, 3)
        if self.classwise_sizes:
            sizes_preds = sizes_preds.reshape(-1, self.num_classes, 3)
            sizes_preds = sizes_preds[torch.arange(sizes_preds.size(0), device=sizes_preds.device), indexing_labels]
        else:
            sizes_preds = sizes_preds.reshape(-1, 3)
        loss_sizes = self.loss_sizes(sizes_preds, sizes_targets, sizes_weights, avg_factor=num_total_pos)

        if self.loss_projection is not None:
            if self.projection_geometry_source == 'target':
                centers_2d_px = centers_2d_targets * factors[:, :2]
                projection_depth = z_targets
                projection_sizes = sizes_targets
            else:
                centers_2d_px = centers_2d_preds * factors[:, :2]
                projection_depth = (
                    torch.exp(z_preds) if self.use_log_z else z_preds)
                projection_sizes = sizes_preds
            projection_intrinsics = []
            for img_meta, per_image_bbox_preds in zip(
                    batch_img_metas, bbox_preds_list):
                intrinsic = self._training_intrinsic_matrix(
                    img_meta, centers_2d_preds)
                projection_intrinsics.append(
                    intrinsic.unsqueeze(0).repeat(
                        per_image_bbox_preds.size(0), 1, 1))
            projection_intrinsics = torch.cat(projection_intrinsics, dim=0)
            loss_projection = self.loss_projection(
                centers_2d_px=centers_2d_px,
                depth=projection_depth,
                rotations=rotation_preds,
                sizes=projection_sizes,
                target_gaussians=obb_gaussian_targets,
                intrinsics=projection_intrinsics,
                weight=obb_gaussian_weights,
                avg_factor=num_total_pos,
            )
        else:
            loss_projection = z_preds.new_tensor(0.0)

        if self.loss_obb_aux is not None and obb_aux_supervision:
            if obb_aux_preds is None:
                raise RuntimeError('loss_obb_aux requires OBB predictions')
            loss_obb_aux = self.loss_obb_aux(
                predicted=flat_obb_aux_preds,
                target=normalized_obb_targets,
                weight=obb_gaussian_weights,
                avg_factor=num_total_pos,
            )
        else:
            loss_obb_aux = z_preds.new_tensor(0.0)

        # ── CoP auxiliary losses (chain outputs, same targets as parallel) ──
        if self.use_cop_chain and sizes_chain_preds is not None:
            if self.classwise_sizes:
                sizes_chain = sizes_chain_preds.reshape(
                    -1, self.num_classes, 3)
                sizes_chain = sizes_chain[
                    torch.arange(sizes_chain.size(0),
                                 device=sizes_chain.device), indexing_labels]
            else:
                sizes_chain = sizes_chain_preds.reshape(-1, 3)
            loss_size_chain = self.loss_sizes(
                sizes_chain, sizes_targets, sizes_weights,
                avg_factor=num_total_pos) * self.cop_loss_weights['size']

            if self.classwise_rotation:
                rot_chain = rotation_chain_preds.reshape(
                    -1, self.num_classes, self.rot_dim)
                rot_chain = rot_chain[
                    torch.arange(rot_chain.size(0),
                                 device=rot_chain.device), indexing_labels]
            else:
                rot_chain = rotation_chain_preds.reshape(-1, self.rot_dim)
            loss_rotation_chain = self.loss_rotation(
                rot_chain, rotation_targets, rotation_weights,
                labels=labels, avg_factor=num_total_pos) * self.cop_loss_weights['rotation']

            z_chain = z_chain_preds.reshape(-1, 1)
            loss_z_chain = self.loss_z(
                z_chain, z_targets, z_weights,
                avg_factor=num_total_pos) * self.cop_loss_weights['z']
        else:
            loss_size_chain = loss_rotation_chain = loss_z_chain = \
                z_preds.new_tensor(0.0)

        # ── GauCho-2D amodal ellipse loss ─────────────────────────────
        loss_ellipse2d = z_preds.new_tensor(0.0)
        loss_ellipse2d_corner = z_preds.new_tensor(0.0)
        loss_ellipse2d_angle = z_preds.new_tensor(0.0)
        if ellipse_cholesky is not None:
            loss_ellipse2d = self.loss_ellipse2d(
                ellipse_mean, ellipse_cholesky, obb_gaussian_targets,
                weight=obb_gaussian_weights, avg_factor=num_total_pos)
            if self.loss_ellipse2d_corner is not None:
                loss_ellipse2d_corner = self.loss_ellipse2d_corner(
                    ellipse_mean, ellipse_sigma, obb_gaussian_targets,
                    weight=obb_gaussian_weights, avg_factor=num_total_pos)
            if self.loss_ellipse2d_angle is not None:
                loss_ellipse2d_angle = self.loss_ellipse2d_angle(
                    ellipse_sigma, obb_gaussian_targets,
                    weight=obb_gaussian_weights, avg_factor=num_total_pos)

        # ── GauCho-3D ellipsoid losses ────────────────────────────────
        loss_ellipsoid = z_preds.new_tensor(0.0)
        loss_ellipsoid_projection = z_preds.new_tensor(0.0)
        loss_ellipsoid_max_axis = z_preds.new_tensor(0.0)
        if self.loss_ellipsoid is not None and ellipsoid_preds is not None:
            if self.gaucho_classwise:
                raw = ellipsoid_preds.reshape(-1, self.num_classes, 6)
                raw = raw[torch.arange(raw.size(0), device=raw.device),
                          indexing_labels]
            else:
                raw = ellipsoid_preds.reshape(-1, 6)

            positive = sizes_weights[:, 0] > 0
            if bool(positive.any()):
                # One float32 island around the whole block.  Autocast demotes
                # matmul but leaves ``torch.linalg.*`` alone, and CUDA ships no
                # bfloat16 kernel for cholesky, triangular solve, LU or eigh --
                # so under the bfloat16 AMP these configs use, the very first
                # iteration would die.  The island has to enclose the loss
                # modules too, not just the calls here, or the next unsupported
                # kernel simply fails a few lines later.
                device_type = z_preds.device.type
                with torch.autocast(device_type=device_type, enabled=False):
                    intrinsics = self._gaucho_query_intrinsics(
                        batch_img_metas, bbox_preds_list,
                        centers_2d_preds.float())
                    selected_intrinsics = intrinsics[positive].float()
                    predicted_cholesky = self._gaucho_cholesky(
                        raw[positive].float())

                    # The GT ellipsoid is the inscribed ellipsoid of the
                    # annotated oriented box.  ``Sigma`` is built once here;
                    # the network is never asked to reproduce ``R`` itself.
                    target_rotation = _rotation_6d_to_matrix(
                        rotation_targets[positive].float())
                    target_sigma = ellipsoid_from_rotation_size(
                        target_rotation, sizes_targets[positive].float())
                    identity = torch.eye(
                        3, dtype=target_sigma.dtype,
                        device=target_sigma.device)
                    target_cholesky = torch.linalg.cholesky(
                        target_sigma + 1e-12 * identity)

                    # The 3D centre reuses the already-supervised 2D centre and
                    # depth branches instead of adding a competing estimator.
                    predicted_depth = (
                        torch.exp(z_preds) if self.use_log_z else z_preds)
                    predicted_center = self._backproject(
                        (centers_2d_preds.reshape(-1, 2) *
                         factors[:, :2])[positive].float(),
                        predicted_depth.reshape(-1, 1)[positive].float(),
                        selected_intrinsics)
                    target_center = self._backproject(
                        (centers_2d_targets * factors[:, :2])[positive].float(),
                        z_targets[positive].float(),
                        selected_intrinsics)

                    # Left in float32 on purpose: summing them with the bf16
                    # siblings promotes to float32 anyway, and casting back
                    # down would discard the range this island exists to keep.
                    # An annotation larger than the objects can physically be
                    # is an annotation error, not a large fruit.  Forcing the
                    # shape KLD to reproduce it fights the max-axis bound
                    # directly, so it is dropped from 3D shape supervision --
                    # and only from that: the same instance still teaches
                    # detection, the 2D ellipse and the centre.
                    shape_weight = None
                    if self.ellipsoid_gt_max_diameter is not None:
                        gt_extent = sizes_targets[positive].float().max(
                            dim=-1).values
                        shape_weight = (
                            gt_extent <= self.ellipsoid_gt_max_diameter
                        ).to(predicted_cholesky.dtype)

                    loss_ellipsoid = self.loss_ellipsoid(
                        predicted_center, predicted_cholesky,
                        target_center, target_cholesky,
                        weight=shape_weight,
                        avg_factor=num_total_pos)

                    if self.loss_ellipsoid_max_axis is not None:
                        # Applied to every positive, including the ones the
                        # shape term ignores: the physical bound does not stop
                        # applying because an annotation is wrong.
                        loss_ellipsoid_max_axis = self.loss_ellipsoid_max_axis(
                            predicted_cholesky, avg_factor=num_total_pos)

                    if self.loss_ellipsoid_projection is not None:
                        loss_ellipsoid_projection = \
                            self.loss_ellipsoid_projection(
                                predicted_center, predicted_cholesky,
                                obb_gaussian_targets[positive].float(),
                                selected_intrinsics,
                                weight=obb_gaussian_weights[positive].float(),
                                avg_factor=num_total_pos)
            else:
                # Keep the branch in the graph so DDP sees a gradient for it
                # even on an image with no positive query.
                loss_ellipsoid = raw.sum() * 0.0
                if self.loss_ellipsoid_projection is not None:
                    loss_ellipsoid_projection = raw.sum() * 0.0
                if self.loss_ellipsoid_max_axis is not None:
                    loss_ellipsoid_max_axis = raw.sum() * 0.0

        return (loss_cls, loss_bbox, loss_iou, loss_centers_2d, loss_z,
                loss_rotation, loss_sizes, loss_projection, loss_obb_aux,
                loss_size_chain, loss_rotation_chain, loss_z_chain,
                loss_ellipsoid, loss_ellipsoid_projection, loss_ellipse2d,
                loss_ellipsoid_max_axis, loss_ellipse2d_corner,
                loss_ellipse2d_angle)

    def get_targets(self, cls_scores_list: List[Tensor], bbox_preds_list: List[Tensor],
                    centers_2d_preds_list: List[Tensor], z_preds_list: List[Tensor],
                    rotation_preds_list: List[Tensor], sizes_preds_list: List[Tensor],
                    batch_gt_instances: InstanceList,
                    batch_img_metas: List[dict],
                    ellipse2d_preds_list: List[Tensor] = None,
                    assigner=None,
                    pose_for_matching: bool = True) -> tuple:
        """Compute regression and classification targets for a batch image."""
        if ellipse2d_preds_list is None:
            ellipse2d_preds_list = [None] * len(batch_img_metas)
        (labels_list, label_weights_list, bbox_targets_list, bbox_weights_list,
         centers_2d_targets_list, centers_2d_weights_list,
         z_targets_list, z_weights_list,
         rotation_targets_list, rotation_weights_list,
         sizes_targets_list, sizes_weights_list,
         obb_gaussian_targets_list, obb_gaussian_weights_list,
         pos_inds_list, neg_inds_list) = multi_apply(self._get_targets_single,
                                      cls_scores_list, bbox_preds_list,
                                      centers_2d_preds_list, z_preds_list,
                                      rotation_preds_list, sizes_preds_list,
                                      batch_gt_instances, batch_img_metas,
                                      ellipse2d_preds_list,
                                      assigner=assigner,
                                      pose_for_matching=pose_for_matching)
        num_total_pos = sum((inds.numel() for inds in pos_inds_list))
        num_total_neg = sum((inds.numel() for inds in neg_inds_list))
        return (labels_list, label_weights_list, bbox_targets_list, bbox_weights_list,
                centers_2d_targets_list, centers_2d_weights_list,
                z_targets_list, z_weights_list,
                rotation_targets_list, rotation_weights_list,
                sizes_targets_list, sizes_weights_list,
                obb_gaussian_targets_list, obb_gaussian_weights_list,
                num_total_pos, num_total_neg)

    @staticmethod
    def _intrinsic_matrix(intrinsic, reference: Tensor) -> Tensor:
        """Normalize 4-value or 3x3 camera intrinsics to a local tensor."""
        if isinstance(intrinsic, Tensor):
            return intrinsic.to(device=reference.device,
                                dtype=reference.dtype).view(3, 3)
        if len(intrinsic) == 4:
            intrinsic = [[intrinsic[0], 0, intrinsic[2]],
                         [0, intrinsic[1], intrinsic[3]],
                         [0, 0, 1]]
        return reference.new_tensor(intrinsic).view(3, 3)

    @classmethod
    def _image_space_intrinsic_matrix(
            cls, img_meta: dict, reference: Tensor) -> Tensor:
        """Map the stored original-image intrinsic into augmented pixels.

        The pose pipeline deliberately keeps the original intrinsic for
        teacher-free inference, where predicted centers are rescaled back to
        original pixels. Training losses and matching operate before that
        rescale, so they need ``K_image = H_image K_original`` instead.

        ``RandomFlipFor9DPose`` historically reflects the stored intrinsic
        using the resized image extent. Undo that reflection first, apply the
        resize, then reapply it; this preserves existing inference metadata
        while making the training geometry exact.
        """
        intrinsic = cls._intrinsic_matrix(img_meta['intrinsic'], reference)
        scale_factor = img_meta.get('scale_factor')
        if scale_factor is None:
            return intrinsic
        scale = reference.new_tensor(scale_factor).flatten()
        if scale.numel() < 2:
            raise ValueError(
                'scale_factor must contain x/y scales, got '
                f'{scale_factor!r}')
        resize = torch.eye(3, device=reference.device, dtype=reference.dtype)
        resize[0, 0] = scale[0]
        resize[1, 1] = scale[1]

        reflection = None
        if img_meta.get('flip', False):
            img_h, img_w = img_meta['img_shape'][:2]
            reflection = torch.eye(
                3, device=reference.device, dtype=reference.dtype)
            direction = img_meta.get('flip_direction')
            if direction == 'horizontal':
                reflection[0, 0] = -1.0
                reflection[0, 2] = float(img_w - 1)
            elif direction == 'vertical':
                reflection[1, 1] = -1.0
                reflection[1, 2] = float(img_h - 1)
            else:
                raise ValueError(
                    'flipped image requires horizontal or vertical '
                    f'flip_direction, got {direction!r}')
            # A pixel-axis reflection is its own inverse.
            intrinsic = reflection @ intrinsic

        intrinsic = resize @ intrinsic
        if reflection is not None:
            intrinsic = reflection @ intrinsic
        return intrinsic

    def _training_intrinsic_matrix(
            self, img_meta: dict, reference: Tensor) -> Tensor:
        if self.train_intrinsic_to_image_space:
            return self._image_space_intrinsic_matrix(img_meta, reference)
        return self._intrinsic_matrix(img_meta['intrinsic'], reference)

    def _recover_translation(
            self, intrinsic: Tensor, centers_2d_h: Tensor,
            z_pred: Tensor) -> Tensor:
        """Back-project image centers in stable float32 under AMP.

        CUDA/CPU linear algebra does not support every low-precision inverse,
        and explicitly solving ``K t = p`` is both more stable and cheaper than
        constructing ``K^-1``.  Casting inside an autocast-disabled island
        preserves gradients to the center/depth predictions while keeping the
        geometric result finite.
        """
        with torch.autocast(
                device_type=centers_2d_h.device.type, enabled=False):
            centers_float = centers_2d_h.float()
            intrinsic_float = intrinsic.float()
            z_float = z_pred.float()
            depth = torch.exp(z_float) if self.use_log_z else z_float
            rays = torch.linalg.solve(
                intrinsic_float, centers_float.transpose(0, 1)
            ).transpose(0, 1)
            return depth * rays

    def _build_matching_pred_instances(
            self, cls_score: Tensor, bbox_pred: Tensor,
            centers_2d_pred: Tensor, z_pred: Tensor,
            rotation_pred: Tensor, sizes_pred: Tensor, img_meta: dict,
            pose_for_matching: bool = True,
            ellipse2d_pred: Tensor = None) -> InstanceData:
        """Build the exact prediction structure consumed by the assigner.

        Keeping this conversion in one place lets offline diagnostics inspect
        the same coordinates and tensors used by the training loss.
        """
        img_h, img_w = img_meta['img_shape']
        factor = bbox_pred.new_tensor([img_w, img_h, img_w, img_h]).unsqueeze(0)
        num_bboxes = bbox_pred.size(0)
        bbox_pred_unnorm = bbox_cxcywh_to_xyxy(bbox_pred) * factor

        ellipse_fields = {}
        if self._assigner_uses_ellipse and ellipse2d_pred is not None:
            if self.gaucho_classwise:
                raw2d = ellipse2d_pred.reshape(
                    num_bboxes, self.num_classes, 5)
                box_shape = (bbox_pred * factor).detach().unsqueeze(1)
            else:
                raw2d = ellipse2d_pred.reshape(num_bboxes, 5)
                box_shape = (bbox_pred * factor).detach()
            # Decoded exactly as the loss decodes it, against the query's own
            # box in pixels and detached for the same reason: matching must not
            # reshape the box branch through the ellipse chart.  Classwise
            # predictions retain all class channels here: each GT column, not
            # the query's pre-assignment argmax, selects the relevant channel
            # inside Ellipse2DKLDCost.
            mean, _, sigma = self._decode_gaucho_ellipse2d(
                raw2d, box_shape)
            compact = torch.stack(
                (mean[..., 0], mean[..., 1], sigma[..., 0, 0],
                 sigma[..., 0, 1], sigma[..., 1, 1]), dim=-1)
            if not self.gaucho_classwise:
                compact = compact.unsqueeze(1).expand(
                    num_bboxes, self.num_classes, 5)
            ellipse_fields = dict(ellipse_gaussians=compact)

        if not pose_for_matching:
            return InstanceData(scores=cls_score, bboxes=bbox_pred_unnorm,
                                **ellipse_fields)

        centers_2d_px = centers_2d_pred * centers_2d_pred.new_tensor(
            [img_w, img_h])
        centers_2d_h = torch.cat([
            centers_2d_px,
            torch.ones_like(centers_2d_pred[:, :1])
        ], dim=1)
        intrinsic = self._training_intrinsic_matrix(img_meta, centers_2d_h)
        t_recovered = self._recover_translation(
            intrinsic, centers_2d_h, z_pred)

        pred_labels = cls_score.argmax(dim=-1)
        if self.classwise_rotation:
            rotation_pred = rotation_pred.reshape(
                num_bboxes, -1, self.rot_dim)
            rotation_pred = rotation_pred[
                torch.arange(num_bboxes, device=cls_score.device),
                pred_labels]
        if self.classwise_sizes:
            sizes_pred = sizes_pred.reshape(num_bboxes, -1, 3)
            sizes_pred = sizes_pred[
                torch.arange(num_bboxes, device=cls_score.device),
                pred_labels]
        return InstanceData(
            scores=cls_score,
            bboxes=bbox_pred_unnorm,
            translations=t_recovered,
            rotations=rotation_pred,
            sizes=sizes_pred,
            **ellipse_fields)

    def matching_diagnostics(
            self, cls_score: Tensor, bbox_pred: Tensor,
            centers_2d_pred: Tensor, z_pred: Tensor,
            rotation_pred: Tensor, sizes_pred: Tensor,
            gt_instances: InstanceData, img_meta: dict, assigner=None,
            pose_for_matching: bool = True) -> dict:
        """Expose component matrices and the native Hungarian assignment."""
        active_assigner = assigner or self.assigner
        pred_instances = self._build_matching_pred_instances(
            cls_score, bbox_pred, centers_2d_pred, z_pred,
            rotation_pred, sizes_pred, img_meta,
            pose_for_matching=pose_for_matching)
        component_costs = {}
        for index, match_cost in enumerate(active_assigner.match_costs):
            name = type(match_cost).__name__
            if name in component_costs:
                name = f'{name}_{index}'
            component_costs[name] = match_cost(
                pred_instances=pred_instances,
                gt_instances=gt_instances,
                img_meta=img_meta)
        assign_result = active_assigner.assign(
            pred_instances=pred_instances,
            gt_instances=gt_instances,
            img_meta=img_meta)
        return dict(
            pred_instances=pred_instances,
            component_costs=component_costs,
            assign_result=assign_result)

    def _get_targets_single(self, cls_score: Tensor, bbox_pred: Tensor,
                           centers_2d_pred: Tensor, z_pred: Tensor,
                           rotation_pred: Tensor, sizes_pred: Tensor,
                           gt_instances: InstanceData, img_meta: dict,
                           ellipse2d_pred: Tensor = None,
                           assigner=None,
                           pose_for_matching: bool = True) -> tuple:
        """Compute regression and classification targets for one image."""
        img_h, img_w = img_meta['img_shape']
        factor = bbox_pred.new_tensor([img_w, img_h, img_w, img_h]).unsqueeze(0)
        num_bboxes = bbox_pred.size(0)
        pred_instances = self._build_matching_pred_instances(
            cls_score, bbox_pred, centers_2d_pred, z_pred,
            rotation_pred, sizes_pred, img_meta,
            pose_for_matching=pose_for_matching,
            ellipse2d_pred=ellipse2d_pred)

        active_assigner = assigner or self.assigner
        assign_result = active_assigner.assign(
            pred_instances=pred_instances,
            gt_instances=gt_instances,
            img_meta=img_meta)

        gt_bboxes = gt_instances.bboxes
        gt_labels = gt_instances.labels
        gt_centers_2d = gt_instances.centers_2d
        gt_z = gt_instances.z
        gt_rotations = gt_instances.rotations
        gt_sizes = gt_instances.sizes

        pos_inds = torch.nonzero(assign_result.gt_inds > 0, as_tuple=False).squeeze(-1).unique()
        neg_inds = torch.nonzero(assign_result.gt_inds == 0, as_tuple=False).squeeze(-1).unique()
        pos_assigned_gt_inds = assign_result.gt_inds[pos_inds] - 1
        pos_gt_bboxes = gt_bboxes[pos_assigned_gt_inds.long(), :]

        # Initialize targets
        labels = gt_bboxes.new_full((num_bboxes, ), self.num_classes, dtype=torch.long)
        labels[pos_inds] = gt_labels[pos_assigned_gt_inds]
        label_weights = gt_bboxes.new_ones(num_bboxes)

        bbox_targets = torch.zeros_like(bbox_pred, dtype=gt_bboxes.dtype)
        bbox_weights = torch.zeros_like(bbox_pred, dtype=gt_bboxes.dtype)
        bbox_weights[pos_inds] = 1.0

        centers_2d_targets = torch.zeros(num_bboxes, 2, device=gt_bboxes.device)
        centers_2d_weights = torch.zeros(num_bboxes, 2, device=gt_bboxes.device)
        centers_2d_weights[pos_inds] = 1.0

        z_targets = torch.zeros(num_bboxes, 1, device=gt_bboxes.device)
        z_weights = torch.zeros(num_bboxes, 1, device=gt_bboxes.device)
        z_weights[pos_inds] = 1.0

        rotation_targets = torch.zeros(num_bboxes, 6, device=gt_bboxes.device)
        rotation_weights = torch.zeros(num_bboxes, 6, device=gt_bboxes.device)
        rotation_weights[pos_inds] = 1.0

        sizes_targets = torch.zeros(num_bboxes, 3, device=gt_bboxes.device)
        sizes_weights = torch.zeros(num_bboxes, 3, device=gt_bboxes.device)
        sizes_weights[pos_inds] = 1.0

        obb_gaussian_targets = torch.zeros(
            num_bboxes, 5, device=gt_bboxes.device)
        obb_gaussian_weights = torch.zeros(
            num_bboxes, device=gt_bboxes.device)
        if self._requires_obb_gaussians:
            if 'obb_gaussians' not in gt_instances:
                raise KeyError(
                    'this loss configuration requires '
                    'gt_instances.obb_gaussians')
            obb_gaussian_weights[pos_inds] = 1.0

        # Set targets for positive samples
        pos_gt_bboxes_normalized = pos_gt_bboxes / factor
        pos_gt_bboxes_targets = bbox_xyxy_to_cxcywh(pos_gt_bboxes_normalized)
        bbox_targets[pos_inds] = pos_gt_bboxes_targets

        gt_centers_2d_normalized = gt_centers_2d / factor[:, :2]
        centers_2d_targets[pos_inds] = gt_centers_2d_normalized[pos_assigned_gt_inds.long(), :]

        z_targets[pos_inds] = gt_z[pos_assigned_gt_inds.long(), :]
        rotation_targets[pos_inds] = gt_rotations[pos_assigned_gt_inds]
        sizes_targets[pos_inds] = gt_sizes[pos_assigned_gt_inds]
        if self._requires_obb_gaussians:
            obb_gaussian_targets[pos_inds] = gt_instances.obb_gaussians[
                pos_assigned_gt_inds]

        return (labels, label_weights, bbox_targets, bbox_weights,
                centers_2d_targets, centers_2d_weights, z_targets, z_weights,
                rotation_targets, rotation_weights, sizes_targets, sizes_weights,
                obb_gaussian_targets, obb_gaussian_weights,
                pos_inds, neg_inds)

    def predict_by_feat(self,
                        all_layers_cls_scores: Tensor,
                        all_layers_bbox_preds: Tensor,
                        all_layers_centers_2d_preds: Tensor,
                        all_layers_z_preds: Tensor,
                        all_layers_rotation_preds: Tensor,
                        all_layers_sizes_preds: Tensor,
                        batch_img_metas: List[Dict],
                        rescale: bool = False) -> InstanceList:
        """Transform a batch of output features extracted from the head into
        bbox results.

        Args:
            all_layers_cls_scores (Tensor): Classification scores of all
                decoder layers, has shape (num_decoder_layers, bs, num_queries,
                cls_out_channels).
            all_layers_bbox_preds (Tensor): Regression outputs of all decoder
                layers. Each is a 4D-tensor with normalized coordinate format
                (cx, cy, w, h) and shape (num_decoder_layers, bs, num_queries,
                4) with the last dimension arranged as (cx, cy, w, h).
            all_layers_centers_2d_preds (Tensor): Sigmoid regression
                outputs of each decoder layers. Each is a 4D-tensor with
                normalized coordinate format (cx, cy) and shape
                (num_decoder_layers, bs, num_queries, 2).
            all_layers_z_preds (Tensor): Direct regression
                outputs of each decoder layers. Each is a 1D-tensor with
                the directly predicted z coordinate and shape
                (num_decoder_layers, bs, num_queries, 1).
            all_layers_rotation_preds (Tensor): Direct regression
                outputs of each decoder layers. Each is a 6D-tensor with
                [r1, r2] that are first two columns of rotation matrix
                and shape (num_decoder_layers, bs, num_queries, 6).
            all_layers_sizes_preds (Tensor): Direct regression
                outputs of each decoder layers. Each is a 3D-tensor with
                object sizes and shape (num_decoder_layers, bs, num_queries, 3).
            batch_img_metas (list[dict]): Meta information of each image.
            rescale (bool, optional): If `True`, return boxes in original
                image space. Default `False`.

        Returns:
            list[obj:`InstanceData`]: Detection results of each image
            after the post process.
        """
        cls_scores = all_layers_cls_scores[-1]
        bbox_preds = all_layers_bbox_preds[-1]
        centers_2d_preds = all_layers_centers_2d_preds[-1]
        z_preds = all_layers_z_preds[-1]
        rotation_preds = all_layers_rotation_preds[-1]
        sizes_preds = all_layers_sizes_preds[-1]

        result_list = []
        for img_id in range(len(batch_img_metas)):
            cls_score = cls_scores[img_id]
            bbox_pred = bbox_preds[img_id]
            centers_2d_pred = centers_2d_preds[img_id]
            z_pred = z_preds[img_id]
            rotation_pred = rotation_preds[img_id]
            sizes_pred = sizes_preds[img_id]

            img_meta = batch_img_metas[img_id]
            results = self._predict_by_feat_single(cls_score, bbox_pred,
                                                   centers_2d_pred, z_pred,
                                                   rotation_pred, sizes_pred,
                                                   img_meta, rescale)
            result_list.append(results)
        return result_list

    def _predict_by_feat_single(self,
                                cls_score: Tensor,
                                bbox_pred: Tensor,
                                centers_2d_pred: Tensor,
                                z_pred: Tensor,
                                rotation_pred: Tensor,
                                sizes_pred: Tensor,
                                img_meta: dict,
                                rescale: bool = True) -> InstanceData:
        """Transform outputs from the last decoder layer into bbox predictions
        for each image.

        Args:
            cls_score (Tensor): Box score logits from the last decoder layer
                for each image. Shape [num_queries, cls_out_channels].
            bbox_pred (Tensor): Sigmoid outputs from the last decoder layer
                for each image, with coordinate format (cx, cy, w, h) and
                shape [num_queries, 4].
            centers_2d_pred (Tensor): Sigmoid outputs from the last decoder
                layer for each image, with coordinate format (cx, cy) and
                shape [num_queries, 2].
            z_pred (Tensor): Direct outputs from the last decoder layer
                for each image, with coordinate format (z) and
                shape [num_queries, 1].
            rotation_pred (Tensor): Direct outputs from the last decoder layer
                for each image, with coordinate format (cx, cy, w, h) and
                shape [num_queries, 6].
            sizes_pred (Tensor): Direct outputs from the last decoder layer
                for each image, with object sizes and shape [num_queries, 3].
            img_meta (dict): Image meta info.
            rescale (bool): If True, return boxes in original image
                space. Default True.

        Returns:
            :obj:`InstanceData`: Detection results of each image
            after the post process.
            Each item usually contains following keys.

                - scores (Tensor): Classification scores, has a shape
                  (num_instance, )
                - labels (Tensor): Labels of bboxes, has a shape
                  (num_instances, ).
                - bboxes (Tensor): Has a shape (num_instances, 4),
                  the last dimension 4 arrange as (x1, y1, x2, y2).
                - T (tensor): 3x4 transformation matrix, 
                    representing the rotation and translation of the object
                    in the camera coordinate system. The tensor has a shape
                    (num_instances, 3, 4).
        """
        assert len(cls_score) == len(bbox_pred)  # num_queries
        max_per_img = self.test_cfg.get('max_per_img', len(cls_score))
        img_shape = img_meta['img_shape']
        # exclude background
        if self.loss_cls.use_sigmoid:
            cls_score = cls_score.sigmoid()
            scores, indexes = cls_score.view(-1).topk(max_per_img)
            det_labels = indexes % self.num_classes
            bbox_index = indexes // self.num_classes
            bbox_pred = bbox_pred[bbox_index]
        else:
            scores, det_labels = F.softmax(cls_score, dim=-1)[..., :-1].max(-1)
            scores, bbox_index = scores.topk(max_per_img)
            bbox_pred = bbox_pred[bbox_index]
            det_labels = det_labels[bbox_index]

        centers_2d_pred = centers_2d_pred[bbox_index]
        z_pred = z_pred[bbox_index]
        rotation_pred = rotation_pred[bbox_index]
        sizes_pred = sizes_pred[bbox_index]

        # Handle classwise rotation and size predictions
        if self.classwise_rotation:
            rotation_pred = rotation_pred.view(rotation_pred.size(0), self.num_classes, -1)
            rotation_pred = rotation_pred[torch.arange(len(det_labels)), det_labels]

        if self.classwise_sizes:
            sizes_pred = sizes_pred.view(sizes_pred.size(0), self.num_classes, -1)
            sizes_pred = sizes_pred[torch.arange(len(det_labels)), det_labels]
  
        det_bboxes = bbox_cxcywh_to_xyxy(bbox_pred)
        det_bboxes[:, 0::2] = det_bboxes[:, 0::2] * img_shape[1]
        det_bboxes[:, 1::2] = det_bboxes[:, 1::2] * img_shape[0]
        det_bboxes[:, 0::2].clamp_(min=0, max=img_shape[1])
        det_bboxes[:, 1::2].clamp_(min=0, max=img_shape[0])

        centers_2d_pred = centers_2d_pred * torch.tensor(
            [img_shape[1], img_shape[0]], device=centers_2d_pred.device)
        centers_2d_pred[:, 0] = centers_2d_pred[:, 0].clamp(
            min=0, max=img_shape[1])
        centers_2d_pred[:, 1] = centers_2d_pred[:, 1].clamp(
            min=0, max=img_shape[0])
        if rescale:
            assert img_meta.get('scale_factor') is not None
            det_bboxes /= det_bboxes.new_tensor(
                img_meta['scale_factor']).repeat((1, 2))
            centers_2d_pred /= det_bboxes.new_tensor(
                img_meta['scale_factor'])
        
        # create the transformation matrix
        if self.rot_dim == 6:
            r1, r2 = torch.split(rotation_pred, 3, dim=1)
            r1 = r1 / torch.norm(r1, dim=1, keepdim=True).clamp_min(1e-6)
            r2 = r2 - torch.sum(r1 * r2, dim=1, keepdim=True) * r1
            r2 = r2 / torch.norm(r2, dim=1, keepdim=True).clamp_min(1e-6)
            r3 = torch.cross(r1, r2, dim=1)
            R = torch.stack([r1, r2, r3], dim=-1)
        elif self.rot_dim == 9:
            m = rotation_pred.view(-1, 3, 3)
            u, s, v = torch.svd(m)
            vt = torch.transpose(v, 1, 2)
            det = torch.det(torch.matmul(u, vt))
            det = det.view(-1, 1, 1)
            vt = torch.cat((vt[:, :2, :], vt[:, -1:, :] * det), 1)
            R = torch.matmul(u, vt)
        else:
            raise ValueError(f'Unsupported rotation dimension: {self.rot_dim}')

        intrinsic = img_meta['intrinsic']
        centers_2d_h = torch.cat([centers_2d_pred, 
                                  torch.ones_like(centers_2d_pred[:, :1])],
                                  dim=1)
        if not isinstance(intrinsic, torch.Tensor):
            if len(intrinsic) == 4:
                # intrinsic is a list of [fx, fy, cx, cy]
                intrinsic = [[intrinsic[0], 0, intrinsic[2]],
                             [0, intrinsic[1], intrinsic[3]],
                             [0, 0, 1]]
            intrinsic = torch.tensor(intrinsic).to(centers_2d_h.device)
        intrinsic = intrinsic.view(3, 3)

        t_recovered = self._recover_translation(
            intrinsic, centers_2d_h, z_pred)

        # generate 3x4 transformation matrix
        T = torch.zeros(det_bboxes.shape[0], 4, 4).to(det_bboxes.device)
        T[:, 0:3, 0:3] = R
        T[:, 0:3, 3] = t_recovered
        T[:, 3, 3] = 1.0

        results = InstanceData()
        results.bboxes = det_bboxes
        results.scores = scores
        results.labels = det_labels
        results.T = T
        results.translations = t_recovered
        results.rotations = rotation_pred
        results.centers_2d = centers_2d_pred
        results.z = z_pred
        results.sizes = sizes_pred
    
        return results

    @staticmethod
    def split_outputs(all_layers_cls_scores: Tensor, all_layers_bbox_preds: Tensor,
                      all_layers_centers_2d_preds: Tensor, all_layers_z_preds: Tensor,
                      all_layers_rotation_preds: Tensor, all_layers_sizes_preds: Tensor,
                      dn_meta: Dict[str, int]) -> Tuple[Tensor]:
        """Split outputs into denoising and matching parts."""
        if dn_meta is not None:
            num_denoising_queries = dn_meta['num_denoising_queries']
            all_layers_denoising_cls_scores = \
                all_layers_cls_scores[:, :, : num_denoising_queries, :]
            all_layers_denoising_bbox_preds = \
                all_layers_bbox_preds[:, :, : num_denoising_queries, :]
            all_layers_denoising_centers_2d_preds = \
                all_layers_centers_2d_preds[:, :, : num_denoising_queries, :]
            all_layers_denoising_z_preds = \
                all_layers_z_preds[:, :, : num_denoising_queries, :]
            all_layers_denoising_rotation_preds = \
                all_layers_rotation_preds[:, :, : num_denoising_queries, :]
            all_layers_denoising_sizes_preds = \
                all_layers_sizes_preds[:, :, : num_denoising_queries, :]
            
            all_layers_matching_cls_scores = \
                all_layers_cls_scores[:, :, num_denoising_queries:, :]
            all_layers_matching_bbox_preds = \
                all_layers_bbox_preds[:, :, num_denoising_queries:, :]
            all_layers_matching_centers_2d_preds = \
                all_layers_centers_2d_preds[:, :, num_denoising_queries:, :]
            all_layers_matching_z_preds = \
                all_layers_z_preds[:, :, num_denoising_queries:, :]
            all_layers_matching_rotation_preds = \
                all_layers_rotation_preds[:, :, num_denoising_queries:, :]
            all_layers_matching_sizes_preds = \
                all_layers_sizes_preds[:, :, num_denoising_queries:, :]
        else:
            all_layers_denoising_cls_scores = None
            all_layers_denoising_bbox_preds = None
            all_layers_denoising_centers_2d_preds = None
            all_layers_denoising_z_preds = None
            all_layers_denoising_rotation_preds = None
            all_layers_denoising_sizes_preds = None
            
            all_layers_matching_cls_scores = all_layers_cls_scores
            all_layers_matching_bbox_preds = all_layers_bbox_preds
            all_layers_matching_centers_2d_preds = all_layers_centers_2d_preds
            all_layers_matching_z_preds = all_layers_z_preds
            all_layers_matching_rotation_preds = all_layers_rotation_preds
            all_layers_matching_sizes_preds = all_layers_sizes_preds

        return (all_layers_matching_cls_scores, all_layers_matching_bbox_preds,
                all_layers_matching_centers_2d_preds, all_layers_matching_z_preds,
                all_layers_matching_rotation_preds, all_layers_matching_sizes_preds,
                all_layers_denoising_cls_scores, all_layers_denoising_bbox_preds,
                all_layers_denoising_centers_2d_preds, all_layers_denoising_z_preds,
                all_layers_denoising_rotation_preds, all_layers_denoising_sizes_preds)
