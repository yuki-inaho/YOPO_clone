"""Dense rotated RTMDet head for RGB-D 2D OBB learning."""

from __future__ import annotations

import math

import numpy as np
import torch
import torch.nn as nn
from mmcv.cnn import ConvModule
from mmcv.ops import nms_rotated
from mmengine.model import bias_init_with_prob, normal_init
from mmengine.structures import InstanceData

from yopo.registry import MODELS, TASK_UTILS
from yopo.structures.bbox import RotatedBoxes, get_box_tensor
from yopo.utils import reduce_mean
from ..utils import unpack_gt_instances
from .base_dense_head import BaseDenseHead


def distance_angle_to_rbox(points, prediction):
    """Decode point-relative ``ltrb+angle`` into absolute rotated boxes."""
    distances = prediction[..., :4]
    angle = prediction[..., 4:5]
    cos_angle = torch.cos(angle)
    sin_angle = torch.sin(angle)
    width_height = distances[..., :2] + distances[..., 2:]
    local_offset = (distances[..., 2:] - distances[..., :2]) * 0.5
    offset_x = (local_offset[..., 0:1] * cos_angle
                - local_offset[..., 1:2] * sin_angle)
    offset_y = (local_offset[..., 0:1] * sin_angle
                + local_offset[..., 1:2] * cos_angle)
    center = points[..., :2] + torch.cat((offset_x, offset_y), dim=-1)
    angle = torch.remainder(angle + math.pi / 2, math.pi) - math.pi / 2
    return torch.cat((center, width_height, angle), dim=-1)


@MODELS.register_module()
class RotatedRTMDetSepBNHead(BaseDenseHead):
    """Anchor-free dense RTMDet head with explicit angle regression.

    Dynamic soft-label assignment creates multiple spatial positives per GT;
    this intentionally differs from decoder one-to-one Hungarian matching.
    """

    def __init__(
        self,
        num_classes,
        in_channels,
        feat_channels=256,
        stacked_convs=2,
        strides=(8, 16, 32, 64),
        norm_cfg=dict(type='BN', momentum=0.03, eps=0.001),
        act_cfg=dict(type='SiLU'),
        loss_cls=dict(
            type='QualityFocalLoss', use_sigmoid=True, beta=2.0,
            loss_weight=1.0),
        loss_bbox=dict(
            type='SmoothL1Loss', beta=1.0 / 9.0, loss_weight=1.0),
        loss_iou=dict(
            type='RotatedIoULoss', mode='linear', loss_weight=2.0),
        train_cfg=None,
        test_cfg=None,
        init_cfg=None,
    ):
        super().__init__(init_cfg=init_cfg)
        self.num_classes = int(num_classes)
        self.in_channels = int(in_channels)
        self.feat_channels = int(feat_channels)
        self.stacked_convs = int(stacked_convs)
        self.strides = tuple(int(value) for value in strides)
        self.train_cfg = train_cfg or {}
        self.test_cfg = test_cfg or {}
        self.loss_cls = MODELS.build(loss_cls)
        self.loss_bbox = MODELS.build(loss_bbox) if loss_bbox else None
        self.loss_iou = MODELS.build(loss_iou)
        self.prior_generator = TASK_UTILS.build(dict(
            type='MlvlPointGenerator', strides=self.strides, offset=0.0))
        self.assigner = TASK_UTILS.build(self.train_cfg.get(
            'assigner',
            dict(
                type='DynamicSoftLabelAssigner',
                topk=13,
                iou_calculator=dict(type='RBboxOverlaps2D'),
            )))
        self._init_layers(norm_cfg, act_cfg)

    def _init_layers(self, norm_cfg, act_cfg):
        self.cls_convs = nn.ModuleList()
        self.reg_convs = nn.ModuleList()
        self.rtm_cls = nn.ModuleList()
        self.rtm_reg = nn.ModuleList()
        self.rtm_ang = nn.ModuleList()
        for _ in self.strides:
            cls_tower = nn.ModuleList()
            reg_tower = nn.ModuleList()
            for index in range(self.stacked_convs):
                channels = self.in_channels if index == 0 else self.feat_channels
                cls_tower.append(ConvModule(
                    channels, self.feat_channels, 3, padding=1,
                    norm_cfg=norm_cfg, act_cfg=act_cfg))
                reg_tower.append(ConvModule(
                    channels, self.feat_channels, 3, padding=1,
                    norm_cfg=norm_cfg, act_cfg=act_cfg))
            self.cls_convs.append(cls_tower)
            self.reg_convs.append(reg_tower)
            self.rtm_cls.append(nn.Conv2d(self.feat_channels, self.num_classes, 1))
            self.rtm_reg.append(nn.Conv2d(self.feat_channels, 4, 1))
            self.rtm_ang.append(nn.Conv2d(self.feat_channels, 1, 1))

    def init_weights(self):
        super().init_weights()
        cls_bias = bias_init_with_prob(0.01)
        for module in self.rtm_cls:
            normal_init(module, std=0.01, bias=cls_bias)
        for module in [*self.rtm_reg, *self.rtm_ang]:
            normal_init(module, std=0.01)

    def forward(self, feats):
        if len(feats) != len(self.strides):
            raise ValueError(
                f'expected {len(self.strides)} pyramid levels, got {len(feats)}')
        cls_scores, box_predictions = [], []
        for level, (feature, stride) in enumerate(zip(feats, self.strides)):
            cls_feature = feature
            reg_feature = feature
            for layer in self.cls_convs[level]:
                cls_feature = layer(cls_feature)
            for layer in self.reg_convs[level]:
                reg_feature = layer(reg_feature)
            cls_scores.append(self.rtm_cls[level](cls_feature))
            raw_distance = self.rtm_reg[level](reg_feature).float().clamp(max=8.0)
            distances = raw_distance.exp() * float(stride)
            angle = self.rtm_ang[level](reg_feature).float()
            box_predictions.append(torch.cat((distances, angle), dim=1))
        return tuple(cls_scores), tuple(box_predictions)

    def _flatten_and_decode(self, cls_scores, box_predictions):
        batch_size = cls_scores[0].shape[0]
        feature_shapes = [score.shape[-2:] for score in cls_scores]
        priors_by_level = self.prior_generator.grid_priors(
            feature_shapes, dtype=torch.float32, device=cls_scores[0].device,
            with_stride=True)
        flat_logits = torch.cat([
            score.permute(0, 2, 3, 1).reshape(batch_size, -1, self.num_classes)
            for score in cls_scores
        ], dim=1)
        flat_raw = torch.cat([
            prediction.permute(0, 2, 3, 1).reshape(batch_size, -1, 5)
            for prediction in box_predictions
        ], dim=1)
        priors = torch.cat(priors_by_level, dim=0).float()
        decoded = distance_angle_to_rbox(priors[None, :, :2], flat_raw)
        return flat_logits, flat_raw, decoded, priors

    def loss(self, x, batch_data_samples):
        cls_scores, box_predictions = self(x)
        batch_gt_instances, _, batch_img_metas = unpack_gt_instances(
            batch_data_samples)
        return self.loss_by_feat(
            cls_scores, box_predictions, batch_gt_instances, batch_img_metas)

    def loss_by_feat(
        self, cls_scores, box_predictions, batch_gt_instances,
        batch_img_metas, batch_gt_instances_ignore=None,
    ):
        del batch_gt_instances_ignore
        logits, _, decoded, priors = self._flatten_and_decode(
            cls_scores, box_predictions)
        all_labels, all_metrics, all_targets, all_positive = [], [], [], []
        for image_index, gt_instances in enumerate(batch_gt_instances):
            gt_instances.bboxes.regularize_boxes(pattern='le90')
            assignment = self.assigner.assign(
                InstanceData(
                    scores=logits[image_index].detach(),
                    bboxes=RotatedBoxes(decoded[image_index].detach()),
                    priors=priors,
                ),
                gt_instances,
            )
            positive = assignment.gt_inds > 0
            labels = logits.new_full(
                (logits.shape[1],), self.num_classes, dtype=torch.long)
            labels[positive] = assignment.labels[positive]
            targets = decoded.new_zeros((decoded.shape[1], 5))
            if positive.any():
                gt_indices = assignment.gt_inds[positive] - 1
                targets[positive] = get_box_tensor(gt_instances.bboxes)[gt_indices]
            metrics = assignment.max_overlaps.to(logits.dtype).clamp_min(0)
            all_labels.append(labels)
            all_metrics.append(metrics)
            all_targets.append(targets)
            all_positive.append(positive)

        labels = torch.cat(all_labels)
        metrics = torch.cat(all_metrics)
        targets = torch.cat(all_targets)
        positive = torch.cat(all_positive)
        flat_logits = logits.reshape(-1, self.num_classes)
        flat_decoded = decoded.reshape(-1, 5)
        normalizers = []
        anchors_per_image = decoded.shape[1]
        for img_meta in batch_img_metas:
            img_h, img_w = img_meta['img_shape'][:2]
            factor = flat_decoded.new_tensor(
                [img_w, img_h, img_w, img_h, math.pi])
            normalizers.append(factor[None].expand(anchors_per_image, -1))
        normalizers = torch.cat(normalizers)
        avg_factor = reduce_mean(metrics.sum()).clamp_min(1).item()
        loss_cls = self.loss_cls(
            flat_logits, (labels, metrics), avg_factor=avg_factor)
        if positive.any():
            weights = metrics[positive]
            loss_iou = self.loss_iou(
                flat_decoded[positive], targets[positive],
                weight=weights, avg_factor=avg_factor)
            if self.loss_bbox is not None:
                loss_bbox = self.loss_bbox(
                    flat_decoded[positive] / normalizers[positive],
                    targets[positive] / normalizers[positive],
                    weight=weights[:, None].expand(-1, 5),
                    avg_factor=avg_factor)
            else:
                loss_bbox = flat_decoded.sum() * 0
        else:
            loss_iou = flat_decoded.sum() * 0
            loss_bbox = flat_decoded.sum() * 0
        return dict(
            loss_cls=loss_cls,
            loss_bbox=loss_bbox,
            loss_iou=loss_iou,
            num_pos=positive.sum().to(flat_logits.dtype).detach(),
        )

    def predict(self, x, batch_data_samples, rescale=False):
        cls_scores, box_predictions = self(x)
        batch_img_metas = [sample.metainfo for sample in batch_data_samples]
        return self.predict_by_feat(
            cls_scores, box_predictions, batch_img_metas, rescale=rescale)

    def predict_by_feat(
        self, cls_scores, box_predictions, batch_img_metas,
        cfg=None, rescale=False, with_nms=True,
    ):
        logits, _, decoded, _ = self._flatten_and_decode(
            cls_scores, box_predictions)
        cfg = self.test_cfg if cfg is None else cfg
        score_thr = cfg.get('score_thr', 0.05)
        nms_pre = cfg.get('nms_pre', 2000)
        nms_iou = cfg.get('nms', {}).get('iou_threshold', 0.1)
        max_per_img = cfg.get('max_per_img', 300)
        results = []
        for image_index, img_meta in enumerate(batch_img_metas):
            scores, labels = logits[image_index].sigmoid().max(dim=-1)
            keep = scores >= score_thr
            scores, labels, boxes = (
                scores[keep], labels[keep], decoded[image_index][keep])
            if 0 < nms_pre < scores.numel():
                topk = scores.topk(nms_pre).indices
                scores, labels, boxes = scores[topk], labels[topk], boxes[topk]
            if with_nms and boxes.numel():
                kept = []
                for label in labels.unique():
                    class_indices = torch.nonzero(labels == label).squeeze(1)
                    _, class_keep = nms_rotated(
                        boxes[class_indices], scores[class_indices], nms_iou)
                    kept.append(class_indices[class_keep])
                keep = torch.cat(kept)
                keep = keep[scores[keep].argsort(descending=True)[:max_per_img]]
                scores, labels, boxes = scores[keep], labels[keep], boxes[keep]
            elif scores.numel() > max_per_img:
                keep = scores.topk(max_per_img).indices
                scores, labels, boxes = scores[keep], labels[keep], boxes[keep]
            box_container = RotatedBoxes(boxes)
            if rescale:
                scale = img_meta.get('scale_factor')
                if scale is None:
                    raise ValueError('scale_factor is required for rescaling')
                scale = np.asarray(scale).reshape(-1)
                box_container.rescale_((1.0 / float(scale[0]),
                                        1.0 / float(scale[1])))
            results.append(InstanceData(
                bboxes=box_container, scores=scores, labels=labels))
        return results
