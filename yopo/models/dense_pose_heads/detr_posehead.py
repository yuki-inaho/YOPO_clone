# Copyright (c) OpenMMLab. All rights reserved.
from typing import Dict, List, Tuple

import torch
import torch.nn as nn
import torch.nn.functional as F
from mmcv.cnn import Linear
from mmcv.cnn.bricks.transformer import FFN
from mmengine.model import BaseModule
from mmengine.structures import InstanceData
from torch import Tensor

from yopo.registry import MODELS, TASK_UTILS
from yopo.structures import SampleList
from yopo.structures.bbox import (bbox_cxcywh_to_xyxy, bbox_overlaps,
                                   bbox_xyxy_to_cxcywh)
from yopo.utils import (ConfigType, InstanceList, OptInstanceList,
                         OptMultiConfig, reduce_mean)
from ..losses import QualityFocalLoss
from ..utils import multi_apply


@MODELS.register_module()
class DETRPoseHead(BaseModule):
    r"""Head of DETR. DETR:End-to-End Object Detection with Transformers.

    More details can be found in the `paper
    <https://arxiv.org/pdf/2005.12872>`_ .

    Args:
        num_classes (int): Number of categories excluding the background.
        embed_dims (int): The dims of Transformer embedding.
        num_reg_fcs (int): Number of fully-connected layers used in `FFN`,
            which is then used for the regression head. Defaults to 2.
        sync_cls_avg_factor (bool): Whether to sync the `avg_factor` of
            all ranks. Default to `False`.
        loss_cls (:obj:`ConfigDict` or dict): Config of the classification
            loss. Defaults to `CrossEntropyLoss`.
        loss_bbox (:obj:`ConfigDict` or dict): Config of the regression bbox
            loss. Defaults to `L1Loss`.
        loss_iou (:obj:`ConfigDict` or dict): Config of the regression iou
            loss. Defaults to `GIoULoss`.
        train_cfg (:obj:`ConfigDict` or dict): Training config of transformer
            head.
        test_cfg (:obj:`ConfigDict` or dict): Testing config of transformer
            head.
        init_cfg (:obj:`ConfigDict` or dict, optional): the config to control
            the initialization. Defaults to None.
    """

    _version = 2

    def __init__(
            self,
            num_classes: int,
            embed_dims: int = 256,
            num_reg_fcs: int = 2,
            sync_cls_avg_factor: bool = False,
            loss_cls: ConfigType = dict(
                type='CrossEntropyLoss',
                bg_cls_weight=0.1,
                use_sigmoid=False,
                loss_weight=1.0,
                class_weight=1.0),
            loss_bbox: ConfigType = dict(type='L1Loss', loss_weight=5.0),
            loss_iou: ConfigType = dict(type='GIoULoss', loss_weight=2.0),
            loss_rotation: ConfigType = dict(type='L2Loss', loss_weight=5.0),
            loss_pose: ConfigType = dict(type='ADDLoss', loss_weight=0.0),
            loss_centers_2d: ConfigType = dict(type='L1Loss', loss_weight=5.0),
            loss_z : ConfigType = dict(type='L2Loss', loss_weight=5.0),
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
            init_cfg: OptMultiConfig = None) -> None:
        super().__init__(init_cfg=init_cfg)
        self.bg_cls_weight = 0
        self.sync_cls_avg_factor = sync_cls_avg_factor
        class_weight = loss_cls.get('class_weight', None)
        if class_weight is not None and (self.__class__ is DETRPoseHead):
            assert isinstance(class_weight, float), 'Expected ' \
                'class_weight to have type float. Found ' \
                f'{type(class_weight)}.'
            # NOTE following the official DETR repo, bg_cls_weight means
            # relative classification weight of the no-object class.
            bg_cls_weight = loss_cls.get('bg_cls_weight', class_weight)
            assert isinstance(bg_cls_weight, float), 'Expected ' \
                'bg_cls_weight to have type float. Found ' \
                f'{type(bg_cls_weight)}.'
            class_weight = torch.ones(num_classes + 1) * class_weight
            # set background class as the last indice
            class_weight[num_classes] = bg_cls_weight
            loss_cls.update({'class_weight': class_weight})
            if 'bg_cls_weight' in loss_cls:
                loss_cls.pop('bg_cls_weight')
            self.bg_cls_weight = bg_cls_weight

        if train_cfg:
            assert 'assigner' in train_cfg, 'assigner should be provided ' \
                                            'when train_cfg is set.'
            assigner = train_cfg['assigner']
            self.assigner = TASK_UTILS.build(assigner)
            if train_cfg.get('sampler', None) is not None:
                raise RuntimeError('DETR do not build sampler.')
        self.num_classes = num_classes
        self.embed_dims = embed_dims
        self.num_reg_fcs = num_reg_fcs
        self.train_cfg = train_cfg
        self.test_cfg = test_cfg
        self.loss_cls = MODELS.build(loss_cls)
        self.loss_bbox = MODELS.build(loss_bbox)
        self.loss_iou = MODELS.build(loss_iou)

        self.loss_rotation = MODELS.build(loss_rotation)
        self.loss_pose = MODELS.build(loss_pose)
        self.loss_centers_2d = MODELS.build(loss_centers_2d)
        self.loss_z = MODELS.build(loss_z)

        if self.loss_cls.use_sigmoid:
            self.cls_out_channels = num_classes
        else:
            self.cls_out_channels = num_classes + 1

        self._init_layers()

    def _init_layers(self) -> None:
        """Initialize layers of the transformer head."""
        # cls branch
        self.fc_cls = Linear(self.embed_dims, self.cls_out_channels)
        # reg branch
        self.activate = nn.ReLU()
        self.reg_ffn = FFN(
            self.embed_dims,
            self.embed_dims,
            self.num_reg_fcs,
            dict(type='ReLU', inplace=True),
            dropout=0.0,
            add_residual=False)
        # NOTE the activations of reg_branch here is the same as
        # those in transformer, but they are actually different
        # in DAB-DETR (prelu in transformer and relu in reg_branch)
        self.fc_reg = Linear(self.embed_dims, 4)
        self.fc_R = Linear(self.embed_dims, 6)
        self.fc_centers_2d = Linear(self.embed_dims, 2)
        self.fc_z = Linear(self.embed_dims, 1)

    def forward(self, hidden_states: Tensor) -> Tuple[Tensor]:
        """"Forward function.

        Args:
            hidden_states (Tensor): Features from transformer decoder. If
                `return_intermediate_dec` in detr.py is True output has shape
                (num_decoder_layers, bs, num_queries, dim), else has shape
                (1, bs, num_queries, dim) which only contains the last layer
                outputs.
        Returns:
            tuple[Tensor]: results of head containing the following tensor.

            - layers_cls_scores (Tensor): Outputs from the classification head,
              shape (num_decoder_layers, bs, num_queries, cls_out_channels).
              Note cls_out_channels should include background.
            - layers_bbox_preds (Tensor): Sigmoid outputs from the regression
              head with normalized coordinate format (cx, cy, w, h), has shape
              (num_decoder_layers, bs, num_queries, 4).
        """
        layers_cls_scores = self.fc_cls(hidden_states)
        reg_feature = self.activate(self.reg_ffn(hidden_states))

        layers_bbox_preds = self.fc_reg(reg_feature).sigmoid()
        layers_R_preds = self.fc_R(reg_feature)
        layers_centers_2d_preds = self.fc_centers_2d(reg_feature).sigmoid()
        layers_z_preds = self.fc_z(reg_feature)

        return (layers_cls_scores, layers_bbox_preds, layers_R_preds, 
                layers_centers_2d_preds, layers_z_preds)

    def loss(self, hidden_states: Tensor,
             batch_data_samples: SampleList) -> dict:
        """Perform forward propagation and loss calculation of the detection
        head on the features of the upstream network.

        Args:
            hidden_states (Tensor): Feature from the transformer decoder, has
                shape (num_decoder_layers, bs, num_queries, cls_out_channels)
                or (num_decoder_layers, num_queries, bs, cls_out_channels).
            batch_data_samples (List[:obj:`DetDataSample`]): The Data
                Samples. It usually includes information such as
                `gt_instance`, `gt_panoptic_seg` and `gt_sem_seg`.

        Returns:
            dict: A dictionary of loss components.
        """
        batch_gt_instances = []
        batch_img_metas = []
        for data_sample in batch_data_samples:
            batch_img_metas.append(data_sample.metainfo)
            batch_gt_instances.append(data_sample.gt_instances)

        outs = self(hidden_states)
        loss_inputs = outs + (batch_gt_instances, batch_img_metas)
        losses = self.loss_by_feat(*loss_inputs)
        return losses

    def loss_by_feat(
        self,
        all_layers_cls_scores: Tensor,
        all_layers_bbox_preds: Tensor,
        all_layers_R_preds: Tensor,
        all_layers_centers_2d_preds: Tensor,
        all_layers_z_preds: Tensor,
        batch_gt_instances: InstanceList,
        batch_img_metas: List[dict],
        batch_gt_instances_ignore: OptInstanceList = None
    ) -> Dict[str, Tensor]:
        """"Loss function.

        Only outputs from the last feature level are used for computing
        losses by default.

        Args:
            all_layers_cls_scores (Tensor): Classification outputs
                of each decoder layers. Each is a 4D-tensor, has shape
                (num_decoder_layers, bs, num_queries, cls_out_channels).
            all_layers_bbox_preds (Tensor): Sigmoid regression
                outputs of each decoder layers. Each is a 4D-tensor with
                normalized coordinate format (cx, cy, w, h) and shape
                (num_decoder_layers, bs, num_queries, 4).
            all_layers_R_preds (Tensor): Direct regression
                outputs of each decoder layers. Each is a 6D-tensor with
                [r1, r2] that are first two columns of rotation matrix
                and shape (num_decoder_layers, bs, num_queries, 6).
            all_layers_centers_2d_preds (Tensor): Sigmoid regression
                outputs of each decoder layers. Each is a 4D-tensor with
                normalized coordinate format (cx, cy) and shape
                (num_decoder_layers, bs, num_queries, 2).
            all_layers_z_preds (Tensor): Direct regression
                outputs of each decoder layers. Each is a 1D-tensor with
                the directly predicted z coordinate and shape
                (num_decoder_layers, bs, num_queries, 1).
            batch_gt_instances (list[:obj:`InstanceData`]): Batch of
                gt_instance. It usually includes ``bboxes`` and ``labels``
                attributes.
            batch_img_metas (list[dict]): Meta information of each image, e.g.,
                image size, scaling factor, etc.
            batch_gt_instances_ignore (list[:obj:`InstanceData`], optional):
                Batch of gt_instances_ignore. It includes ``bboxes`` attribute
                data that is ignored during training and testing.
                Defaults to None.

        Returns:
            dict[str, Tensor]: A dictionary of loss components.
        """
        assert batch_gt_instances_ignore is None, \
            f'{self.__class__.__name__} only supports ' \
            'for batch_gt_instances_ignore setting to None.'

        losses_cls, losses_bbox, losses_iou, losses_rotation, losses_pose, losses_centers_2d, losses_z = multi_apply(
            self.loss_by_feat_single,
            all_layers_cls_scores,
            all_layers_bbox_preds,
            all_layers_R_preds,
            all_layers_centers_2d_preds,
            all_layers_z_preds,
            batch_gt_instances=batch_gt_instances,
            batch_img_metas=batch_img_metas)

        loss_dict = dict()
        # loss from the last decoder layer
        loss_dict['loss_cls'] = losses_cls[-1]
        loss_dict['loss_bbox'] = losses_bbox[-1]
        loss_dict['loss_iou'] = losses_iou[-1]

        loss_dict['loss_rotation'] = losses_rotation[-1]
        loss_dict['loss_pose'] = losses_pose[-1]
        loss_dict['loss_centers_2d'] = losses_centers_2d[-1]
        loss_dict['loss_z'] = losses_z[-1]
        # loss from other decoder layers
        num_dec_layer = 0
        for loss_cls_i, loss_bbox_i, loss_iou_i, loss_rotation_i, loss_pose_i, loss_centers_2d_i, loss_z_i in \
                zip(losses_cls[:-1], losses_bbox[:-1], losses_iou[:-1],
                    losses_rotation[:-1], losses_pose[:-1], 
                    losses_centers_2d[:-1], losses_z[:-1]):
            loss_dict[f'd{num_dec_layer}.loss_cls'] = loss_cls_i
            loss_dict[f'd{num_dec_layer}.loss_bbox'] = loss_bbox_i
            loss_dict[f'd{num_dec_layer}.loss_iou'] = loss_iou_i

            loss_dict[f'd{num_dec_layer}.loss_rotation'] = loss_rotation_i
            loss_dict[f'd{num_dec_layer}.loss_pose'] = loss_pose_i
            loss_dict[f'd{num_dec_layer}.loss_centers_2d'] = loss_centers_2d_i
            loss_dict[f'd{num_dec_layer}.loss_z'] = loss_z_i
            num_dec_layer += 1
        return loss_dict

    def loss_by_feat_single(self, cls_scores: Tensor, bbox_preds: Tensor,
                            R_preds: Tensor, centers_2d_preds: Tensor,
                            z_preds: Tensor,
                            batch_gt_instances: InstanceList,
                            batch_img_metas: List[dict]) -> Tuple[Tensor]:
        """Loss function for outputs from a single decoder layer of a single
        feature level.

        Args:
            cls_scores (Tensor): Box score logits from a single decoder layer
                for all images, has shape (bs, num_queries, cls_out_channels).
            bbox_preds (Tensor): Sigmoid outputs from a single decoder layer
                for all images, with normalized coordinate (cx, cy, w, h) and
                shape (bs, num_queries, 4).
            R_preds (Tensor): Direct outputs from a single decoder layer
                for all images, with normalized coordinate (cx, cy, w, h) and
                shape (bs, num_queries, 6).
            centers_2d_preds (Tensor): Sigmoid outputs from a single decoder layer
                for all images, with normalized coordinate (cx, cy) and
                shape (bs, num_queries, 2).
            z_preds (Tensor): Direct outputs from a single decoder layer
                for all images, with normalized coordinate (z) and
                shape (bs, num_queries, 1).
            batch_gt_instances (list[:obj:`InstanceData`]): Batch of
                gt_instance. It usually includes ``bboxes`` and ``labels``
                attributes.
            batch_img_metas (list[dict]): Meta information of each image, e.g.,
                image size, scaling factor, etc.

        Returns:
            Tuple[Tensor]: A tuple including `loss_cls`, `loss_box` and
            `loss_iou`.
        """
        num_imgs = cls_scores.size(0)
        cls_scores_list = [cls_scores[i] for i in range(num_imgs)]
        bbox_preds_list = [bbox_preds[i] for i in range(num_imgs)]

        R_preds_list = [R_preds[i] for i in range(num_imgs)]
        centers_2d_preds_list = [centers_2d_preds[i] for i in range(num_imgs)]
        z_preds_list = [z_preds[i] for i in range(num_imgs)]

        cls_reg_targets = self.get_targets(cls_scores_list, bbox_preds_list,
                                           R_preds_list, centers_2d_preds_list,
                                           z_preds_list,
                                           batch_gt_instances, batch_img_metas)
        (labels_list, label_weights_list,
         bbox_targets_list, bbox_weights_list,
         R_targets_list, R_weights_list,
         centers_2d_targets_list, centers_2d_weights_list,
         z_targets_list, z_weights_list,
         num_total_pos, num_total_neg) = cls_reg_targets

        labels = torch.cat(labels_list, 0)
        label_weights = torch.cat(label_weights_list, 0)
        bbox_targets = torch.cat(bbox_targets_list, 0)
        bbox_weights = torch.cat(bbox_weights_list, 0)

        R_targets = torch.cat(R_targets_list, 0)
        R_weights = torch.cat(R_weights_list, 0)
        centers_2d_targets = torch.cat(centers_2d_targets_list, 0)
        centers_2d_weights = torch.cat(centers_2d_weights_list, 0)
        z_targets = torch.cat(z_targets_list, 0)
        z_weights = torch.cat(z_weights_list, 0)

        # classification loss
        cls_scores = cls_scores.reshape(-1, self.cls_out_channels)
        # construct weighted avg_factor to match with the official DETR repo
        cls_avg_factor = num_total_pos * 1.0 + \
            num_total_neg * self.bg_cls_weight
        if self.sync_cls_avg_factor:
            cls_avg_factor = reduce_mean(
                cls_scores.new_tensor([cls_avg_factor]))
        cls_avg_factor = max(cls_avg_factor, 1)

        if isinstance(self.loss_cls, QualityFocalLoss):
            raise NotImplementedError(
                'QualityFocalLoss for DETRPoseHead is not supported yet.')
            bg_class_ind = self.num_classes
            pos_inds = ((labels >= 0)
                        & (labels < bg_class_ind)).nonzero().squeeze(1)
            scores = label_weights.new_zeros(labels.shape)
            pos_bbox_targets = bbox_targets[pos_inds]
            pos_decode_bbox_targets = bbox_cxcywh_to_xyxy(pos_bbox_targets)
            pos_bbox_pred = bbox_preds.reshape(-1, 4)[pos_inds]
            pos_decode_bbox_pred = bbox_cxcywh_to_xyxy(pos_bbox_pred)
            scores[pos_inds] = bbox_overlaps(
                pos_decode_bbox_pred.detach(),
                pos_decode_bbox_targets,
                is_aligned=True)
            loss_cls = self.loss_cls(
                cls_scores, (labels, scores),
                label_weights,
                avg_factor=cls_avg_factor)
        else:
            loss_cls = self.loss_cls(
                cls_scores, labels, label_weights, avg_factor=cls_avg_factor)

        # Compute the average number of gt boxes across all gpus, for
        # normalization purposes
        num_total_pos = loss_cls.new_tensor([num_total_pos])
        num_total_pos = torch.clamp(reduce_mean(num_total_pos), min=1).item()

        # construct factors used for rescale bboxes
        factors = []
        for img_meta, bbox_pred in zip(batch_img_metas, bbox_preds):
            img_h, img_w, = img_meta['img_shape']
            factor = bbox_pred.new_tensor([img_w, img_h, img_w,
                                           img_h]).unsqueeze(0).repeat(
                                               bbox_pred.size(0), 1)
            factors.append(factor)
        factors = torch.cat(factors, 0)

        # DETR regress the relative position of boxes (cxcywh) in the image,
        # thus the learning target is normalized by the image size. So here
        # we need to re-scale them for calculating IoU loss
        bbox_preds = bbox_preds.reshape(-1, 4)
        bboxes = bbox_cxcywh_to_xyxy(bbox_preds) * factors
        bboxes_gt = bbox_cxcywh_to_xyxy(bbox_targets) * factors

        # regression IoU loss, defaultly GIoU loss
        loss_iou = self.loss_iou(
            bboxes, bboxes_gt, bbox_weights, avg_factor=num_total_pos)

        # regression L1 loss
        loss_bbox = self.loss_bbox(
            bbox_preds, bbox_targets, bbox_weights, avg_factor=num_total_pos)
        
        # regression centers_2d loss
        centers_2d_preds = centers_2d_preds.reshape(-1, 2)
        # centers_2d_preds = centers_2d_preds * factors[:, :2]
        # centers_2d_targets = centers_2d_targets * factors[:, :2]
        loss_centers_2d = self.loss_centers_2d(
            centers_2d_preds, centers_2d_targets, centers_2d_weights,
            avg_factor=num_total_pos)
        # regression z loss
        z_preds = z_preds.reshape(-1, 1)
        loss_z = self.loss_z(
            z_preds, z_targets, z_weights,
            avg_factor=num_total_pos)

        # regression pose loss
        R_preds = R_preds.reshape(-1, 6)

        loss_rotation = self.loss_rotation(
            R_preds, R_targets, R_weights,
            avg_factor=num_total_pos)

        loss_pose = self.loss_pose(
            R_preds, R_targets, R_weights, 
            centers_2d_pred=centers_2d_preds,
            centers_2d_target=centers_2d_targets,
            z_pred=z_preds,
            z_target=z_targets,
            batch_img_metas=batch_img_metas,
            labels=labels,
            avg_factor=num_total_pos)

        return loss_cls, loss_bbox, loss_iou, loss_rotation, loss_pose, loss_centers_2d, loss_z

    def get_targets(self, cls_scores_list: List[Tensor],
                    bbox_preds_list: List[Tensor],
                    R_preds_list: List[Tensor],
                    centers_2d_preds_list: List[Tensor],
                    z_preds_list: List[Tensor],
                    batch_gt_instances: InstanceList,
                    batch_img_metas: List[dict]) -> tuple:
        """Compute regression and classification targets for a batch image.

        Outputs from a single decoder layer of a single feature level are used.

        Args:
            cls_scores_list (list[Tensor]): Box score logits from a single
                decoder layer for each image, has shape [num_queries,
                cls_out_channels].
            bbox_preds_list (list[Tensor]): Sigmoid outputs from a single
                decoder layer for each image, with normalized coordinate
                (cx, cy, w, h) and shape [num_queries, 4].
            R_preds_list (list[Tensor]): Direct outputs from a single
                decoder layer for each image, with normalized coordinate
                (cx, cy, w, h) and shape [num_queries, 6].
            centers_2d_preds_list (list[Tensor]): Sigmoid outputs from a single
                decoder layer for each image, with normalized coordinate
                (cx, cy) and shape [num_queries, 2].
            z_preds_list (list[Tensor]): Direct outputs from a single
                decoder layer for each image, with normalized coordinate
                (z) and shape [num_queries, 1].
            batch_gt_instances (list[:obj:`InstanceData`]): Batch of
                gt_instance. It usually includes ``bboxes`` and ``labels``
                attributes.
            batch_img_metas (list[dict]): Meta information of each image, e.g.,
                image size, scaling factor, etc.

        Returns:
            tuple: a tuple containing the following targets.

            - labels_list (list[Tensor]): Labels for all images.
            - label_weights_list (list[Tensor]): Label weights for all images.
            - bbox_targets_list (list[Tensor]): BBox targets for all images.
            - bbox_weights_list (list[Tensor]): BBox weights for all images.
            - num_total_pos (int): Number of positive samples in all images.
            - num_total_neg (int): Number of negative samples in all images.
        """
        (labels_list, label_weights_list, bbox_targets_list, bbox_weights_list,
            pose_targets_list, pose_weights_list,
            centers_2d_targets_list, centers_2d_weights_list,
            z_targets_list, z_weights_list,
            pos_inds_list, neg_inds_list) = multi_apply(self._get_targets_single,
                                      cls_scores_list, bbox_preds_list,
                                        R_preds_list, centers_2d_preds_list,
                                        z_preds_list, 
                                      batch_gt_instances, batch_img_metas)
        num_total_pos = sum((inds.numel() for inds in pos_inds_list))
        num_total_neg = sum((inds.numel() for inds in neg_inds_list))
        return (labels_list, label_weights_list, bbox_targets_list,
                bbox_weights_list, pose_targets_list, pose_weights_list,
                centers_2d_targets_list, centers_2d_weights_list,
                z_targets_list, z_weights_list, num_total_pos, num_total_neg)

    def _get_targets_single(self, cls_score: Tensor, bbox_pred: Tensor,
                            R_pred: Tensor, centers_2d_pred: Tensor,
                            z_pred: Tensor,
                            gt_instances: InstanceData,
                            img_meta: dict) -> tuple:
        """Compute regression and classification targets for one image.

        Outputs from a single decoder layer of a single feature level are used.

        Args:
            cls_score (Tensor): Box score logits from a single decoder layer
                for one image. Shape [num_queries, cls_out_channels].
            bbox_pred (Tensor): Sigmoid outputs from a single decoder layer
                for one image, with normalized coordinate (cx, cy, w, h) and
                shape [num_queries, 4].
            R_pred (Tensor): Direct outputs from a single decoder layer
                for one image, with normalized coordinate (cx, cy, w, h) and
                shape [num_queries, 6].
            centers_2d_pred (Tensor): Sigmoid outputs from a single decoder
                layer for one image, with normalized coordinate (cx, cy) and
                shape [num_queries, 2].
            z_pred (Tensor): Direct outputs from a single decoder layer
                for one image, with normalized coordinate (z) and
                shape [num_queries, 1].
            gt_instances (:obj:`InstanceData`): Ground truth of instance
                annotations. It should includes ``bboxes`` and ``labels``
                attributes.
            img_meta (dict): Meta information for one image.

        Returns:
            tuple[Tensor]: a tuple containing the following for one image.

            - labels (Tensor): Labels of each image.
            - label_weights (Tensor]): Label weights of each image.
            - bbox_targets (Tensor): BBox targets of each image.
            - bbox_weights (Tensor): BBox weights of each image.
            - pose_targets (Tensor): Pose targets of each image.
            - pose_weights (Tensor): Pose weights of each image.
            - centers_2d_targets (Tensor): Center 2D targets of each image.
            - centers_2d_weights (Tensor): Center 2D weights of each image.
            - z_targets (Tensor): Z targets of each image.
            - z_weights (Tensor): Z weights of each image.
            - pos_inds (Tensor): Sampled positive indices for each image.
            - neg_inds (Tensor): Sampled negative indices for each image.
        """
        img_h, img_w = img_meta['img_shape']
        factor = bbox_pred.new_tensor([img_w, img_h, img_w,
                                       img_h]).unsqueeze(0)
        num_bboxes = bbox_pred.size(0)
        # convert bbox_pred from xywh, normalized to xyxy, unnormalized
        bbox_pred = bbox_cxcywh_to_xyxy(bbox_pred)
        bbox_pred = bbox_pred * factor

        pred_instances = InstanceData(scores=cls_score, bboxes=bbox_pred,
                                      rotations=R_pred,
                                      centers_2d=centers_2d_pred,
                                      z=z_pred)
        # assigner and sampler
        assign_result = self.assigner.assign(
            pred_instances=pred_instances,
            gt_instances=gt_instances,
            img_meta=img_meta)

        gt_bboxes = gt_instances.bboxes
        gt_labels = gt_instances.labels
        gt_R = gt_instances.rotations
        gt_centers_2d = gt_instances.centers_2d
        gt_z = gt_instances.z


        pos_inds = torch.nonzero(
            assign_result.gt_inds > 0, as_tuple=False).squeeze(-1).unique()
        neg_inds = torch.nonzero(
            assign_result.gt_inds == 0, as_tuple=False).squeeze(-1).unique()
        pos_assigned_gt_inds = assign_result.gt_inds[pos_inds] - 1
        pos_gt_bboxes = gt_bboxes[pos_assigned_gt_inds.long(), :]

        # label targets
        labels = gt_bboxes.new_full((num_bboxes, ),
                                    self.num_classes,
                                    dtype=torch.long)
        labels[pos_inds] = gt_labels[pos_assigned_gt_inds]
        label_weights = gt_bboxes.new_ones(num_bboxes)

        # bbox targets
        bbox_targets = torch.zeros_like(bbox_pred, dtype=gt_bboxes.dtype)
        bbox_weights = torch.zeros_like(bbox_pred, dtype=gt_bboxes.dtype)
        bbox_weights[pos_inds] = 1.0

        # pose targets
        R_targets = torch.zeros_like(R_pred, dtype=gt_bboxes.dtype)
        R_weights = torch.zeros_like(R_pred, dtype=gt_bboxes.dtype)
        R_weights[pos_inds] = 1.0
        R_targets[pos_inds] = gt_R[pos_assigned_gt_inds.long(), :]
        # centers_2d targets
        centers_2d_targets = torch.zeros_like(centers_2d_pred,
            dtype=gt_bboxes.dtype)
        centers_2d_weights = torch.zeros_like(centers_2d_pred,
            dtype=gt_bboxes.dtype)
        centers_2d_weights[pos_inds] = 1.0
        gt_centers_2d = gt_centers_2d / factor[:, :2]
        centers_2d_targets[pos_inds] = gt_centers_2d[
            pos_assigned_gt_inds.long(), :]
        # z targets
        z_targets = torch.zeros_like(z_pred, dtype=gt_bboxes.dtype)
        z_weights = torch.zeros_like(z_pred, dtype=gt_bboxes.dtype)
        z_weights[pos_inds] = 1.0
        z_targets[pos_inds] = gt_z[pos_assigned_gt_inds.long(), :]

        # DETR regress the relative position of boxes (cxcywh) in the image.
        # Thus the learning target should be normalized by the image size, also
        # the box format should be converted from defaultly x1y1x2y2 to cxcywh.
        pos_gt_bboxes_normalized = pos_gt_bboxes / factor
        pos_gt_bboxes_targets = bbox_xyxy_to_cxcywh(pos_gt_bboxes_normalized)
        bbox_targets[pos_inds] = pos_gt_bboxes_targets
        return (labels, label_weights, bbox_targets, bbox_weights,
                R_targets, R_weights, centers_2d_targets, centers_2d_weights,
                z_targets, z_weights, pos_inds, neg_inds)

    def loss_and_predict(
            self, hidden_states: Tuple[Tensor],
            batch_data_samples: SampleList) -> Tuple[dict, InstanceList]:
        """Perform forward propagation of the head, then calculate loss and
        predictions from the features and data samples. Over-write because
        img_metas are needed as inputs for bbox_head.

        Args:
            hidden_states (tuple[Tensor]): Feature from the transformer
                decoder, has shape (num_decoder_layers, bs, num_queries, dim).
            batch_data_samples (list[:obj:`DetDataSample`]): Each item contains
                the meta information of each image and corresponding
                annotations.

        Returns:
            tuple: the return value is a tuple contains:

            - losses: (dict[str, Tensor]): A dictionary of loss components.
            - predictions (list[:obj:`InstanceData`]): Detection
              results of each image after the post process.
        """
        batch_gt_instances = []
        batch_img_metas = []
        for data_sample in batch_data_samples:
            batch_img_metas.append(data_sample.metainfo)
            batch_gt_instances.append(data_sample.gt_instances)

        outs = self(hidden_states)
        loss_inputs = outs + (batch_gt_instances, batch_img_metas)
        losses = self.loss_by_feat(*loss_inputs)

        predictions = self.predict_by_feat(
            *outs, batch_img_metas=batch_img_metas)
        return losses, predictions

    def predict(self,
                hidden_states: Tuple[Tensor],
                batch_data_samples: SampleList,
                rescale: bool = True) -> InstanceList:
        """Perform forward propagation of the detection head and predict
        detection results on the features of the upstream network. Over-write
        because img_metas are needed as inputs for bbox_head.

        Args:
            hidden_states (tuple[Tensor]): Multi-level features from the
                upstream network, each is a 4D-tensor.
            batch_data_samples (List[:obj:`DetDataSample`]): The Data
                Samples. It usually includes information such as
                `gt_instance`, `gt_panoptic_seg` and `gt_sem_seg`.
            rescale (bool, optional): Whether to rescale the results.
                Defaults to True.

        Returns:
            list[obj:`InstanceData`]: Detection results of each image
            after the post process.
        """
        batch_img_metas = [
            data_samples.metainfo for data_samples in batch_data_samples
        ]

        last_layer_hidden_state = hidden_states[-1].unsqueeze(0)
        outs = self(last_layer_hidden_state)

        predictions = self.predict_by_feat(
            *outs, batch_img_metas=batch_img_metas, rescale=rescale)

        return predictions

    def predict_by_feat(self,
                        layer_cls_scores: Tensor,
                        layer_bbox_preds: Tensor,
                        batch_img_metas: List[dict],
                        rescale: bool = True) -> InstanceList:
        """Transform network outputs for a batch into bbox predictions.

        Args:
            layer_cls_scores (Tensor): Classification outputs of the last or
                all decoder layer. Each is a 4D-tensor, has shape
                (num_decoder_layers, bs, num_queries, cls_out_channels).
            layer_bbox_preds (Tensor): Sigmoid regression outputs of the last
                or all decoder layer. Each is a 4D-tensor with normalized
                coordinate format (cx, cy, w, h) and shape
                (num_decoder_layers, bs, num_queries, 4).
            batch_img_metas (list[dict]): Meta information of each image.
            rescale (bool, optional): If `True`, return boxes in original
                image space. Defaults to `True`.

        Returns:
            list[:obj:`InstanceData`]: Object detection results of each image
            after the post process. Each item usually contains following keys.

                - scores (Tensor): Classification scores, has a shape
                  (num_instance, )
                - labels (Tensor): Labels of bboxes, has a shape
                  (num_instances, ).
                - bboxes (Tensor): Has a shape (num_instances, 4),
                  the last dimension 4 arrange as (x1, y1, x2, y2).
        """
        # NOTE only using outputs from the last feature level,
        # and only the outputs from the last decoder layer is used.
        cls_scores = layer_cls_scores[-1]
        bbox_preds = layer_bbox_preds[-1]

        result_list = []
        for img_id in range(len(batch_img_metas)):
            cls_score = cls_scores[img_id]
            bbox_pred = bbox_preds[img_id]
            img_meta = batch_img_metas[img_id]
            results = self._predict_by_feat_single(cls_score, bbox_pred,
                                                   img_meta, rescale)
            result_list.append(results)
        return result_list

    def _predict_by_feat_single(self,
                                cls_score: Tensor,
                                bbox_pred: Tensor,
                                R_pred: Tensor,
                                centers_2d_pred: Tensor,
                                z_pred: Tensor,
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
            R_pred (Tensor): Direct outputs from the last decoder layer
                for each image, with coordinate format (cx, cy, w, h) and
                shape [num_queries, 6].
            centers_2d_pred (Tensor): Sigmoid outputs from the last decoder
                layer for each image, with coordinate format (cx, cy) and
                shape [num_queries, 2].
            z_pred (Tensor): Direct outputs from the last decoder layer
                for each image, with coordinate format (z) and
                shape [num_queries, 1].
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

        R_pred = R_pred[bbox_index]
        centers_2d_pred = centers_2d_pred[bbox_index]
        z_pred = z_pred[bbox_index]

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
        # r1 = R_pred[:, [0, 2, 4]]
        # r2 = R_pred[:, [1, 3, 5]]
        r1, r2 = torch.split(R_pred, 3, dim=1)
        r1 = r1 / torch.norm(r1, dim=1, keepdim=True).clamp_min(1e-6)
        r2 = r2 - torch.sum(r1 * r2, dim=1, keepdim=True) * r1
        r2 = r2 / torch.norm(r2, dim=1, keepdim=True).clamp_min(1e-6)
        r3 = torch.cross(r1, r2, dim=1)
        R = torch.stack([r1, r2, r3], dim=-1)

        intrinsic = img_meta['intrinsic']
        centers_2d_h = torch.cat([centers_2d_pred, 
                                  torch.ones_like(centers_2d_pred[:, :1])],
                                  dim=1)
        if not isinstance(intrinsic, torch.Tensor):
            intrinsic = torch.tensor(intrinsic).to(centers_2d_h.device)
        intrinsic = intrinsic.view(3, 3)
        # depth = z_pred * 1800 # TODO: edit this as variable
        # depth = z_pred * (1740.27 - 141.37) + 141.37
        depth = torch.exp(z_pred)

        t_recovered = depth * (torch.inverse(intrinsic) @ centers_2d_h.T).T

        # generate 3x4 transformation matrix
        T = torch.zeros(det_bboxes.shape[0], 3, 4).to(det_bboxes.device)
        T[:, 0:3, 0:3] = R
        T[:, 0:3, 3] = t_recovered

        results = InstanceData()
        results.bboxes = det_bboxes
        results.scores = scores
        results.labels = det_labels
        results.T = T
        results.rotations = R_pred
        results.centers_2d = centers_2d_pred
        results.z = z_pred
    
        return results
