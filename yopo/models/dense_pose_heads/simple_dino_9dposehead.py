# Copyright (c) OpenMMLab. All rights reserved.
import copy
from typing import Dict, List, Tuple

import torch.nn as nn
from mmcv.cnn import Linear
from mmcv.cnn.bricks.transformer import FFN
import torch
from mmengine.structures import InstanceData
from mmengine.model import BaseModule, bias_init_with_prob, constant_init
from torch import Tensor

from yopo.registry import MODELS, TASK_UTILS
from yopo.structures import SampleList
from yopo.structures.bbox import (bbox_cxcywh_to_xyxy, bbox_overlaps,
                                   bbox_xyxy_to_cxcywh)
from yopo.utils import (InstanceList, OptInstanceList, reduce_mean,
    ConfigType, OptMultiConfig)
from ..layers import inverse_sigmoid
from ..losses import QualityFocalLoss
from ..utils import multi_apply
from ..dense_heads import DeformableDETRHead


@MODELS.register_module()
class SimpleDINO9DPoseHead(DeformableDETRHead):
    r"""Head of DINO with 9D pose estimation for End-to-End Object Detection.

    This head extends DINO (DETR with Improved DeNoising Anchor Boxes) to predict
    9D pose parameters including 3D translation, 6D rotation representation, and 
    3D object dimensions alongside standard 2D bounding boxes and object classes.

    The 9D pose representation consists of:
        - 3D translation: (tx, ty, tz) in camera coordinates
        - 6D rotation: First two columns of rotation matrix (r1, r2)
        - 3D dimensions: (width, height, depth) of the object

    Code is modified from the `official DINO repository
    <https://github.com/IDEA-Research/DINO>`_.

    More details about DINO can be found in the `paper
    <https://arxiv.org/abs/2203.03605>`_.

    Args:
        num_classes (int): Number of object classes.
        embed_dims (int): Embedding dimensions. Defaults to 256.
        num_reg_fcs (int): Number of fully connected layers in regression branches. 
            Defaults to 2.
        sync_cls_avg_factor (bool): Whether to synchronize classification average 
            factor across GPUs. Defaults to False.
        share_pred_layer (bool): Whether to share prediction layers across decoder 
            layers. Defaults to False.
        num_pred_layer (int): Number of prediction layers. Defaults to 6.
        as_two_stage (bool): Whether to use two-stage training. Defaults to False.
        loss_cls (ConfigType): Configuration for classification loss. 
            Defaults to CrossEntropyLoss.
        loss_bbox (ConfigType): Configuration for bounding box regression loss. 
            Defaults to L1Loss.
        loss_iou (ConfigType): Configuration for IoU loss. Defaults to GIoULoss.
        loss_translation (ConfigType): Configuration for 3D translation loss. 
            Defaults to PoseL2Loss.
        loss_rotation (ConfigType): Configuration for rotation loss. 
            Defaults to PoseL2Loss.
        loss_sizes (ConfigType): Configuration for 3D size loss. 
            Defaults to PoseL2Loss.
        train_cfg (ConfigType): Training configuration including assigner settings.
        test_cfg (ConfigType): Testing configuration. Defaults to dict(max_per_img=100).
        init_cfg (OptMultiConfig, optional): Initialization configuration. 
            Defaults to None.
    """
    def __init__(
            self,
            num_classes: int,
            embed_dims: int = 256,
            num_reg_fcs: int = 2,
            sync_cls_avg_factor: bool = False,
            share_pred_layer: bool = False,
            num_pred_layer: int = 6,
            as_two_stage: bool = False,
            loss_cls: ConfigType = dict(
                type='CrossEntropyLoss',
                bg_cls_weight=0.1,
                use_sigmoid=False,
                loss_weight=1.0,
                class_weight=1.0),
            loss_bbox: ConfigType = dict(type='L1Loss', loss_weight=5.0),
            loss_iou: ConfigType = dict(type='GIoULoss', loss_weight=2.0),
            loss_translation : ConfigType = dict(type='L2PoseLoss', loss_weight=5.0),
            loss_rotation: ConfigType = dict(type='L2PoseLoss', loss_weight=5.0),
            loss_sizes: ConfigType = dict(type='L2PoseLoss', loss_weight=5.0),
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
        # super().__init__(init_cfg=init_cfg)
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
        self.num_classes = num_classes
        self.embed_dims = embed_dims
        self.num_reg_fcs = num_reg_fcs
        self.train_cfg = train_cfg
        self.test_cfg = test_cfg
        self.loss_cls = MODELS.build(loss_cls)
        self.loss_bbox = MODELS.build(loss_bbox)
        self.loss_iou = MODELS.build(loss_iou)

        self.loss_translation = MODELS.build(loss_translation)
        self.loss_rotation = MODELS.build(loss_rotation)
        self.loss_sizes = MODELS.build(loss_sizes)

        if self.loss_cls.use_sigmoid:
            self.cls_out_channels = num_classes
        else:
            self.cls_out_channels = num_classes + 1
        
        self.share_pred_layer = share_pred_layer
        self.num_pred_layer = num_pred_layer
        self.as_two_stage = as_two_stage

        self._init_layers()

    def replicate(self, layer, num_layers):
        """Replicate a layer using shared instances or deep copies based on self.share_pred_layer."""
        if self.share_pred_layer:
            return nn.ModuleList([layer] * num_layers)
        else:
            return nn.ModuleList([copy.deepcopy(layer) for _ in range(num_layers)])

    def _init_layers(self) -> None:
        """Initialize classification branch and pose regression branches.
        
        Creates separate regression branches for:
        - Classification: Object class prediction
        - Bounding box: 2D box coordinates (cx, cy, w, h)
        - Translation: 3D translation (tx, ty, tz) 
        - Rotation: 6D rotation representation (first two columns of rotation matrix)
        - Sizes: 3D object dimensions (width, height, depth)
        """
        # maintain the same interface as Deformable DETR
        # to utilize the pre-trained model
        fc_cls = Linear(self.embed_dims, self.cls_out_channels)
        self.cls_branches = self.replicate(fc_cls, self.num_pred_layer)

        reg_branch = []
        for _ in range(self.num_reg_fcs):
            reg_branch.append(Linear(self.embed_dims, self.embed_dims))
            reg_branch.append(nn.ReLU())
        reg_branch.append(Linear(self.embed_dims, 4))
        reg_branch = nn.Sequential(*reg_branch)

        self.reg_branches = self.replicate(reg_branch, self.num_pred_layer)

        all_branches = []
        for i in range(3): # translation, rotation, sizes
            reg_branch = []
            for _ in range(self.num_reg_fcs):
                # set the input dim of the first layer
                embed_dims = self.embed_dims 
                reg_branch.append(Linear(embed_dims, embed_dims))
                reg_branch.append(nn.ReLU())
            if i == 0: # translation
                reg_branch.append(Linear(embed_dims, 3))
            elif i == 1: # rotation
                reg_branch.append(Linear(embed_dims, 6))
            elif i == 2: # sizes
                reg_branch.append(Linear(embed_dims, 3))
            reg_branch = nn.Sequential(*reg_branch)
            all_branches.append(reg_branch)

        self.reg_translation_branch = self.replicate(all_branches[0], self.num_pred_layer)
        self.reg_rotation_branch = self.replicate(all_branches[1], self.num_pred_layer)
        self.reg_size_branch = self.replicate(all_branches[2], self.num_pred_layer)

    def init_weights(self) -> None:
        """Initialize weights of the DINO 9D pose head."""
        if self.loss_cls.use_sigmoid:
            bias_init = bias_init_with_prob(0.01)
            for m in self.cls_branches:
                if hasattr(m, 'bias') and m.bias is not None:
                    nn.init.constant_(m.bias, bias_init)
        # Initialize bbox regression branches
        for m in self.reg_branches:
            constant_init(m[-1], 0, bias=0)
        nn.init.constant_(self.reg_branches[0][-1].bias.data[2:], -2.0)
        if self.as_two_stage:
            for m in self.reg_branches:
                nn.init.constant_(m[-1].bias.data[2:], 0.0)
        
        # Initialize translation regression branches
        for m in self.reg_translation_branch:
            constant_init(m[-1], 0, bias=0)
        # nn.init.constant_(self.reg_translation_branch[0][-1].bias.data, 0.0)
            
        # Initialize rotation regression branches  
        for m in self.reg_rotation_branch:
            constant_init(m[-1], 0, bias=0)
        # nn.init.constant_(self.reg_rotation_branch[0][-1].bias.data, 0.0)
            
        # Initialize size regression branches
        for m in self.reg_size_branch:
            constant_init(m[-1], 0, bias=0)
        # nn.init.constant_(self.reg_size_branch[0][-1].bias.data, 0.2)

    def forward(self, hidden_states: Tensor,
                references: List[Tensor]) -> Tuple[Tensor, Tensor, Tensor, Tensor, Tensor]:
        """Forward function for 9D pose estimation.

        Args:
            hidden_states (Tensor): Hidden states output from each decoder
                layer, has shape (num_decoder_layers, bs, num_queries, dim).
            references (list[Tensor]): List of the reference from the decoder.
                The first reference is the `init_reference` (initial) and the
                other num_decoder_layers(6) references are `inter_references`
                (intermediate). The `init_reference` has shape (bs,
                num_queries, 4) when `as_two_stage` of the detector is `True`,
                otherwise (bs, num_queries, 2). Each `inter_reference` has
                shape (bs, num_queries, 4) when `with_box_refine` of the
                detector is `True`, otherwise (bs, num_queries, 2). The
                coordinates are arranged as (cx, cy) when the last dimension is
                2, and (cx, cy, w, h) when it is 4.

        Returns:
            tuple[Tensor]: A tuple containing the following tensors:

            - all_layers_outputs_classes (Tensor): Classification outputs from 
              all decoder layers, has shape (num_decoder_layers, bs, num_queries, 
              cls_out_channels).
            - all_layers_outputs_coords (Tensor): Sigmoid outputs from the
              bbox regression head with normalized coordinate format (cx, cy, w,
              h), has shape (num_decoder_layers, bs, num_queries, 4).
            - all_layers_outputs_translations (Tensor): Direct regression outputs
              from the translation head with 3D coordinates (tx, ty, tz) and shape
              (num_decoder_layers, bs, num_queries, 3).
            - all_layers_outputs_rotations (Tensor): Direct regression outputs 
              from the rotation head. Each is a 6D-tensor with [r1, r2] that are 
              the first two columns of rotation matrix and shape 
              (num_decoder_layers, bs, num_queries, 6).
            - all_layers_outputs_sizes (Tensor): Direct regression outputs from 
              the size head with 3D dimensions (width, height, depth) and shape
              (num_decoder_layers, bs, num_queries, 3).
        """
        all_layers_outputs_classes = []
        all_layers_outputs_coords = []
        all_layers_outputs_translations = []
        all_layers_outputs_rotations = []
        all_layers_outputs_sizes = []

        for layer_id in range(hidden_states.shape[0]):
            reference = inverse_sigmoid(references[layer_id])
            # NOTE The last reference will not be used.
            hidden_state = hidden_states[layer_id]
            outputs_class = self.cls_branches[layer_id](hidden_state)
            tmp_reg_bbox_preds = self.reg_branches[layer_id](hidden_state)

            tmp_translation_preds = self.reg_translation_branch[layer_id](hidden_state)
            tmp_rotation_preds = self.reg_rotation_branch[layer_id](hidden_state)
            tmp_sizes = self.reg_size_branch[layer_id](hidden_state)

            if reference.shape[-1] == 4:
                # When `layer` is 0 and `as_two_stage` of the detector
                # is `True`, or when `layer` is greater than 0 and
                # `with_box_refine` of the detector is `True`.
                # tmp_reg_preds += reference
                tmp_reg_bbox_preds += reference

            else:
                # When `layer` is 0 and `as_two_stage` of the detector
                # is `False`, or when `layer` is greater than 0 and
                # `with_box_refine` of the detector is `False`.
                assert reference.shape[-1] == 2
                # tmp_reg_preds[..., :2] += reference
                tmp_reg_bbox_preds[..., :2] += reference
            outputs_coord = tmp_reg_bbox_preds.sigmoid()

            all_layers_outputs_classes.append(outputs_class)
            all_layers_outputs_coords.append(outputs_coord)
            all_layers_outputs_translations.append(tmp_translation_preds)
            all_layers_outputs_rotations.append(tmp_rotation_preds)
            all_layers_outputs_sizes.append(tmp_sizes)

        all_layers_outputs_classes = torch.stack(all_layers_outputs_classes)
        all_layers_outputs_coords = torch.stack(all_layers_outputs_coords)
        all_layers_outputs_translations = torch.stack(all_layers_outputs_translations)
        all_layers_outputs_rotations = torch.stack(all_layers_outputs_rotations)
        all_layers_outputs_sizes = torch.stack(all_layers_outputs_sizes)


        return (all_layers_outputs_classes, all_layers_outputs_coords,
                all_layers_outputs_translations, all_layers_outputs_rotations,
                all_layers_outputs_sizes)

    def loss(self, hidden_states: Tensor, references: List[Tensor],
             enc_outputs_class: Tensor, enc_outputs_coord: Tensor,
             enc_outputs_translation: Tensor,
             enc_outputs_rotation: Tensor,
             enc_outputs_size: Tensor,
             batch_data_samples: SampleList, dn_meta: Dict[str, int]) -> dict:
        """Perform forward propagation and loss calculation of the detection
        head on the queries of the upstream network.

        Args:
            hidden_states (Tensor): Hidden states output from each decoder
                layer, has shape (num_decoder_layers, bs, num_queries_total,
                dim), where `num_queries_total` is the sum of
                `num_denoising_queries` and `num_matching_queries` when
                `self.training` is `True`, else `num_matching_queries`.
            references (list[Tensor]): List of the reference from the decoder.
                The first reference is the `init_reference` (initial) and the
                other num_decoder_layers(6) references are `inter_references`
                (intermediate). The `init_reference` has shape (bs,
                num_queries_total, 4) and each `inter_reference` has shape
                (bs, num_queries, 4) with the last dimension arranged as
                (cx, cy, w, h). When `as_two_stage` is `True`, the first
                `num_queries` of `inter_reference` are used for reference
                initialization, and the remaining queries are ignored.
            enc_outputs_class (Tensor): The score of each point on encode
                feature map, has shape (bs, num_feat_points, cls_out_channels).
            enc_outputs_coord (Tensor): The proposal generate from the
                encode feature map, has shape (bs, num_feat_points, 4) with the
                last dimension arranged as (cx, cy, w, h).
            enc_outputs_R (Tensor): Direct regression
                outputs of each decoder layers. Each is a 6D-tensor with
                [r1, r2] that are first two columns of rotation matrix
                and shape (bs, num_feat_points, 6).
            enc_outputs_centers_2d (Tensor): Sigmoid regression
                outputs of each decoder layers. Each is a 4D-tensor with
                normalized coordinate format (cx, cy) and shape
                (bs, num_feat_points, 2).
            enc_outputs_z (Tensor): Direct regression outputs of each decoder
                layers. Each is a 1D-tensor with the directly predicted z
                coordinate and shape (bs, num_feat_points, 1).
            batch_data_samples (list[:obj:`DetDataSample`]): The Data
                Samples. It usually includes information such as
                `gt_instance`, `gt_panoptic_seg` and `gt_sem_seg`.
            dn_meta (Dict[str, int]): The dictionary saves information about
              group collation, including 'num_denoising_queries' and
              'num_denoising_groups'. It will be used for split outputs of
              denoising and matching parts and loss calculation.

        Returns:
            dict: A dictionary of loss components.
        """
        batch_gt_instances = []
        batch_img_metas = []
        for data_sample in batch_data_samples:
            batch_img_metas.append(data_sample.metainfo)
            batch_gt_instances.append(data_sample.gt_instances)

        outs = self(hidden_states, references)
        loss_inputs = outs + (enc_outputs_class, enc_outputs_coord,
                              enc_outputs_translation, enc_outputs_rotation, enc_outputs_size,
                              batch_gt_instances, batch_img_metas, dn_meta)
        losses = self.loss_by_feat(*loss_inputs)
        return losses

    def loss_by_feat(
        self,
        all_layers_cls_scores: Tensor,
        all_layers_bbox_preds: Tensor,
        all_layers_translation_preds: Tensor,
        all_layers_rotation_preds: Tensor,
        all_layers_sizes_preds: Tensor,
        enc_cls_scores: Tensor,
        enc_bbox_preds: Tensor,
        enc_outputs_translation: Tensor,
        enc_outputs_rotation: Tensor,
        enc_outputs_size: Tensor,
        batch_gt_instances: InstanceList,
        batch_img_metas: List[dict],
        dn_meta: Dict[str, int],
        batch_gt_instances_ignore: OptInstanceList = None
    ) -> Dict[str, Tensor]:
        """Loss function.

        Args:
            all_layers_cls_scores (Tensor): Classification scores of all
                decoder layers, has shape (num_decoder_layers, bs,
                num_queries_total, cls_out_channels), where
                `num_queries_total` is the sum of `num_denoising_queries`
                and `num_matching_queries`.
            all_layers_bbox_preds (Tensor): Regression outputs of all decoder
                layers. Each is a 4D-tensor with normalized coordinate format
                (cx, cy, w, h) and has shape (num_decoder_layers, bs,
                num_queries_total, 4).
            all_layers_translation_preds (Tensor): Direct regression
                outputs of each decoder layers. Each is a 3D-tensor with
                the directly predicted translation coordinate and shape
                (num_decoder_layers, bs, num_queries_total, 3).
            all_layers_rotation_preds (Tensor): Direct regression
                outputs of each decoder layers. Each is a 6D-tensor with
                [r1, r2] that are first two columns of rotation matrix
                and shape (num_decoder_layers, bs, num_queries_total, 6).
            all_layers_sizes_preds (Tensor): Direct regression
                outputs of each decoder layers. Each is a 3D-tensor with
                the directly predicted size coordinate and shape
                (num_decoder_layers, bs, num_queries_total, 3).
            enc_cls_scores (Tensor): The score of each point on encode
                feature map, has shape (bs, num_feat_points, cls_out_channels).
            enc_bbox_preds (Tensor): The proposal generate from the encode
                feature map, has shape (bs, num_feat_points, 4) with the last
                dimension arranged as (cx, cy, w, h).
            enc_outputs_translation (Tensor): The translation proposal
                generate from the encode feature map, has shape
                (bs, num_feat_points, 3).
            enc_outputs_rotation (Tensor): The rotation proposal
                generate from the encode feature map, has shape
                (bs, num_feat_points, 6).
            enc_outputs_size (Tensor): The size proposal
                generate from the encode feature map, has shape
                (bs, num_feat_points, 3).
            batch_gt_instances (list[:obj:`InstanceData`]): Batch of
                gt_instance. It usually includes ``bboxes`` and ``labels``
                attributes.
            batch_img_metas (list[dict]): Meta information of each image, e.g.,
                image size, scaling factor, etc.
            dn_meta (Dict[str, int]): The dictionary saves information about
                group collation, including 'num_denoising_queries' and
                'num_denoising_groups'. It will be used for split outputs of
                denoising and matching parts and loss calculation.
            batch_gt_instances_ignore (list[:obj:`InstanceData`], optional):
                Batch of gt_instances_ignore. It includes ``bboxes`` attribute
                data that is ignored during training and testing.
                Defaults to None.

        Returns:
            dict[str, Tensor]: A dictionary of loss components.
        """
        # extract denoising and matching part of outputs
        (all_layers_matching_cls_scores, all_layers_matching_bbox_preds,
         all_layers_matching_translation_preds,
         all_layers_matching_rotation_preds, all_layers_matching_sizes_preds,
            all_layers_denoising_cls_scores, all_layers_denoising_bbox_preds,
            all_layers_denoising_translation_preds, all_layers_denoising_rotation_preds,
            all_layers_denoising_sizes_preds) = \
            self.split_outputs(
                all_layers_cls_scores, all_layers_bbox_preds, 
                all_layers_translation_preds, all_layers_rotation_preds,
                all_layers_sizes_preds,
                dn_meta)

        loss_dict = self.loss_by_feat_simple(
            all_layers_matching_cls_scores, all_layers_matching_bbox_preds,
            all_layers_matching_translation_preds, all_layers_matching_rotation_preds,
            all_layers_matching_sizes_preds,
            batch_gt_instances, batch_img_metas, batch_gt_instances_ignore)
        # NOTE DETRHead.loss_by_feat but not DeformableDETRHead.loss_by_feat
        # is called, because the encoder loss calculations are different
        # between DINO and DeformableDETR.

        # loss of proposal generated from encode feature map.
        if enc_cls_scores is not None:
            # NOTE The enc_loss calculation of the DINO is
            # different from that of Deformable DETR.
            (enc_loss_cls, enc_losses_bbox, enc_losses_iou, 
             enc_loss_translation, enc_loss_rotation,
             enc_loss_size) = \
                self.loss_by_feat_single(
                    enc_cls_scores, enc_bbox_preds,
                    enc_outputs_translation, enc_outputs_rotation,
                    enc_outputs_size,
                    batch_gt_instances=batch_gt_instances,
                    batch_img_metas=batch_img_metas)
            loss_dict['enc_loss_cls'] = enc_loss_cls
            loss_dict['enc_loss_bbox'] = enc_losses_bbox
            loss_dict['enc_loss_iou'] = enc_losses_iou
            loss_dict['enc_loss_translation'] = enc_loss_translation
            loss_dict['enc_loss_rotation'] = enc_loss_rotation
            loss_dict['enc_loss_size'] = enc_loss_size

        if all_layers_denoising_cls_scores is not None:
            # calculate denoising loss from all decoder layers
            (dn_losses_cls, dn_losses_bbox, dn_losses_iou,
             dn_losses_translation, dn_losses_rotation, dn_losses_sizes) = \
            self.loss_dn(
                all_layers_denoising_cls_scores,
                all_layers_denoising_bbox_preds,
                all_layers_denoising_translation_preds,
                all_layers_denoising_rotation_preds,
                all_layers_denoising_sizes_preds,
                batch_gt_instances=batch_gt_instances,
                batch_img_metas=batch_img_metas,
                dn_meta=dn_meta)
            # collate denoising loss
            loss_dict['dn_loss_cls'] = dn_losses_cls[-1]
            loss_dict['dn_loss_bbox'] = dn_losses_bbox[-1]
            loss_dict['dn_loss_iou'] = dn_losses_iou[-1]
            loss_dict['dn_loss_translation'] = dn_losses_translation[-1]
            loss_dict['dn_loss_rotation'] = dn_losses_rotation[-1]
            loss_dict['dn_loss_size'] = dn_losses_sizes[-1]

            for num_dec_layer, (loss_cls_i, loss_bbox_i, loss_iou_i, 
                                loss_translation_i, loss_rotation_i, loss_sizes_i) in \
                    enumerate(zip(dn_losses_cls[:-1], dn_losses_bbox[:-1],
                                  dn_losses_iou[:-1],
                                    dn_losses_translation[:-1],
                                    dn_losses_rotation[:-1],
                                    dn_losses_sizes[:-1])):
                loss_dict[f'd{num_dec_layer}.dn_loss_cls'] = loss_cls_i
                loss_dict[f'd{num_dec_layer}.dn_loss_bbox'] = loss_bbox_i
                loss_dict[f'd{num_dec_layer}.dn_loss_iou'] = loss_iou_i
                loss_dict[f'd{num_dec_layer}.dn_loss_translation'] = loss_translation_i
                loss_dict[f'd{num_dec_layer}.dn_loss_rotation'] = loss_rotation_i
                loss_dict[f'd{num_dec_layer}.dn_loss_size'] = loss_sizes_i

        return loss_dict

    def loss_by_feat_simple(
        self,
        all_layers_cls_scores: Tensor,
        all_layers_bbox_preds: Tensor,
        all_layers_translation_preds: Tensor,
        all_layers_rotation_preds: Tensor,
        all_layers_sizes_preds: Tensor,
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

        (losses_cls, losses_bbox, losses_iou, losses_translation, 
            losses_rotation, losses_sizes)= multi_apply(
            self.loss_by_feat_single,
            all_layers_cls_scores,
            all_layers_bbox_preds,
            all_layers_translation_preds,
            all_layers_rotation_preds,
            all_layers_sizes_preds,
            batch_gt_instances=batch_gt_instances,
            batch_img_metas=batch_img_metas)

        loss_dict = dict()
        # loss from the last decoder layer
        loss_dict['loss_cls'] = losses_cls[-1]
        loss_dict['loss_bbox'] = losses_bbox[-1]
        loss_dict['loss_iou'] = losses_iou[-1]
        loss_dict['loss_translation'] = losses_translation[-1]
        loss_dict['loss_rotation'] = losses_rotation[-1]
        loss_dict['loss_size'] = losses_sizes[-1]

        # loss from other decoder layers
        num_dec_layer = 0
        for loss_cls_i, loss_bbox_i, loss_iou_i, loss_translation_i, loss_rotation_i, loss_sizes_i in \
                zip(losses_cls[:-1], losses_bbox[:-1], losses_iou[:-1],
                   losses_translation[:-1],
                   losses_rotation[:-1], losses_sizes[:-1]):
            loss_dict[f'd{num_dec_layer}.loss_cls'] = loss_cls_i
            loss_dict[f'd{num_dec_layer}.loss_bbox'] = loss_bbox_i
            loss_dict[f'd{num_dec_layer}.loss_iou'] = loss_iou_i

            loss_dict[f'd{num_dec_layer}.loss_translation'] = loss_translation_i
            loss_dict[f'd{num_dec_layer}.loss_rotation'] = loss_rotation_i
            loss_dict[f'd{num_dec_layer}.loss_size'] = loss_sizes_i
            num_dec_layer += 1
        return loss_dict

    def loss_dn(self, all_layers_denoising_cls_scores: Tensor,
                all_layers_denoising_bbox_preds: Tensor,
                all_layers_denoising_translation_preds: Tensor,
                all_layers_denoising_rotation_preds: Tensor,
                all_layers_denoising_sizes_preds: Tensor,
                batch_gt_instances: InstanceList, batch_img_metas: List[dict],
                dn_meta: Dict[str, int]) -> Tuple[List[Tensor]]:
        """Calculate denoising loss.

        Args:
            all_layers_denoising_cls_scores (Tensor): Classification scores of
                all decoder layers in denoising part, has shape (
                num_decoder_layers, bs, num_denoising_queries,
                cls_out_channels).
            all_layers_denoising_bbox_preds (Tensor): Regression outputs of all
                decoder layers in denoising part. Each is a 4D-tensor with
                normalized coordinate format (cx, cy, w, h) and has shape
                (num_decoder_layers, bs, num_denoising_queries, 4).
            all_layers_denoising_R_preds (Tensor): Direct regression
                outputs of each decoder layers in denoising part. Each is a
                6D-tensor with [r1, r2] that are first two columns of rotation
                matrix and shape (num_decoder_layers, bs,
                num_denoising_queries, 6).
            all_layers_denoising_centers_2d_preds (Tensor): Sigmoid regression
                outputs of each decoder layers in denoising part. Each is a
                2D-tensor with normalized coordinate format (cx, cy) and has
                shape (num_decoder_layers, bs, num_denoising_queries, 2).
            all_layers_denoising_z_preds (Tensor): Direct regression 
                outputs of each decoder layers in denoising part. Each is a
                1D-tensor with the directly predicted z coordinate and shape
                (num_decoder_layers, bs, num_denoising_queries, 1).
            batch_gt_instances (list[:obj:`InstanceData`]): Batch of
                gt_instance. It usually includes ``bboxes`` and ``labels``
                attributes.
            batch_img_metas (list[dict]): Meta information of each image, e.g.,
                image size, scaling factor, etc.
            dn_meta (Dict[str, int]): The dictionary saves information about
              group collation, including 'num_denoising_queries' and
              'num_denoising_groups'. It will be used for split outputs of
              denoising and matching parts and loss calculation.

        Returns:
            Tuple[List[Tensor]]: The loss_dn_cls, loss_dn_bbox, and loss_dn_iou
            of each decoder layers.
        """
        return multi_apply(
            self._loss_dn_single,
            all_layers_denoising_cls_scores,
            all_layers_denoising_bbox_preds,
            all_layers_denoising_translation_preds,
            all_layers_denoising_rotation_preds,
            all_layers_denoising_sizes_preds,
            batch_gt_instances=batch_gt_instances,
            batch_img_metas=batch_img_metas,
            dn_meta=dn_meta)

    def _loss_dn_single(self, dn_cls_scores: Tensor, dn_bbox_preds: Tensor,
                        dn_translation_preds: Tensor,
                        dn_rotation_preds: Tensor,
                        dn_sizes_preds: Tensor,
                        batch_gt_instances: InstanceList,
                        batch_img_metas: List[dict],
                        dn_meta: Dict[str, int]) -> Tuple[Tensor]:
        """Denoising loss for outputs from a single decoder layer.

        Args:
            dn_cls_scores (Tensor): Classification scores of a single decoder
                layer in denoising part, has shape (bs, num_denoising_queries,
                cls_out_channels).
            dn_bbox_preds (Tensor): Regression outputs of a single decoder
                layer in denoising part. Each is a 4D-tensor with normalized
                coordinate format (cx, cy, w, h) and has shape
                (bs, num_denoising_queries, 4).
            dn_R_preds (Tensor): Direct regression outputs of a single decoder
                layer in denoising part. Each is a 6D-tensor with [r1, r2] that
                are first two columns of rotation matrix and shape
                (bs, num_denoising_queries, 6).
            dn_centers_2d_preds (Tensor): Sigmoid regression outputs of a
                single decoder layer in denoising part. Each is a 2D-tensor
                with normalized coordinate format (cx, cy) and has shape
                (bs, num_denoising_queries, 2).
            dn_z_preds (Tensor): Direct regression outputs of a single decoder
                layer in denoising part. Each is a 1D-tensor with the directly
                predicted z coordinate and shape (bs, num_denoising_queries, 1).
            batch_gt_instances (list[:obj:`InstanceData`]): Batch of
                gt_instance. It usually includes ``bboxes`` and ``labels``
                attributes.
            batch_img_metas (list[dict]): Meta information of each image, e.g.,
                image size, scaling factor, etc.
            dn_meta (Dict[str, int]): The dictionary saves information about
              group collation, including 'num_denoising_queries' and
              'num_denoising_groups'. It will be used for split outputs of
              denoising and matching parts and loss calculation.

        Returns:
            Tuple[Tensor]: A tuple including `loss_cls`, `loss_box` and
            `loss_iou`.
        """
        cls_reg_targets = self.get_dn_targets(batch_gt_instances,
                                              batch_img_metas, dn_meta)
        (labels_list, label_weights_list, bbox_targets_list, bbox_weights_list,
            dn_translation_targets_list, dn_translation_weights_list,
            dn_rotation_targets_list, dn_rotation_weights_list,
            dn_sizes_targets_list, dn_sizes_weights_list,
            num_total_pos, num_total_neg) = cls_reg_targets
        labels = torch.cat(labels_list, 0)
        label_weights = torch.cat(label_weights_list, 0)
        bbox_targets = torch.cat(bbox_targets_list, 0)
        bbox_weights = torch.cat(bbox_weights_list, 0)

        translation_targets = torch.cat(dn_translation_targets_list, 0)
        translation_weights = torch.cat(dn_translation_weights_list, 0)
        rotation_targets = torch.cat(dn_rotation_targets_list, 0)
        rotation_weights = torch.cat(dn_rotation_weights_list, 0)
        sizes_targets = torch.cat(dn_sizes_targets_list, 0)
        sizes_weights = torch.cat(dn_sizes_weights_list, 0)

        # classification loss
        cls_scores = dn_cls_scores.reshape(-1, self.cls_out_channels)
        # construct weighted avg_factor to match with the official DETR repo
        cls_avg_factor = \
            num_total_pos * 1.0 + num_total_neg * self.bg_cls_weight
        if self.sync_cls_avg_factor:
            cls_avg_factor = reduce_mean(
                cls_scores.new_tensor([cls_avg_factor]))
        cls_avg_factor = max(cls_avg_factor, 1)

        if len(cls_scores) > 0:
            if isinstance(self.loss_cls, QualityFocalLoss):
                raise NotImplementedError(
                    'QualityFocalLoss for DETRPoseHead is not supported yet.')
                bg_class_ind = self.num_classes
                pos_inds = ((labels >= 0)
                            & (labels < bg_class_ind)).nonzero().squeeze(1)
                scores = label_weights.new_zeros(labels.shape)
                pos_bbox_targets = bbox_targets[pos_inds]
                pos_decode_bbox_targets = bbox_cxcywh_to_xyxy(pos_bbox_targets)
                pos_bbox_pred = dn_bbox_preds.reshape(-1, 4)[pos_inds]
                pos_decode_bbox_pred = bbox_cxcywh_to_xyxy(pos_bbox_pred)
                scores[pos_inds] = bbox_overlaps(
                    pos_decode_bbox_pred.detach(),
                    pos_decode_bbox_targets,
                    is_aligned=True)
                loss_cls = self.loss_cls(
                    cls_scores, (labels, scores),
                    weight=label_weights,
                    avg_factor=cls_avg_factor)
            else:
                loss_cls = self.loss_cls(
                    cls_scores,
                    labels,
                    label_weights,
                    avg_factor=cls_avg_factor)
        else:
            loss_cls = torch.zeros(
                1, dtype=cls_scores.dtype, device=cls_scores.device)

        # Compute the average number of gt boxes across all gpus, for
        # normalization purposes
        num_total_pos = loss_cls.new_tensor([num_total_pos])
        num_total_pos = torch.clamp(reduce_mean(num_total_pos), min=1).item()

        # construct factors used for rescale bboxes
        factors = []
        for img_meta, bbox_pred in zip(batch_img_metas, dn_bbox_preds):
            img_h, img_w = img_meta['img_shape']
            factor = bbox_pred.new_tensor([img_w, img_h, img_w,
                                           img_h]).unsqueeze(0).repeat(
                                               bbox_pred.size(0), 1)
            factors.append(factor)
        factors = torch.cat(factors)

        # DETR regress the relative position of boxes (cxcywh) in the image,
        # thus the learning target is normalized by the image size. So here
        # we need to re-scale them for calculating IoU loss
        bbox_preds = dn_bbox_preds.reshape(-1, 4)
        bboxes = bbox_cxcywh_to_xyxy(bbox_preds) * factors
        bboxes_gt = bbox_cxcywh_to_xyxy(bbox_targets) * factors

        # regression IoU loss, defaultly GIoU loss
        loss_iou = self.loss_iou(
            bboxes, bboxes_gt, bbox_weights, avg_factor=num_total_pos)

        # regression L1 loss
        loss_bbox = self.loss_bbox(
            bbox_preds, bbox_targets, bbox_weights, avg_factor=num_total_pos)
        
        # translation loss
        translation_preds = dn_translation_preds.reshape(-1, 3)
        loss_translation = self.loss_translation(
            translation_preds, translation_targets, translation_weights,
            avg_factor=num_total_pos)

        # rotation loss
        rotation_preds = dn_rotation_preds.reshape(-1, 6)
        loss_rotation = self.loss_rotation(
            rotation_preds, rotation_targets, rotation_weights,
            avg_factor=num_total_pos)

        # sizes loss
        sizes_preds = dn_sizes_preds.reshape(-1, 3)
        loss_sizes = self.loss_sizes(
            sizes_preds, sizes_targets, sizes_weights,
            avg_factor=num_total_pos)

        # return losses

        return loss_cls, loss_bbox, loss_iou, loss_translation, loss_rotation, loss_sizes


    def get_dn_targets(self, batch_gt_instances: InstanceList,
                       batch_img_metas: dict, dn_meta: Dict[str,
                                                            int]) -> tuple:
        """Get targets in denoising part for a batch of images.

        Args:
            batch_gt_instances (list[:obj:`InstanceData`]): Batch of
                gt_instance. It usually includes ``bboxes`` and ``labels``
                attributes.
            batch_img_metas (list[dict]): Meta information of each image, e.g.,
                image size, scaling factor, etc.
            dn_meta (Dict[str, int]): The dictionary saves information about
              group collation, including 'num_denoising_queries' and
              'num_denoising_groups'. It will be used for split outputs of
              denoising and matching parts and loss calculation.

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
         translation_targets_list, translation_weights_list,
         rotation_targets_list, rotation_weights_list,
         sizes_targets_list, sizes_weights_list,
         pos_inds_list, neg_inds_list) = multi_apply(
             self._get_dn_targets_single,
             batch_gt_instances,
             batch_img_metas,
             dn_meta=dn_meta)
        num_total_pos = sum((inds.numel() for inds in pos_inds_list))
        num_total_neg = sum((inds.numel() for inds in neg_inds_list))
        return (labels_list, label_weights_list, bbox_targets_list,
                bbox_weights_list,
                translation_targets_list, translation_weights_list,
                rotation_targets_list, rotation_weights_list,
                sizes_targets_list, sizes_weights_list,
                num_total_pos, num_total_neg)

    def _get_dn_targets_single(self, gt_instances: InstanceData,
                               img_meta: dict, dn_meta: Dict[str,
                                                             int]) -> tuple:
        """Get targets in denoising part for one image.

        Args:
            gt_instances (:obj:`InstanceData`): Ground truth of instance
                annotations. It should includes ``bboxes`` and ``labels``
                attributes.
            img_meta (dict): Meta information for one image.
            dn_meta (Dict[str, int]): The dictionary saves information about
              group collation, including 'num_denoising_queries' and
              'num_denoising_groups'. It will be used for split outputs of
              denoising and matching parts and loss calculation.

        Returns:
            tuple[Tensor]: a tuple containing the following for one image.

            - labels (Tensor): Labels of each image.
            - label_weights (Tensor]): Label weights of each image.
            - bbox_targets (Tensor): BBox targets of each image.
            - bbox_weights (Tensor): BBox weights of each image.
            - pos_inds (Tensor): Sampled positive indices for each image.
            - neg_inds (Tensor): Sampled negative indices for each image.
        """
        # DETR regress the relative position of boxes (cxcywh) in the image.
        # Thus the learning target should be normalized by the image size, also
        # the box format should be converted from defaultly x1y1x2y2 to cxcywh.
        gt_bboxes = gt_instances.bboxes
        gt_labels = gt_instances.labels
        gt_translations = gt_instances.translations
        gt_rotations = gt_instances.rotations
        gt_sizes = gt_instances.sizes

        img_h, img_w = img_meta['img_shape']
        factor = gt_bboxes.new_tensor([img_w, img_h, img_w,
                                       img_h]).unsqueeze(0)

        num_groups = dn_meta['num_denoising_groups']
        num_denoising_queries = dn_meta['num_denoising_queries']
        num_queries_each_group = int(num_denoising_queries / num_groups)
        device = gt_bboxes.device

        if len(gt_labels) > 0:
            t = torch.arange(len(gt_labels), dtype=torch.long, device=device)
            t = t.unsqueeze(0).repeat(num_groups, 1)
            pos_assigned_gt_inds = t.flatten()
            pos_inds = torch.arange(
                num_groups, dtype=torch.long, device=device)
            pos_inds = pos_inds.unsqueeze(1) * num_queries_each_group + t
            pos_inds = pos_inds.flatten()
        else:
            pos_inds = pos_assigned_gt_inds = \
                gt_bboxes.new_tensor([], dtype=torch.long)

        neg_inds = pos_inds + num_queries_each_group // 2

        # label targets
        labels = gt_bboxes.new_full((num_denoising_queries, ),
                                    self.num_classes,
                                    dtype=torch.long)
        labels[pos_inds] = gt_labels[pos_assigned_gt_inds]
        label_weights = gt_bboxes.new_ones(num_denoising_queries)

        # bbox targets
        bbox_targets = torch.zeros(num_denoising_queries, 4, device=device)
        bbox_weights = torch.zeros(num_denoising_queries, 4, device=device)
        bbox_weights[pos_inds] = 1.0

        # translation targets
        translation_targets = torch.zeros(num_denoising_queries, 3,
                                          device=device)
        translation_weights = torch.zeros(num_denoising_queries, 3,
                                          device=device)
        translation_weights[pos_inds] = 1.0

        # rotation targets
        rotation_targets = torch.zeros(num_denoising_queries, 6,
                                       device=device)
        rotation_weights = torch.zeros(num_denoising_queries, 6,
                                       device=device)
        rotation_weights[pos_inds] = 1.0

        # sizes targets
        sizes_targets = torch.zeros(num_denoising_queries, 3, device=device)
        sizes_weights = torch.zeros(num_denoising_queries, 3, device=device)
        sizes_weights[pos_inds] = 1.0

        gt_bboxes_normalized = gt_bboxes / factor
        gt_bboxes_targets = bbox_xyxy_to_cxcywh(gt_bboxes_normalized)
        bbox_targets[pos_inds] = gt_bboxes_targets.repeat([num_groups, 1])

        return (labels, label_weights, bbox_targets, bbox_weights,
                translation_targets, translation_weights,
                rotation_targets, rotation_weights,
                sizes_targets, sizes_weights,
                pos_inds, neg_inds)

    @staticmethod
    def split_outputs(all_layers_cls_scores: Tensor,
                      all_layers_bbox_preds: Tensor,
                      all_layers_translation_preds: Tensor,
                      all_layers_rotation_preds: Tensor,
                      all_layers_sizes_preds: Tensor,
                      dn_meta: Dict[str, int]) -> Tuple[Tensor]:
        """Split outputs of the denoising part and the matching part.

        For the total outputs of `num_queries_total` length, the former
        `num_denoising_queries` outputs are from denoising queries, and
        the rest `num_matching_queries` ones are from matching queries,
        where `num_queries_total` is the sum of `num_denoising_queries` and
        `num_matching_queries`.

        Args:
            all_layers_cls_scores (Tensor): Classification scores of all
                decoder layers, has shape (num_decoder_layers, bs,
                num_queries_total, cls_out_channels).
            all_layers_bbox_preds (Tensor): Regression outputs of all decoder
                layers. Each is a 4D-tensor with normalized coordinate format
                (cx, cy, w, h) and has shape (num_decoder_layers, bs,
                num_queries_total, 4).
            all_layers_R_preds (Tensor): Direct regression
                outputs of each decoder layers. Each is a 6D-tensor with
                [r1, r2] that are first two columns of rotation matrix
                and shape (num_decoder_layers, bs, num_queries, 6).
            all_layers_centers_2d_preds (Tensor): Sigmoid regression
                outputs of each decoder layers. Each is a 2D-tensor with
                normalized coordinate format (cx, cy) and shape
                (num_decoder_layers, bs, num_queries, 2).
            all_layers_z_preds (Tensor): Direct regression outputs of each
                decoder layers. Each is a 1D-tensor with the directly
                predicted z coordinate and shape (num_decoder_layers, bs,
                num_queries, 1).
            dn_meta (Dict[str, int): The dictionary saves information about
              group collation, including 'num_denoising_queries' and
              'num_denoising_groups'.

        Returns:
            Tuple[Tensor]: a tuple containing the following outputs.

            - all_layers_matching_cls_scores (Tensor): Classification scores
              of all decoder layers in matching part, has shape
              (num_decoder_layers, bs, num_matching_queries, cls_out_channels).
            - all_layers_matching_bbox_preds (Tensor): Regression outputs of
              all decoder layers in matching part. Each is a 4D-tensor with
              normalized coordinate format (cx, cy, w, h) and has shape
              (num_decoder_layers, bs, num_matching_queries, 4).
            - all_layers_denoising_cls_scores (Tensor): Classification scores
              of all decoder layers in denoising part, has shape
              (num_decoder_layers, bs, num_denoising_queries,
              cls_out_channels).
            - all_layers_denoising_bbox_preds (Tensor): Regression outputs of
              all decoder layers in denoising part. Each is a 4D-tensor with
              normalized coordinate format (cx, cy, w, h) and has shape
              (num_decoder_layers, bs, num_denoising_queries, 4).
        """
        num_denoising_queries = dn_meta['num_denoising_queries']
        if dn_meta is not None:
            all_layers_denoising_cls_scores = \
                all_layers_cls_scores[:, :, : num_denoising_queries, :]
            all_layers_denoising_bbox_preds = \
                all_layers_bbox_preds[:, :, : num_denoising_queries, :]
            all_layers_denoising_translation_preds = \
                all_layers_translation_preds[:, :, : num_denoising_queries, :]
            all_layers_denoising_rotation_preds = \
                all_layers_rotation_preds[:, :, : num_denoising_queries, :]
            all_layers_denoising_sizes_preds = \
                all_layers_sizes_preds[:, :, : num_denoising_queries, :]

            all_layers_matching_cls_scores = \
                all_layers_cls_scores[:, :, num_denoising_queries:, :]
            all_layers_matching_bbox_preds = \
                all_layers_bbox_preds[:, :, num_denoising_queries:, :]
            
            all_layers_matching_translation_preds = \
                all_layers_translation_preds[:, :, num_denoising_queries:, :]
            all_layers_matching_rotation_preds = \
                all_layers_rotation_preds[:, :, num_denoising_queries:, :]
            all_layers_matching_sizes_preds = \
                all_layers_sizes_preds[:, :, num_denoising_queries:, :]
            
        else:
            all_layers_denoising_cls_scores = None
            all_layers_denoising_bbox_preds = None
            all_layers_denoising_translation_preds = None
            all_layers_denoising_rotation_preds = None
            all_layers_denoising_sizes_preds = None

            all_layers_matching_cls_scores = all_layers_cls_scores
            all_layers_matching_bbox_preds = all_layers_bbox_preds
            all_layers_matching_translation_preds = \
                all_layers_translation_preds
            all_layers_matching_rotation_preds = \
                all_layers_rotation_preds
            all_layers_matching_sizes_preds = \
                all_layers_sizes_preds

        return (all_layers_matching_cls_scores, all_layers_matching_bbox_preds,
                all_layers_matching_translation_preds,
                all_layers_matching_rotation_preds,
                all_layers_matching_sizes_preds,
                all_layers_denoising_cls_scores,
                all_layers_denoising_bbox_preds,
                all_layers_denoising_translation_preds,
                all_layers_denoising_rotation_preds,
                all_layers_denoising_sizes_preds
        )

    def get_targets(self, cls_scores_list: List[Tensor],
                    bbox_preds_list: List[Tensor],
                    translation_preds_list: List[Tensor],
                    rotation_preds_list: List[Tensor],
                    sizes_preds_list: List[Tensor],
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
            z_preds_list (list[Tensor]): Direct outputs from a single decoder
                layer for each image, with normalized coordinate
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
            translation_targets_list, translation_weights_list,
            rotation_targets_list, rotation_weights_list,
            sizes_targets_list, sizes_weights_list,
            pos_inds_list, neg_inds_list) = multi_apply(self._get_targets_single,
                                      cls_scores_list, bbox_preds_list,
                                        translation_preds_list,
                                        rotation_preds_list,
                                        sizes_preds_list,
                                      batch_gt_instances, batch_img_metas)
        num_total_pos = sum((inds.numel() for inds in pos_inds_list))
        num_total_neg = sum((inds.numel() for inds in neg_inds_list))
        return (labels_list, label_weights_list, bbox_targets_list,
                bbox_weights_list, translation_targets_list, translation_weights_list,
                rotation_targets_list, rotation_weights_list,
                sizes_targets_list, sizes_weights_list, num_total_pos, num_total_neg)



    def _get_targets_single(self, cls_score: Tensor, bbox_pred: Tensor,
                            translation_pred: Tensor,
                            rotation_pred: Tensor,
                            sizes_pred: Tensor,
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
                                      translations=translation_pred,
                                      rotations=rotation_pred,
                                      sizes=sizes_pred)
        # assigner and sampler
        assign_result = self.assigner.assign(
            pred_instances=pred_instances,
            gt_instances=gt_instances,
            img_meta=img_meta)

        gt_bboxes = gt_instances.bboxes
        gt_labels = gt_instances.labels
        gt_translations = gt_instances.translations
        gt_rotations = gt_instances.rotations
        gt_sizes = gt_instances.sizes

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

        # translation targets
        translation_targets = torch.zeros_like(translation_pred,
                                               dtype=gt_bboxes.dtype)
        translation_weights = torch.zeros_like(translation_pred,
                                               dtype=gt_bboxes.dtype)
        translation_weights[pos_inds] = 1.0
        translation_targets[pos_inds] = gt_translations[pos_assigned_gt_inds.long(), :]

        # rotation targets
        rotation_targets = torch.zeros_like(rotation_pred, dtype=gt_bboxes.dtype)
        rotation_weights = torch.zeros_like(rotation_pred, dtype=gt_bboxes.dtype)
        rotation_weights[pos_inds] = 1.0
        rotation_targets[pos_inds] = gt_rotations[pos_assigned_gt_inds.long(), :]

        # sizes targets
        sizes_targets = torch.zeros_like(sizes_pred, dtype=gt_bboxes.dtype)
        sizes_weights = torch.zeros_like(sizes_pred, dtype=gt_bboxes.dtype)
        sizes_weights[pos_inds] = 1.0
        sizes_targets[pos_inds] = gt_sizes[pos_assigned_gt_inds.long(), :]

        # DETR regress the relative position of boxes (cxcywh) in the image.
        # Thus the learning target should be normalized by the image size, also
        # the box format should be converted from defaultly x1y1x2y2 to cxcywh.
        pos_gt_bboxes_normalized = pos_gt_bboxes / factor
        pos_gt_bboxes_targets = bbox_xyxy_to_cxcywh(pos_gt_bboxes_normalized)
        bbox_targets[pos_inds] = pos_gt_bboxes_targets
        return (labels, label_weights, bbox_targets, bbox_weights,
                translation_targets, translation_weights,
                rotation_targets, rotation_weights,
                sizes_targets, sizes_weights,
                pos_inds, neg_inds)

    def loss_by_feat_single(self, cls_scores: Tensor, bbox_preds: Tensor,
                            translation_preds: Tensor,
                            rotation_preds: Tensor,
                            sizes_preds: Tensor,
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

        translation_preds_list = [translation_preds[i] for i in range(num_imgs)]
        rotation_preds_list = [rotation_preds[i] for i in range(num_imgs)]
        sizes_preds_list = [sizes_preds[i] for i in range(num_imgs)]

        cls_reg_targets = self.get_targets(cls_scores_list, bbox_preds_list,
                                           translation_preds_list, rotation_preds_list, sizes_preds_list,  
                                           batch_gt_instances, batch_img_metas)
        (labels_list, label_weights_list,
         bbox_targets_list, bbox_weights_list,
         translation_targets_list, translation_weights_list,
         rotation_targets_list, rotation_weights_list,
         sizes_targets_list, sizes_weights_list,
         num_total_pos, num_total_neg) = cls_reg_targets

        labels = torch.cat(labels_list, 0)
        label_weights = torch.cat(label_weights_list, 0)
        bbox_targets = torch.cat(bbox_targets_list, 0)
        bbox_weights = torch.cat(bbox_weights_list, 0)

        translation_targets = torch.cat(translation_targets_list, 0)
        translation_weights = torch.cat(translation_weights_list, 0)
        rotation_targets = torch.cat(rotation_targets_list, 0)
        rotation_weights = torch.cat(rotation_weights_list, 0)
        sizes_targets = torch.cat(sizes_targets_list, 0)
        sizes_weights = torch.cat(sizes_weights_list, 0)

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
        
        # translation loss
        translation_preds = translation_preds.reshape(-1, 3)
        loss_translation = self.loss_translation(
            translation_preds, translation_targets, translation_weights,
            avg_factor=num_total_pos)

        # rotation loss
        rotation_preds = rotation_preds.reshape(-1, 6)
        loss_rotation = self.loss_rotation(
            rotation_preds, rotation_targets, rotation_weights,
            avg_factor=num_total_pos)

        # sizes loss
        sizes_preds = sizes_preds.reshape(-1, 3)
        loss_sizes = self.loss_sizes(
            sizes_preds, sizes_targets, sizes_weights,
            avg_factor=num_total_pos)

        return (loss_cls, loss_bbox, loss_iou, loss_translation, loss_rotation, loss_sizes)


    def predict(self,
                hidden_states: Tensor,
                references: List[Tensor],
                batch_data_samples: SampleList,
                rescale: bool = True,
                depth_features=None) -> InstanceList:
        """Perform forward propagation and loss calculation of the detection
        head on the queries of the upstream network.

        Args:
            hidden_states (Tensor): Hidden states output from each decoder
                layer, has shape (num_decoder_layers, num_queries, bs, dim).
            references (list[Tensor]): List of the reference from the decoder.
                The first reference is the `init_reference` (initial) and the
                other num_decoder_layers(6) references are `inter_references`
                (intermediate). The `init_reference` has shape (bs,
                num_queries, 4) when `as_two_stage` of the detector is `True`,
                otherwise (bs, num_queries, 2). Each `inter_reference` has
                shape (bs, num_queries, 4) when `with_box_refine` of the
                detector is `True`, otherwise (bs, num_queries, 2). The
                coordinates are arranged as (cx, cy) when the last dimension is
                2, and (cx, cy, w, h) when it is 4.
            batch_data_samples (list[:obj:`DetDataSample`]): The Data
                Samples. It usually includes information such as
                `gt_instance`, `gt_panoptic_seg` and `gt_sem_seg`.
            rescale (bool, optional): If `True`, return boxes in original
                image space. Defaults to `True`.

        Returns:
            list[obj:`InstanceData`]: Detection results of each image
            after the post process.
        """
        batch_img_metas = [
            data_samples.metainfo for data_samples in batch_data_samples
        ]

        if depth_features is None:
            outs = self(hidden_states, references)
        else:
            # RGB-D center heads consume query-aligned pre-fusion depth
            # features while ordinary 9D heads keep the historical signature.
            outs = self(
                hidden_states, references, depth_features=depth_features)

        # Subclasses (e.g. DINO9DCenter2DPoseHead) may return auxiliary CoP
        # outputs after the first 6. The first 6 are always the selected
        # primary path: parallel outputs in parallel/auxiliary mode and CoP
        # outputs in chain mode.
        predictions = self.predict_by_feat(
            *outs[:6], batch_img_metas=batch_img_metas, rescale=rescale)
        return predictions


    def predict_by_feat(self,
                        all_layers_cls_scores: Tensor,
                        all_layers_bbox_preds: Tensor,
                        all_layers_translation_preds: Tensor,
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
            all_layers_R_preds (Tensor): Direct regression
                outputs of each decoder layers. Each is a 6D-tensor with
                [r1, r2] that are first two columns of rotation matrix
                and shape (num_decoder_layers, bs, num_queries, 6).
            all_layers_center_2d_preds (Tensor): Sigmoid regression
                outputs of each decoder layers. Each is a 4D-tensor with
                normalized coordinate format (cx, cy) and shape
                (num_decoder_layers, bs, num_queries, 2).
            all_layers_z_preds (Tensor): Direct regression
                outputs of each decoder layers. Each is a 1D-tensor with
                the directly predicted z coordinate and shape
                (num_decoder_layers, bs, num_queries, 1).
            batch_img_metas (list[dict]): Meta information of each image.
            rescale (bool, optional): If `True`, return boxes in original
                image space. Default `False`.

        Returns:
            list[obj:`InstanceData`]: Detection results of each image
            after the post process.
        """
        cls_scores = all_layers_cls_scores[-1]
        bbox_preds = all_layers_bbox_preds[-1]
        translation_preds = all_layers_translation_preds[-1]
        rotation_preds = all_layers_rotation_preds[-1]
        sizes_preds = all_layers_sizes_preds[-1]

        result_list = []
        for img_id in range(len(batch_img_metas)):
            cls_score = cls_scores[img_id]
            bbox_pred = bbox_preds[img_id]
            translation_pred = translation_preds[img_id]
            rotation_pred = rotation_preds[img_id]
            sizes_pred = sizes_preds[img_id]

            img_meta = batch_img_metas[img_id]
            results = self._predict_by_feat_single(cls_score, bbox_pred,
                                                   translation_pred, rotation_pred,
                                                   sizes_pred, img_meta, rescale)
            result_list.append(results)
        return result_list

    def _predict_by_feat_single(self,
                                cls_score: Tensor,
                                bbox_pred: Tensor,
                                translation_pred: Tensor,
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

        translation_pred = translation_pred[bbox_index]
        rotation_pred = rotation_pred[bbox_index]
        sizes_pred = sizes_pred[bbox_index]

        det_bboxes = bbox_cxcywh_to_xyxy(bbox_pred)
        det_bboxes[:, 0::2] = det_bboxes[:, 0::2] * img_shape[1]
        det_bboxes[:, 1::2] = det_bboxes[:, 1::2] * img_shape[0]
        det_bboxes[:, 0::2].clamp_(min=0, max=img_shape[1])
        det_bboxes[:, 1::2].clamp_(min=0, max=img_shape[0])

        if rescale:
            assert img_meta.get('scale_factor') is not None
            det_bboxes /= det_bboxes.new_tensor(
                img_meta['scale_factor']).repeat((1, 2))
        
        # create the transformation matrix
        # r1 = R_pred[:, [0, 2, 4]]
        # r2 = R_pred[:, [1, 3, 5]]
        r1, r2 = torch.split(rotation_pred, 3, dim=1)
        r1 = r1 / torch.norm(r1, dim=1, keepdim=True).clamp_min(1e-6)
        r2 = r2 - torch.sum(r1 * r2, dim=1, keepdim=True) * r1
        r2 = r2 / torch.norm(r2, dim=1, keepdim=True).clamp_min(1e-6)
        r3 = torch.cross(r1, r2, dim=1)
        R = torch.stack([r1, r2, r3], dim=-1)

        intrinsic = img_meta['intrinsic']

        # generate 4x4 transformation matrix R | t
        T = torch.zeros((det_bboxes.shape[0], 4, 4), dtype=det_bboxes.dtype, device=det_bboxes.device)
        T[:, :3, :3] = R
        T[:, :3, 3] = translation_pred
        T[:, 3, 3] = 1.0

        results = InstanceData()
        results.bboxes = det_bboxes
        results.scores = scores
        results.labels = det_labels
        results.T = T
        results.translations = translation_pred
        results.rotations = rotation_pred
        results.sizes = sizes_pred
    
        return results
