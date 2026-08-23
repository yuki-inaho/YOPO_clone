# Copyright (c) OpenMMLab. All rights reserved.
import copy
from typing import Dict, List, Tuple

import torch
import torch.nn as nn
import torch.nn.functional as F
from mmcv.cnn import Linear
from mmcv.cnn.bricks.transformer import FFN
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
from .simple_dino_9dposehead import SimpleDINO9DPoseHead

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
            loss_bbox: ConfigType = dict(type='L1Loss', loss_weight=5.0),
            loss_iou: ConfigType = dict(type='GIoULoss', loss_weight=2.0),
            loss_centers_2d: ConfigType = dict(type='L1PoseLoss', loss_weight=5.0),
            loss_z: ConfigType = dict(type='L2PoseLoss', loss_weight=5.0),
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
        self.rot_dim = rot_dim
        self.train_cfg = train_cfg
        self.test_cfg = test_cfg

        self.loss_cls = MODELS.build(loss_cls)
        self.loss_bbox = MODELS.build(loss_bbox)
        self.loss_iou = MODELS.build(loss_iou)

        self.use_cuboid_conditioning = use_cuboid_conditioning
        self.use_intrinsinc_for_bbox = use_intrinsinc_for_bbox
        self.use_bbox_for_centers_2d = use_bbox_for_centers_2d
        self.use_bbox_for_z = use_bbox_for_z
        self.use_bbox_for_rotation = use_bbox_for_rotation
        self.use_bbox_for_size = use_bbox_for_size

        self.use_log_z = use_log_z

        self.use_cop_chain = use_cop_chain
        if use_cop_chain:
            self.cop_loss_weights = dict(
                size=3.0, rotation=2.0, z=1.0)

        self.loss_centers_2d = MODELS.build(loss_centers_2d)
        self.loss_z = MODELS.build(loss_z)

        self.loss_rotation = MODELS.build(loss_rotation)
        self.loss_sizes = MODELS.build(loss_sizes)

        if self.loss_cls.use_sigmoid:
            self.cls_out_channels = num_classes
        else:
            self.cls_out_channels = num_classes + 1
        
        self.share_pred_layer = share_pred_layer
        self.num_pred_layer = num_pred_layer
        self.as_two_stage = as_two_stage
        self.classwise_rotation = classwise_rotation
        self.classwise_sizes = classwise_sizes


        self._init_layers()

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

        self.reg_centers_2d_branch = self.replicate(all_branches[0], self.num_pred_layer)
        self.reg_z_branch = self.replicate(all_branches[1], self.num_pred_layer)
        self.reg_rotation_branch = self.replicate(all_branches[2], self.num_pred_layer)
        self.reg_size_branch = self.replicate(all_branches[3], self.num_pred_layer)

        # ── CoP (Chain-of-Prediction) auxiliary nets ──────────────────────
        # AttributeNet A(·) per attribute (MonoCoP Eq.6): two Linear layers
        # with ReLU between. The chain follows the best prediction order
        # size -> rotation -> depth, with residual aggregation (Eq.8):
        #   f_s  = A_s(q);          f̃_s = f_s + q
        #   f_a  = A_a(f̃_s);       f̃_a = f_a + f̃_s
        #   f_d  = A_d(f̃_a);       f̃_d = f_d + f̃_a
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

        if self.as_two_stage:
            for m in self.reg_branches:
                nn.init.constant_(m[-1].bias.data[2:], 0.0)
            # for m in self.reg_centers_2d_branch:
                # nn.init.constant_(m[-1].bias.data, 0.0)

    def forward(self, hidden_states: Tensor,
                references: List[Tensor],
                batch_img_metas=None) -> Tuple[Tensor, Tensor, Tensor, Tensor, Tensor, Tensor]:
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
            tmp_reg_z_preds = self.reg_z_branch[layer_id](tmp_z_input)
            tmp_rotation_preds = self.reg_rotation_branch[layer_id](tmp_rotation_input)
            tmp_sizes = self.reg_size_branch[layer_id](tmp_size_input)

            if self.use_cop_chain:
                # CoP chain: size -> rotation -> depth with residual aggregation.
                f_s = self.cop_size_net[layer_id](hidden_state)
                f_s_a = f_s + hidden_state
                f_a = self.cop_rotation_net[layer_id](f_s_a)
                f_a_a = f_a + f_s_a
                f_d = self.cop_z_net[layer_id](f_a_a)
                f_d_a = f_d + f_a_a
                # Classwise indexing is applied later in loss (same as parallel).
                tmp_sizes_chain = self.cop_size_out[layer_id](f_s_a)
                tmp_rotation_chain = self.cop_rotation_out[layer_id](f_a_a)
                tmp_z_chain = self.cop_z_out[layer_id](f_d_a)
            else:
                tmp_sizes_chain = tmp_rotation_chain = tmp_z_chain = None


            all_layers_outputs_classes.append(outputs_class)
            all_layers_outputs_coords.append(outputs_coord)
            all_layers_outputs_centers_2d.append(tmp_reg_centers_2d_preds)
            all_layers_outputs_z.append(tmp_reg_z_preds)
            all_layers_outputs_rotations.append(tmp_rotation_preds)
            all_layers_outputs_sizes.append(tmp_sizes)
            all_layers_outputs_size_chain.append(tmp_sizes_chain)
            all_layers_outputs_rotation_chain.append(tmp_rotation_chain)
            all_layers_outputs_z_chain.append(tmp_z_chain)

        all_layers_outputs_classes = torch.stack(all_layers_outputs_classes)
        all_layers_outputs_coords = torch.stack(all_layers_outputs_coords)
        all_layers_outputs_centers_2d = torch.stack(all_layers_outputs_centers_2d)
        all_layers_outputs_z = torch.stack(all_layers_outputs_z)
        all_layers_outputs_rotations = torch.stack(all_layers_outputs_rotations)
        all_layers_outputs_sizes = torch.stack(all_layers_outputs_sizes)
        all_layers_outputs_size_chain = torch.stack(all_layers_outputs_size_chain)
        all_layers_outputs_rotation_chain = torch.stack(all_layers_outputs_rotation_chain)
        all_layers_outputs_z_chain = torch.stack(all_layers_outputs_z_chain)

        return (all_layers_outputs_classes, all_layers_outputs_coords,
                all_layers_outputs_centers_2d, all_layers_outputs_z,
                all_layers_outputs_rotations, all_layers_outputs_sizes,
                all_layers_outputs_size_chain, all_layers_outputs_rotation_chain,
                all_layers_outputs_z_chain)

    def loss(self, hidden_states: Tensor, references: List[Tensor],
             enc_outputs_class: Tensor, enc_outputs_coord: Tensor,
             enc_outputs_centers_2d: Tensor, enc_outputs_z: Tensor,
             enc_outputs_rotation: Tensor, enc_outputs_size: Tensor,
             batch_data_samples: SampleList, dn_meta: Dict[str, int]) -> dict:
        """Perform forward propagation and loss calculation."""
        batch_gt_instances = []
        batch_img_metas = []
        for data_sample in batch_data_samples:
            batch_img_metas.append(data_sample.metainfo)
            batch_gt_instances.append(data_sample.gt_instances)

        outs = self(hidden_states, references)
        loss_inputs = outs + (enc_outputs_class, enc_outputs_coord,
                              enc_outputs_centers_2d, enc_outputs_z,
                              enc_outputs_rotation, enc_outputs_size,
                              batch_gt_instances, batch_img_metas, dn_meta)
        losses = self.loss_by_feat(*loss_inputs)
        return losses
    def loss_by_feat(self, all_layers_cls_scores: Tensor, all_layers_bbox_preds: Tensor,
                     all_layers_centers_2d_preds: Tensor, all_layers_z_preds: Tensor,
                     all_layers_rotation_preds: Tensor, all_layers_sizes_preds: Tensor,
                     all_layers_sizes_chain_preds: Tensor,
                     all_layers_rotation_chain_preds: Tensor,
                     all_layers_z_chain_preds: Tensor,
                     enc_cls_scores: Tensor, enc_bbox_preds: Tensor,
                     enc_outputs_centers_2d: Tensor, enc_outputs_z: Tensor,
                     enc_outputs_rotation: Tensor, enc_outputs_size: Tensor,
                     batch_gt_instances: InstanceList, batch_img_metas: List[dict],
                     dn_meta: Dict[str, int], batch_gt_instances_ignore: OptInstanceList = None) -> Dict[str, Tensor]:
        """Loss function."""
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
        if self.use_cop_chain and dn_meta is not None:
            n_dn = dn_meta['num_denoising_queries']
            all_m_sizes_chain = all_layers_sizes_chain_preds[:, :, n_dn:, :]
            all_m_rotation_chain = all_layers_rotation_chain_preds[:, :, n_dn:, :]
            all_m_z_chain = all_layers_z_chain_preds[:, :, n_dn:, :]
        else:
            all_m_sizes_chain = all_layers_sizes_chain_preds
            all_m_rotation_chain = all_layers_rotation_chain_preds
            all_m_z_chain = all_layers_z_chain_preds

        loss_dict = self.loss_by_feat_simple(
            all_layers_matching_cls_scores, all_layers_matching_bbox_preds,
            all_layers_matching_centers_2d_preds, all_layers_matching_z_preds,
            all_layers_matching_rotation_preds, all_layers_matching_sizes_preds,
            all_m_sizes_chain, all_m_rotation_chain, all_m_z_chain,
            batch_gt_instances, batch_img_metas, batch_gt_instances_ignore)

        # loss of proposal generated from encode feature map.
        if enc_cls_scores is not None:
            (enc_loss_cls, enc_losses_bbox, enc_losses_iou, 
             enc_loss_centers_2d, enc_loss_z, enc_loss_rotation, enc_loss_size,
             _enc_size_chain, _enc_rotation_chain, _enc_z_chain) = \
                self.loss_by_feat_single(enc_cls_scores, enc_bbox_preds,
                                       enc_outputs_centers_2d, enc_outputs_z,
                                       enc_outputs_rotation, enc_outputs_size,
                                       None, None, None,
                                       batch_gt_instances=batch_gt_instances,
                                       batch_img_metas=batch_img_metas)
            loss_dict['enc_loss_cls'] = enc_loss_cls
            loss_dict['enc_loss_bbox'] = enc_losses_bbox
            loss_dict['enc_loss_iou'] = enc_losses_iou
            loss_dict['enc_loss_centers_2d'] = enc_loss_centers_2d
            loss_dict['enc_loss_z'] = enc_loss_z
            loss_dict['enc_loss_rotation'] = enc_loss_rotation
            loss_dict['enc_loss_size'] = enc_loss_size

        if all_layers_denoising_cls_scores is not None:
            # calculate denoising loss from all decoder layers
            (dn_losses_cls, dn_losses_bbox, dn_losses_iou,
             dn_losses_centers_2d, dn_losses_z, dn_losses_rotation, dn_losses_sizes) = \
            self.loss_dn(all_layers_denoising_cls_scores, all_layers_denoising_bbox_preds,
                        all_layers_denoising_centers_2d_preds, all_layers_denoising_z_preds,
                        all_layers_denoising_rotation_preds, all_layers_denoising_sizes_preds,
                        batch_gt_instances=batch_gt_instances, batch_img_metas=batch_img_metas,
                        dn_meta=dn_meta)
            # collate denoising loss
            loss_dict['dn_loss_cls'] = dn_losses_cls[-1]
            loss_dict['dn_loss_bbox'] = dn_losses_bbox[-1]
            loss_dict['dn_loss_iou'] = dn_losses_iou[-1]
            loss_dict['dn_loss_centers_2d'] = dn_losses_centers_2d[-1]
            loss_dict['dn_loss_z'] = dn_losses_z[-1]
            loss_dict['dn_loss_rotation'] = dn_losses_rotation[-1]
            loss_dict['dn_loss_size'] = dn_losses_sizes[-1]

            for num_dec_layer, (loss_cls_i, loss_bbox_i, loss_iou_i, 
                               loss_centers_2d_i, loss_z_i, loss_rotation_i, loss_sizes_i) in \
                    enumerate(zip(dn_losses_cls[:-1], dn_losses_bbox[:-1], dn_losses_iou[:-1],
                                 dn_losses_centers_2d[:-1], dn_losses_z[:-1],
                                 dn_losses_rotation[:-1], dn_losses_sizes[:-1])):
                loss_dict[f'd{num_dec_layer}.dn_loss_cls'] = loss_cls_i
                loss_dict[f'd{num_dec_layer}.dn_loss_bbox'] = loss_bbox_i
                loss_dict[f'd{num_dec_layer}.dn_loss_iou'] = loss_iou_i
                loss_dict[f'd{num_dec_layer}.dn_loss_centers_2d'] = loss_centers_2d_i
                loss_dict[f'd{num_dec_layer}.dn_loss_z'] = loss_z_i
                loss_dict[f'd{num_dec_layer}.dn_loss_rotation'] = loss_rotation_i
                loss_dict[f'd{num_dec_layer}.dn_loss_size'] = loss_sizes_i

        return loss_dict

    def loss_by_feat_simple(self, all_layers_cls_scores: Tensor, all_layers_bbox_preds: Tensor,
                           all_layers_centers_2d_preds: Tensor, all_layers_z_preds: Tensor,
                           all_layers_rotation_preds: Tensor, all_layers_sizes_preds: Tensor,
                           all_layers_sizes_chain_preds: Tensor,
                           all_layers_rotation_chain_preds: Tensor,
                           all_layers_z_chain_preds: Tensor,
                           batch_gt_instances: InstanceList, batch_img_metas: List[dict],
                           batch_gt_instances_ignore: OptInstanceList = None) -> Dict[str, Tensor]:
        """Loss function."""
        assert batch_gt_instances_ignore is None, \
            f'{self.__class__.__name__} only supports ' \
            'for batch_gt_instances_ignore setting to None.'

        (losses_cls, losses_bbox, losses_iou, losses_centers_2d, 
         losses_z, losses_rotation, losses_sizes,
         losses_size_chain, losses_rotation_chain, losses_z_chain) = multi_apply(
            self.loss_by_feat_single, all_layers_cls_scores, all_layers_bbox_preds,
            all_layers_centers_2d_preds, all_layers_z_preds,
            all_layers_rotation_preds, all_layers_sizes_preds,
            all_layers_sizes_chain_preds, all_layers_rotation_chain_preds,
            all_layers_z_chain_preds,
            batch_gt_instances=batch_gt_instances, batch_img_metas=batch_img_metas)

        loss_dict = dict()
        # loss from the last decoder layer
        loss_dict['loss_cls'] = losses_cls[-1]
        loss_dict['loss_bbox'] = losses_bbox[-1]
        loss_dict['loss_iou'] = losses_iou[-1]
        loss_dict['loss_centers_2d'] = losses_centers_2d[-1]
        loss_dict['loss_z'] = losses_z[-1]
        loss_dict['loss_rotation'] = losses_rotation[-1]
        loss_dict['loss_size'] = losses_sizes[-1]
        if self.use_cop_chain:
            loss_dict['loss_size_chain'] = losses_size_chain[-1]
            loss_dict['loss_rotation_chain'] = losses_rotation_chain[-1]
            loss_dict['loss_z_chain'] = losses_z_chain[-1]

        # loss from other decoder layers
        num_dec_layer = 0
        for i, (loss_cls_i, loss_bbox_i, loss_iou_i, loss_centers_2d_i, loss_z_i, loss_rotation_i, loss_sizes_i) in \
                enumerate(zip(losses_cls[:-1], losses_bbox[:-1], losses_iou[:-1],
                   losses_centers_2d[:-1], losses_z[:-1], losses_rotation[:-1], losses_sizes[:-1])):
            loss_dict[f'd{num_dec_layer}.loss_cls'] = loss_cls_i
            loss_dict[f'd{num_dec_layer}.loss_bbox'] = loss_bbox_i
            loss_dict[f'd{num_dec_layer}.loss_iou'] = loss_iou_i
            loss_dict[f'd{num_dec_layer}.loss_centers_2d'] = loss_centers_2d_i
            loss_dict[f'd{num_dec_layer}.loss_z'] = loss_z_i
            loss_dict[f'd{num_dec_layer}.loss_rotation'] = loss_rotation_i
            loss_dict[f'd{num_dec_layer}.loss_size'] = loss_sizes_i
            if self.use_cop_chain:
                loss_dict[f'd{num_dec_layer}.loss_size_chain'] = losses_size_chain[i]
                loss_dict[f'd{num_dec_layer}.loss_rotation_chain'] = losses_rotation_chain[i]
                loss_dict[f'd{num_dec_layer}.loss_z_chain'] = losses_z_chain[i]
            num_dec_layer += 1
        return loss_dict

    def loss_dn(self, all_layers_denoising_cls_scores: Tensor, all_layers_denoising_bbox_preds: Tensor,
                all_layers_denoising_centers_2d_preds: Tensor, all_layers_denoising_z_preds: Tensor,
                all_layers_denoising_rotation_preds: Tensor, all_layers_denoising_sizes_preds: Tensor,
                batch_gt_instances: InstanceList, batch_img_metas: List[dict],
                dn_meta: Dict[str, int]) -> Tuple[List[Tensor]]:
        """Calculate denoising loss."""
        return multi_apply(self._loss_dn_single, all_layers_denoising_cls_scores,
                          all_layers_denoising_bbox_preds, all_layers_denoising_centers_2d_preds,
                          all_layers_denoising_z_preds, all_layers_denoising_rotation_preds,
                          all_layers_denoising_sizes_preds, batch_gt_instances=batch_gt_instances,
                          batch_img_metas=batch_img_metas, dn_meta=dn_meta)

    def _loss_dn_single(self, dn_cls_scores: Tensor, dn_bbox_preds: Tensor,
                        dn_centers_2d_preds: Tensor, dn_z_preds: Tensor,
                        dn_rotation_preds: Tensor, dn_sizes_preds: Tensor,
                        batch_gt_instances: InstanceList, batch_img_metas: List[dict],
                        dn_meta: Dict[str, int]) -> Tuple[Tensor]:
        """Denoising loss for outputs from a single decoder layer."""
        cls_reg_targets = self.get_dn_targets(batch_gt_instances, batch_img_metas, dn_meta)
        (labels_list, label_weights_list, bbox_targets_list, bbox_weights_list,
         dn_centers_2d_targets_list, dn_centers_2d_weights_list,
         dn_z_targets_list, dn_z_weights_list,
         dn_rotation_targets_list, dn_rotation_weights_list,
         dn_sizes_targets_list, dn_sizes_weights_list,
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

        # classification loss
        cls_scores = dn_cls_scores.reshape(-1, self.cls_out_channels)
        cls_avg_factor = num_total_pos * 1.0 + num_total_neg * self.bg_cls_weight
        if self.sync_cls_avg_factor:
            cls_avg_factor = reduce_mean(cls_scores.new_tensor([cls_avg_factor]))
        cls_avg_factor = max(cls_avg_factor, 1)

        if len(cls_scores) > 0:
            if isinstance(self.loss_cls, QualityFocalLoss):
                raise NotImplementedError('QualityFocalLoss for DETRPoseHead is not supported yet.')
            else:
                loss_cls = self.loss_cls(cls_scores, labels, label_weights, avg_factor=cls_avg_factor)
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
            rotation_preds = rotation_preds[torch.arange(rotation_preds.size(0)), indexing_labels]
        else:
            rotation_preds = dn_rotation_preds.reshape(-1, self.rot_dim)
        loss_rotation = self.loss_rotation(rotation_preds, rotation_targets, rotation_weights, labels=labels, avg_factor=num_total_pos)

        # sizes loss
        # sizes_preds = dn_sizes_preds.reshape(-1, 3)
        if self.classwise_sizes:
            sizes_preds = dn_sizes_preds.reshape(-1, self.num_classes, 3)
            sizes_preds = sizes_preds[torch.arange(sizes_preds.size(0)), indexing_labels]
        else:
            sizes_preds = dn_sizes_preds.reshape(-1, 3)
        loss_sizes = self.loss_sizes(sizes_preds, sizes_targets, sizes_weights, avg_factor=num_total_pos)

        return loss_cls, loss_bbox, loss_iou, loss_centers_2d, loss_z, loss_rotation, loss_sizes

    def get_dn_targets(self, batch_gt_instances: InstanceList, batch_img_metas: List[dict],
                       dn_meta: Dict[str, int]) -> tuple:
        """Get targets in denoising part for a batch of images."""
        (labels_list, label_weights_list, bbox_targets_list, bbox_weights_list,
         centers_2d_targets_list, centers_2d_weights_list,
         z_targets_list, z_weights_list,
         rotation_targets_list, rotation_weights_list,
         sizes_targets_list, sizes_weights_list,
         pos_inds_list, neg_inds_list) = multi_apply(
             self._get_dn_targets_single, batch_gt_instances, batch_img_metas, dn_meta=dn_meta)
        num_total_pos = sum((inds.numel() for inds in pos_inds_list))
        num_total_neg = sum((inds.numel() for inds in neg_inds_list))
        return (labels_list, label_weights_list, bbox_targets_list, bbox_weights_list,
                centers_2d_targets_list, centers_2d_weights_list,
                z_targets_list, z_weights_list,
                rotation_targets_list, rotation_weights_list,
                sizes_targets_list, sizes_weights_list,
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

        # Set targets for positive samples
        gt_bboxes_normalized = gt_bboxes / factor
        gt_bboxes_targets = bbox_xyxy_to_cxcywh(gt_bboxes_normalized)
        bbox_targets[pos_inds] = gt_bboxes_targets.repeat([num_groups, 1])

        gt_centers_2d_normalized = gt_centers_2d / factor[:, :2]
        centers_2d_targets[pos_inds] = gt_centers_2d_normalized.repeat([num_groups, 1])

        z_targets[pos_inds] = gt_z.repeat([num_groups, 1])
        rotation_targets[pos_inds] = gt_rotations.repeat([num_groups, 1])
        sizes_targets[pos_inds] = gt_sizes.repeat([num_groups, 1])

        return (labels, label_weights, bbox_targets, bbox_weights,
                centers_2d_targets, centers_2d_weights, z_targets, z_weights,
                rotation_targets, rotation_weights, sizes_targets, sizes_weights,
                pos_inds, neg_inds)

    def loss_by_feat_single(self, cls_scores: Tensor, bbox_preds: Tensor,
                           centers_2d_preds: Tensor, z_preds: Tensor,
                           rotation_preds: Tensor, sizes_preds: Tensor,
                           sizes_chain_preds: Tensor,
                           rotation_chain_preds: Tensor, z_chain_preds: Tensor,
                           batch_gt_instances: InstanceList, batch_img_metas: List[dict]) -> Tuple[Tensor]:
        """Loss function for outputs from a single decoder layer."""
        num_imgs = cls_scores.size(0)
        cls_scores_list = [cls_scores[i] for i in range(num_imgs)]
        bbox_preds_list = [bbox_preds[i] for i in range(num_imgs)]
        centers_2d_preds_list = [centers_2d_preds[i] for i in range(num_imgs)]
        z_preds_list = [z_preds[i] for i in range(num_imgs)]
        rotation_preds_list = [rotation_preds[i] for i in range(num_imgs)]
        sizes_preds_list = [sizes_preds[i] for i in range(num_imgs)]

        cls_reg_targets = self.get_targets(cls_scores_list, bbox_preds_list,
                                          centers_2d_preds_list, z_preds_list,
                                          rotation_preds_list, sizes_preds_list,  
                                          batch_gt_instances, batch_img_metas)
        (labels_list, label_weights_list, bbox_targets_list, bbox_weights_list,
         centers_2d_targets_list, centers_2d_weights_list,
         z_targets_list, z_weights_list,
         rotation_targets_list, rotation_weights_list,
         sizes_targets_list, sizes_weights_list,
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

        # classification loss
        cls_scores = cls_scores.reshape(-1, self.cls_out_channels)
        cls_avg_factor = num_total_pos * 1.0 + num_total_neg * self.bg_cls_weight
        if self.sync_cls_avg_factor:
            cls_avg_factor = reduce_mean(cls_scores.new_tensor([cls_avg_factor]))
        cls_avg_factor = max(cls_avg_factor, 1)

        if isinstance(self.loss_cls, QualityFocalLoss):
            raise NotImplementedError('QualityFocalLoss for DETRPoseHead is not supported yet.')
        else:
            loss_cls = self.loss_cls(cls_scores, labels, label_weights, avg_factor=cls_avg_factor)

        num_total_pos = loss_cls.new_tensor([num_total_pos])
        num_total_pos = torch.clamp(reduce_mean(num_total_pos), min=1).item()

        # construct factors used for rescale bboxes
        factors = []
        for img_meta, bbox_pred in zip(batch_img_metas, bbox_preds):
            img_h, img_w = img_meta['img_shape']
            factor = bbox_pred.new_tensor([img_w, img_h, img_w, img_h]).unsqueeze(0).repeat(bbox_pred.size(0), 1)
            factors.append(factor)
        factors = torch.cat(factors, 0)

        bbox_preds = bbox_preds.reshape(-1, 4)
        bboxes = bbox_cxcywh_to_xyxy(bbox_preds) * factors
        bboxes_gt = bbox_cxcywh_to_xyxy(bbox_targets) * factors

        loss_iou = self.loss_iou(bboxes, bboxes_gt, bbox_weights, avg_factor=num_total_pos)
        loss_bbox = self.loss_bbox(bbox_preds, bbox_targets, bbox_weights, avg_factor=num_total_pos)
        
        # centers_2d loss
        centers_2d_preds = centers_2d_preds.reshape(-1, 2)
        loss_centers_2d = self.loss_centers_2d(centers_2d_preds, centers_2d_targets, centers_2d_weights, avg_factor=num_total_pos)

        # z loss
        z_preds = z_preds.reshape(-1, 1)
        loss_z = self.loss_z(z_preds, z_targets, z_weights, avg_factor=num_total_pos)

        # indexing labels for rotation and sizes. background labels are replaced with 0
        indexing_labels = labels.clone()
        indexing_labels[indexing_labels == self.num_classes] = 0

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

        return (loss_cls, loss_bbox, loss_iou, loss_centers_2d, loss_z,
                loss_rotation, loss_sizes, loss_size_chain,
                loss_rotation_chain, loss_z_chain)

    def get_targets(self, cls_scores_list: List[Tensor], bbox_preds_list: List[Tensor],
                    centers_2d_preds_list: List[Tensor], z_preds_list: List[Tensor],
                    rotation_preds_list: List[Tensor], sizes_preds_list: List[Tensor],
                    batch_gt_instances: InstanceList, batch_img_metas: List[dict]) -> tuple:
        """Compute regression and classification targets for a batch image."""
        (labels_list, label_weights_list, bbox_targets_list, bbox_weights_list,
         centers_2d_targets_list, centers_2d_weights_list,
         z_targets_list, z_weights_list,
         rotation_targets_list, rotation_weights_list,
         sizes_targets_list, sizes_weights_list,
         pos_inds_list, neg_inds_list) = multi_apply(self._get_targets_single,
                                      cls_scores_list, bbox_preds_list,
                                      centers_2d_preds_list, z_preds_list,
                                      rotation_preds_list, sizes_preds_list,
                                      batch_gt_instances, batch_img_metas)
        num_total_pos = sum((inds.numel() for inds in pos_inds_list))
        num_total_neg = sum((inds.numel() for inds in neg_inds_list))
        return (labels_list, label_weights_list, bbox_targets_list, bbox_weights_list,
                centers_2d_targets_list, centers_2d_weights_list,
                z_targets_list, z_weights_list,
                rotation_targets_list, rotation_weights_list,
                sizes_targets_list, sizes_weights_list, num_total_pos, num_total_neg)

    def _get_targets_single(self, cls_score: Tensor, bbox_pred: Tensor,
                           centers_2d_pred: Tensor, z_pred: Tensor,
                           rotation_pred: Tensor, sizes_pred: Tensor,
                           gt_instances: InstanceData, img_meta: dict) -> tuple:
        """Compute regression and classification targets for one image."""
        img_h, img_w = img_meta['img_shape']
        factor = bbox_pred.new_tensor([img_w, img_h, img_w, img_h]).unsqueeze(0)
        num_bboxes = bbox_pred.size(0)
        
        # convert bbox_pred from xywh, normalized to xyxy, unnormalized
        bbox_pred_unnorm = bbox_cxcywh_to_xyxy(bbox_pred) * factor

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

        if self.use_log_z:
            depth = torch.exp(z_pred)
        else:
            depth = z_pred

        t_recovered = depth * (torch.inverse(intrinsic) @ centers_2d_h.T).T

        # choose the elements with the highest class score
        pred_labels = cls_score.argmax(dim=-1)
        if self.classwise_rotation:
            rotation_pred = rotation_pred.reshape(num_bboxes, -1, self.rot_dim)
            rotation_pred = rotation_pred[torch.arange(num_bboxes, device=cls_score.device), pred_labels]
        
        if self.classwise_sizes:
            sizes_pred = sizes_pred.reshape(num_bboxes, -1, 3)
            sizes_pred = sizes_pred[torch.arange(num_bboxes, device=cls_score.device), pred_labels]
        

        pred_instances = InstanceData(scores=cls_score, bboxes=bbox_pred_unnorm,
                                     translations=t_recovered,
                                     rotations=rotation_pred, sizes=sizes_pred)
        
        assign_result = self.assigner.assign(pred_instances=pred_instances,
                                           gt_instances=gt_instances, img_meta=img_meta)

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

        # Set targets for positive samples
        pos_gt_bboxes_normalized = pos_gt_bboxes / factor
        pos_gt_bboxes_targets = bbox_xyxy_to_cxcywh(pos_gt_bboxes_normalized)
        bbox_targets[pos_inds] = pos_gt_bboxes_targets

        gt_centers_2d_normalized = gt_centers_2d / factor[:, :2]
        centers_2d_targets[pos_inds] = gt_centers_2d_normalized[pos_assigned_gt_inds.long(), :]

        z_targets[pos_inds] = gt_z[pos_assigned_gt_inds.long(), :]
        rotation_targets[pos_inds] = gt_rotations[pos_assigned_gt_inds]
        sizes_targets[pos_inds] = gt_sizes[pos_assigned_gt_inds]

        return (labels, label_weights, bbox_targets, bbox_weights,
                centers_2d_targets, centers_2d_weights, z_targets, z_weights,
                rotation_targets, rotation_weights, sizes_targets, sizes_weights,
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

        if self.use_log_z:
            depth = torch.exp(z_pred)
        else:
            depth = z_pred

        t_recovered = depth * (torch.inverse(intrinsic) @ centers_2d_h.T).T

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
        num_denoising_queries = dn_meta['num_denoising_queries']
        if dn_meta is not None:
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
