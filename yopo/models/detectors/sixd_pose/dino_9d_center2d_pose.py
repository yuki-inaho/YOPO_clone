# Copyright (c) OpenMMLab. All rights reserved.
from typing import Dict, Optional, Tuple

import torch
from torch import Tensor, nn
from torch.nn.init import normal_

from yopo.registry import MODELS
from yopo.structures import OptSampleList
from yopo.utils import OptConfigType
from ...layers import (CdnQueryGenerator, DeformableDetrTransformerEncoder,
                      DinoTransformerDecoder, SinePositionalEncoding)
from .deformable_pose_detr import DeformablePoseDETR, MultiScaleDeformableAttention
from ..deformable_detr import DeformableDETR


@MODELS.register_module()
class DINO9DCenter2DPose(DeformablePoseDETR):
    r"""Implementation of DINO for 9D pose estimation with separate centers_2d and z prediction.
    
    This detector uses the DINO9DCenter2DPoseHead which predicts centers_2d (2D center coordinates)
    and z (depth) separately instead of combined translation. This allows for better control
    and potentially improved performance for 6D pose estimation tasks.

    Code is modified from the `official github repo
    <https://github.com/IDEA-Research/DINO>`_.

    Args:
        dn_cfg (:obj:`ConfigDict` or dict, optional): Config of denoising
            query generator. Defaults to `None`.
    """

    def __init__(self, *args, dn_cfg: OptConfigType = None,
                 dense_aux_head: OptConfigType = None,
                 dense_aux_loss_weight: float = 1.0, **kwargs) -> None:
        super().__init__(*args, **kwargs)
        # A dense, training-only head on the neck features.  Measurement says
        # the 2D limit is positional: three independent centre estimates -- the
        # box, ``centers_2d`` and the ellipse -- all sit at 0.087 of object
        # scale, so the shared representation, not any one branch, is what runs
        # out.  A one-to-one matcher gives each object a single positive; this
        # gives the same features many, the way the dense reference detector
        # does.  Inference is untouched: nothing here is read at predict time.
        self.dense_aux_loss_weight = float(dense_aux_loss_weight)
        self.dense_aux_head = (
            MODELS.build(dense_aux_head) if dense_aux_head is not None
            else None)
        assert self.as_two_stage, 'as_two_stage must be True for DINO'
        assert self.with_box_refine, 'with_box_refine must be True for DINO'

        if dn_cfg is not None:
            assert 'num_classes' not in dn_cfg and \
                   'num_queries' not in dn_cfg and \
                   'hidden_dim' not in dn_cfg, \
                'The three keyword args `num_classes`, `embed_dims`, and ' \
                '`num_matching_queries` are set in `detector.__init__()`, ' \
                'users should not set them in `dn_cfg` config.'
            dn_cfg['num_classes'] = self.bbox_head.num_classes
            dn_cfg['embed_dims'] = self.embed_dims
            dn_cfg['num_matching_queries'] = self.num_queries
            self.dn_query_generator = CdnQueryGenerator(**dn_cfg)
        else:
            self.dn_query_generator = None

    def _init_layers(self) -> None:
        """Initialize layers except for backbone, neck and bbox_head."""
        self.positional_encoding = SinePositionalEncoding(
            **self.positional_encoding)
        self.encoder = DeformableDetrTransformerEncoder(**self.encoder)
        self.decoder = DinoTransformerDecoder(**self.decoder)
        self.embed_dims = self.encoder.embed_dims
        self.query_embedding = nn.Embedding(self.num_queries, self.embed_dims)
        # NOTE In DINO, the query_embedding only contains content
        # queries, while in Deformable DETR, the query_embedding
        # contains both content and spatial queries, and in DETR,
        # it only contains spatial queries.

        num_feats = self.positional_encoding.num_feats
        assert num_feats * 2 == self.embed_dims, \
            f'embed_dims should be exactly 2 times of num_feats. ' \
            f'Found {self.embed_dims} and {num_feats}.'

        self.level_embed = nn.Parameter(
            torch.Tensor(self.num_feature_levels, self.embed_dims))
        self.memory_trans_fc = nn.Linear(self.embed_dims, self.embed_dims)
        self.memory_trans_norm = nn.LayerNorm(self.embed_dims)

    def init_weights(self) -> None:
        """Initialize weights for Transformer and other components."""
        super(DeformableDETR, self).init_weights()
        for coder in self.encoder, self.decoder:
            for p in coder.parameters():
                if p.dim() > 1:
                    nn.init.xavier_uniform_(p)
        for m in self.modules():
            if isinstance(m, MultiScaleDeformableAttention):
                m.init_weights()
        nn.init.xavier_uniform_(self.memory_trans_fc.weight)
        nn.init.xavier_uniform_(self.query_embedding.weight)
        normal_(self.level_embed)

    def loss(self, batch_inputs: Tensor, batch_data_samples):
        """Main losses, plus the dense auxiliary when one is configured."""
        img_feats = self.extract_feat(batch_inputs)
        head_inputs_dict = self.forward_transformer(img_feats,
                                                    batch_data_samples)
        losses = self.bbox_head.loss(
            **head_inputs_dict, batch_data_samples=batch_data_samples)

        if self.dense_aux_head is not None:
            features = (img_feats['fused_features']
                        if isinstance(img_feats, dict) else img_feats)
            aux = self.dense_aux_head.loss(
                features, self._rotated_aux_samples(batch_data_samples))
            weight = self.dense_aux_loss_weight
            for name, value in aux.items():
                key = f'dense_aux_{name}'
                # The head also reports counters such as ``num_pos``.  mmengine
                # sums every value whose key contains "loss", so a counter that
                # is passed through unfiltered would be added to the objective;
                # it is kept for the log but detached and left unweighted.
                if 'loss' not in name:
                    losses[key] = (value.detach()
                                   if torch.is_tensor(value) else value)
                    continue
                losses[key] = (
                    [v * weight for v in value]
                    if isinstance(value, (list, tuple)) else value * weight)
        return losses

    def _rotated_aux_samples(self, batch_data_samples):
        """Re-express the annotations as rotated boxes for the dense auxiliary.

        The dense head assigns on oriented boxes and calls ``regularize_boxes``
        on them, while this detector's samples carry horizontal ones.  The
        oriented annotation is already present as the compact Gaussian the
        ellipse branch is trained against, so it is decoded here rather than
        approximated from the horizontal box -- otherwise the auxiliary would be
        supervising a different target from the head it is meant to help.

        A shallow copy per sample: the originals must reach the main head with
        their horizontal boxes untouched.
        """
        from mmengine.structures import InstanceData

        from yopo.evaluation.metrics.ellipse_rotated_iou_metric import (
            compact_gaussian_to_obb)
        from yopo.structures.bbox import RotatedBoxes

        converted = []
        for sample in batch_data_samples:
            gt = sample.gt_instances
            if not hasattr(gt, 'obb_gaussians'):
                raise ValueError(
                    'dense_aux_head needs obb_gaussians on the annotations; '
                    'load them with with_obb_gaussian=True')
            compact = gt.obb_gaussians
            boxes = (compact_gaussian_to_obb(compact.float())
                     if compact.numel() else compact.new_zeros((0, 5)))
            aux = sample.new()
            aux.set_metainfo(sample.metainfo)
            aux.gt_instances = InstanceData(
                bboxes=RotatedBoxes(boxes.to(compact.device)),
                labels=gt.labels)
            aux.ignored_instances = InstanceData(
                bboxes=RotatedBoxes(boxes.new_zeros((0, 5))),
                labels=gt.labels.new_zeros((0,)))
            converted.append(aux)
        return converted

    def _unnormalized_depth(self, batch_inputs: Tensor) -> Tensor:
        """Undo the preprocessor's normalization of the depth channel.

        ``extract_feat`` sees inputs after ``data_preprocessor``, which
        normalizes all four channels -- depth included, with its own mean
        and std.  A metric reading has to come from the packed value, so
        the statistics the preprocessor actually holds are used to invert
        it rather than a constant written into a config, which would go
        silently wrong the moment the normalization changed.
        """
        depth = batch_inputs[:, 3:4].detach()
        preprocessor = getattr(self, 'data_preprocessor', None)
        mean = getattr(preprocessor, 'mean', None)
        std = getattr(preprocessor, 'std', None)
        if mean is None or std is None or mean.numel() <= 3:
            return depth
        return depth * std.flatten()[3] + mean.flatten()[3]

    def extract_feat(self, batch_inputs: Tensor):
        """Extract fused transformer maps and optional explicit depth maps."""
        # The head can consume the *metric* depth, not just depth features.
        # Measured on this data: reading the depth channel at an object's
        # centre is accurate to 5.7 mm, while the regressed range is off by
        # 21.9 mm, and replacing the range with the truth multiplies the
        # primary 3D metric by fifteen.  The raw channel never reaches the
        # head through the feature path, so it is handed over here.  It is a
        # transient input, not state: no parameter, no buffer, overwritten
        # every forward.
        if getattr(self.bbox_head, 'sensor_depth_scale', None) is not None:
            self.bbox_head.sensor_depth_map = (
                self._unnormalized_depth(batch_inputs)
                if batch_inputs.shape[1] > 3 else None)
        if not getattr(self.bbox_head, 'requires_depth_features', False):
            return super().extract_feat(batch_inputs)
        if not hasattr(self.backbone, 'forward_with_depth_features'):
            raise TypeError(
                'depth_dense CoP requires a backbone implementing '
                'forward_with_depth_features()')
        fused_features, depth_features = \
            self.backbone.forward_with_depth_features(batch_inputs)
        if self.with_neck:
            fused_features = self.neck(fused_features)
        return dict(
            fused_features=fused_features,
            depth_features=depth_features,
        )

    def forward_transformer(
        self,
        img_feats: Tuple[Tensor],
        batch_data_samples: OptSampleList = None,
    ) -> Dict:
        """Forward process of Transformer.

        The forward procedure of the transformer is defined as:
        'pre_transformer' -> 'encoder' -> 'pre_decoder' -> 'decoder'
        More details can be found at `TransformerDetector.forward_transformer`
        in `yopo/detector/base_detr.py`.
        The difference is that the ground truth in `batch_data_samples` is
        required for the `pre_decoder` to prepare the query of DINO.
        Additionally, DINO inherits the `pre_transformer` method and the
        `forward_encoder` method of DeformableDETR. More details about the
        two methods can be found in `yopo/detector/deformable_detr.py`.

        Args:
            img_feats (tuple[Tensor]): Tuple of feature maps from neck. Each
                feature map has shape (bs, dim, H, W).
            batch_data_samples (list[:obj:`DetDataSample`]): The batch
                data samples. It usually includes information such
                as `gt_instance` or `gt_panoptic_seg` or `gt_sem_seg`.
                Defaults to None.

        Returns:
            dict: The dictionary of bbox_head function inputs, which always
            includes the `hidden_states` of the decoder output and may contain
            `references` including the initial and intermediate references.
        """
        depth_features = None
        if isinstance(img_feats, dict):
            depth_features = img_feats['depth_features']
            img_feats = img_feats['fused_features']

        encoder_inputs_dict, decoder_inputs_dict = self.pre_transformer(
            img_feats, batch_data_samples)

        encoder_outputs_dict = self.forward_encoder(**encoder_inputs_dict)

        tmp_dec_in, head_inputs_dict = self.pre_decoder(
            **encoder_outputs_dict, batch_data_samples=batch_data_samples)
        decoder_inputs_dict.update(tmp_dec_in)

        decoder_outputs_dict = self.forward_decoder(**decoder_inputs_dict)
        head_inputs_dict.update(decoder_outputs_dict)
        if depth_features is not None:
            head_inputs_dict['depth_features'] = depth_features
        return head_inputs_dict

    def pre_decoder(
        self,
        memory: Tensor,
        memory_mask: Tensor,
        spatial_shapes: Tensor,
        batch_data_samples: OptSampleList = None,
    ) -> Tuple[Dict]:
        """Prepare intermediate variables before entering Transformer decoder,
        such as `query`, `query_pos`, and `reference_points`.
        
        This method is modified to work with separate centers_2d and z predictions
        instead of combined translation.

        Args:
            memory (Tensor): The output embeddings of the Transformer encoder,
                has shape (bs, num_feat_points, dim).
            memory_mask (Tensor): ByteTensor, the padding mask of the memory,
                has shape (bs, num_feat_points). Will only be used when
                `as_two_stage` is `True`.
            spatial_shapes (Tensor): Spatial shapes of features in all levels.
                With shape (num_levels, 2), last dimension represents (h, w).
                Will only be used when `as_two_stage` is `True`.
            batch_data_samples (list[:obj:`DetDataSample`]): The batch
                data samples. It usually includes information such
                as `gt_instance` or `gt_panoptic_seg` or `gt_sem_seg`.
                Defaults to None.

        Returns:
            tuple[dict]: The decoder_inputs_dict and head_inputs_dict.

            - decoder_inputs_dict (dict): The keyword dictionary args of
              `self.forward_decoder()`, which includes 'query', 'memory',
              `reference_points`, and `dn_mask`. The reference points of
              decoder input here are 4D boxes, although it has `points`
              in its name.
            - head_inputs_dict (dict): The keyword dictionary args of the
              bbox_head functions, which includes `topk_score`, `topk_coords`,
              and `dn_meta` when `self.training` is `True`, else is empty.
        """
        bs, _, c = memory.shape
        cls_out_features = self.bbox_head.cls_branches[
            self.decoder.num_layers].out_features

        output_memory, output_proposals = self.gen_encoder_output_proposals(
            memory, memory_mask, spatial_shapes)
        enc_outputs_class = self.bbox_head.cls_branches[
            self.decoder.num_layers](
                output_memory)
        enc_outputs_coord_unact = self.bbox_head.reg_branches[
            self.decoder.num_layers](output_memory) + output_proposals

        # Generate encoder outputs for separate centers_2d and z predictions
        tmp_enc_outputs_coords = enc_outputs_coord_unact.sigmoid()
        
        # Centers 2D prediction - using bbox information if configured
        if hasattr(self.bbox_head, 'use_bbox_for_centers_2d') and self.bbox_head.use_bbox_for_centers_2d:
            centers_2d_input = torch.cat([output_memory, tmp_enc_outputs_coords], dim=-1)
        else:
            centers_2d_input = output_memory
        enc_outputs_centers_2d = self.bbox_head.reg_centers_2d_branch[
            self.decoder.num_layers](centers_2d_input)

        encoder_pose_supervision = getattr(
            self.bbox_head, 'cop_encoder_pose_supervision', True)
        if encoder_pose_supervision:
            z_input = output_memory
            if self.bbox_head.use_bbox_for_z:
                z_input = torch.cat(
                    [output_memory, tmp_enc_outputs_coords], dim=-1)
            rotation_input = output_memory
            if self.bbox_head.use_bbox_for_rotation:
                rotation_input = torch.cat(
                    [output_memory, tmp_enc_outputs_coords], dim=-1)
            size_input = output_memory
            if self.bbox_head.use_bbox_for_size:
                size_input = torch.cat(
                    [output_memory, tmp_enc_outputs_coords], dim=-1)
            enc_outputs_z = self.bbox_head.reg_z_branch[
                self.decoder.num_layers](z_input)
            enc_outputs_rotation = self.bbox_head.reg_rotation_branch[
                self.decoder.num_layers](rotation_input)
            enc_outputs_sizes = self.bbox_head.reg_size_branch[
                self.decoder.num_layers](size_input)
        else:
            enc_outputs_z = enc_outputs_rotation = enc_outputs_sizes = None

        # NOTE The DINO selects top-k proposals according to scores of
        # multi-class classification, while DeformDETR, where the input
        # is `enc_outputs_class[..., 0]` selects according to scores of
        # binary classification.
        topk_indices = torch.topk(
            enc_outputs_class.max(-1)[0], k=self.num_queries, dim=1)[1]
        topk_score = torch.gather(
            enc_outputs_class, 1,
            topk_indices.unsqueeze(-1).repeat(1, 1, cls_out_features))
        topk_coords_unact = torch.gather(
            enc_outputs_coord_unact, 1,
            topk_indices.unsqueeze(-1).repeat(1, 1, 4))
        
        # Gather top-k predictions for centers_2d, z, rotation, and sizes
        topk_centers_2d = torch.gather(
            enc_outputs_centers_2d, 1,
            topk_indices.unsqueeze(-1).repeat(1, 1, 2))
        if encoder_pose_supervision:
            topk_z = torch.gather(
                enc_outputs_z, 1,
                topk_indices.unsqueeze(-1).repeat(1, 1, 1))
            topk_rotation = torch.gather(
                enc_outputs_rotation, 1,
                topk_indices.unsqueeze(-1).repeat(
                    1, 1, enc_outputs_rotation.shape[-1]))
            topk_sizes = torch.gather(
                enc_outputs_sizes, 1,
                topk_indices.unsqueeze(-1).repeat(
                    1, 1, enc_outputs_sizes.shape[-1]))
        else:
            topk_z = topk_rotation = topk_sizes = None

        topk_coords = topk_coords_unact.sigmoid()
        topk_coords_unact = topk_coords_unact.detach()
        
        # Process centers_2d predictions with bbox offset
        topk_centers_2d = topk_centers_2d.sigmoid() + topk_coords[..., :2] - 0.5

        query = self.query_embedding.weight[:, None, :]
        query = query.repeat(1, bs, 1).transpose(0, 1)
        if self.training and self.dn_query_generator is not None:
            dn_label_query, dn_bbox_query, dn_mask, dn_meta = \
                self.dn_query_generator(batch_data_samples)
            query = torch.cat([dn_label_query, query], dim=1)
            reference_points = torch.cat([dn_bbox_query, topk_coords_unact],
                                         dim=1)
        else:
            reference_points = topk_coords_unact
            dn_mask, dn_meta = None, None
        reference_points = reference_points.sigmoid()

        decoder_inputs_dict = dict(
            query=query,
            memory=memory,
            reference_points=reference_points,
            dn_mask=dn_mask)
        # NOTE DINO calculates encoder losses on scores and coordinates
        # of selected top-k encoder queries, while DeformDETR is of all
        # encoder queries.
        head_inputs_dict = dict(
            enc_outputs_class=topk_score,
            enc_outputs_coord=topk_coords,
            enc_outputs_centers_2d=topk_centers_2d,
            enc_outputs_z=topk_z,
            enc_outputs_rotation=topk_rotation,
            enc_outputs_size=topk_sizes,
            dn_meta=dn_meta) if self.training else dict()
        return decoder_inputs_dict, head_inputs_dict

    def forward_decoder(self,
                        query: Tensor,
                        memory: Tensor,
                        memory_mask: Tensor,
                        reference_points: Tensor,
                        spatial_shapes: Tensor,
                        level_start_index: Tensor,
                        valid_ratios: Tensor,
                        dn_mask: Optional[Tensor] = None,
                        points: Optional[Tensor] = None,
                        **kwargs) -> Dict:
        """Forward with Transformer decoder.

        The forward procedure of the transformer is defined as:
        'pre_transformer' -> 'encoder' -> 'pre_decoder' -> 'decoder'
        More details can be found at `TransformerDetector.forward_transformer`
        in `yopo/detector/base_detr.py`.

        Args:
            query (Tensor): The queries of decoder inputs, has shape
                (bs, num_queries_total, dim), where `num_queries_total` is the
                sum of `num_denoising_queries` and `num_matching_queries` when
                `self.training` is `True`, else `num_matching_queries`.
            memory (Tensor): The output embeddings of the Transformer encoder,
                has shape (bs, num_feat_points, dim).
            memory_mask (Tensor): ByteTensor, the padding mask of the memory,
                has shape (bs, num_feat_points).
            reference_points (Tensor): The initial reference, has shape
                (bs, num_queries_total, 4) with the last dimension arranged as
                (cx, cy, w, h).
            spatial_shapes (Tensor): Spatial shapes of features in all levels,
                has shape (num_levels, 2), last dimension represents (h, w).
            level_start_index (Tensor): The start index of each level.
                A tensor has shape (num_levels, ) and can be represented
                as [0, h_0*w_0, h_0*w_0+h_1*w_1, ...].
            valid_ratios (Tensor): The ratios of the valid width and the valid
                height relative to the width and the height of features in all
                levels, has shape (bs, num_levels, 2).
            dn_mask (Tensor, optional): The attention mask to prevent
                information leakage from different denoising groups and
                matching parts, will be used as `self_attn_mask` of the
                `self.decoder`, has shape (num_queries_total,
                num_queries_total).
                It is `None` when `self.training` is `False`.

        Returns:
            dict: The dictionary of decoder outputs, which includes the
            `hidden_states` of the decoder output and `references` including
            the initial and intermediate reference_points.
        """
        inter_states, references = self.decoder(
            query=query,
            value=memory,
            key_padding_mask=memory_mask,
            self_attn_mask=dn_mask,
            reference_points=reference_points,
            spatial_shapes=spatial_shapes,
            level_start_index=level_start_index,
            valid_ratios=valid_ratios,
            reg_branches=self.bbox_head.reg_branches,
            **kwargs)

        if (self.dn_query_generator is not None and
                len(query) == self.num_queries):
            # NOTE: This is to make sure label_embeding can be involved to
            # produce loss even if there is no denoising query (no ground truth
            # target in this GPU), otherwise, this will raise runtime error in
            # distributed training.
            inter_states[0] += \
                self.dn_query_generator.label_embedding.weight[0, 0] * 0.0

        decoder_outputs_dict = dict(
            hidden_states=inter_states, references=list(references))
        return decoder_outputs_dict
