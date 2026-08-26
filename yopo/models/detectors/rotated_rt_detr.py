"""RT-DETR-style dense proposal selection for rotated RGB-D detection."""

from __future__ import annotations

import math

import torch

from yopo.registry import MODELS
from yopo.models.layers import inverse_sigmoid
from .deformable_detr import DeformableDETR


@MODELS.register_module()
class RotatedRTDETR(DeformableDETR):
    """Combine dense O2M spatial learning with an O2O query decoder.

    The dense rotated RTMDet head is trained over all pyramid cells.  Its
    highest-quality cells select both encoder memory content and xywh decoder
    references, following RT-DETR's query-selection role split.  Final query
    predictions remain one-to-one and NMS-free.
    """

    def __init__(self, *args, auxiliary_dense_head, dense_loss_weight=1.0,
                 prediction_mode='query', **kwargs):
        if not kwargs.get('as_two_stage', False):
            raise ValueError('RotatedRTDETR requires as_two_stage=True')
        if not kwargs.get('with_box_refine', False):
            raise ValueError('RotatedRTDETR requires with_box_refine=True')
        super().__init__(*args, **kwargs)
        self.auxiliary_dense_head = MODELS.build(auxiliary_dense_head)
        self.dense_loss_weight = float(dense_loss_weight)
        if prediction_mode not in {'query', 'dense'}:
            raise ValueError(
                f'prediction_mode must be query or dense, got {prediction_mode}')
        self.prediction_mode = prediction_mode

    def _pre_decoder_with_dense(
        self,
        memory,
        memory_mask,
        spatial_shapes,
        dense_outputs,
        batch_data_samples,
    ):
        del spatial_shapes
        dense_logits, _, dense_boxes, _ = \
            self.auxiliary_dense_head._flatten_and_decode(*dense_outputs)
        if dense_logits.shape[1] != memory.shape[1]:
            raise RuntimeError(
                'dense pyramid and encoder memory ordering/count diverged: '
                f'{dense_logits.shape[1]} vs {memory.shape[1]}')
        quality = dense_logits.sigmoid().amax(dim=-1)
        if memory_mask is not None:
            quality = quality.masked_fill(memory_mask, -1)
        topk_indices = quality.topk(self.num_queries, dim=1).indices

        batch_h, batch_w = batch_data_samples[0].batch_input_shape
        factor = dense_boxes.new_tensor(
            [batch_w, batch_h, batch_w, batch_h])
        spatial_boxes = (dense_boxes[..., :4] / factor).clamp(
            min=1e-4, max=1 - 1e-4)
        reference_points = torch.gather(
            spatial_boxes, 1,
            topk_indices[..., None].expand(-1, -1, 4)).detach()
        normalized_angles = torch.remainder(
            dense_boxes[..., 4:5], math.pi) / math.pi
        reference_angles = torch.gather(
            normalized_angles, 1,
            topk_indices[..., None]).clamp(min=1e-4, max=1 - 1e-4).detach()

        output_memory = memory
        if memory_mask is not None:
            output_memory = output_memory.masked_fill(
                memory_mask.unsqueeze(-1), 0)
        output_memory = self.memory_trans_norm(
            self.memory_trans_fc(output_memory))
        query = torch.gather(
            output_memory, 1,
            topk_indices[..., None].expand(-1, -1, output_memory.shape[-1]))
        query = query.detach()
        reference_unact = inverse_sigmoid(reference_points)
        position = self.pos_trans_norm(
            self.pos_trans_fc(self.get_proposal_pos_embed(reference_unact)))
        query_pos, _ = torch.split(position, self.embed_dims, dim=-1)
        decoder_inputs = dict(
            query=query,
            query_pos=query_pos,
            memory=memory,
            reference_points=reference_points,
            reference_angles=reference_angles,
        )
        # Dense supervision is handled by the O2M head itself.  The decoder
        # head is supervised only on actual decoder outputs.
        head_inputs = (
            dict(enc_outputs_class=None, enc_outputs_coord=None)
            if self.training else {})
        return decoder_inputs, head_inputs

    def forward_decoder(
        self,
        query,
        query_pos,
        memory,
        memory_mask,
        reference_points,
        reference_angles,
        spatial_shapes,
        level_start_index,
        valid_ratios,
    ):
        """Refine xywh for attention and angle for the rotated head."""
        hidden_states, spatial_references = self.decoder(
            query=query,
            value=memory,
            query_pos=query_pos,
            key_padding_mask=memory_mask,
            reference_points=reference_points,
            spatial_shapes=spatial_shapes,
            level_start_index=level_start_index,
            valid_ratios=valid_ratios,
            reg_branches=self.bbox_head.reg_branches,
        )
        spatial_references = [reference_points, *spatial_references]
        angle_references = [reference_angles]
        current_angle = reference_angles
        for layer_index, hidden_state in enumerate(hidden_states):
            angle_delta = self.bbox_head.reg_branches[layer_index](
                hidden_state)[..., 4:5]
            current_angle = (
                angle_delta + inverse_sigmoid(current_angle)
            ).sigmoid().detach()
            angle_references.append(current_angle)
        references = [
            torch.cat((spatial, angle), dim=-1)
            for spatial, angle in zip(spatial_references, angle_references)
        ]
        return dict(hidden_states=hidden_states, references=references)

    def forward_transformer(
        self, img_feats, batch_data_samples=None, dense_outputs=None,
    ):
        if batch_data_samples is None:
            raise ValueError('batch_data_samples are required for query selection')
        if dense_outputs is None:
            dense_outputs = self.auxiliary_dense_head(img_feats)
        encoder_inputs, decoder_inputs = self.pre_transformer(
            img_feats, batch_data_samples)
        encoder_outputs = self.forward_encoder(**encoder_inputs)
        new_decoder_inputs, head_inputs = self._pre_decoder_with_dense(
            **encoder_outputs,
            dense_outputs=dense_outputs,
            batch_data_samples=batch_data_samples,
        )
        decoder_inputs.update(new_decoder_inputs)
        head_inputs.update(self.forward_decoder(**decoder_inputs))
        return head_inputs

    def loss(self, batch_inputs, batch_data_samples):
        img_feats = self.extract_feat(batch_inputs)
        dense_outputs = self.auxiliary_dense_head(img_feats)
        batch_gt_instances = [
            sample.gt_instances for sample in batch_data_samples
        ]
        batch_img_metas = [sample.metainfo for sample in batch_data_samples]
        dense_losses = self.auxiliary_dense_head.loss_by_feat(
            *dense_outputs,
            batch_gt_instances=batch_gt_instances,
            batch_img_metas=batch_img_metas,
        )
        head_inputs = self.forward_transformer(
            img_feats, batch_data_samples, dense_outputs=dense_outputs)
        losses = self.bbox_head.loss(
            **head_inputs, batch_data_samples=batch_data_samples)
        for name, value in dense_losses.items():
            key = f'dense_{name}'
            losses[key] = (
                value * self.dense_loss_weight
                if name.startswith('loss_') else value)
        return losses

    def predict(self, batch_inputs, batch_data_samples, rescale=True):
        if self.prediction_mode == 'query':
            return super().predict(
                batch_inputs, batch_data_samples, rescale=rescale)
        img_feats = self.extract_feat(batch_inputs)
        results = self.auxiliary_dense_head.predict(
            img_feats, batch_data_samples, rescale=rescale)
        return self.add_pred_to_datasample(batch_data_samples, results)
