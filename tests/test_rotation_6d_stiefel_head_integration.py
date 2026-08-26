"""Head integration contracts for the optional raw-rotation frame loss."""

from __future__ import annotations

import pytest
import torch

from yopo.models.dense_pose_heads.dino_9d_center2d_posehead import (
    DINO9DCenter2DPoseHead,
)


def _head(
    *,
    mode: str = 'parallel',
    num_pred_layer: int = 2,
    num_classes: int = 1,
    classwise_rotation: bool = False,
    rot_dim: int = 6,
    enable_frame_loss: bool = True,
) -> DINO9DCenter2DPoseHead:
    loss_rotation_frame = None
    if enable_frame_loss:
        loss_rotation_frame = dict(
            type='Rotation6DStiefelLoss', beta=1.0, loss_weight=0.04)
    return DINO9DCenter2DPoseHead(
        num_classes=num_classes,
        embed_dims=8,
        num_reg_fcs=1,
        num_pred_layer=num_pred_layer,
        rot_dim=rot_dim,
        classwise_rotation=classwise_rotation,
        train_cfg=None,
        cop_prediction_mode=mode,
        cop_chain_order=('z', 'size', 'rotation'),
        loss_rotation_frame=loss_rotation_frame,
        loss_cls=dict(
            type='FocalLoss',
            use_sigmoid=True,
            gamma=2.0,
            alpha=0.25,
            loss_weight=1.0,
        ),
    )


def _loss_inputs(rotation_predictions: torch.Tensor) -> tuple:
    layers, batch, queries = rotation_predictions.shape[:3]
    return (
        torch.zeros(layers, batch, queries, 1),
        torch.zeros(layers, batch, queries, 4),
        torch.zeros(layers, batch, queries, 2),
        torch.zeros(layers, batch, queries, 1),
        rotation_predictions,
        torch.zeros(layers, batch, queries, 3),
        None,
        None,
        None,
        None,
    )


def _stub_semantic_losses(head: DINO9DCenter2DPoseHead) -> None:
    def fake_loss_by_feat_single(cls_scores, *_args, **_kwargs):
        zero = cls_scores.sum() * 0.0
        values = [zero] * 12
        values[5] = zero + 7.0
        return tuple(values)

    head.loss_by_feat_single = fake_loss_by_feat_single


def _simple_losses(
    head: DINO9DCenter2DPoseHead,
    rotation_predictions: torch.Tensor,
) -> dict[str, torch.Tensor]:
    _stub_semantic_losses(head)
    return head.loss_by_feat_simple(
        *_loss_inputs(rotation_predictions),
        batch_gt_instances=[],
        batch_img_metas=[],
    )


def test_disabled_frame_loss_is_backward_compatible_and_adds_no_keys():
    head = _head(enable_frame_loss=False)
    predictions = torch.zeros(2, 1, 2, 6)

    losses = _simple_losses(head, predictions)

    assert head.loss_rotation_frame is None
    assert all('rotation_frame' not in key for key in losses)


def test_main_and_intermediate_frame_losses_are_separate_from_so3_loss():
    head = _head()
    parallel = torch.tensor([1.0, 0.0, 0.0, 1.0, 0.0, 0.0])
    orthonormal = torch.tensor([1.0, 0.0, 0.0, 0.0, 1.0, 0.0])
    predictions = torch.stack((parallel, orthonormal)).view(2, 1, 1, 6)

    losses = _simple_losses(head, predictions)

    # The raw parallel frame has base loss 0.25 and module weight 0.04.
    torch.testing.assert_close(
        losses['d0.loss_rotation_frame'], torch.tensor(0.01))
    torch.testing.assert_close(
        losses['loss_rotation_frame'], torch.tensor(0.0), atol=0, rtol=0)
    torch.testing.assert_close(losses['loss_rotation'], torch.tensor(7.0))
    torch.testing.assert_close(losses['d0.loss_rotation'], torch.tensor(7.0))


def test_chain_mode_regularizes_the_cop_stream_used_for_inference():
    head = _head(mode='chain', num_pred_layer=1)
    head.init_weights()
    hidden = torch.randn(1, 1, 2, 8, requires_grad=True)
    references = [torch.full((1, 2, 4), 0.5)]
    outputs = head(hidden, references)
    _stub_semantic_losses(head)

    losses = head.loss_by_feat_simple(
        *outputs,
        batch_gt_instances=[],
        batch_img_metas=[],
    )
    losses['loss_rotation_frame'].backward()

    assert head.cop_rotation_out[0].bias.grad is not None
    assert torch.isfinite(head.cop_rotation_out[0].bias.grad).all()
    assert head.cop_rotation_out[0].bias.grad.norm() > 0
    assert all(
        parameter.grad is None
        for parameter in head.reg_rotation_branch.parameters()
    )


def test_denoising_and_encoder_streams_do_not_receive_frame_loss_keys():
    head = _head(num_pred_layer=1)
    _stub_semantic_losses(head)
    orthonormal = torch.tensor([1.0, 0.0, 0.0, 0.0, 1.0, 0.0])
    invalid_dn = torch.full((6,), 10.0)
    rotations = torch.stack((invalid_dn, orthonormal, orthonormal)).view(
        1, 1, 3, 6)
    inputs = _loss_inputs(rotations)
    zero = rotations.sum() * 0.0
    head.loss_dn = lambda *_args, **_kwargs: tuple([[zero]] * 7)

    losses = head.loss_by_feat(
        *inputs,
        None,
        None,
        None,
        None,
        None,
        None,
        batch_gt_instances=[],
        batch_img_metas=[],
        dn_meta=dict(num_denoising_queries=1),
    )

    torch.testing.assert_close(
        losses['loss_rotation_frame'], torch.tensor(0.0), atol=0, rtol=0)
    assert 'dn_loss_rotation_frame' not in losses
    assert 'enc_loss_rotation_frame' not in losses


def test_classwise_rotation_regularizes_every_inference_candidate():
    head = _head(
        num_pred_layer=1,
        num_classes=2,
        classwise_rotation=True,
    )
    orthonormal = torch.tensor([1.0, 0.0, 0.0, 0.0, 1.0, 0.0])
    parallel = torch.tensor([1.0, 0.0, 0.0, 1.0, 0.0, 0.0])
    predictions = torch.cat((orthonormal, parallel)).view(1, 1, 1, 12)

    losses = _simple_losses(head, predictions)

    # Mean over the zero-loss class and the 0.25-loss class, then * 0.04.
    torch.testing.assert_close(
        losses['loss_rotation_frame'], torch.tensor(0.005))


def test_frame_loss_rejects_non_6d_rotation_parameterization():
    with pytest.raises(ValueError, match='rot_dim=6'):
        _head(rot_dim=9)
