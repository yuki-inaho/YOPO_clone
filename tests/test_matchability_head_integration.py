from __future__ import annotations

import pytest
import torch

from yopo.models.dense_pose_heads.dino_9d_center2d_posehead import (
    DINO9DCenter2DPoseHead,
)
from yopo.models.dense_pose_heads.matchability_quality import (
    MatchabilityQualityPolicy,
)


def _head(*, source: str = 'hbb_iou', **policy_options):
    return DINO9DCenter2DPoseHead(
        num_classes=1,
        embed_dims=8,
        num_reg_fcs=1,
        num_pred_layer=1,
        train_cfg=None,
        loss_cls=dict(
            type='MatchabilityAwareLoss',
            use_sigmoid=True,
            gamma=1.5,
            loss_weight=1.0,
        ),
        quality_target=dict(
            type='MatchabilityQualityPolicy',
            source=source,
            **policy_options,
        ),
    )


def test_head_builds_configured_stateless_quality_policy():
    head = _head(
        source='blend', obb_weight=0.25, missing_obb='hbb_iou')

    assert head.uses_quality_target is True
    assert isinstance(
        head.quality_target_policy, MatchabilityQualityPolicy)
    assert head.quality_target_policy.num_classes == 1
    assert head.quality_target_policy.source == 'blend'
    assert list(head.quality_target_policy.parameters()) == []


def test_quality_policy_is_rejected_for_hard_label_loss():
    with pytest.raises(ValueError, match='quality_target requires'):
        DINO9DCenter2DPoseHead(
            num_classes=1,
            embed_dims=8,
            num_reg_fcs=1,
            num_pred_layer=1,
            train_cfg=None,
            loss_cls=dict(
                type='FocalLoss', use_sigmoid=True, gamma=2.0,
                alpha=0.25, loss_weight=1.0),
            quality_target=dict(
                type='MatchabilityQualityPolicy', source='hbb_iou'),
        )


def test_head_adapter_detaches_hbb_and_obb_geometry():
    head = _head(
        source='blend', obb_weight=0.25, missing_obb='hbb_iou')
    logits = torch.tensor([[0.1], [-0.2]], requires_grad=True)
    labels = torch.tensor([0, 1])
    label_weights = torch.ones(2)
    hbb_predictions = torch.tensor(
        [[0.5, 0.5, 0.2, 0.2], [0.2, 0.2, 0.1, 0.1]],
        requires_grad=True)
    hbb_targets = torch.tensor(
        [[0.5, 0.5, 0.2, 0.2], [0.0, 0.0, 0.0, 0.0]])
    obb_predictions = torch.tensor(
        [[0.0, 0.0, 0.01, 0.0, 0.04],
         [0.0, 0.0, 0.01, 0.0, 0.01]],
        requires_grad=True)
    obb_targets = torch.tensor(
        [[0.0, 0.0, 0.01, 0.0, 0.04],
         [0.0, 0.0, 0.0, 0.0, 0.0]])

    loss = head._classification_loss(
        cls_scores=logits,
        labels=labels,
        label_weights=label_weights,
        bbox_predictions=hbb_predictions,
        bbox_targets=hbb_targets,
        obb_predictions=obb_predictions,
        obb_targets=obb_targets,
        avg_factor=1,
    )
    loss.backward()

    assert torch.isfinite(loss)
    assert logits.grad is not None and logits.grad.norm() > 0
    assert hbb_predictions.grad is None
    assert obb_predictions.grad is None


def test_blend_policy_falls_back_to_hbb_for_encoder_and_dn_queries():
    head = _head(
        source='blend', obb_weight=0.25, missing_obb='hbb_iou')
    logits = torch.zeros((2, 1), requires_grad=True)
    labels = torch.tensor([0, 1])
    predictions = torch.tensor([
        [0.5, 0.5, 0.2, 0.2],
        [0.2, 0.2, 0.1, 0.1],
    ])
    targets = torch.tensor([
        [0.5, 0.5, 0.2, 0.2],
        [0.0, 0.0, 0.0, 0.0],
    ])

    actual = head._classification_loss(
        logits, labels, torch.ones(2), predictions, targets, avg_factor=1)
    hbb_head = _head(source='hbb_iou')
    expected = hbb_head._classification_loss(
        logits, labels, torch.ones(2), predictions, targets, avg_factor=1)

    torch.testing.assert_close(actual, expected)


def test_compact_gaussian_normalization_is_shared_and_exact():
    targets = torch.tensor([
        [100.0, 50.0, 400.0, 20.0, 100.0],
    ])
    factors = torch.tensor([[200.0, 100.0, 200.0, 100.0]])

    normalized = DINO9DCenter2DPoseHead._normalize_obb_gaussian_targets(
        targets, factors)

    torch.testing.assert_close(
        normalized,
        torch.tensor([[0.5, 0.5, 0.01, 0.001, 0.01]]),
    )

