"""Contracts for ellipse-aware query assignment, ranking, and DN training."""

from __future__ import annotations

import torch
from mmengine.structures import InstanceData

from yopo.models.dense_pose_heads.dino_9d_center2d_posehead import (
    DINO9DCenter2DPoseHead,
)
from yopo.models.dense_pose_heads.matchability_quality import (
    MatchabilityQualityPolicy,
)
from yopo.models.task_modules.assigners.match_cost import Ellipse2DKLDCost


def _compact(mean_x, mean_y, xx, xy, yy):
    return torch.tensor([mean_x, mean_y, xx, xy, yy], dtype=torch.float32)


def _ellipse_head(**overrides) -> DINO9DCenter2DPoseHead:
    kwargs = dict(
        num_classes=1,
        embed_dims=8,
        num_reg_fcs=1,
        num_pred_layer=1,
        train_cfg=None,
        use_cop_chain=False,
        cop_prediction_mode="parallel",
        loss_cls=dict(
            type="MatchabilityAwareLoss",
            use_sigmoid=True,
            gamma=1.5,
            loss_weight=1.0,
        ),
        quality_target=dict(
            type="MatchabilityQualityPolicy",
            source="ellipse_kld_blend",
            obb_weight=0.5,
            include_center=True,
            missing_obb="hbb_iou",
        ),
        gaucho_ellipse2d=True,
        gaucho_ellipse2d_dn=True,
        loss_ellipse2d=dict(
            type="Ellipse2DKLDLoss",
            loss_weight=1.0,
            include_center=True,
            fail_on_invalid=True,
        ),
        loss_rotation=dict(type="Rotation3DLoss", loss_weight=0.0),
    )
    kwargs.update(overrides)
    return DINO9DCenter2DPoseHead(**kwargs)


def test_ellipse_kld_cost_prefers_the_matching_gaussian():
    first = _compact(20.0, 30.0, 25.0, 0.0, 9.0)
    second = _compact(80.0, 60.0, 16.0, 0.0, 36.0)
    predictions = torch.stack((first, second))[:, None, :]
    targets = torch.stack((first, second))
    cost = Ellipse2DKLDCost(weight=2.0)(
        pred_instances=InstanceData(ellipse_gaussians=predictions),
        gt_instances=InstanceData(
            obb_gaussians=targets,
            labels=torch.zeros(2, dtype=torch.long),
        ),
    )

    assert cost.shape == (2, 2)
    assert torch.all(cost.diagonal() < 1e-6)
    assert cost[0, 1] > cost[0, 0]
    assert cost[1, 0] > cost[1, 1]


def test_ellipse_kld_cost_selects_the_gt_class_prediction():
    class_zero = _compact(20.0, 20.0, 16.0, 0.0, 9.0)
    class_one = _compact(70.0, 70.0, 25.0, 0.0, 4.0)
    predictions = torch.stack((class_zero, class_one))[None, :, :]
    cost = Ellipse2DKLDCost()(
        pred_instances=InstanceData(ellipse_gaussians=predictions),
        gt_instances=InstanceData(
            obb_gaussians=class_one[None, :],
            labels=torch.ones(1, dtype=torch.long),
        ),
    )
    assert cost.item() < 1e-6


def test_ellipse_quality_is_one_when_aligned_and_detached_when_shifted():
    target = torch.stack((
        _compact(20.0, 30.0, 25.0, 0.0, 9.0),
        _compact(0.0, 0.0, 1.0, 0.0, 1.0),
    ))
    predicted = target.clone().requires_grad_(True)
    predicted.data[0, 0] += 3.0
    labels = torch.tensor([0, 1])
    hbb = torch.tensor([
        [0.5, 0.5, 0.2, 0.2],
        [0.1, 0.1, 0.1, 0.1],
    ])
    policy = MatchabilityQualityPolicy(
        num_classes=1,
        source="ellipse_kld",
        include_center=True,
    )
    shifted = policy(labels, hbb, hbb, predicted, target)
    exact = policy(labels, hbb, hbb, target, target)

    assert 0.0 < shifted[0] < exact[0]
    assert exact[0].item() == 1.0
    assert shifted[1].item() == 0.0
    assert shifted.requires_grad is False


def test_head_decodes_classwise_ellipse_for_hungarian_cost():
    head = _ellipse_head(train_cfg=dict(assigner=dict(
        type="HungarianAssigner",
        match_costs=[dict(type="Ellipse2DKLDCost", weight=1.0)],
    )))
    bbox = torch.tensor([
        [0.5, 0.5, 0.2, 0.1],
        [0.2, 0.3, 0.1, 0.2],
    ])
    raw = torch.zeros(2, 5)
    predictions = head._build_matching_pred_instances(
        cls_score=torch.zeros(2, 1),
        bbox_pred=bbox,
        centers_2d_pred=torch.zeros(2, 2),
        z_pred=torch.ones(2, 1),
        rotation_pred=torch.zeros(2, 6),
        sizes_pred=torch.ones(2, 3),
        img_meta=dict(img_shape=(100, 200)),
        pose_for_matching=False,
        ellipse2d_pred=raw,
    )

    assert predictions.ellipse_gaussians.shape == (2, 1, 5)
    assert torch.isfinite(predictions.ellipse_gaussians).all()
    torch.testing.assert_close(
        predictions.ellipse_gaussians[:, 0, :2],
        torch.tensor([[100.0, 50.0], [40.0, 30.0]]),
    )


def test_denoising_queries_receive_ellipse_loss_and_gradient():
    head = _ellipse_head()
    gt = InstanceData(
        bboxes=torch.tensor([[40.0, 40.0, 60.0, 60.0]]),
        labels=torch.zeros(1, dtype=torch.long),
        centers_2d=torch.tensor([[50.0, 50.0]]),
        z=torch.tensor([[0.9]]),
        rotations=torch.tensor([[1.0, 0.0, 0.0, 0.0, 1.0, 0.0]]),
        sizes=torch.tensor([[0.04, 0.03, 0.03]]),
        obb_gaussians=torch.tensor([[50.0, 50.0, 100.0, 0.0, 25.0]]),
    )
    raw_ellipse = torch.zeros(1, 2, 5, requires_grad=True)
    losses = head._loss_dn_single(
        dn_cls_scores=torch.zeros(1, 2, 1),
        dn_bbox_preds=torch.tensor([[[0.5, 0.5, 0.2, 0.2],
                                     [0.5, 0.5, 0.2, 0.2]]]),
        dn_centers_2d_preds=torch.full((1, 2, 2), 0.5),
        dn_z_preds=torch.full((1, 2, 1), 0.9),
        dn_rotation_preds=torch.tensor(
            [[[1.0, 0.0, 0.0, 0.0, 1.0, 0.0],
              [1.0, 0.0, 0.0, 0.0, 1.0, 0.0]]]),
        dn_sizes_preds=torch.full((1, 2, 3), 0.04),
        batch_gt_instances=[gt],
        batch_img_metas=[dict(img_shape=(100, 100))],
        dn_meta=dict(num_denoising_groups=1, num_denoising_queries=2),
        dn_ellipse2d_preds=raw_ellipse,
    )

    assert len(losses) == 8
    dn_ellipse_loss = losses[-1]
    assert torch.isfinite(dn_ellipse_loss)
    assert dn_ellipse_loss.item() > 0.0
    dn_ellipse_loss.backward()
    assert raw_ellipse.grad is not None
    assert torch.isfinite(raw_ellipse.grad).all()
    assert raw_ellipse.grad.norm() > 0
