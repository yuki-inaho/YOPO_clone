from __future__ import annotations

import torch
from mmengine.config import Config
from mmengine.structures import InstanceData

from yopo.models.dense_pose_heads.dino_9d_center2d_posehead import (
    DINO9DCenter2DPoseHead,
    aligned_iou_quality_targets,
    one_to_many_bbox_targets,
)


def test_aligned_iou_quality_targets_are_detached_and_zero_for_background():
    labels = torch.tensor([0, 0, 1])
    predicted = torch.tensor(
        [
            [0.5, 0.5, 0.2, 0.2],
            [0.55, 0.5, 0.2, 0.2],
            [0.5, 0.5, 0.2, 0.2],
        ],
        requires_grad=True,
    )
    target = torch.tensor(
        [
            [0.5, 0.5, 0.2, 0.2],
            [0.5, 0.5, 0.2, 0.2],
            [0.5, 0.5, 0.2, 0.2],
        ]
    )

    quality = aligned_iou_quality_targets(
        labels, predicted, target, num_classes=1
    )

    torch.testing.assert_close(quality, torch.tensor([1.0, 0.6, 0.0]))
    assert quality.requires_grad is False


def _quality_head() -> DINO9DCenter2DPoseHead:
    return DINO9DCenter2DPoseHead(
        num_classes=1,
        embed_dims=8,
        num_reg_fcs=1,
        num_pred_layer=1,
        train_cfg=None,
        use_cop_chain=True,
        cop_prediction_mode="chain",
        cop_chain_order=("z", "size", "rotation"),
        loss_cls=dict(
            type="QualityFocalLoss",
            use_sigmoid=True,
            beta=2.0,
            loss_weight=1.0,
        ),
    )


def test_quality_focal_one_to_one_loss_is_finite_and_does_not_move_boxes():
    head = _quality_head()
    labels = torch.tensor([0, 1])
    label_weights = torch.ones(2)
    bbox_targets = torch.tensor(
        [[0.5, 0.5, 0.2, 0.2], [0.0, 0.0, 0.0, 0.0]]
    )
    bbox_weights = torch.tensor(
        [[1.0, 1.0, 1.0, 1.0], [0.0, 0.0, 0.0, 0.0]]
    )
    zeros_2 = torch.zeros(2, 2)
    zeros_1 = torch.zeros(2, 1)
    zeros_3 = torch.zeros(2, 3)
    zeros_6 = torch.zeros(2, 6)
    zeros_5 = torch.zeros(2, 5)
    zeros_w = torch.zeros(2)

    head.get_targets = lambda *args, **kwargs: (
        [labels],
        [label_weights],
        [bbox_targets],
        [bbox_weights],
        [zeros_2],
        [zeros_2],
        [zeros_1],
        [zeros_1],
        [zeros_6],
        [zeros_6],
        [zeros_3],
        [zeros_3],
        [zeros_5],
        [zeros_w],
        1,
        1,
    )
    cls_scores = torch.tensor([[[0.0], [0.0]]], requires_grad=True)
    bbox_preds = torch.tensor(
        [[[0.5, 0.5, 0.2, 0.2], [0.2, 0.2, 0.1, 0.1]]],
        requires_grad=True,
    )
    centers = torch.zeros(1, 2, 2, requires_grad=True)

    losses = head.loss_by_feat_single(
        cls_scores,
        bbox_preds,
        centers,
        torch.zeros(1, 2, 1),
        torch.zeros(1, 2, 6),
        torch.zeros(1, 2, 3),
        None,
        None,
        None,
        None,
        batch_gt_instances=[],
        batch_img_metas=[dict(img_shape=(100, 100))],
        pose_supervision=False,
    )

    assert torch.isfinite(losses[0])
    losses[0].backward()
    assert cls_scores.grad is not None and cls_scores.grad.norm() > 0
    assert bbox_preds.grad is None


def test_quality_focal_ablation_changes_only_score_loss_contract():
    control = Config.fromfile(
        "configs/yopo/nocs_custom_fruit_rgbd_nmsfree_control_continue5.py"
    )
    quality = Config.fromfile(
        "configs/yopo/nocs_custom_fruit_rgbd_nmsfree_quality_focal.py"
    )

    assert control.model.num_queries == quality.model.num_queries == 150
    assert control.model.bbox_head.loss_cls.type == "FocalLoss"
    assert quality.model.bbox_head.loss_cls.type == "QualityFocalLoss"
    assert quality.model.bbox_head.loss_cls.beta == 2.0
    assert control.model.train_cfg == quality.model.train_cfg
    assert control.model.bbox_head.test_cfg == quality.model.bbox_head.test_cfg
    assert control.load_from == quality.load_from


def test_nms_free_diagnostic_configs_disable_teachers_and_expose_query_obb():
    for name in (
        "nocs_custom_fruit_rgbd_nmsfree_control_continue5_diagnostic.py",
        "nocs_custom_fruit_rgbd_nmsfree_quality_focal_diagnostic.py",
    ):
        config = Config.fromfile(f"configs/yopo/{name}")
        head = config.model.bbox_head
        assert head.expose_obb_aux_predictions is True
        assert tuple(head.distill_attributes) == ()
        assert head.obb_center_teacher_checkpoint is None
        assert head.pose_teacher_checkpoint is None
        assert config.load_from is None


def test_one_to_many_targets_seed_every_gt_then_add_unique_queries():
    predictions = torch.tensor([
        [0.20, 0.20, 0.20, 0.20],
        [0.22, 0.20, 0.20, 0.20],
        [0.70, 0.70, 0.20, 0.20],
        [0.72, 0.70, 0.20, 0.20],
        [0.45, 0.45, 0.10, 0.10],
    ])
    targets_xyxy = torch.tensor([
        [0.10, 0.10, 0.30, 0.30],
        [0.60, 0.60, 0.80, 0.80],
    ])
    labels = torch.zeros(2, dtype=torch.long)

    result = one_to_many_bbox_targets(
        predictions, targets_xyxy, labels, num_classes=1, topk=2)

    positive = result["assigned_gt_indices"] >= 0
    assert positive.sum().item() == 4
    assert torch.bincount(
        result["assigned_gt_indices"][positive], minlength=2
    ).tolist() == [2, 2]
    assert result["bbox_weights"][positive].eq(1).all()
    assert result["labels"][~positive].eq(1).all()


def test_one_to_many_auxiliary_head_is_separate_and_training_only_by_contract():
    config = Config.fromfile(
        "configs/yopo/nocs_custom_fruit_rgbd_nmsfree_o2m_auxiliary.py"
    )
    head = DINO9DCenter2DPoseHead(
        num_classes=1,
        embed_dims=8,
        num_reg_fcs=1,
        num_pred_layer=1,
        train_cfg=None,
        loss_cls=dict(
            type="FocalLoss", use_sigmoid=True, gamma=2.0, alpha=0.25,
            loss_weight=1.0,
        ),
        o2m_aux_topk=config.model.bbox_head.o2m_aux_topk,
        o2m_aux_loss_weight=config.model.bbox_head.o2m_aux_loss_weight,
    )

    assert head.o2m_aux_topk == 2
    assert head.o2m_aux_loss_weight == 0.1
    assert head.o2m_cls_branch is not head.cls_branches[-1]
    assert head.o2m_reg_branch is not head.reg_branches[-1]
    assert "o2m" not in str(head.forward.__annotations__).lower()


def test_one_to_many_auxiliary_loss_updates_shared_features_not_main_predictor():
    head = DINO9DCenter2DPoseHead(
        num_classes=1,
        embed_dims=8,
        num_reg_fcs=1,
        num_pred_layer=1,
        train_cfg=None,
        loss_cls=dict(
            type="FocalLoss", use_sigmoid=True, gamma=2.0, alpha=0.25,
            loss_weight=1.0,
        ),
        o2m_aux_topk=2,
        o2m_aux_loss_weight=0.1,
    )
    hidden = torch.randn(1, 1, 5, 8, requires_grad=True)
    references = [torch.tensor([[[
        0.2, 0.2, 0.2, 0.2,
    ], [
        0.22, 0.2, 0.2, 0.2,
    ], [
        0.7, 0.7, 0.2, 0.2,
    ], [
        0.72, 0.7, 0.2, 0.2,
    ], [
        0.45, 0.45, 0.1, 0.1,
    ]]])]
    gt = InstanceData(
        bboxes=torch.tensor([
            [10.0, 10.0, 30.0, 30.0],
            [60.0, 60.0, 80.0, 80.0],
        ]),
        labels=torch.zeros(2, dtype=torch.long),
    )

    losses = head._loss_o2m_auxiliary(
        hidden, references, [gt], [dict(img_shape=(100, 100))], None)
    total = sum(losses.values())
    assert torch.isfinite(total)
    total.backward()

    assert hidden.grad is not None and hidden.grad.norm() > 0
    assert head.o2m_cls_branch.weight.grad is not None
    assert head.o2m_reg_branch[-1].weight.grad is not None
    assert head.cls_branches[-1].weight.grad is None
    assert head.reg_branches[-1][-1].weight.grad is None
