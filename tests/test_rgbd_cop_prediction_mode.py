from types import SimpleNamespace

import pytest
import torch
from torch import nn
from mmengine.config import Config

from yopo.models.dense_pose_heads.dino_9d_center2d_posehead import (
    DINO9DCenter2DPoseHead,
    compact_gaussian_orientation_descriptor,
)
from yopo.models.dense_pose_heads.depth_query_context import (
    CoPStageFusion,
    MultiScaleDepthQuerySampler,
)


def _head(mode, **kwargs):
    head_kwargs = dict(
        num_classes=1,
        embed_dims=8,
        num_reg_fcs=1,
        num_pred_layer=1,
        train_cfg=None,
        use_cop_chain=True,
        cop_prediction_mode=mode,
        cop_chain_order=("z", "size", "rotation"),
        cop_use_bbox_conditioning=True,
        loss_cls=dict(
            type="FocalLoss",
            use_sigmoid=True,
            gamma=2.0,
            alpha=0.25,
            loss_weight=1.0,
        ),
    )
    head_kwargs.update(kwargs)
    return DINO9DCenter2DPoseHead(**head_kwargs)


def _forward(head, **kwargs):
    hidden = torch.zeros(1, 1, 2, 8)
    references = [torch.full((1, 2, 4), 0.5)]
    return head(hidden, references, **kwargs)


def test_cop_prediction_mode_rejects_unknown_value():
    with pytest.raises(ValueError, match="cop_prediction_mode"):
        _head("unknown")


def test_chain_mode_routes_chain_pose_to_primary_outputs():
    head = _head("chain")
    head.init_weights()
    with torch.no_grad():
        head.reg_z_branch[0][-1].bias.fill_(1.0)
        head.reg_rotation_branch[0][-1].bias.fill_(1.0)
        head.reg_size_branch[0][-1].bias.fill_(1.0)
        head.cop_z_out[0].bias.fill_(2.0)
        head.cop_rotation_out[0].bias.fill_(2.0)
        head.cop_size_out[0].bias.fill_(2.0)

    outputs = _forward(head)

    assert torch.all(outputs[3] == 2.0)
    assert torch.all(outputs[4] == 2.0)
    assert torch.all(outputs[5] == 2.0)
    assert outputs[6:] == (None, None, None, None)
    assert head.cop_chain_order == ("z", "size", "rotation")


def test_chain_primary_backward_does_not_train_parallel_pose_branches():
    head = _head("chain")
    hidden = torch.randn(1, 1, 2, 8, requires_grad=True)
    references = [torch.full((1, 2, 4), 0.5)]

    outputs = head(hidden, references)
    sum(output.sum() for output in outputs[3:6]).backward()

    parallel_parameters = [
        parameter
        for branches in (
            head.reg_z_branch,
            head.reg_rotation_branch,
            head.reg_size_branch,
        )
        for parameter in branches.parameters()
    ]
    cop_parameters = [
        parameter
        for branches in (head.cop_z_out, head.cop_size_out, head.cop_rotation_out)
        for parameter in branches.parameters()
    ]
    assert all(parameter.grad is None for parameter in parallel_parameters)
    assert any(parameter.grad is not None for parameter in cop_parameters)


def test_auxiliary_mode_preserves_parallel_primary_and_chain_auxiliary():
    head = _head("auxiliary")
    head.init_weights()
    with torch.no_grad():
        head.reg_z_branch[0][-1].bias.fill_(1.0)
        head.cop_z_out[0].bias.fill_(2.0)

    outputs = _forward(head)

    assert torch.all(outputs[3] == 1.0)
    assert torch.all(outputs[8] == 2.0)


def test_query_obb_auxiliary_is_spd_and_reaches_rotation_stage():
    head = _head(
        "chain",
        loss_obb_aux=dict(
            type="GaussianGWDLoss",
            loss_weight=5.0,
            include_center=False,
        ),
    )
    head.init_weights()
    hidden = torch.randn(1, 1, 2, 8, requires_grad=True)
    references = [torch.full((1, 2, 4), 0.5)]
    outputs = head(hidden, references)
    compact = outputs[9]
    sigma = torch.stack(
        (compact[..., 2], compact[..., 3],
         compact[..., 3], compact[..., 4]), dim=-1).reshape(*compact.shape[:-1], 2, 2)

    assert compact.shape == (1, 1, 2, 5)
    assert torch.linalg.eigvalsh(sigma).min() > 0
    compact[..., 2:].sum().backward()
    assert hidden.grad is not None and hidden.grad.norm() > 0


def test_obb_orientation_descriptor_is_scale_invariant_spin2_encoding():
    compact = torch.tensor([
        [0.0, 0.0, 3.0, 0.0, 1.0],
        [0.0, 0.0, 12.0, 0.0, 4.0],
        [0.0, 0.0, 2.0, 1.0, 2.0],
    ])

    descriptor = compact_gaussian_orientation_descriptor(compact)

    assert torch.allclose(descriptor[0], torch.tensor([0.5, 0.0]))
    assert torch.allclose(descriptor[1], descriptor[0])
    assert torch.allclose(descriptor[2], torch.tensor([0.0, 0.5]))


def test_obb_rotation_conditioning_routes_rotation_gradient_to_obb_predictor():
    head = _head(
        "chain",
        loss_obb_aux=dict(
            type="GaussianGWDLoss",
            loss_weight=1.0,
            include_center=False,
        ),
        cop_obb_rotation_conditioning=True,
    )
    head.init_weights()
    with torch.no_grad():
        head.cop_rotation_out[0].weight.fill_(0.1)
    outputs = _forward(head)

    outputs[4].sum().backward()

    assert any(
        parameter.grad is not None and parameter.grad.norm() > 0
        for parameter in head.cop_obb_out.parameters()
    )
    assert any(
        parameter.grad is not None and parameter.grad.norm() > 0
        for parameter in head.cop_obb_rotation_embed.parameters()
    )


def test_post_rotation_obb_refinement_preserves_legacy_obb_predictor_location():
    common = dict(
        loss_obb_aux=dict(
            type="GaussianGWDLoss",
            loss_weight=1.0,
            include_center=False,
        ),
    )
    torch.manual_seed(7)
    legacy = _head("chain", **common)
    legacy.init_weights()
    refinement = _head(
        "chain", cop_obb_rotation_refinement=True, **common)
    refinement.load_state_dict(legacy.state_dict(), strict=False)
    with torch.no_grad():
        refinement.cop_obb_rotation_embed[0].weight.zero_()
        refinement.cop_obb_rotation_embed[0].bias.zero_()

    legacy_outputs = _forward(legacy)
    refinement_outputs = _forward(refinement)

    assert torch.equal(legacy_outputs[9], refinement_outputs[9])
    assert torch.equal(legacy_outputs[4], refinement_outputs[4])


def test_post_rotation_obb_refinement_routes_rotation_gradient_to_obb_predictor():
    head = _head(
        "chain",
        loss_obb_aux=dict(
            type="GaussianGWDLoss",
            loss_weight=1.0,
            include_center=False,
        ),
        cop_obb_rotation_refinement=True,
    )
    head.init_weights()
    with torch.no_grad():
        head.cop_rotation_out[0].weight.fill_(0.1)
    outputs = _forward(head)

    outputs[4].sum().backward()

    assert any(
        parameter.grad is not None and parameter.grad.norm() > 0
        for parameter in head.cop_obb_out.parameters()
    )


def test_obb_conditioning_and_refinement_are_mutually_exclusive():
    with pytest.raises(ValueError, match="mutually exclusive"):
        _head(
            "chain",
            loss_obb_aux=dict(
                type="GaussianGWDLoss",
                loss_weight=1.0,
                include_center=False,
            ),
            cop_obb_rotation_conditioning=True,
            cop_obb_rotation_refinement=True,
        )


def test_legacy_obb_aux_does_not_route_rotation_gradient_to_obb_predictor():
    head = _head(
        "chain",
        loss_obb_aux=dict(
            type="GaussianGWDLoss",
            loss_weight=1.0,
            include_center=False,
        ),
    )
    head.init_weights()
    outputs = _forward(head)

    outputs[4].sum().backward()

    assert all(parameter.grad is None for parameter in head.cop_obb_out.parameters())


def test_obb_rotation_conditioning_requires_obb_loss_and_rotation_last():
    with pytest.raises(ValueError, match="requires loss_obb_aux"):
        _head("chain", cop_obb_rotation_conditioning=True)
    with pytest.raises(ValueError, match="rotation to be the final"):
        _head(
            "chain",
            cop_chain_order=("z", "rotation", "size"),
            loss_obb_aux=dict(
                type="GaussianGWDLoss",
                loss_weight=1.0,
                include_center=False,
            ),
            cop_obb_rotation_conditioning=True,
        )


def test_obb_diagnostic_predictions_are_opt_in_and_spd():
    common = dict(
        loss_obb_aux=dict(
            type="GaussianGWDLoss",
            loss_weight=1.0,
            include_center=False,
        ),
        test_cfg=dict(max_per_img=2),
    )
    hidden = torch.zeros(1, 1, 2, 8)
    references = [torch.full((1, 2, 4), 0.5)]
    sample = SimpleNamespace(metainfo=dict(
        img_shape=(32, 32),
        scale_factor=(1.0, 1.0),
        intrinsic=(10.0, 10.0, 16.0, 16.0),
    ))

    ordinary = _head("chain", **common)
    ordinary.init_weights()
    ordinary_result = ordinary.predict(
        hidden, references, [sample], rescale=False)[0]
    assert "obb_gaussians" not in ordinary_result

    diagnostic = _head(
        "chain", expose_obb_aux_predictions=True, **common)
    diagnostic.init_weights()
    diagnostic_result = diagnostic.predict(
        hidden, references, [sample], rescale=False)[0]
    compact = diagnostic_result.obb_gaussians
    sigma = torch.stack(
        (compact[:, 2], compact[:, 3], compact[:, 3], compact[:, 4]),
        dim=-1).reshape(-1, 2, 2)
    assert compact.shape == (2, 5)
    assert torch.linalg.eigvalsh(sigma).min() > 0


def test_obb_diagnostic_predictions_require_obb_head():
    with pytest.raises(ValueError, match="requires loss_obb_aux"):
        _head("chain", expose_obb_aux_predictions=True)


def test_legacy_use_cop_chain_maps_to_auxiliary_mode():
    head = _head(None)
    assert head.cop_prediction_mode == "auxiliary"


def test_chain_config_exposes_one_max_objects_knob():
    cfg = Config.fromfile(
        "configs/yopo/nocs_custom_fruit_rgbd_3dbbox_cop_depth_size_rotation.py"
    )
    assert cfg.max_objects == 100
    assert cfg.model.num_queries == cfg.max_objects
    assert cfg.model.bbox_head.test_cfg.max_per_img == cfg.max_objects
    assert cfg.model.bbox_head.cop_prediction_mode == "chain"
    assert tuple(cfg.model.bbox_head.cop_chain_order) == ("z", "size", "rotation")
    assert cfg.model.bbox_head.cop_use_bbox_conditioning is True


def test_q150_projection_gwd_config_is_rotation_only_and_stage4_initialized():
    cfg = Config.fromfile(
        "configs/yopo/"
        "nocs_custom_fruit_rgbd_3dbbox_cop_q150_stage4_projection_gwd.py"
    )
    projection = cfg.model.bbox_head.loss_projection

    assert projection.type == "ProjectedEllipsoidGWDLoss"
    assert cfg.model.bbox_head.projection_geometry_source == "target"
    assert projection.loss_weight == 1.0
    assert projection.detach_center is True
    assert projection.detach_depth is True
    assert projection.detach_size is True
    assert projection.fail_on_invalid is True
    assert cfg.load_from.endswith("best_3d_iou_0.50_epoch_5.pth")
    assert cfg.train_dataloader.dataset.obb_coordinate_scale == 0.8
    assert any(
        step.type == "ResizeOBBGaussians"
        for step in cfg.train_dataloader.dataset.pipeline
    )


def test_q150_obb_aux_config_regularizes_rotation_stage_with_gwd():
    cfg = Config.fromfile(
        "configs/yopo/"
        "nocs_custom_fruit_rgbd_3dbbox_cop_q150_stage4_projection_gwd_obb_aux.py"
    )
    auxiliary = cfg.model.bbox_head.loss_obb_aux
    assert auxiliary.type == "GaussianGWDLoss"
    assert auxiliary.loss_weight == 5.0
    assert auxiliary.include_center is False
    assert cfg.model.bbox_head.projection_geometry_source == "target"


def test_q150_obb_aux_w1_config_changes_only_auxiliary_weight():
    cfg = Config.fromfile(
        "configs/yopo/"
        "nocs_custom_fruit_rgbd_3dbbox_cop_q150_stage4_projection_gwd_obb_aux_w1.py"
    )
    auxiliary = cfg.model.bbox_head.loss_obb_aux
    assert auxiliary.type == "GaussianGWDLoss"
    assert auxiliary.loss_weight == 1.0
    assert cfg.model.bbox_head.loss_projection.loss_weight == 1.0


def test_q150_obb_aux_w1_inference_is_teacher_free():
    cfg = Config.fromfile(
        "configs/yopo/"
        "nocs_custom_fruit_rgbd_3dbbox_cop_q150_stage4_projection_gwd_obb_aux_w1_inference.py"
    )
    head = cfg.model.bbox_head
    assert tuple(head.distill_attributes) == ()
    assert head.obb_center_teacher_checkpoint is None
    assert head.pose_teacher_checkpoint is None
    assert head.loss_obb_aux.loss_weight == 1.0
    assert cfg.load_from is None


def test_q150_projection_anisotropy_config_changes_only_observability_weighting():
    cfg = Config.fromfile(
        "configs/yopo/"
        "nocs_custom_fruit_rgbd_3dbbox_cop_q150_stage4_projection_gwd_anisotropy.py"
    )
    head = cfg.model.bbox_head
    assert head.loss_projection.type == "ProjectedEllipsoidGWDLoss"
    assert head.loss_projection.loss_weight == 1.0
    assert head.loss_projection.target_anisotropy_power == 1.0
    assert head.projection_geometry_source == "target"
    assert head.get("loss_obb_aux") is None


def test_q150_projection_anisotropy_inference_is_teacher_free():
    cfg = Config.fromfile(
        "configs/yopo/"
        "nocs_custom_fruit_rgbd_3dbbox_cop_q150_stage4_projection_gwd_anisotropy_inference.py"
    )
    head = cfg.model.bbox_head
    assert tuple(head.distill_attributes) == ()
    assert head.obb_center_teacher_checkpoint is None
    assert head.pose_teacher_checkpoint is None
    assert head.loss_projection.target_anisotropy_power == 1.0
    assert cfg.load_from is None


def test_q150_obb_rotation_conditioning_config_is_explicit():
    cfg = Config.fromfile(
        "configs/yopo/"
        "nocs_custom_fruit_rgbd_3dbbox_cop_q150_stage4_obb_rotation_conditioning.py"
    )
    head = cfg.model.bbox_head
    assert head.cop_obb_rotation_conditioning is True
    assert head.loss_obb_aux.type == "GaussianGWDLoss"
    assert head.loss_obb_aux.loss_weight == 1.0
    assert tuple(head.cop_chain_order)[-1] == "rotation"
    assert cfg.load_from.endswith("best_3d_iou_0.50_epoch_5.pth")


def test_q150_obb_rotation_conditioning_inference_is_teacher_free():
    cfg = Config.fromfile(
        "configs/yopo/"
        "nocs_custom_fruit_rgbd_3dbbox_cop_q150_stage4_obb_rotation_conditioning_inference.py"
    )
    head = cfg.model.bbox_head
    assert head.cop_obb_rotation_conditioning is True
    assert tuple(head.distill_attributes) == ()
    assert head.obb_center_teacher_checkpoint is None
    assert head.pose_teacher_checkpoint is None
    assert cfg.load_from is None


def test_q150_post_rotation_obb_refinement_config_is_single_change():
    cfg = Config.fromfile(
        "configs/yopo/"
        "nocs_custom_fruit_rgbd_3dbbox_cop_q150_stage4_obb_rotation_refinement.py"
    )
    head = cfg.model.bbox_head
    assert head.cop_obb_rotation_refinement is True
    assert head.get("cop_obb_rotation_conditioning", False) is False
    assert head.loss_obb_aux.loss_weight == 1.0
    assert tuple(head.cop_chain_order)[-1] == "rotation"
    assert cfg.load_from.endswith("best_3d_iou_0.50_epoch_5.pth")


def test_q150_2d_obb_foundation_disables_3d_objectives_for_full_training():
    cfg = Config.fromfile(
        "configs/yopo/"
        "nocs_custom_fruit_rgbd_2d_obb_foundation_full.py"
    )
    head = cfg.model.bbox_head
    assert cfg.train_cfg.max_epochs == 20
    assert cfg.train_cfg.val_interval == 5
    assert head.loss_obb_aux.type == "GaussianGWDLoss"
    assert head.loss_obb_aux.loss_weight == 5.0
    assert head.loss_projection is None
    assert head.loss_z.loss_weight == 0.0
    assert head.loss_sizes.loss_weight == 0.0
    assert head.loss_rotation.loss_weight == 0.0
    assert tuple(head.distill_attributes) == ()
    assert head.loss_cls.loss_weight > 0.0
    assert head.loss_bbox.loss_weight > 0.0
    assert cfg.load_from.endswith("best_3d_iou_0.50_epoch_5.pth")


def test_q150_2d_obb_foundation_diagnostic_is_teacher_free_and_exposed():
    cfg = Config.fromfile(
        "configs/yopo/"
        "nocs_custom_fruit_rgbd_2d_obb_foundation_diagnostic.py"
    )
    head = cfg.model.bbox_head
    assert head.expose_obb_aux_predictions is True
    assert tuple(head.distill_attributes) == ()
    assert head.obb_center_teacher_checkpoint is None
    assert head.pose_teacher_checkpoint is None
    assert cfg.load_from is None


def test_q150_obb_foundation_stage4_control_restores_3d_without_projection():
    cfg = Config.fromfile(
        "configs/yopo/"
        "nocs_custom_fruit_rgbd_2d_obb_foundation_stage4_control.py"
    )
    head = cfg.model.bbox_head
    assert cfg.train_cfg.max_epochs == 5
    assert cfg.train_cfg.val_interval == 5
    assert cfg.load_from.endswith("2d_obb_foundation_full/epoch_20.pth")
    assert head.loss_projection is None
    assert head.loss_obb_aux.loss_weight == 1.0
    assert head.loss_z.loss_weight == 50.0
    assert head.loss_sizes.loss_weight == 50.0
    assert head.loss_rotation.loss_weight == 5.0
    assert tuple(head.distill_attributes) == (
        "center", "z", "size", "rotation")
    assert head.get("cop_obb_rotation_refinement", False) is False


def test_q150_obb_foundation_stage4_refinement_is_single_matched_change():
    control = Config.fromfile(
        "configs/yopo/"
        "nocs_custom_fruit_rgbd_2d_obb_foundation_stage4_control.py"
    )
    refinement = Config.fromfile(
        "configs/yopo/"
        "nocs_custom_fruit_rgbd_2d_obb_foundation_stage4_refinement.py"
    )
    control_head = control.model.bbox_head
    refinement_head = refinement.model.bbox_head
    assert refinement.load_from == control.load_from
    assert refinement_head.cop_obb_rotation_refinement is True
    assert refinement_head.get("cop_obb_rotation_conditioning", False) is False
    for field in (
            "loss_projection", "loss_obb_aux", "loss_z", "loss_sizes",
            "loss_rotation", "distill_attributes"):
        assert refinement_head[field] == control_head[field]


@pytest.mark.parametrize("variant", ["control", "refinement"])
def test_q150_obb_foundation_stage4_inference_is_teacher_free(variant):
    cfg = Config.fromfile(
        "configs/yopo/"
        f"nocs_custom_fruit_rgbd_2d_obb_foundation_stage4_{variant}_inference.py"
    )
    head = cfg.model.bbox_head
    assert cfg.load_from is None
    assert tuple(head.distill_attributes) == ()
    assert head.obb_center_teacher_checkpoint is None
    assert head.pose_teacher_checkpoint is None
    assert head.loss_projection is None
    assert head.get("cop_obb_rotation_refinement", False) is (
        variant == "refinement")


def test_q150_obb_refinement_recovery_schedule_keeps_accepted_path():
    cfg = Config.fromfile(
        "configs/yopo/"
        "nocs_custom_fruit_rgbd_2d_obb_foundation_stage4_refinement_continue15.py"
    )
    head = cfg.model.bbox_head
    assert cfg.load_from.endswith(
        "2d_obb_foundation_stage4_refinement/"
        "best_3d_iou_0.50_epoch_5.pth")
    assert cfg.train_cfg.max_epochs == 15
    assert cfg.train_cfg.val_interval == 5
    assert cfg.param_scheduler[0].T_max == 15
    assert cfg.default_hooks.checkpoint.interval == 5
    assert cfg.default_hooks.checkpoint.save_best == "3d_iou_0.50"
    assert head.cop_obb_rotation_refinement is True
    assert head.loss_projection is None
    assert head.loss_obb_aux.loss_weight == 1.0


def test_q150_obb_refinement_recovery_diagnostic_is_teacher_free():
    cfg = Config.fromfile(
        "configs/yopo/"
        "nocs_custom_fruit_rgbd_2d_obb_foundation_stage4_refinement_continue15_diagnostic.py"
    )
    head = cfg.model.bbox_head
    assert cfg.load_from is None
    assert head.expose_obb_aux_predictions is True
    assert tuple(head.distill_attributes) == ()
    assert head.obb_center_teacher_checkpoint is None
    assert head.pose_teacher_checkpoint is None
    assert head.cop_obb_rotation_refinement is True


@pytest.mark.parametrize("config_name", [
    "nocs_custom_fruit_rgbd_3dbbox_cop_q150_stage4_obb_aux_w1_diagnostic.py",
    "nocs_custom_fruit_rgbd_3dbbox_cop_q150_stage4_obb_rotation_conditioning_diagnostic.py",
])
def test_obb_diagnostic_configs_explicitly_expose_teacher_free_gaussians(
        config_name):
    cfg = Config.fromfile(f"configs/yopo/{config_name}")
    head = cfg.model.bbox_head
    assert head.expose_obb_aux_predictions is True
    assert tuple(head.distill_attributes) == ()
    assert head.obb_center_teacher_checkpoint is None
    assert head.pose_teacher_checkpoint is None
    assert cfg.load_from is None


def _write_teacher_checkpoints(tmp_path):
    pose_head = _head("parallel")
    pose_state = {
        f"bbox_head.{key}": value.clone()
        for key, value in pose_head.state_dict().items()
    }
    pose_path = tmp_path / "pose_teacher.pth"
    torch.save({"state_dict": pose_state}, pose_path)

    obb_state = {}
    for layer_id in range(1):
        prefix = f"bbox_head.reg_branches.{layer_id}"
        obb_state[f"{prefix}.0.weight"] = torch.eye(8)
        obb_state[f"{prefix}.0.bias"] = torch.zeros(8)
        obb_state[f"{prefix}.2.weight"] = torch.zeros(5, 8)
        obb_state[f"{prefix}.2.bias"] = torch.zeros(5)
        obb_state[f"bbox_head.cls_branches.{layer_id}.weight"] = torch.zeros(1, 8)
        obb_state[f"bbox_head.cls_branches.{layer_id}.bias"] = torch.zeros(1)
    obb_path = tmp_path / "obb_teacher.pth"
    torch.save({"state_dict": obb_state}, obb_path)
    return str(obb_path), str(pose_path)


def test_distillation_attributes_reject_unknown_value():
    with pytest.raises(ValueError, match="distill_attributes"):
        _head("chain", distill_attributes=("unknown",))


class _FailDistillationTeacher(nn.Module):
    def forward(self, *_args, **_kwargs):
        raise RuntimeError("distillation teacher was called")


def test_distillation_teacher_is_skipped_by_normal_forward():
    head = _head("chain")
    head.distillation_teacher = _FailDistillationTeacher()

    outputs = _forward(head)

    assert head._last_distillation_targets == {}
    assert head._last_distillation_scores is None
    with pytest.raises(RuntimeError, match="targets are unavailable"):
        head._distillation_losses(outputs, dn_meta=None)
    with pytest.raises(RuntimeError, match="distillation teacher was called"):
        _forward(head, compute_distillation_targets=True)

    hidden = torch.zeros(1, 1, 2, 8)
    references = [torch.full((1, 2, 4), 0.5)]
    sample = SimpleNamespace(metainfo={}, gt_instances=None)
    with pytest.raises(RuntimeError, match="distillation teacher was called"):
        head.loss(
            hidden,
            references,
            None,
            None,
            None,
            None,
            None,
            None,
            batch_data_samples=[sample],
            dn_meta=None,
        )


def test_frozen_teacher_adapters_emit_cumulative_targets(tmp_path):
    obb_path, pose_path = _write_teacher_checkpoints(tmp_path)
    head = _head(
        "chain",
        distill_attributes=("center", "z", "size"),
        obb_center_teacher_checkpoint=obb_path,
        pose_teacher_checkpoint=pose_path,
        distill_loss_weights=dict(center=2.0, z=3.0, size=4.0),
    )

    outputs = _forward(head, compute_distillation_targets=True)

    assert set(head._last_distillation_targets) == {"center", "z", "size"}
    assert head._last_distillation_scores.shape == (1, 1, 2)
    teacher_parameters = [
        parameter
        for name, parameter in head.named_parameters()
        if "teacher" in name
    ]
    assert teacher_parameters
    assert all(not parameter.requires_grad for parameter in teacher_parameters)
    assert head.distill_loss_weights == {"center": 2.0, "z": 3.0, "size": 4.0}
    losses = head._distillation_losses(outputs, dn_meta=None)
    assert set(losses) == {
        "loss_distill_center", "loss_distill_z", "loss_distill_size"
    }
    assert all(torch.isfinite(loss) for loss in losses.values())

    head.train()
    assert head.distillation_teacher.training is False


def test_pose_center_teacher_matches_checkpoint_initialized_student(tmp_path):
    obb_path, pose_path = _write_teacher_checkpoints(tmp_path)
    head = _head(
        "chain",
        distill_attributes=("center",),
        center_teacher_source="pose_center",
        obb_center_teacher_checkpoint=obb_path,
        pose_teacher_checkpoint=pose_path,
    )
    checkpoint = torch.load(pose_path, map_location="cpu", weights_only=False)
    head_state = {
        key.removeprefix("bbox_head."): value
        for key, value in checkpoint["state_dict"].items()
    }
    head.load_state_dict(head_state, strict=False)

    outputs = _forward(head, compute_distillation_targets=True)

    assert head.distillation_teacher_report["center"]["source"] == (
        "frozen_pose_center_head"
    )
    assert torch.allclose(
        outputs[2], head._last_distillation_targets["center"], atol=1e-7
    )
    assert torch.isfinite(head._last_distillation_targets["center"]).all()


def test_incompatible_obb_center_checkpoint_is_rejected(tmp_path):
    obb_path, pose_path = _write_teacher_checkpoints(tmp_path)
    checkpoint = torch.load(obb_path, map_location="cpu", weights_only=False)
    checkpoint["state_dict"]["bbox_head.reg_branches.0.2.weight"] = (
        torch.zeros(1, 8)
    )
    bad_path = tmp_path / "bad_obb_teacher.pth"
    torch.save(checkpoint, bad_path)

    with pytest.raises(RuntimeError, match="shape mismatch"):
        _head(
            "chain",
            distill_attributes=("center",),
            center_teacher_source="obb_adapter",
            obb_center_teacher_checkpoint=str(bad_path),
            pose_teacher_checkpoint=pose_path,
        )


def test_center_distillation_masks_teacher_below_confidence_threshold(tmp_path):
    obb_path, pose_path = _write_teacher_checkpoints(tmp_path)
    head = _head(
        "chain",
        distill_attributes=("center",),
        center_teacher_source="pose_center",
        obb_center_teacher_checkpoint=obb_path,
        pose_teacher_checkpoint=pose_path,
        distill_score_threshold=0.9,
    )
    outputs = _forward(head, compute_distillation_targets=True)
    losses = head._distillation_losses(outputs, dn_meta=None)
    assert losses["loss_distill_center"].item() == 0.0


def test_obb_center_adapter_remains_explicit_ablation(tmp_path):
    obb_path, pose_path = _write_teacher_checkpoints(tmp_path)
    head = _head(
        "chain",
        distill_attributes=("center",),
        center_teacher_source="obb_adapter",
        obb_center_teacher_checkpoint=obb_path,
        pose_teacher_checkpoint=pose_path,
    )
    assert head.distillation_teacher_report["center"]["source"] == (
        "frozen_obb_head_adapter"
    )


def test_curriculum_configs_are_chain_only_then_parallel_control():
    expected = {
        "nocs_custom_fruit_rgbd_3dbbox_cop_stage1_obb.py": ("center",),
        "nocs_custom_fruit_rgbd_3dbbox_cop_stage2_obb_depth.py": ("center", "z"),
        "nocs_custom_fruit_rgbd_3dbbox_cop_stage3_obb_depth_size.py": (
            "center", "z", "size"
        ),
        "nocs_custom_fruit_rgbd_3dbbox_cop_stage4_full.py": (
            "center", "z", "size", "rotation"
        ),
    }
    for filename, attributes in expected.items():
        cfg = Config.fromfile(f"configs/yopo/{filename}")
        assert cfg.max_objects == 100
        assert cfg.model.num_queries == cfg.max_objects
        assert cfg.model.bbox_head.cop_prediction_mode == "chain"
        assert tuple(cfg.model.bbox_head.distill_attributes) == attributes

    parallel = Config.fromfile(
        "configs/yopo/nocs_custom_fruit_rgbd_3dbbox_parallel_control.py"
    )
    assert parallel.model.bbox_head.cop_prediction_mode == "parallel"
    assert tuple(parallel.model.bbox_head.distill_attributes) == ()


def test_q150_3d_curriculum_starts_from_2d_foundation_and_boosts_only_cop():
    expected = {
        "nocs_custom_fruit_rgbd_3dbbox_cop_q150_from_2d_stage2_z.py": (
            "center", "z"
        ),
        "nocs_custom_fruit_rgbd_3dbbox_cop_q150_from_2d_stage3_size.py": (
            "center", "z", "size"
        ),
        "nocs_custom_fruit_rgbd_3dbbox_cop_q150_from_2d_stage4_full.py": (
            "center", "z", "size", "rotation"
        ),
    }
    foundation = (
        "work_dirs/nocs_custom_fruit_rgbd_2d_foundation_full/"
        "best_AP50_epoch_50.pth"
    )
    for filename, attributes in expected.items():
        cfg = Config.fromfile(f"configs/yopo/{filename}")
        assert cfg.max_objects == 150
        assert cfg.model.num_queries == 150
        assert cfg.model.test_cfg.max_per_img == 150
        assert cfg.model.bbox_head.test_cfg.max_per_img == 150
        assert cfg.model.bbox_head.pose_teacher_checkpoint == foundation
        assert tuple(cfg.model.bbox_head.distill_attributes) == attributes
        assert cfg.train_dataloader.batch_size == 22
        keys = cfg.optim_wrapper.paramwise_cfg.custom_keys
        assert keys["bbox_head.cop_"].lr_mult == 50.0
        assert keys["bbox_head.depth_query_sampler"].lr_mult == 50.0
        assert cfg.optim_wrapper.optimizer.lr == pytest.approx(1e-6)

    stage2 = Config.fromfile(
        "configs/yopo/"
        "nocs_custom_fruit_rgbd_3dbbox_cop_q150_from_2d_stage2_z.py"
    )
    assert stage2.load_from == foundation
    assert stage2.model.bbox_head.loss_z.loss_weight == 50.0
    assert stage2.model.bbox_head.loss_sizes.loss_weight == 0.0
    assert stage2.model.bbox_head.loss_rotation.loss_weight == 0.0


def test_depth_query_sampler_preserves_layer_batch_query_shape_and_is_spatial():
    sampler = MultiScaleDepthQuerySampler(
        embed_dims=8, num_levels=3, roi_size=3
    )
    depth_features = []
    for height, width in ((8, 10), (4, 5), (2, 3)):
        x_ramp = torch.linspace(0, 1, width).view(1, 1, 1, width)
        depth_features.append(x_ramp.expand(2, 8, height, width).clone())
    boxes = torch.tensor(
        [
            [
                [[0.2, 0.5, 0.2, 0.2], [0.8, 0.5, 0.2, 0.2]],
                [[0.3, 0.5, 0.2, 0.2], [0.7, 0.5, 0.2, 0.2]],
            ],
            [
                [[0.1, 0.5, 0.2, 0.2], [0.9, 0.5, 0.2, 0.2]],
                [[0.4, 0.5, 0.2, 0.2], [0.6, 0.5, 0.2, 0.2]],
            ],
        ],
        dtype=torch.float32,
    )

    contexts = sampler.forward_layers(depth_features, boxes)

    assert contexts.shape == (2, 2, 2, 8)
    assert torch.isfinite(contexts).all()
    assert not torch.allclose(contexts[0, 0, 0], contexts[0, 0, 1])


def test_depth_query_sampler_vectorizes_decoder_layers(monkeypatch):
    from yopo.models.dense_pose_heads import depth_query_context

    torch.manual_seed(11)
    sampler = MultiScaleDepthQuerySampler(
        embed_dims=8, num_levels=3, roi_size=3
    )
    depth_features = [
        torch.randn(2, 8, height, width, requires_grad=True)
        for height, width in ((8, 10), (4, 5), (2, 3))
    ]
    boxes = torch.rand(4, 2, 5, 4)
    expected = torch.stack([
        sampler(depth_features, layer_boxes) for layer_boxes in boxes
    ])

    original_grid_sample = depth_query_context.F.grid_sample
    call_count = 0

    def counted_grid_sample(*args, **kwargs):
        nonlocal call_count
        call_count += 1
        return original_grid_sample(*args, **kwargs)

    monkeypatch.setattr(
        depth_query_context.F, "grid_sample", counted_grid_sample)
    actual = sampler.forward_layers(depth_features, boxes)

    assert call_count == 3
    assert actual.shape == expected.shape
    assert torch.allclose(actual, expected, atol=1e-6, rtol=1e-5)
    assert actual.is_contiguous()
    sampler.vectorize_layers = False
    fallback = sampler.forward_layers(depth_features, boxes)
    assert torch.allclose(fallback, expected, atol=1e-6, rtol=1e-5)
    actual.sum().backward()
    assert all(feature.grad is not None for feature in depth_features)


def test_cop_stage_fusion_modes_validate_depth_and_keep_query_shape():
    current = torch.randn(2, 5, 8)
    original = torch.randn(2, 5, 8)
    depth = torch.randn(2, 5, 8)

    for mode in ("residual", "query_dense", "depth_dense"):
        fusion = CoPStageFusion(embed_dims=8, mode=mode)
        output = fusion(current, original, depth if mode == "depth_dense" else None)
        assert output.shape == current.shape

    with pytest.raises(ValueError, match="depth_query"):
        CoPStageFusion(embed_dims=8, mode="depth_dense")(
            current, original, None
        )


def test_chain_head_depth_dense_requires_depth_features_and_uses_them():
    torch.manual_seed(7)
    head = _head(
        "chain",
        cop_fusion_mode="depth_dense",
        cop_depth_context=dict(num_levels=3, roi_size=3),
    )
    hidden = torch.randn(1, 1, 2, 8)
    references = [torch.full((1, 2, 4), 0.5)]
    zeros = [torch.zeros(1, 8, h, w) for h, w in ((8, 10), (4, 5), (2, 3))]
    ramps = [
        torch.linspace(0, 1, w).view(1, 1, 1, w).expand(1, 8, h, w)
        for h, w in ((8, 10), (4, 5), (2, 3))
    ]

    with pytest.raises(ValueError, match="depth_features"):
        head(hidden, references)
    outputs_zero = head(hidden, references, depth_features=zeros)
    outputs_ramp = head(hidden, references, depth_features=ramps)

    assert head.last_depth_context_shape == (1, 1, 2, 8)
    assert head.last_dense_input_shape == (1, 1, 2, 24)
    assert any(
        not torch.allclose(outputs_zero[index], outputs_ramp[index])
        for index in (3, 4, 5)
    )


def test_curriculum_depth_context_is_enabled_from_depth_stage():
    expected_modes = {
        "nocs_custom_fruit_rgbd_3dbbox_cop_stage1_obb.py": "query_dense",
        "nocs_custom_fruit_rgbd_3dbbox_cop_stage2_obb_depth.py": "depth_dense",
        "nocs_custom_fruit_rgbd_3dbbox_cop_stage3_obb_depth_size.py": "depth_dense",
        "nocs_custom_fruit_rgbd_3dbbox_cop_stage4_full.py": "depth_dense",
        "nocs_custom_fruit_rgbd_3dbbox_parallel_control.py": "residual",
    }
    for filename, mode in expected_modes.items():
        cfg = Config.fromfile(f"configs/yopo/{filename}")
        assert cfg.model.bbox_head.cop_fusion_mode == mode
        assert cfg.model.bbox_head.cop_depth_context.vectorize_layers is True
    standalone = Config.fromfile(
        "configs/yopo/nocs_custom_fruit_rgbd_3dbbox_cop_depth_size_rotation.py"
    )
    assert standalone.model.bbox_head.cop_fusion_mode == "depth_dense"
    assert standalone.model.bbox_head.cop_depth_context.vectorize_layers is True


def test_curriculum_pairs_pinned_batches_with_nonblocking_transfers():
    cfg = Config.fromfile(
        "configs/yopo/nocs_custom_fruit_rgbd_3dbbox_cop_stage4_full.py"
    )
    assert cfg.train_dataloader.pin_memory is True
    assert cfg.val_dataloader.pin_memory is True
    assert cfg.model.data_preprocessor.non_blocking is True

    cfg.merge_from_dict({
        "train_dataloader.pin_memory": False,
        "val_dataloader.pin_memory": False,
        "model.data_preprocessor.non_blocking": False,
    })
    assert cfg.train_dataloader.pin_memory is False
    assert cfg.val_dataloader.pin_memory is False
    assert cfg.model.data_preprocessor.non_blocking is False


def test_curriculum_uses_fp32_schedulefree_for_deformable_attention():
    for filename in (
        "nocs_custom_fruit_rgbd_3dbbox_parallel_control.py",
        "nocs_custom_fruit_rgbd_3dbbox_cop_stage1_obb.py",
        "nocs_custom_fruit_rgbd_3dbbox_cop_stage4_full.py",
    ):
        cfg = Config.fromfile(f"configs/yopo/{filename}")
        assert cfg.optim_wrapper.type == "ScheduleFreeOptimWrapper"
        assert "dtype" not in cfg.optim_wrapper
        assert "loss_scale" not in cfg.optim_wrapper
        assert cfg.optim_wrapper.optimizer.type == "AdamWScheduleFreeOptimizer"
        assert cfg.optim_wrapper.optimizer.lr == 1e-6


def test_2d_foundation_config_removes_3d_objectives_and_covers_dense_images():
    cfg = Config.fromfile(
        "configs/yopo/nocs_custom_fruit_rgbd_2d_foundation_full.py"
    )
    assert cfg.max_objects == 150
    assert cfg.model.num_queries == 150
    assert cfg.model.test_cfg.max_per_img == 150
    assert cfg.model.bbox_head.test_cfg.max_per_img == 150
    assert tuple(cfg.model.bbox_head.distill_attributes) == ()
    assert cfg.model.bbox_head.obb_center_teacher_checkpoint is None
    assert cfg.model.bbox_head.pose_teacher_checkpoint is None
    assert cfg.model.bbox_head.loss_z.loss_weight == 0
    assert cfg.model.bbox_head.loss_sizes.loss_weight == 0
    assert cfg.model.bbox_head.loss_rotation.loss_weight == 0
    assert [cost.type for cost in cfg.model.train_cfg.assigner.match_costs] == [
        "FocalLossCost",
        "BBoxL1Cost",
        "IoUCost",
    ]
    assert cfg.optim_wrapper.type == "ScheduleFreeOptimWrapper"
    assert cfg.optim_wrapper.optimizer.lr == 5e-5
    assert cfg.train_cfg.max_epochs == 50
    assert cfg.train_cfg.val_interval == 5
    assert cfg.default_hooks.checkpoint.save_best == "AP50"


def test_main_curriculum_uses_safe_pose_center_teacher_and_keeps_obb_ablation():
    stage1 = Config.fromfile(
        "configs/yopo/nocs_custom_fruit_rgbd_3dbbox_cop_stage1_obb.py"
    )
    assert stage1.model.bbox_head.center_teacher_source == "pose_center"
    assert stage1.model.bbox_head.distill_score_threshold == 0.2

    ablation = Config.fromfile(
        "configs/yopo/"
        "nocs_custom_fruit_rgbd_3dbbox_cop_stage1_obb_adapter_ablation.py"
    )
    assert ablation.model.bbox_head.center_teacher_source == "obb_adapter"


def test_stage4_inference_config_has_no_teacher_checkpoint_dependency():
    from yopo.registry import MODELS
    from yopo.utils import register_all_modules

    register_all_modules()
    cfg = Config.fromfile(
        "configs/yopo/"
        "nocs_custom_fruit_rgbd_3dbbox_cop_stage4_inference.py"
    )
    assert tuple(cfg.model.bbox_head.distill_attributes) == ()
    assert cfg.model.bbox_head.obb_center_teacher_checkpoint is None
    assert cfg.model.bbox_head.pose_teacher_checkpoint is None

    model = MODELS.build(cfg.model)
    assert model.bbox_head.distillation_teacher is None
    assert model.bbox_head.cop_prediction_mode == "chain"
    assert tuple(model.bbox_head.cop_chain_order) == (
        "z", "size", "rotation"
    )


def test_q150_stage4_inference_is_teacher_free_and_keeps_capacity():
    from yopo.registry import MODELS
    from yopo.utils import register_all_modules

    register_all_modules()
    cfg = Config.fromfile(
        "configs/yopo/"
        "nocs_custom_fruit_rgbd_3dbbox_cop_q150_from_2d_stage4_inference.py"
    )
    assert cfg.load_from is None
    assert cfg.model.num_queries == 150
    assert cfg.model.test_cfg.max_per_img == 150
    assert cfg.model.bbox_head.test_cfg.max_per_img == 150
    assert tuple(cfg.model.bbox_head.distill_attributes) == ()
    assert cfg.model.bbox_head.obb_center_teacher_checkpoint is None
    assert cfg.model.bbox_head.pose_teacher_checkpoint is None

    model = MODELS.build(cfg.model)
    assert model.bbox_head.distillation_teacher is None
    assert model.bbox_head.cop_prediction_mode == "chain"


def test_chain_encoder_pose_disable_requires_explicit_2d_assigner():
    with pytest.raises(ValueError, match="encoder_assigner"):
        _head(
            "chain",
            cop_encoder_pose_supervision=False,
            train_cfg=dict(
                assigner=dict(
                    type="HungarianAssigner",
                    match_costs=[dict(type="FocalLossCost", weight=1.0)],
                )
            ),
        )


def test_curriculum_disables_encoder_parallel_pose_but_control_keeps_it():
    for filename in (
        "nocs_custom_fruit_rgbd_3dbbox_cop_stage1_obb.py",
        "nocs_custom_fruit_rgbd_3dbbox_cop_stage2_obb_depth.py",
        "nocs_custom_fruit_rgbd_3dbbox_cop_stage3_obb_depth_size.py",
        "nocs_custom_fruit_rgbd_3dbbox_cop_stage4_full.py",
    ):
        cfg = Config.fromfile(f"configs/yopo/{filename}")
        assert cfg.model.bbox_head.cop_encoder_pose_supervision is False
        cost_types = [
            cost.type for cost in cfg.model.train_cfg.encoder_assigner.match_costs
        ]
        assert cost_types == ["FocalLossCost", "BBoxL1Cost", "IoUCost"]

    parallel = Config.fromfile(
        "configs/yopo/nocs_custom_fruit_rgbd_3dbbox_parallel_control.py"
    )
    assert parallel.model.bbox_head.cop_encoder_pose_supervision is True


class _FailIfCalled(nn.Module):
    def forward(self, _inputs):
        raise RuntimeError("encoder pose branch was called")


def test_chain_pre_decoder_skips_encoder_pose_branches():
    from yopo.registry import MODELS
    from yopo.utils import register_all_modules

    register_all_modules()
    shapes = torch.tensor([[8, 10], [4, 5], [2, 3], [1, 2]])
    num_tokens = int((shapes[:, 0] * shapes[:, 1]).sum())
    memory = torch.randn(1, num_tokens, 256)
    mask = torch.zeros(1, num_tokens, dtype=torch.bool)

    chain_cfg = Config.fromfile(
        "configs/yopo/nocs_custom_fruit_rgbd_3dbbox_cop_stage2_obb_depth.py"
    )
    chain = MODELS.build(chain_cfg.model).eval()
    layer_id = chain.decoder.num_layers
    chain.bbox_head.reg_z_branch[layer_id] = _FailIfCalled()
    chain.bbox_head.reg_rotation_branch[layer_id] = _FailIfCalled()
    chain.bbox_head.reg_size_branch[layer_id] = _FailIfCalled()
    _, chain_head_inputs = chain.pre_decoder(memory, mask, shapes)
    assert chain_head_inputs == {}

    parallel_cfg = Config.fromfile(
        "configs/yopo/nocs_custom_fruit_rgbd_3dbbox_parallel_control.py"
    )
    parallel = MODELS.build(parallel_cfg.model).eval()
    layer_id = parallel.decoder.num_layers
    parallel.bbox_head.reg_z_branch[layer_id] = _FailIfCalled()
    with pytest.raises(RuntimeError, match="encoder pose branch was called"):
        parallel.pre_decoder(memory, mask, shapes)


def test_chain_loss_accepts_absent_auxiliary_prediction_tensors():
    head = _head("chain")

    def fake_loss_by_feat_single(cls_scores, *_args, **_kwargs):
        zero = cls_scores.sum() * 0
        return (zero,) * 12

    head.loss_by_feat_single = fake_loss_by_feat_single
    cls_scores = torch.zeros(2, 1, 3, 1)
    bbox_preds = torch.zeros(2, 1, 3, 4)
    center_preds = torch.zeros(2, 1, 3, 2)
    z_preds = torch.zeros(2, 1, 3, 1)
    rotation_preds = torch.zeros(2, 1, 3, 6)
    size_preds = torch.zeros(2, 1, 3, 3)

    losses = head.loss_by_feat_simple(
        cls_scores,
        bbox_preds,
        center_preds,
        z_preds,
        rotation_preds,
        size_preds,
        None,
        None,
        None,
        None,
        batch_gt_instances=[],
        batch_img_metas=[],
    )

    assert "loss_z" in losses
    assert "d0.loss_z" in losses
    assert all("chain" not in key for key in losses)


def test_encoder_loss_keys_follow_chain_and_parallel_supervision_modes():
    assigner = dict(
        type="HungarianAssigner",
        match_costs=[dict(type="FocalLossCost", weight=1.0)],
    )
    chain = _head(
        "chain",
        cop_encoder_pose_supervision=False,
        train_cfg=dict(assigner=assigner, encoder_assigner=assigner),
    )
    parallel = _head("parallel", train_cfg=dict(assigner=assigner))

    cls_scores = torch.zeros(1, 1, 2, 1)
    bbox_preds = torch.zeros(1, 1, 2, 4)
    center_preds = torch.zeros(1, 1, 2, 2)
    z_preds = torch.zeros(1, 1, 2, 1)
    rotation_preds = torch.zeros(1, 1, 2, 6)
    size_preds = torch.zeros(1, 1, 2, 3)

    def encoder_keys(head):
        head.split_outputs = lambda *outputs: (*outputs[:6],) + (None,) * 6
        head.loss_by_feat_simple = lambda *_args, **_kwargs: {}

        def fake_loss_by_feat_single(cls_score, *_args, **_kwargs):
            zero = cls_score.sum() * 0
            return (zero,) * 12

        head.loss_by_feat_single = fake_loss_by_feat_single
        losses = head.loss_by_feat(
            cls_scores,
            bbox_preds,
            center_preds,
            z_preds,
            rotation_preds,
            size_preds,
            None,
            None,
            None,
            None,
            cls_scores[0],
            bbox_preds[0],
            center_preds[0],
            z_preds[0],
            rotation_preds[0],
            size_preds[0],
            batch_gt_instances=[],
            batch_img_metas=[],
            dn_meta=None,
        )
        return {key for key in losses if key.startswith("enc_")}

    assert encoder_keys(chain) == {
        "enc_loss_cls",
        "enc_loss_bbox",
        "enc_loss_iou",
        "enc_loss_centers_2d",
    }
    assert encoder_keys(parallel) == {
        "enc_loss_cls",
        "enc_loss_bbox",
        "enc_loss_iou",
        "enc_loss_centers_2d",
        "enc_loss_z",
        "enc_loss_rotation",
        "enc_loss_size",
    }
