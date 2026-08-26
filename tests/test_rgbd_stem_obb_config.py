"""Contracts for the standalone RGB-D encoder + 2D OBB detector."""

from pathlib import Path

import torch
from mmengine.config import Config
from mmengine.utils import import_modules_from_strings

from yopo.registry import DATASETS
from yopo.utils import register_all_modules


STAGE1 = "configs/yopo/rotated_deformable_detr_stem_rgbd_obb_riou_stage1.py"
STAGE2 = "configs/yopo/rotated_deformable_detr_stem_rgbd_obb_gwd_stage2.py"
SMOKE = "configs/yopo/rotated_deformable_detr_stem_rgbd_obb_smoke.py"
MODULAR_RIOU = (
    "configs/yopo/"
    "rotated_deformable_detr_stem_rgbd_obb_modular_riou_full.py"
)
MODULAR_GWD = (
    "configs/yopo/"
    "rotated_deformable_detr_stem_rgbd_obb_modular_gwd_full.py"
)
MODULAR_GWD_CONTINUE20 = (
    "configs/yopo/"
    "rotated_deformable_detr_stem_rgbd_obb_modular_gwd_continue20_lr1e4.py"
)
DENSE_RIOU = (
    'configs/yopo/rotated_rtmdet_stem_rgbd_obb_dense_riou_full.py'
)
DENSE_GWD = (
    'configs/yopo/rotated_rtmdet_stem_rgbd_obb_dense_gwd_full.py'
)
HYBRID_RIOU = (
    'configs/yopo/rotated_rt_detr_stem_rgbd_obb_hybrid_riou_full.py'
)
HYBRID_GWD = (
    'configs/yopo/rotated_rt_detr_stem_rgbd_obb_hybrid_gwd_full.py'
)


def test_rgbd_stem_obb_config_is_direct_and_query_complete() -> None:
    cfg = Config.fromfile(STAGE1)

    assert cfg.model.type == "DeformableDETR"
    assert cfg.model.backbone.type == "RGBDDualBackbone"
    assert cfg.model.backbone.rgb_backbone.in_channels == 3
    assert cfg.model.backbone.depth_backbone.in_channels == 1
    assert cfg.model.backbone.rgb_backbone.freeze_at == -1
    assert cfg.model.backbone.depth_backbone.freeze_at == -1
    assert cfg.model.bbox_head.type == "RotatedDeformableDETRHead"
    assert cfg.model.bbox_head.loss_iou.type == "RotatedIoULoss"
    assert cfg.model.num_queries == 256
    assert cfg.model.test_cfg.max_per_img == 256
    assert "pose" not in repr(cfg.model).lower()
    assert "translation" not in repr(cfg.model).lower()

    pipeline = cfg.train_dataloader.dataset.pipeline
    assert [step.type for step in pipeline] == [
        "LoadImageFromFile",
        "LoadRawDepthImageWithValidMask",
        "LoadAnnotations",
        "YOLOXHSVRandomAug",
        "ConcatRawDepthToImage",
        "Resize",
        "RandomFlip",
        "PackDetInputs",
    ]
    assert pipeline[4].bgr_to_rgb is True
    assert pipeline[6].direction == "vertical"

    train = cfg.train_dataloader.dataset
    val = cfg.val_dataloader.dataset
    assert tuple(train.metainfo.classes) == ("stem",)
    assert train.strict_loading is True
    assert train.data_prefix.depth_path == "train/depth"
    assert val.ann_file == "valid/labels"
    assert val.data_prefix.depth_path == "valid/depth"
    assert cfg.train_dataloader.batch_size == 32


def test_rgbd_stem_obb_gwd_stage_does_not_replay_rgb_transfer() -> None:
    cfg = Config.fromfile(STAGE2)

    assert cfg.model.bbox_head.loss_iou.type == "GDLoss"
    assert cfg.model.bbox_head.loss_iou.loss_type == "gwd"
    assert cfg.train_cfg.max_epochs == 15
    assert cfg.custom_hooks == []
    assert cfg.load_from.endswith("rddetr_stem_rgbd_obb_riou_stage1/selected_best.pth")


def test_rgbd_stem_obb_dataset_emits_four_aligned_channels() -> None:
    register_all_modules()
    cfg = Config.fromfile(SMOKE)
    import_modules_from_strings(**cfg.custom_imports)
    dataset = DATASETS.build(cfg.train_dataloader.dataset)
    packed = dataset[0]

    assert len(dataset) == 64
    assert packed["inputs"].shape == (4, 512, 736)
    assert packed["inputs"].dtype == torch.float32
    assert Path(dataset.get_data_info(0)["depth_path"]).suffix == ".png"
    assert len(packed["data_samples"].gt_instances) > 0


def test_rgbd_obb_transfer_expands_queries_reproducibly() -> None:
    from yopo.engine.hooks.rgbd_obb_transfer import build_rgbd_obb_transfer_state

    source = {
        "backbone.weight": torch.ones(2, 3),
        "encoder.weight": torch.full((2, 2), 2.0),
        "query_embedding.weight": torch.arange(12, dtype=torch.float32).reshape(3, 4),
        "neck.weight": torch.ones(1),
    }
    target = {
        "backbone.rgb_backbone.weight": torch.zeros(2, 3),
        "encoder.weight": torch.zeros(2, 2),
        "query_embedding.weight": torch.zeros(5, 4),
        "backbone.depth_backbone.weight": torch.zeros(1, 3),
    }

    selected_a, report = build_rgbd_obb_transfer_state(source, target, query_seed=7)
    selected_b, _ = build_rgbd_obb_transfer_state(source, target, query_seed=7)

    assert torch.equal(
        selected_a["query_embedding.weight"][:3],
        source["query_embedding.weight"],
    )
    assert torch.equal(
        selected_a["query_embedding.weight"],
        selected_b["query_embedding.weight"],
    )
    assert report["source_queries"] == 3
    assert report["target_queries"] == 5
    assert report["ignored_groups"] == {"neck": 1}


def test_rgbd_feature_transfer_loads_exact_pretrained_neck() -> None:
    from yopo.engine.hooks.rgbd_obb_transfer import (
        build_rgbd_feature_transfer_state,
    )

    source = {
        'backbone.weight': torch.ones(2, 3),
        'neck.proj.weight': torch.full((4, 2, 1, 1), 2.0),
        'encoder.layer.weight': torch.full((2, 2), 3.0),
        'bbox_head.weight': torch.ones(1),
    }
    target = {
        'backbone.rgb_backbone.weight': torch.zeros(2, 3),
        'backbone.depth_backbone.weight': torch.zeros(1, 1),
        'neck.proj.weight': torch.zeros(4, 2, 1, 1),
        'encoder.layer.weight': torch.zeros(2, 2),
        'bbox_head.weight': torch.zeros(9),
    }

    selected, report = build_rgbd_feature_transfer_state(
        source, target, ('encoder',))

    assert set(selected) == {
        'backbone.rgb_backbone.weight',
        'neck.proj.weight',
        'encoder.layer.weight',
    }
    assert report['loaded_groups']['neck'] == 1
    assert report['loaded_groups']['encoder'] == 1

    bad_target = dict(target)
    bad_target['neck.proj.weight'] = torch.zeros(4, 3, 1, 1)
    try:
        build_rgbd_feature_transfer_state(source, bad_target)
    except ValueError as error:
        assert 'shape_mismatch' in str(error)
    else:
        raise AssertionError('expected incompatible pretrained neck to fail')


def test_modular_full_configs_change_only_the_neck_and_loss_stage() -> None:
    riou = Config.fromfile(MODULAR_RIOU)
    gwd = Config.fromfile(MODULAR_GWD)

    assert riou.model.neck.type == "ComposablePyramidNeck"
    assert riou.model.neck.mapper.type == "ChannelMapper"
    assert riou.model.neck.refiner.type == "ResidualPyramidRefiner"
    assert riou.model.neck.refiner.beta_init == 0.0
    assert riou.model.bbox_head.loss_iou.type == "RotatedIoULoss"
    assert riou.train_cfg.max_epochs == 5
    assert "neck_target_prefix" not in riou.custom_hooks[0]

    assert gwd.model.neck.type == "ComposablePyramidNeck"
    assert gwd.model.bbox_head.loss_iou.type == "GDLoss"
    assert gwd.model.bbox_head.loss_iou.loss_type == "gwd"
    assert gwd.train_cfg.max_epochs == 15
    assert gwd.custom_hooks == []
    assert gwd.load_from is None

    continuation = Config.fromfile(MODULAR_GWD_CONTINUE20)
    assert continuation.model.neck.type == "ComposablePyramidNeck"
    assert continuation.model.bbox_head.loss_iou.loss_type == "gwd"
    assert continuation.train_cfg.max_epochs == 20
    assert continuation.optim_wrapper.optimizer.muon_lr == 1e-4
    assert continuation.optim_wrapper.optimizer.sf_lr == 1e-5
    assert continuation.param_scheduler[0].milestones == [16]
    assert continuation.load_from is None
    assert continuation.resume is False


def test_dense_config_preserves_pretrained_neck_and_uses_o2m_assignment() -> None:
    riou = Config.fromfile(DENSE_RIOU)
    gwd = Config.fromfile(DENSE_GWD)

    assert riou.model.type == 'RTMDet'
    assert riou.model.backbone.type == 'RGBDResidualBackbone'
    assert riou.model.backbone.beta_init == 0.0
    assert riou.model.neck.type == 'ChannelMapper'
    assert riou.model.neck.in_channels == [384, 768, 1536]
    assert riou.model.bbox_head.type == 'RotatedRTMDetSepBNHead'
    assert riou.model.train_cfg.assigner.type == 'DynamicSoftLabelAssigner'
    assert riou.model.train_cfg.assigner.iou_calculator.type == 'RBboxOverlaps2D'
    assert riou.custom_hooks[0].type == 'RGBDFeatureTransferHook'
    assert riou.model.bbox_head.loss_iou.type == 'RotatedIoULoss'
    assert gwd.model.bbox_head.loss_iou.type == 'GDLoss'
    assert gwd.model.bbox_head.loss_iou.loss_type == 'gwd'


def test_hybrid_config_connects_dense_o2m_to_refined_o2o_queries() -> None:
    riou = Config.fromfile(HYBRID_RIOU)
    gwd = Config.fromfile(HYBRID_GWD)

    assert riou.model.type == 'RotatedRTDETR'
    assert riou.model.as_two_stage is True
    assert riou.model.with_box_refine is True
    assert riou.model.prediction_mode == 'query'
    assert riou.model.backbone.type == 'RGBDResidualBackbone'
    assert riou.model.neck.in_channels == [384, 768, 1536]
    assert riou.model.auxiliary_dense_head.type == 'RotatedRTMDetSepBNHead'
    assert riou.model.auxiliary_dense_head.train_cfg.assigner.type == \
        'DynamicSoftLabelAssigner'
    assert riou.model.bbox_head.type == 'RotatedDeformableDETRHead'
    assert riou.model.train_cfg.assigner.type == 'HungarianAssigner'
    assert riou.model.bbox_head.loss_bbox.type == 'SmoothL1Loss'
    assert gwd.model.auxiliary_dense_head.loss_iou.loss_type == 'gwd'
    assert gwd.model.bbox_head.loss_iou.loss_type == 'gwd'
