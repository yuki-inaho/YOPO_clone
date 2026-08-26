from pathlib import Path

import cv2
import numpy as np
import pytest
from mmengine.config import Config

from yopo.datasets import DOTAOBBDataset, DOTATomatoDataset


def _write_image(path: Path, shape=(24, 32)) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    image = np.zeros((*shape, 3), dtype=np.uint8)
    assert cv2.imwrite(str(path), image)


def _write_depth(path: Path, shape=(24, 32)) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    depth = np.full(shape, 750, dtype=np.uint16)
    assert cv2.imwrite(str(path), depth)


def _write_label(path: Path, text: str) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(text)


def _dataset(root: Path, **kwargs) -> DOTAOBBDataset:
    return DOTAOBBDataset(
        data_root=str(root),
        ann_file="labels",
        data_prefix=dict(img_path="images"),
        metainfo=dict(classes=("tomato",)),
        pipeline=[],
        serialize_data=False,
        **kwargs,
    )


def test_generic_dota_obb_loader_supports_classes_suffixes_and_auto_shape(
    tmp_path: Path,
) -> None:
    _write_image(tmp_path / "images" / "sample.png", shape=(24, 32))
    _write_label(
        tmp_path / "labels" / "sample.txt",
        "1 2 9 2 9 10 1 10 tomato 0\n",
    )

    dataset = _dataset(
        tmp_path,
        img_shape=None,
        img_suffixes=(".png", ".jpg"),
        strict_loading=True,
    )
    info = dataset.get_data_info(0)

    assert len(dataset) == 1
    assert info["width"] == 32
    assert info["height"] == 24
    assert info["img_path"].endswith("sample.png")
    assert info["instances"][0]["bbox_label"] == 0
    assert len(info["instances"][0]["bbox"]) == 5
    assert np.isfinite(info["instances"][0]["bbox"]).all()


def test_strict_loader_accepts_uppercase_suffix_and_checks_declared_shape(
    tmp_path: Path,
) -> None:
    _write_image(tmp_path / "images" / "sample.JPG", shape=(24, 32))
    _write_label(
        tmp_path / "labels" / "sample.txt",
        "1 2 9 2 9 10 1 10 tomato 0\n",
    )

    dataset = _dataset(
        tmp_path,
        img_shape=(24, 32),
        img_suffixes=("jpg",),
        strict_loading=True,
    )
    assert len(dataset) == 1

    with pytest.raises(ValueError, match="image shape mismatch"):
        _dataset(
            tmp_path,
            img_shape=(48, 64),
            img_suffixes=(".jpg",),
            strict_loading=True,
        )


@pytest.mark.parametrize(
    ("label", "message"),
    [
        ("1 2 9 2 9 10 1 10 stem 0\n", "unknown class"),
        ("1 2 9 2 tomato 0\n", "9 or 10 columns"),
        ("1 2 nan 2 9 10 1 10 tomato 0\n", "non-finite"),
        ("1 2 1 2 1 2 1 2 tomato 0\n", "degenerate"),
    ],
)
def test_strict_dota_obb_loader_rejects_invalid_rows(
    tmp_path: Path, label: str, message: str
) -> None:
    _write_image(tmp_path / "images" / "sample.jpg")
    _write_label(tmp_path / "labels" / "sample.txt", label)

    with pytest.raises(ValueError, match=message):
        _dataset(tmp_path, img_shape=(24, 32), strict_loading=True)


def test_strict_dota_obb_loader_rejects_unpaired_files(tmp_path: Path) -> None:
    _write_image(tmp_path / "images" / "image_only.jpg")
    _write_label(
        tmp_path / "labels" / "label_only.txt",
        "1 2 9 2 9 10 1 10 tomato 0\n",
    )

    with pytest.raises(FileNotFoundError, match="image/label stem mismatch"):
        _dataset(tmp_path, img_shape=(24, 32), strict_loading=True)


def test_strict_dota_obb_loader_adds_paired_depth_path(tmp_path: Path) -> None:
    _write_image(tmp_path / "images" / "sample.jpg")
    _write_depth(tmp_path / "depth" / "sample.png")
    _write_label(
        tmp_path / "labels" / "sample.txt",
        "1 2 9 2 9 10 1 10 tomato 0\n",
    )

    dataset = DOTAOBBDataset(
        data_root=str(tmp_path),
        ann_file="labels",
        data_prefix=dict(img_path="images", depth_path="depth"),
        metainfo=dict(classes=("tomato",)),
        img_shape=(24, 32),
        img_suffixes=(".jpg",),
        depth_suffixes=(".png",),
        strict_loading=True,
        pipeline=[],
        serialize_data=False,
    )

    assert dataset.get_data_info(0)["depth_path"].endswith("sample.png")


def test_strict_dota_obb_loader_rejects_missing_depth(tmp_path: Path) -> None:
    _write_image(tmp_path / "images" / "sample.jpg")
    (tmp_path / "depth").mkdir()
    _write_label(
        tmp_path / "labels" / "sample.txt",
        "1 2 9 2 9 10 1 10 tomato 0\n",
    )

    with pytest.raises(FileNotFoundError, match="RGB/depth/label stem mismatch"):
        DOTAOBBDataset(
            data_root=str(tmp_path),
            ann_file="labels",
            data_prefix=dict(img_path="images", depth_path="depth"),
            metainfo=dict(classes=("tomato",)),
            img_shape=(24, 32),
            strict_loading=True,
            pipeline=[],
            serialize_data=False,
        )


def test_strict_dota_obb_loader_rejects_depth_shape_mismatch(
    tmp_path: Path,
) -> None:
    _write_image(tmp_path / "images" / "sample.jpg")
    _write_depth(tmp_path / "depth" / "sample.png", shape=(12, 16))
    _write_label(
        tmp_path / "labels" / "sample.txt",
        "1 2 9 2 9 10 1 10 tomato 0\n",
    )

    with pytest.raises(ValueError, match="RGB/depth shape mismatch"):
        DOTAOBBDataset(
            data_root=str(tmp_path),
            ann_file="labels",
            data_prefix=dict(img_path="images", depth_path="depth"),
            metainfo=dict(classes=("tomato",)),
            img_shape=(24, 32),
            strict_loading=True,
            pipeline=[],
            serialize_data=False,
        )


def test_legacy_tomato_loader_remains_registered_with_stem_class() -> None:
    assert DOTATomatoDataset.METAINFO["classes"] == ("stem",)


def test_corrected_dota_riou_stage_is_rgb_only_strict_and_query_complete() -> None:
    cfg = Config.fromfile(
        "configs/yopo/"
        "rotated_deformable_detr_tomato_obb_corrected_riou_stage1.py"
    )

    assert cfg.model.type == "DeformableDETR"
    assert cfg.model.num_queries == 150
    assert cfg.model.backbone.in_channels == 3
    assert cfg.model.backbone.freeze_at == 0
    assert cfg.model.backbone.freeze_stem_only is True
    assert cfg.model.bbox_head.type == "RotatedDeformableDETRHead"
    assert cfg.model.bbox_head.loss_iou.type == "RotatedIoULoss"
    assert cfg.model.bbox_head.loss_iou.mode == "linear"
    assert "depth" not in repr(cfg.model).lower()

    train = cfg.train_dataloader.dataset
    val = cfg.val_dataloader.dataset
    assert train.type == "DOTAOBBDataset"
    assert val.type == "DOTAOBBDataset"
    assert tuple(train.metainfo.classes) == ("tomato",)
    assert train.strict_loading is True
    assert val.strict_loading is True
    assert tuple(train.img_shape) == (600, 800)
    assert train.ann_file == "train/labels"
    assert train.data_prefix.img_path == "train/images"
    assert val.ann_file == "test/labels"
    assert val.data_prefix.img_path == "test/images"
    assert cfg.train_cfg.max_epochs == 5
    assert cfg.train_cfg.val_interval == 1
    assert cfg.train_dataloader.batch_size == 32
    assert cfg.load_from.endswith("rddetr_tomato_riou_epoch20_q150.pth")


def test_corrected_dota_gwd_stage_changes_only_refinement_policy() -> None:
    stage1 = Config.fromfile(
        "configs/yopo/"
        "rotated_deformable_detr_tomato_obb_corrected_riou_stage1.py"
    )
    stage2 = Config.fromfile(
        "configs/yopo/"
        "rotated_deformable_detr_tomato_obb_corrected_gwd_stage2.py"
    )

    assert stage2.model.bbox_head.loss_iou.type == "GDLoss"
    assert stage2.model.bbox_head.loss_iou.loss_type == "gwd"
    assert stage2.model.train_cfg == stage1.model.train_cfg
    assert stage2.train_dataloader.dataset == stage1.train_dataloader.dataset
    assert stage2.val_dataloader.dataset == stage1.val_dataloader.dataset
    assert stage2.train_cfg.max_epochs == 15
    assert stage2.optim_wrapper.optimizer.muon_lr == pytest.approx(5e-5)
    assert stage2.optim_wrapper.optimizer.sf_lr == pytest.approx(2.5e-6)
    assert stage2.load_from.endswith("selected_best.pth")

    inference = Config.fromfile(
        "configs/yopo/"
        "rotated_deformable_detr_tomato_obb_corrected_inference.py"
    )
    assert inference.load_from is None
    assert inference.resume is False


def test_corrected_dota_weak_da_matches_reference_policy() -> None:
    stage1 = Config.fromfile(
        "configs/yopo/"
        "rotated_deformable_detr_tomato_obb_corrected_da_weak_riou_stage1.py"
    )
    pipeline = stage1.train_dataloader.dataset.pipeline

    assert [transform.type for transform in pipeline] == [
        "LoadImageFromFile",
        "LoadAnnotations",
        "Resize",
        "RandomFlip",
        "YOLOXHSVRandomAug",
        "PackDetInputs",
    ]
    assert pipeline[3].prob == pytest.approx(0.75)
    assert pipeline[3].direction == "vertical"
    assert pipeline[4].hue_delta == 5
    assert pipeline[4].saturation_delta == 30
    assert pipeline[4].value_delta == 30
    assert all(
        transform.type != "YOLOXHSVRandomAug"
        for transform in stage1.val_dataloader.dataset.pipeline
    )
    assert stage1.val_evaluator.score_thr == pytest.approx(0.05)

    stage2 = Config.fromfile(
        "configs/yopo/"
        "rotated_deformable_detr_tomato_obb_corrected_da_weak_gwd_stage2.py"
    )
    assert stage2.train_dataloader.dataset == stage1.train_dataloader.dataset
    assert stage2.model.bbox_head.loss_iou.type == "GDLoss"
    assert stage2.model.bbox_head.loss_iou.loss_type == "gwd"
    assert stage2.load_from.endswith(
        "rddetr_tomato_obb_corrected_da_weak_riou_stage1/selected_best.pth"
    )
