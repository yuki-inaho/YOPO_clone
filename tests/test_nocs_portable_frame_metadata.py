from __future__ import annotations

import pickle
from pathlib import Path

import cv2
import numpy as np
import pytest
from mmengine.config import Config

from yopo.datasets.pose_estimation.nocs_custom_fruit_dataset import (
    NOCSCustomFruitDataset,
)


def _write_portable_sample(root: Path) -> None:
    scene = root / "real" / "scene_train_camera_a"
    scene.mkdir(parents=True)
    assert cv2.imwrite(
        str(scene / "0000_color.png"), np.zeros((12, 20, 3), dtype=np.uint8)
    )
    assert cv2.imwrite(
        str(scene / "0000_depth.png"), np.full((12, 20), 800, dtype=np.uint16)
    )
    label = {
        "class_ids": np.ones(1, dtype=np.int32),
        "instance_ids": np.ones(1, dtype=np.int32),
        "bboxes": np.asarray([[2.0, 3.0, 8.0, 13.0]], dtype=np.float32),
        "translations": np.asarray([[0.1, 0.2, 1.0]], dtype=np.float32),
        "rotations": np.eye(3, dtype=np.float32)[None],
        "sizes": np.asarray([[0.06, 0.05, 0.04]], dtype=np.float32),
        "scales": np.ones(1, dtype=np.float32),
        "obb_cxcywha_rad": np.asarray([[8.0, 5.0, 10.0, 6.0, 0.0]], dtype=np.float32),
        "intrinsic": np.asarray([100.0, 120.0, 10.0, 6.0], dtype=np.float32),
        "image_size_wh": np.asarray([20, 12], dtype=np.int32),
        "obb_coordinate_scale": 1.0,
    }
    with (scene / "0000_label.pkl").open("wb") as stream:
        pickle.dump(label, stream)
    (root / "real" / "train_list.txt").write_text(
        "scene_train_camera_a/0000\n", encoding="utf-8"
    )


def test_portable_label_overrides_split_k_shape_and_legacy_obb_scale(tmp_path: Path) -> None:
    _write_portable_sample(tmp_path)

    dataset = NOCSCustomFruitDataset(
        data_root=str(tmp_path),
        split="real_train",
        intrinsic=[1.0, 1.0, 0.0, 0.0],
        obb_coordinate_scale=0.8,
        pipeline=[],
        serialize_data=False,
    )
    info = dataset.get_data_info(0)
    instance = info["instances"][0]

    assert info["intrinsic"] == [100.0, 120.0, 10.0, 6.0]
    assert (info["width"], info["height"]) == (20, 12)
    assert instance["center_2d"] == pytest.approx([20.0, 30.0])
    # Metadata scale 1.0 wins over the legacy constructor fallback 0.8.
    assert instance["obb_gaussian"][:2].tolist() == pytest.approx([8.0, 5.0])
    assert instance["obb_gaussian"][[2, 4]].tolist() == pytest.approx([25.0, 9.0])


def test_dataset_intrinsic_override_does_not_mutate_split_defaults(tmp_path: Path) -> None:
    _write_portable_sample(tmp_path)
    original = list(NOCSCustomFruitDataset.SPLIT_INFO["real_train"]["intrinsic"])

    NOCSCustomFruitDataset(
        data_root=str(tmp_path),
        split="real_train",
        intrinsic=[9.0, 8.0, 7.0, 6.0],
        pipeline=[],
        serialize_data=False,
    )

    assert NOCSCustomFruitDataset.SPLIT_INFO["real_train"]["intrinsic"] == original


def test_portable_jun30_config_targets_generated_dataset() -> None:
    cfg = Config.fromfile(
        "configs/yopo/nocs_fruits_detection_Jun30_2025_rgbd_3dbbox.py"
    )

    expected = "data/fruits_detection_Jun30-2025_stem_rgbd_736x512/"
    assert cfg.train_dataloader.dataset.data_root == expected
    assert cfg.val_dataloader.dataset.data_root == expected


def test_portable_2026_config_targets_generated_dataset() -> None:
    cfg = Config.fromfile(
        "configs/yopo/nocs_fruit_obb_rgbd_2026_800x600_3dbbox.py"
    )
    expected = "data/fruit_obb_rgbd_train1345-test181_20260825_yopo_3dobb_800x600/"
    assert cfg.train_dataloader.dataset.data_root == expected
    assert cfg.val_dataloader.dataset.data_root == expected


def test_portable_2026_resized_config_targets_generated_dataset() -> None:
    cfg = Config.fromfile(
        "configs/yopo/nocs_fruit_obb_rgbd_2026_736x512_3dbbox.py"
    )
    expected = "data/fruit_obb_rgbd_train1345-test181_20260825_yopo_3dobb_736x512/"
    assert cfg.train_dataloader.dataset.data_root == expected
    assert cfg.val_dataloader.dataset.data_root == expected
    assert cfg.train_dataloader.dataset.type == "NOCSCustomFruitDataset"
