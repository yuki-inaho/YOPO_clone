"""Regression coverage for the custom single-class fruit pose contract."""

from __future__ import annotations

import pickle
from pathlib import Path

import numpy as np
from mmengine.config import Config

from yopo.registry import DATASETS
from yopo.utils import register_all_modules


CONFIG_PATH = "configs/yopo/nocs_custom_real_hgnetv2_rgbd_deim_cop_dual.py"


def _build_stock_fruit_dataset():
    """Build the currently inherited NOCS dataset before the fruit adapter."""
    register_all_modules()
    cfg = Config.fromfile(CONFIG_PATH)
    return DATASETS.build(cfg.train_dataloader.dataset)


def test_nocs_rotation_finite_contract():
    """Keep parsed targets aligned with the current finite raw annotations.

    Dataset revisions may change the number of fruit instances.  The contract
    is therefore derived from the raw label files rather than a historical
    hard-coded count.
    """
    dataset = _build_stock_fruit_dataset()
    target_count = 0
    nonfinite_target_count = 0
    raw_rotation_count = 0

    for sample_index in range(len(dataset)):
        data_info = dataset.get_data_info(sample_index)
        label_path = Path(data_info["img_path"].replace("_color.png", "_label.pkl"))
        with label_path.open("rb") as label_file:
            raw_label = pickle.load(label_file)
        raw_rotations = np.asarray(raw_label["rotations"], dtype=np.float32)
        assert np.isfinite(raw_rotations).all(), label_path
        raw_rotation_count += len(raw_rotations)

        for instance in data_info["instances"]:
            target_count += 1
            nonfinite_target_count += int(
                not np.isfinite(np.asarray(instance["rotation"], dtype=np.float32)).all()
            )

    assert raw_rotation_count == target_count
    assert target_count > 0
    assert nonfinite_target_count == 0


def test_custom_fruit_dataset_keeps_all_rotation_targets_finite():
    """The dedicated fruit adapter must not inherit stock NOCS symmetry IDs."""
    from yopo.datasets.pose_estimation.nocs_custom_fruit_dataset import (
        NOCSCustomFruitDataset,
    )

    register_all_modules()
    cfg = Config.fromfile(CONFIG_PATH)
    dataset_cfg = cfg.train_dataloader.dataset.copy()
    dataset_cfg.type = "NOCSCustomFruitDataset"
    dataset = DATASETS.build(dataset_cfg)

    assert isinstance(dataset, NOCSCustomFruitDataset)
    assert dataset.sym_ids == []

    target_count = 0
    for sample_index in range(len(dataset)):
        for instance in dataset.get_data_info(sample_index)["instances"]:
            target_count += 1
            assert np.isfinite(np.asarray(instance["rotation"], dtype=np.float32)).all()

    assert target_count > 0
