"""Data-contract coverage for the custom raw-depth 3D BBOX pipeline."""

from __future__ import annotations

import cv2
import numpy as np
import pytest
import torch
from mmengine.config import Config
from mmengine.utils import import_modules_from_strings

from yopo.registry import DATASETS
from yopo.utils import register_all_modules


CONFIG_PATH = "configs/yopo/nocs_custom_fruit_rgbd_3dbbox_transfer.py"
CUSTOM_INTRINSIC = [443.9066, 449.1953, 321.3503, 230.8687]


def test_concat_raw_depth_can_swap_only_rgb_channels():
    from yopo.datasets.transforms.raw_depth import ConcatRawDepthToImage

    result = ConcatRawDepthToImage(
        depth_scale=255.0,
        bgr_to_rgb=True,
    ).transform(
        {
            "img": np.array([[[10, 20, 30]]], dtype=np.uint8),
            "depth": np.array([[0.5]], dtype=np.float32),
            "depth_valid_mask": np.array([[True]]),
        }
    )

    assert result["img"].shape == (1, 1, 4)
    assert result["img"][0, 0].tolist() == pytest.approx(
        [30.0, 20.0, 10.0, 127.5]
    )


def test_rgbd_3dbbox_sample_contract():
    """RGB/raw-depth, K and 3D pose targets must survive one val sample."""
    from yopo.datasets.transforms.raw_depth import LoadRawDepthImageWithValidMask

    register_all_modules()
    config = Config.fromfile(CONFIG_PATH)
    import_modules_from_strings(**config.custom_imports)
    dataset = DATASETS.build(config.val_dataloader.dataset)

    raw_info = dataset.get_data_info(0)
    raw_depth = cv2.imread(raw_info["depth_path"], cv2.IMREAD_UNCHANGED)
    depth_result = LoadRawDepthImageWithValidMask().transform(
        {"depth_path": raw_info["depth_path"]}
    )
    assert depth_result["depth"].dtype == np.float32
    assert depth_result["depth_valid_mask"].dtype == np.bool_
    assert np.array_equal(depth_result["depth_valid_mask"], raw_depth > 0)
    assert depth_result["depth"].min() >= 0.0
    assert np.isclose(
        depth_result["depth"][raw_depth > 0].max(), raw_depth[raw_depth > 0].max() / 1000.0
    )

    packed = next(dataset[index] for index in range(len(dataset))
                  if len(dataset[index]["data_samples"].gt_instances) > 0)
    inputs = packed["inputs"]
    instances = packed["data_samples"].gt_instances
    metainfo = packed["data_samples"].metainfo

    assert inputs.shape == (4, 480, 640)
    assert inputs.dtype == torch.float32
    assert np.allclose(metainfo["intrinsic"], CUSTOM_INTRINSIC)
    assert instances.labels.ndim == 1
    assert instances.translations.shape[-1] == 3
    assert instances.rotations.shape[-1] == 6
    assert instances.sizes.shape[-1] == 3
    assert torch.isfinite(instances.translations).all()
    assert torch.isfinite(instances.rotations).all()
    assert torch.isfinite(instances.sizes).all()
