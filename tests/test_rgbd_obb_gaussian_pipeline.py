from __future__ import annotations

import numpy as np
import torch
from mmengine.config import Config

from yopo.datasets.transforms.pose_transform import (
    RandomFlipFor9DPose,
    ResizeOBBGaussians,
)
from yopo.registry import DATASETS
from yopo.utils import register_all_modules


CONFIG_PATH = "configs/yopo/nocs_custom_fruit_rgbd_3dbbox_transfer.py"


def _unpack(compact: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    center = compact[:2]
    sigma = np.array(
        [[compact[2], compact[3]], [compact[3], compact[4]]],
        dtype=np.float64,
    )
    return center, sigma


def test_custom_dataset_scales_raw_obb_to_image_gaussian():
    register_all_modules()
    cfg = Config.fromfile(CONFIG_PATH)
    dataset = DATASETS.build(cfg.val_dataloader.dataset)
    info = dataset.get_data_info(0)
    instance = info["instances"][0]

    center, sigma = _unpack(np.asarray(instance["obb_gaussian"]))
    bbox = np.asarray(instance["bbox"])
    bbox_center = (bbox[:2] + bbox[2:]) * 0.5

    np.testing.assert_allclose(center, bbox_center, atol=5e-5)
    assert np.linalg.eigvalsh(sigma).min() > 0


def test_resize_obb_gaussian_applies_affine_covariance_rule():
    compact = np.array([[10.0, 20.0, 9.0, 2.0, 4.0]], dtype=np.float32)
    result = ResizeOBBGaussians().transform(
        {"obb_gaussian": compact.copy(), "scale_factor": (2.0, 3.0)})
    center, sigma = _unpack(result["obb_gaussian"][0])

    np.testing.assert_allclose(center, [20.0, 60.0])
    np.testing.assert_allclose(sigma, [[36.0, 12.0], [12.0, 36.0]])


def test_horizontal_flip_obb_gaussian_reflects_center_and_cross_covariance():
    compact = np.array([[10.0, 20.0, 9.0, 2.0, 4.0]], dtype=np.float32)
    result = RandomFlipFor9DPose(prob=1.0).transform({
        "img": np.zeros((100, 200, 3), dtype=np.uint8),
        "obb_gaussian": compact.copy(),
    })
    center, sigma = _unpack(result["obb_gaussian"][0])

    np.testing.assert_allclose(center, [189.0, 20.0])
    np.testing.assert_allclose(sigma, [[9.0, -2.0], [-2.0, 4.0]])


def test_packed_val_sample_contains_finite_spd_obb_gaussians():
    register_all_modules()
    cfg = Config.fromfile(CONFIG_PATH)
    dataset = DATASETS.build(cfg.val_dataloader.dataset)
    packed = dataset[0]["data_samples"].gt_instances

    assert packed.obb_gaussians.shape[-1] == 5
    assert torch.isfinite(packed.obb_gaussians).all()
    sigma = torch.stack(
        (packed.obb_gaussians[:, 2], packed.obb_gaussians[:, 3],
         packed.obb_gaussians[:, 3], packed.obb_gaussians[:, 4]),
        dim=-1,
    ).reshape(-1, 2, 2)
    assert torch.linalg.eigvalsh(sigma).min() > 0
