"""Native 736x512 RGB-D input and identity-geometry contracts."""

from pathlib import Path

import numpy as np
import pytest
import torch
from mmengine.config import Config
from mmengine.utils import import_modules_from_strings

from yopo.datasets.transforms.identity_geometry import (
    AssertIdentityImageGeometry,
)
from yopo.registry import DATASETS, MODELS
from yopo.structures.bbox import BaseBoxes, HorizontalBoxes
from yopo.utils import register_all_modules


CONFIG_ROOT = Path('configs/yopo')
BASE = CONFIG_ROOT / (
    'nocs_fruits_detection_Jun30_2025_rgbd_3dbbox_'
    'stage12_native736x512_base.py'
)
CAPACITY = CONFIG_ROOT / (
    'nocs_fruits_detection_Jun30_2025_rgbd_3dbbox_'
    'stage12_native736x512_kfiou_capacity_smoke.py'
)
FULL = CONFIG_ROOT / (
    'nocs_fruits_detection_Jun30_2025_rgbd_3dbbox_'
    'stage12_native736x512_kfiou_full.py'
)


def _identity_results() -> dict:
    return dict(
        img=np.zeros((512, 736, 4), dtype=np.float32),
        depth=np.ones((512, 736), dtype=np.float32),
        depth_valid_mask=np.ones((512, 736), dtype=bool),
        ori_shape=(512, 736),
        gt_bboxes=HorizontalBoxes([[10.0, 20.0, 100.0, 120.0]]),
        obb_gaussian=np.array(
            [[55.0, 70.0, 400.0, 15.0, 625.0]], dtype=np.float32),
        intrinsic=[500.0, 501.0, 320.0, 256.0],
        center_2d=np.array([[55.0, 70.0]], dtype=np.float32),
        rotation=np.array(
            [[1.0, 0.0, 0.0, 0.0, 1.0, 0.0]], dtype=np.float32),
        translation=np.array([[0.1, 0.2, 0.8]], dtype=np.float32),
        size=np.array([[0.05, 0.06, 0.07]], dtype=np.float32),
        z=np.array([[0.8]], dtype=np.float32),
        T=np.eye(4, dtype=np.float32)[None],
    )


def test_identity_geometry_records_metadata_without_mutating_any_field() -> None:
    transform = AssertIdentityImageGeometry(
        image_size=(736, 512), channels=4)
    results = _identity_results()
    untouched = {
        key: value for key, value in results.items()
        if key not in {'img_shape', 'scale', 'scale_factor', 'keep_ratio'}
    }

    transformed = transform(results)

    assert transformed is results
    for key, original in untouched.items():
        assert transformed[key] is original
    assert transformed['img_shape'] == (512, 736)
    assert transformed['scale'] == (736, 512)
    assert transformed['scale_factor'] == (1.0, 1.0)
    assert transformed['keep_ratio'] is True


@pytest.mark.parametrize(
    ('field', 'value', 'message'),
    [
        ('img', np.zeros((511, 736, 4), dtype=np.float32),
         'native image shape changed'),
        ('depth', np.zeros((511, 736), dtype=np.float32),
         'depth is not aligned'),
        ('depth_valid_mask', np.zeros((512, 735), dtype=bool),
         'depth_valid_mask is not aligned'),
    ],
)
def test_identity_geometry_fails_fast_instead_of_resizing(
        field, value, message) -> None:
    results = _identity_results()
    results[field] = value
    transform = AssertIdentityImageGeometry(
        image_size=(736, 512), channels=4)
    with pytest.raises(ValueError, match=message):
        transform(results)


def test_native_configs_remove_resize_and_pad_from_train_and_validation(
        ) -> None:
    base = Config.fromfile(BASE)
    full = Config.fromfile(FULL)
    capacity = Config.fromfile(CAPACITY)

    for cfg in (base, full, capacity):
        assert cfg.model.data_preprocessor.pad_size_divisor == 1
        train_names = [
            step.type for step in cfg.train_dataloader.dataset.pipeline]
        assert train_names == [
            'LoadImageFromFile',
            'LoadRawDepthImageWithValidMask',
            'Load9DPoseAnnotations',
            'ConcatRawDepthToImage',
            'AssertIdentityImageGeometry',
            'RandomFlipFor9DPose',
            'FilterAnnotations',
            'Pack9DPoseInputs',
        ]
        assert not {'Resize', 'ResizeforPose', 'ResizeOBBGaussians', 'Pad'}.intersection(
            train_names)

        val_loader = cfg.get('val_dataloader')
        if val_loader is not None:
            assert cfg.val_evaluator.two_phase_3d_iou is True
            val_names = [step.type for step in val_loader.dataset.pipeline]
            assert val_names == [
                'LoadImageFromFile',
                'LoadRawDepthImageWithValidMask',
                'Load9DPoseAnnotations',
                'ConcatRawDepthToImage',
                'AssertIdentityImageGeometry',
                'Pack9DPoseInputs',
            ]

    assert base.train_dataloader.batch_size == 16
    assert full.train_dataloader.batch_size == 16
    assert full.load_from is None
    assert full.resume is False
    assert full.train_cfg.max_epochs == 50
    assert capacity.train_dataloader.batch_size == 16
    assert capacity.train_cfg.type == 'IterBasedTrainLoop'
    assert capacity.train_cfg.max_iters == 2
    assert capacity.val_dataloader is None
    assert capacity.val_evaluator is None


def _as_tensor(value):
    return value.tensor if isinstance(value, BaseBoxes) else value


@pytest.mark.parametrize(
    ('loader_name', 'training'),
    [('train_dataloader', True), ('val_dataloader', False)],
)
def test_real_train_and_validation_batches_are_4x512x736_and_geometry_stable(
        loader_name: str, training: bool) -> None:
    register_all_modules()
    cfg = Config.fromfile(FULL)
    import_modules_from_strings(**cfg.custom_imports)
    loader_cfg = cfg[loader_name]
    dataset = DATASETS.build(loader_cfg.dataset)
    packed = dataset[0]

    assert packed['inputs'].shape == (4, 512, 736)
    sample = packed['data_samples']
    assert tuple(sample.scale_factor) == (1.0, 1.0)
    intrinsic_before = list(sample.intrinsic)
    geometry_before = {
        key: _as_tensor(value).clone()
        for key, value in sample.gt_instances.items()
        if key in {
            'bboxes', 'obb_gaussians', 'centers_2d', 'rotations',
            'translations', 'sizes', 'z', 'T'
        }
    }

    preprocessor = MODELS.build(cfg.model.data_preprocessor)
    output = preprocessor(
        dict(inputs=[packed['inputs']], data_samples=[sample]),
        training=training,
    )

    assert output['inputs'].shape == (1, 4, 512, 736)
    expected = (
        packed['inputs'].float() - preprocessor.mean.cpu()
    ) / preprocessor.std.cpu()
    torch.testing.assert_close(output['inputs'][0], expected)

    processed_sample = output['data_samples'][0]
    assert processed_sample.img_shape == (512, 736)
    assert processed_sample.pad_shape == (512, 736)
    assert processed_sample.batch_input_shape == (512, 736)
    assert processed_sample.intrinsic == intrinsic_before
    for key, expected_geometry in geometry_before.items():
        torch.testing.assert_close(
            _as_tensor(processed_sample.gt_instances[key]),
            expected_geometry,
        )
