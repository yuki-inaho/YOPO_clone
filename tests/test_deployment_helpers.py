from pathlib import Path

import cv2
import numpy as np

from yopo.deployment.onnx import postprocess_pose_outputs
from yopo.deployment.tomato import (discover_rgbd_samples,
                                    inspect_rgbd_sample)


def test_postprocess_recovers_camera_translation_and_original_coordinates():
    outputs = [
        np.array([[[10.0, -10.0]]], dtype=np.float32),
        np.array([[[0.5, 0.5, 0.4, 0.2]]], dtype=np.float32),
        np.array([[[0.5, 0.5]]], dtype=np.float32),
        np.array([[[2.0]]], dtype=np.float32),
        np.array([[[1.0, 0.0, 0.0, 0.0, 1.0, 0.0]]], dtype=np.float32),
        np.array([[[0.2, 0.3, 0.4]]], dtype=np.float32),
    ]
    prediction = postprocess_pose_outputs(
        outputs,
        intrinsic=[100.0, 100.0, 50.0, 40.0],
        original_shape=(160, 120),
        input_size=(100, 80),
        num_classes=2,
        max_per_img=1,
        classwise_rotation=False,
        classwise_sizes=False,
    )
    assert prediction['labels'].tolist() == [0]
    np.testing.assert_allclose(
        prediction['bboxes'], [[36.0, 64.0, 84.0, 96.0]], rtol=1e-6,
        atol=1e-5)
    np.testing.assert_allclose(
        prediction['centers_2d'], [[60.0, 80.0]], rtol=1e-6, atol=1e-6)
    np.testing.assert_allclose(
        prediction['translations'], [[0.2, 0.8, 2.0]], rtol=1e-6, atol=1e-6)


def test_tomato_pair_discovery_supports_rgb_depth_suffixes(tmp_path: Path):
    scene = tmp_path / 'scene-01'
    (scene / 'rgb').mkdir(parents=True)
    (scene / 'depth').mkdir()
    (scene / 'camera_parameters').mkdir()
    rgb_path = scene / 'rgb' / 'frame_rgb.jpg'
    depth_path = scene / 'depth' / 'frame_depth.png'
    assert cv2.imwrite(str(rgb_path), np.zeros((6, 8, 3), dtype=np.uint8))
    assert cv2.imwrite(str(depth_path), np.ones((4, 5), dtype=np.uint16))
    (scene / 'camera_parameters' / 'rgb_camera_param.yaml').write_text(
        'K: [100, 0, 4, 0, 100, 3, 0, 0, 1]\n', encoding='utf-8')

    samples = discover_rgbd_samples(str(tmp_path), max_scenes=1)
    assert len(samples) == 1
    assert samples[0]['depth_path'] == str(depth_path)
    assert samples[0]['rgb_intrinsic'] == [100.0, 100.0, 4.0, 3.0]
    inspected = inspect_rgbd_sample(samples[0])
    assert inspected['rgb_shape'] == [6, 8, 3]
    assert inspected['depth_shape'] == [4, 5]
    assert inspected['depth_dtype'] == 'uint16'
    assert inspected['depth_valid_fraction'] == 1.0
