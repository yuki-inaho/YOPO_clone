import cv2
import numpy as np
import pytest
import torch

from yopo.datasets.transforms.pose_transform import (
    RandomFlipFor9DPose,
    RandomRotationFor9DPose,
    RandomTranslatePixels,
)
from yopo.structures.bbox import HorizontalBoxes, RotatedBoxes


def _project(intrinsic, translation):
    intrinsic = np.asarray(intrinsic, dtype=np.float64)
    if intrinsic.shape == (4,):
        intrinsic = np.asarray([
            [intrinsic[0], 0, intrinsic[2]],
            [0, intrinsic[1], intrinsic[3]],
            [0, 0, 1],
        ])
    projected = intrinsic @ np.asarray(translation, dtype=np.float64)
    return projected[:2] / projected[2]


@pytest.mark.parametrize(
    ('direction', 'expected_center'),
    [('horizontal', [138.0, 52.0]), ('vertical', [61.0, 47.0])],
)
def test_pose_flip_preserves_projection_and_proper_rotation(
    direction, expected_center
) -> None:
    intrinsic = [100.0, 120.0, 50.0, 40.0]
    translation = np.asarray([[0.22, 0.20, 2.0]], dtype=np.float32)
    center = _project(intrinsic, translation[0])
    assert center.tolist() == pytest.approx([61.0, 52.0])
    rotation = np.eye(3, dtype=np.float32)
    transform = np.eye(4, dtype=np.float32)[None]
    transform[0, :3, :3] = rotation
    transform[0, :3, 3] = translation[0]
    depth = np.arange(100 * 200, dtype=np.float32).reshape(100, 200)
    valid_mask = depth > 10
    gaussian = np.asarray([[61.0, 52.0, 9.0, 2.0, 4.0]],
                          dtype=np.float32)
    results = dict(
        img=np.zeros((100, 200, 4), dtype=np.float32),
        depth=depth.copy(),
        depth_valid_mask=valid_mask.copy(),
        gt_bboxes=HorizontalBoxes([[50.0, 45.0, 72.0, 59.0]]),
        intrinsic=intrinsic.copy(),
        translation=translation.copy(),
        rotation=np.asarray([[1, 0, 0, 0, 1, 0]], dtype=np.float32),
        T=transform.copy(),
        center_2d=center[None].astype(np.float32),
        obb_gaussian=gaussian.copy(),
    )

    transformed = RandomFlipFor9DPose(
        prob=1.0, direction=direction).transform(results)

    np.testing.assert_allclose(
        transformed['center_2d'][0], expected_center, atol=1e-6)
    np.testing.assert_allclose(
        _project(transformed['intrinsic'], transformed['translation'][0]),
        expected_center,
        atol=1e-6,
    )
    np.testing.assert_array_equal(transformed['translation'], translation)
    np.testing.assert_array_equal(transformed['rotation'],
                                  [[1, 0, 0, 0, 1, 0]])
    np.testing.assert_array_equal(transformed['T'], transform)
    assert np.linalg.det(transformed['T'][0, :3, :3]) == pytest.approx(1.0)
    assert transformed['obb_gaussian'][0, 3] == pytest.approx(-2.0)
    if direction == 'horizontal':
        np.testing.assert_array_equal(transformed['depth'], depth[:, ::-1])
        np.testing.assert_array_equal(
            transformed['depth_valid_mask'], valid_mask[:, ::-1])
    else:
        np.testing.assert_array_equal(transformed['depth'], depth[::-1])
        np.testing.assert_array_equal(
            transformed['depth_valid_mask'], valid_mask[::-1])


def test_pose_flip_rejects_unrepresentable_intrinsic() -> None:
    with pytest.raises(ValueError, match='intrinsic must be'):
        RandomFlipFor9DPose(prob=1.0).transform({
            'img': np.zeros((10, 20, 3), dtype=np.uint8),
            'intrinsic': [np.eye(3), np.eye(3)],
        })


def test_pose_translation_uses_one_consistent_image_homography() -> None:
    image = np.zeros((5, 6, 3), dtype=np.uint8)
    image[2, 2] = 255
    depth = np.arange(30, dtype=np.float32).reshape(5, 6)
    valid_mask = depth > 0
    intrinsic = [10.0, 10.0, 2.0, 2.0]
    translation = np.asarray([[0.0, 0.0, 2.0]], dtype=np.float32)
    transform_matrix = np.eye(4, dtype=np.float32)[None]
    transform_matrix[0, :3, 3] = translation[0]
    transform = RandomTranslatePixels(
        prob=1.0, max_translate_offset=2, filter_thr_px=0)
    transform._get_offset = lambda: (2, -1)

    transformed = transform.transform(dict(
        img=image,
        depth=depth.copy(),
        depth_valid_mask=valid_mask.copy(),
        gt_bboxes=HorizontalBoxes([[1.0, 1.0, 3.0, 3.0]]),
        gt_bboxes_labels=np.asarray([0], dtype=np.int64),
        gt_ignore_flags=np.asarray([False]),
        intrinsic=intrinsic,
        translation=translation.copy(),
        rotation=np.asarray([[1, 0, 0, 0, 1, 0]], dtype=np.float32),
        size=np.ones((1, 3), dtype=np.float32),
        T=transform_matrix.copy(),
        center_2d=np.asarray([[2.0, 2.0]], dtype=np.float32),
        z=np.asarray([2.0], dtype=np.float32),
        obb_gaussian=np.asarray([[2.0, 2.0, 9.0, 2.0, 4.0]],
                                dtype=np.float32),
    ))

    assert transformed is not None
    np.testing.assert_array_equal(transformed['img'][1, 4], [255, 255, 255])
    assert transformed['depth'][1, 4] == depth[2, 2]
    assert transformed['depth_valid_mask'][1, 4] == valid_mask[2, 2]
    np.testing.assert_allclose(
        transformed['gt_bboxes'].tensor.numpy(), [[3.0, 0.0, 5.0, 2.0]])
    np.testing.assert_allclose(transformed['center_2d'], [[4.0, 1.0]])
    np.testing.assert_allclose(
        transformed['obb_gaussian'], [[4.0, 1.0, 9.0, 2.0, 4.0]])
    np.testing.assert_allclose(transformed['intrinsic'], [10., 10., 4., 1.])
    np.testing.assert_allclose(
        _project(transformed['intrinsic'], transformed['translation'][0]),
        transformed['center_2d'][0])
    np.testing.assert_array_equal(transformed['translation'], translation)
    np.testing.assert_array_equal(transformed['T'], transform_matrix)
    assert intrinsic == [10.0, 10.0, 2.0, 2.0]


def test_pose_rotation_preserves_3d_and_projects_all_image_fields() -> None:
    image = np.zeros((8, 8, 3), dtype=np.uint8)
    image[2, 5] = 255
    depth = np.zeros((8, 8), dtype=np.float32)
    depth[2, 5] = 7.0
    valid_mask = depth > 0
    intrinsic = [10.0, 10.0, 3.0, 3.0]
    translation = np.asarray([[0.2, 0.1, 2.0]], dtype=np.float32)
    center = _project(intrinsic, translation[0])
    transform_matrix = np.eye(4, dtype=np.float32)[None]
    transform_matrix[0, :3, 3] = translation[0]
    box = RotatedBoxes([[4.0, 3.5, 2.0, 1.0, 0.2]])
    expected_box = box.clone()
    transform = RandomRotationFor9DPose(prob=1.0, max_rotate_degree=90)
    transform._get_rotate_flag = lambda: True
    transform._get_random_rotation_info = lambda: (np.eye(3), 90.0)
    affine = cv2.getRotationMatrix2D((4.0, 4.0), 90.0, 1.0)
    homography = np.vstack((affine, [0.0, 0.0, 1.0])).astype(np.float32)
    expected_box.project_(homography)
    gaussian = np.asarray([[4.0, 3.5, 9.0, 2.0, 4.0]],
                          dtype=np.float32)

    transformed = transform.transform(dict(
        img=image,
        depth=depth,
        depth_valid_mask=valid_mask,
        gt_bboxes=box,
        intrinsic=intrinsic,
        translation=translation.copy(),
        rotation=np.asarray([[1, 0, 0, 0, 1, 0]], dtype=np.float32),
        T=transform_matrix.copy(),
        center_2d=center[None].astype(np.float32),
        obb_gaussian=gaussian,
        z=np.asarray([2.0], dtype=np.float32),
    ))

    center_h = homography @ np.r_[center, 1.0]
    np.testing.assert_allclose(
        transformed['center_2d'][0], center_h[:2], atol=1e-6)
    np.testing.assert_allclose(
        _project(transformed['intrinsic'], transformed['translation'][0]),
        center_h[:2], atol=1e-6)
    torch.testing.assert_close(
        transformed['gt_bboxes'].tensor, expected_box.tensor,
        atol=1e-5, rtol=0)
    linear = homography[:2, :2]
    covariance = np.asarray([[9.0, 2.0], [2.0, 4.0]], dtype=np.float32)
    expected_covariance = linear @ covariance @ linear.T
    np.testing.assert_allclose(
        transformed['obb_gaussian'][0, 2:],
        [expected_covariance[0, 0], expected_covariance[0, 1],
         expected_covariance[1, 1]], atol=1e-6)
    assert np.count_nonzero(transformed['depth']) == 1
    assert transformed['depth_valid_mask'].sum() == 1
    np.testing.assert_array_equal(transformed['translation'], translation)
    np.testing.assert_array_equal(transformed['T'], transform_matrix)
    assert np.linalg.det(transformed['T'][0, :3, :3]) == pytest.approx(1.0)


def test_vertical_obb_flip_matches_reflected_quad_geometry() -> None:
    image_shape = (100, 200)
    box = RotatedBoxes([[61.0, 52.0, 20.0, 8.0, 0.37]])
    original_corners = RotatedBoxes.rbox2corner(box.tensor).squeeze(0)

    box.flip_(image_shape, direction='vertical')
    actual_corners = RotatedBoxes.rbox2corner(box.tensor).squeeze(0)
    expected_corners = original_corners.clone()
    expected_corners[:, 1] = image_shape[0] - expected_corners[:, 1]

    actual_sorted = actual_corners[torch.argsort(
        actual_corners[:, 0] * 1000 + actual_corners[:, 1])]
    expected_sorted = expected_corners[torch.argsort(
        expected_corners[:, 0] * 1000 + expected_corners[:, 1])]
    torch.testing.assert_close(actual_sorted, expected_sorted, atol=1e-5, rtol=0)
