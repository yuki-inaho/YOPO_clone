import numpy as np

from yopo.datasets.transforms import YOLOXHSVRandomAug


def test_yolox_hsv_accepts_negative_stride_image_after_flip() -> None:
    original = np.arange(12 * 16 * 3, dtype=np.uint8).reshape(12, 16, 3)
    flipped = original[::-1]
    assert not flipped.flags.c_contiguous

    results = YOLOXHSVRandomAug().transform(dict(img=flipped))

    assert results['img'].shape == flipped.shape
    assert results['img'].dtype == np.uint8
    assert results['img'].flags.c_contiguous


def test_yolox_hsv_does_not_mutate_annotation_fields() -> None:
    boxes = np.asarray([[8.0, 6.0, 4.0, 2.0, 0.3]], dtype=np.float32)
    labels = np.asarray([0], dtype=np.int64)
    results = dict(
        img=np.full((12, 16, 3), 127, dtype=np.uint8),
        gt_bboxes=boxes.copy(),
        gt_bboxes_labels=labels.copy(),
    )

    transformed = YOLOXHSVRandomAug().transform(results)

    np.testing.assert_array_equal(transformed['gt_bboxes'], boxes)
    np.testing.assert_array_equal(transformed['gt_bboxes_labels'], labels)
