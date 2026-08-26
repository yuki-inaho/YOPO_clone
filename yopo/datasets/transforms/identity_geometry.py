"""Identity image-geometry contracts for native-resolution RGB-D data."""

from __future__ import annotations

import numpy as np
from mmcv.transforms import BaseTransform

from yopo.registry import TRANSFORMS


@TRANSFORMS.register_module()
class AssertIdentityImageGeometry(BaseTransform):
    """Validate native geometry and record an identity scale transform.

    This transform performs no resize, pad, copy, or annotation mutation.  It
    exists to fail fast when a dataset no longer satisfies its declared native
    resolution and to provide ``scale_factor=(1, 1)`` for standard rescaled
    prediction code paths.

    Args:
        image_size (tuple[int, int]): Expected ``(width, height)``.
        channels (int, optional): Expected image channel count.
        validate_depth_fields (bool): Validate standalone ``depth`` and
            ``depth_valid_mask`` against the same spatial shape.
    """

    def __init__(
        self,
        image_size: tuple[int, int],
        channels: int | None = None,
        validate_depth_fields: bool = True,
    ) -> None:
        if len(image_size) != 2 or any(int(v) <= 0 for v in image_size):
            raise ValueError(
                'image_size must be a positive (width, height) pair, '
                f'got {image_size!r}')
        if channels is not None and channels <= 0:
            raise ValueError(f'channels must be positive, got {channels}')
        self.image_size = tuple(int(v) for v in image_size)
        self.channels = channels
        self.validate_depth_fields = bool(validate_depth_fields)

    def transform(self, results: dict) -> dict:
        if 'img' not in results:
            raise KeyError('AssertIdentityImageGeometry requires img')

        image = results['img']
        expected_hw = (self.image_size[1], self.image_size[0])
        if tuple(image.shape[:2]) != expected_hw:
            raise ValueError(
                'native image shape changed; identity geometry cannot resize: '
                f'expected={expected_hw}, actual={tuple(image.shape[:2])}')
        if self.channels is not None:
            actual_channels = 1 if image.ndim == 2 else image.shape[2]
            if actual_channels != self.channels:
                raise ValueError(
                    'native image channel mismatch: '
                    f'expected={self.channels}, actual={actual_channels}')

        if self.validate_depth_fields:
            for key in ('depth', 'depth_valid_mask'):
                value = results.get(key)
                if value is not None and tuple(np.asarray(value).shape[:2]) != expected_hw:
                    raise ValueError(
                        f'{key} is not aligned with the native image: '
                        f'expected={expected_hw}, '
                        f'actual={tuple(np.asarray(value).shape[:2])}')

        ori_shape = results.get('ori_shape')
        if ori_shape is not None and tuple(ori_shape[:2]) != expected_hw:
            raise ValueError(
                'ori_shape disagrees with native image geometry: '
                f'expected={expected_hw}, actual={tuple(ori_shape[:2])}')

        results['img_shape'] = expected_hw
        results['scale'] = self.image_size
        results['scale_factor'] = (1.0, 1.0)
        results['keep_ratio'] = True
        return results

    def __repr__(self) -> str:
        return (
            f'{self.__class__.__name__}(image_size={self.image_size}, '
            f'channels={self.channels}, '
            f'validate_depth_fields={self.validate_depth_fields})'
        )
