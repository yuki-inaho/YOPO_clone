"""Discovery and validation helpers for the standard tomato RGB-D layout."""

from __future__ import annotations

from pathlib import Path
from typing import Dict, List, Optional

import cv2
import numpy as np
import yaml


def _read_intrinsic(path: Path) -> List[float]:
    with path.open('r', encoding='utf-8') as stream:
        document = yaml.safe_load(stream) or {}
    matrix = document.get('K') or document.get('P')
    if matrix is None or len(matrix) < 9:
        raise ValueError('camera file has no 3x3 K/P matrix: {}'.format(path))
    matrix = [float(value) for value in matrix[:9]]
    return [matrix[0], matrix[4], matrix[2], matrix[5]]


def discover_rgbd_samples(root: str, max_scenes: Optional[int] = 10
                          ) -> List[Dict]:
    """Return one valid RGB-D pair from each scene directory.

    The standard dataset stores RGB and depth in sibling directories. In the
    observed exports, pairs use either the same stem with different image
    extensions or the ``*_rgb``/``*_depth`` suffix convention. Subsampled
    directories are treated as separate, directly addressable dataset roots.
    """
    root_path = Path(root)
    if not root_path.is_dir():
        raise FileNotFoundError('RGB-D root does not exist: {}'.format(root))

    samples = []
    for scene_dir in sorted(root_path.iterdir()):
        if not scene_dir.is_dir():
            continue
        rgb_dir = scene_dir / 'rgb'
        depth_dir = scene_dir / 'depth'
        if not rgb_dir.is_dir() or not depth_dir.is_dir():
            continue
        rgb_path = None
        depth_path = None
        rgb_candidates = sorted(
            candidate for candidate in rgb_dir.iterdir()
            if candidate.is_file() and candidate.suffix.lower() in
            ('.png', '.jpg', '.jpeg'))
        for candidate in rgb_candidates:
            stem = candidate.stem
            depth_candidates = [
                depth_dir / candidate.name,
                depth_dir / '{}.png'.format(stem),
            ]
            if stem.endswith('_rgb'):
                depth_candidates.insert(
                    0, depth_dir / '{}_depth.png'.format(stem[:-4]))
            if stem.endswith('_color'):
                depth_candidates.insert(
                    0, depth_dir / '{}_depth.png'.format(stem[:-6]))
            paired_depth = next(
                (path for path in depth_candidates if path.is_file()), None)
            if paired_depth is not None:
                rgb_path = candidate
                depth_path = paired_depth
                break
        if rgb_path is None:
            continue

        rgb_camera_file = scene_dir / 'camera_parameters' / 'rgb_camera_param.yaml'
        depth_camera_file = scene_dir / 'camera_parameters' / 'depth_camera_param.yaml'
        if not rgb_camera_file.is_file():
            raise FileNotFoundError('RGB camera parameters missing: {}'.format(
                rgb_camera_file))
        sample = {
            'scene': scene_dir.name,
            'rgb_path': str(rgb_path),
            'depth_path': str(depth_path),
            'rgb_intrinsic': _read_intrinsic(rgb_camera_file),
        }
        if depth_camera_file.is_file():
            sample['depth_intrinsic'] = _read_intrinsic(depth_camera_file)
        samples.append(sample)
        if max_scenes is not None and len(samples) >= int(max_scenes):
            break

    if not samples:
        raise RuntimeError('no RGB-D pairs found below {}'.format(root))
    return samples


def inspect_rgbd_sample(sample: Dict) -> Dict:
    """Read one pair and append shapes, dtype, and depth validity statistics."""
    rgb = cv2.imread(sample['rgb_path'], cv2.IMREAD_COLOR)
    depth = cv2.imread(sample['depth_path'], cv2.IMREAD_UNCHANGED)
    if rgb is None:
        raise RuntimeError('failed to read RGB: {}'.format(sample['rgb_path']))
    if depth is None:
        raise RuntimeError('failed to read depth: {}'.format(sample['depth_path']))
    if depth.ndim != 2:
        raise ValueError('depth must be single-channel uint16: {}'.format(
            sample['depth_path']))
    if depth.dtype != np.uint16:
        raise ValueError('depth must be uint16, got {}: {}'.format(
            depth.dtype, sample['depth_path']))

    result = dict(sample)
    result.update({
        'rgb_shape': [int(v) for v in rgb.shape],
        'rgb_dtype': str(rgb.dtype),
        'depth_shape': [int(v) for v in depth.shape],
        'depth_dtype': str(depth.dtype),
        'depth_valid_fraction': float(np.count_nonzero(depth) / depth.size),
        'depth_min_valid': int(depth[depth > 0].min())
        if np.any(depth > 0) else None,
        'depth_max': int(depth.max()),
    })
    return result
