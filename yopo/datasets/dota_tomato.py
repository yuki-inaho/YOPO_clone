# Copyright (c) OpenMMLab. All rights reserved.
import glob
import os.path as osp
from pathlib import Path
from typing import Optional, Sequence

import cv2
import numpy as np

from yopo.registry import DATASETS
from .base_det_dataset import BaseDetDataset


def qbox2rbox_np(points):
    """Convert a quadrilateral (8 values) to a rotated box (cx, cy, w, h, t).

    Args:
        points (np.ndarray): Shape (8,). 8 polygon vertex coordinates.

    Returns:
        np.ndarray: Shape (5,) rotated box in <cx, cy, w, h, t> (radian).
    """
    pts = points.reshape(4, 2).astype(np.float32)
    (x, y), (w, h), angle = cv2.minAreaRect(pts)
    if h > w:
        w, h = h, w
        angle += 90
    return np.array([x, y, w, h, np.deg2rad(angle)], dtype=np.float32)


@DATASETS.register_module()
class DOTAOBBDataset(BaseDetDataset):
    """Configurable, optionally strict DOTA-format OBB dataset.

    Reads the DOTA-style annotation layout: ``ann_file`` is a directory of
    ``*.txt`` files, one per image. Each line in a txt file is
    ``x1 y1 x2 y2 x3 y3 x4 y4 class difficulty``. The 8-vertex polygon (qbox)
    is converted on load into a 5-dim rotated box (rbox, <cx, cy, w, h, rad>).

    Args:
        img_shape: Known ``(height, width)`` shared by every image. Set to
            ``None`` to read the real shape from each image.
        img_suffixes: Ordered image extensions used to resolve label stems.
        depth_suffixes: Optional depth extensions. When ``data_prefix`` also
            contains ``depth_path``, every RGB/label stem must have one depth
            file and ``depth_path`` is added to each sample.
        strict_loading: Fail on malformed/non-finite/degenerate annotations,
            unknown classes, corrupt images/depth, shape disagreement, and
            RGB/depth/label stem mismatch.
        diff_thr (int): Difficulty threshold; instances with difficulty
            larger than this are ignored. Defaults to 100.
    """

    METAINFO = {"classes": (), "palette": [(255, 255, 0)]}

    def __init__(self,
                 img_shape: Optional[tuple[int, int]] = None,
                 img_suffixes: Sequence[str] = ('.jpg', '.jpeg', '.png'),
                 depth_suffixes: Sequence[str] = ('.png',),
                 strict_loading: bool = True,
                 diff_thr: int = 100,
                 **kwargs):
        if img_shape is not None:
            if (len(img_shape) != 2 or any(int(value) <= 0
                                           for value in img_shape)):
                raise ValueError(
                    f'img_shape must be positive (height, width), got '
                    f'{img_shape!r}')
            img_shape = tuple(int(value) for value in img_shape)
        self.img_shape = img_shape
        self.img_suffixes = self._normalize_suffixes(
            img_suffixes, 'img_suffixes')
        self.depth_suffixes = self._normalize_suffixes(
            depth_suffixes, 'depth_suffixes')
        self.strict_loading = bool(strict_loading)
        self.diff_thr = diff_thr
        super().__init__(**kwargs)

    @staticmethod
    def _normalize_suffixes(suffixes: Sequence[str],
                            name: str) -> tuple[str, ...]:
        if not suffixes:
            raise ValueError(f'{name} must not be empty')
        normalized_suffixes: list[str] = []
        for suffix in suffixes:
            suffix = str(suffix).lower()
            if not suffix.startswith('.'):
                suffix = f'.{suffix}'
            if suffix in normalized_suffixes:
                raise ValueError(f'duplicate {name} entry: {suffix}')
            normalized_suffixes.append(suffix)
        return tuple(normalized_suffixes)

    @staticmethod
    def _files_by_stem(root: Path, suffixes: Sequence[str],
                       kind: str) -> dict[str, Path]:
        files_by_stem: dict[str, Path] = {}
        paths = sorted(path for path in root.iterdir() if path.is_file())
        for suffix in suffixes:
            for path in paths:
                if path.suffix.lower() != suffix:
                    continue
                if path.stem in files_by_stem:
                    raise ValueError(
                        f'multiple {kind} files share DOTA label stem '
                        f'{path.stem!r}')
                files_by_stem[path.stem] = path
        return files_by_stem

    def _image_by_stem(self) -> dict[str, Path]:
        return self._files_by_stem(
            Path(self.data_prefix['img_path']), self.img_suffixes, 'image')

    def _depth_by_stem(self) -> dict[str, Path]:
        return self._files_by_stem(
            Path(self.data_prefix['depth_path']), self.depth_suffixes, 'depth')

    def _image_shape(self, image_path: Path) -> tuple[int, int]:
        if self.img_shape is not None and not self.strict_loading:
            return self.img_shape
        image = cv2.imread(str(image_path), cv2.IMREAD_UNCHANGED)
        if image is None:
            raise ValueError(f'failed to decode DOTA image: {image_path}')
        actual_shape = (int(image.shape[0]), int(image.shape[1]))
        if self.img_shape is not None and actual_shape != self.img_shape:
            raise ValueError(
                f'DOTA image shape mismatch for {image_path}: expected '
                f'{self.img_shape}, got {actual_shape}')
        return actual_shape

    def _validate_depth(self, depth_path: Path,
                        image_shape: tuple[int, int]) -> None:
        if not self.strict_loading:
            return
        depth = cv2.imread(str(depth_path), cv2.IMREAD_UNCHANGED)
        if depth is None:
            raise ValueError(f'failed to decode DOTA depth: {depth_path}')
        depth_shape = (int(depth.shape[0]), int(depth.shape[1]))
        if depth_shape != image_shape:
            raise ValueError(
                f'DOTA RGB/depth shape mismatch for {depth_path}: '
                f'rgb={image_shape}, depth={depth_shape}')

    def _parse_instance(self, line: str, txt_file: str, line_number: int,
                        cls_map: dict[str, int]):
        parts = line.split()
        location = f'{txt_file}:{line_number}'
        if len(parts) not in (9, 10):
            if self.strict_loading:
                raise ValueError(
                    f'{location}: expected 9 or 10 columns, got {len(parts)}')
            return None
        try:
            bbox_q = np.asarray(parts[:8], dtype=np.float64)
        except ValueError as error:
            if self.strict_loading:
                raise ValueError(
                    f'{location}: invalid quad coordinates') from error
            return None
        if not np.isfinite(bbox_q).all():
            if self.strict_loading:
                raise ValueError(f'{location}: non-finite quad coordinates')
            return None

        cls_name = parts[8]
        if cls_name not in cls_map:
            if self.strict_loading:
                raise ValueError(
                    f'{location}: unknown class {cls_name!r}; expected '
                    f'{tuple(cls_map)!r}')
            return None
        try:
            difficulty = int(parts[9]) if len(parts) == 10 else 0
        except ValueError as error:
            if self.strict_loading:
                raise ValueError(
                    f'{location}: invalid difficulty {parts[9]!r}') from error
            return None

        bbox = qbox2rbox_np(bbox_q)
        if (not np.isfinite(bbox).all() or bbox[2] <= 0.0 or bbox[3] <= 0.0):
            if self.strict_loading:
                raise ValueError(f'{location}: degenerate DOTA quadrilateral')
            return None
        return {
            # Native Python lists let LoadAnnotations build one tensor in a
            # single operation instead of converting a list of ndarrays.
            'bbox': bbox.tolist(),
            'bbox_label': cls_map[cls_name],
            'ignore_flag': int(difficulty > self.diff_thr)
        }

    def load_data_list(self):
        cls_map = {c: i for i, c in enumerate(self.metainfo["classes"])}
        if not cls_map:
            raise ValueError(
                'DOTAOBBDataset requires non-empty metainfo.classes')
        if len(cls_map) != len(self.metainfo["classes"]):
            raise ValueError(
                'DOTAOBBDataset metainfo.classes must not contain duplicates')
        data_list = []
        txt_files = sorted(glob.glob(osp.join(self.ann_file, "*.txt")))
        image_by_stem = self._image_by_stem()
        depth_by_stem = (
            self._depth_by_stem() if 'depth_path' in self.data_prefix else None)
        label_stems = {Path(txt_file).stem for txt_file in txt_files}
        image_stems = set(image_by_stem)
        if self.strict_loading and label_stems != image_stems:
            missing_images = sorted(label_stems - image_stems)
            missing_labels = sorted(image_stems - label_stems)
            raise FileNotFoundError(
                'DOTA image/label stem mismatch: '
                f'missing_images={missing_images[:5]} '
                f'missing_labels={missing_labels[:5]}')
        if self.strict_loading and depth_by_stem is not None:
            depth_stems = set(depth_by_stem)
            if label_stems != depth_stems:
                missing_depth = sorted(label_stems - depth_stems)
                orphan_depth = sorted(depth_stems - label_stems)
                raise FileNotFoundError(
                    'DOTA RGB/depth/label stem mismatch: '
                    f'missing_depth={missing_depth[:5]} '
                    f'orphan_depth={orphan_depth[:5]}')
        for txt_file in txt_files:
            img_id = Path(txt_file).stem
            image_path = image_by_stem.get(img_id)
            if image_path is None:
                continue
            instances = []
            with open(txt_file) as f:
                for line_number, line in enumerate(f, start=1):
                    if not line.strip():
                        continue
                    instance = self._parse_instance(
                        line, txt_file, line_number, cls_map)
                    if instance is not None:
                        instances.append(instance)
            height, width = self._image_shape(image_path)
            data_info = {
                "img_id": img_id,
                "file_name": image_path.name,
                "img_path": str(image_path),
                "height": height,
                "width": width,
                "instances": instances,
            }
            if depth_by_stem is not None:
                depth_path = depth_by_stem.get(img_id)
                if depth_path is None:
                    continue
                self._validate_depth(depth_path, (height, width))
                data_info['depth_path'] = str(depth_path)
            data_list.append(data_info)
        return data_list

    def filter_data(self):
        if self.test_mode:
            return self.data_list
        filter_empty_gt = (self.filter_cfg.get("filter_empty_gt", False)
                           if self.filter_cfg is not None else False)
        valid_data_infos = []
        for data_info in self.data_list:
            if filter_empty_gt and len(data_info["instances"]) == 0:
                continue
            valid_data_infos.append(data_info)
        return valid_data_infos

    def get_cat_ids(self, idx):
        data_info = self.get_data_info(idx)
        return [inst["bbox_label"] for inst in data_info["instances"]]


@DATASETS.register_module()
class DOTATomatoDataset(DOTAOBBDataset):
    """Backward-compatible loader for the historical ``stem`` DOTA data."""

    METAINFO = {"classes": ("stem",), "palette": [(255, 255, 0)]}

    def __init__(self,
                 img_shape=(512, 736),
                 img_suffixes=('.jpg',),
                 strict_loading=False,
                 **kwargs):
        super().__init__(
            img_shape=img_shape,
            img_suffixes=img_suffixes,
            strict_loading=strict_loading,
            **kwargs)
