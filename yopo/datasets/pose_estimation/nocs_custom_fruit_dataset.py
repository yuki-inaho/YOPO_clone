"""NOCS adapter for the single-class custom fruit dataset.

The custom annotations use class ID 1 (mapped to label 0), but label 0 means
``bottle`` in stock NOCS.  Stock ``NOCSDataset`` therefore applies its bottle
symmetry canonicalization to fruit rotations.  The adapter keeps the proven
NOCS parsing implementation while making the custom symmetry contract explicit.
"""

from __future__ import annotations

from typing import List, Optional

import numpy as np

from yopo.registry import DATASETS

from .nocs_dataset import NOCSDataset


@DATASETS.register_module()
class NOCSCustomFruitDataset(NOCSDataset):
    """Parse custom fruit labels without inheriting stock NOCS symmetries.

    Args:
        sym_ids: Zero-indexed custom labels that are rotationally symmetric.
            The supplied fruit dataset has no such declared classes, so the
            safe default is an empty list.
    """

    METAINFO = {
        "classes": ("fruit",),
        "palette": [(255, 165, 0)],
    }

    def __init__(
        self,
        sym_ids: Optional[List[int]] = None,
        obb_coordinate_scale: float = 0.8,
        **kwargs,
    ) -> None:
        if obb_coordinate_scale <= 0.0:
            raise ValueError(
                "obb_coordinate_scale must be positive, got "
                f"{obb_coordinate_scale}")
        self._custom_sym_ids = [] if sym_ids is None else list(sym_ids)
        self.obb_coordinate_scale = float(obb_coordinate_scale)
        super().__init__(**kwargs)
        # `NOCSDataset.__init__` installs its stock `[0, 1, 3]` IDs before
        # BaseDataset may parse samples, so restore the custom contract after
        # construction as well as during the virtual parse hook below.
        self.sym_ids = self._custom_sym_ids

    def _parse_instance_info(self, gt_info: dict, intrinsic: list[float]) -> List[dict]:
        """Delegate parsing while preventing stock symmetry canonicalization."""
        inherited_sym_ids = self.sym_ids
        self.sym_ids = self._custom_sym_ids
        try:
            instances = super()._parse_instance_info(gt_info, intrinsic)
        finally:
            self.sym_ids = inherited_sym_ids
        raw_obbs = gt_info.get("obb_cxcywha_rad")
        if raw_obbs is None:
            raise KeyError(
                "custom fruit projection supervision requires "
                "obb_cxcywha_rad in every label pkl")
        if len(raw_obbs) != len(instances):
            raise ValueError(
                "OBB/instance count mismatch: "
                f"{len(raw_obbs)} vs {len(instances)}")
        for instance, raw_obb in zip(instances, raw_obbs):
            obb = np.asarray(raw_obb, dtype=np.float64).copy()
            if obb.shape != (5,) or not np.isfinite(obb).all() or \
                    np.any(obb[2:4] <= 0.0):
                raise ValueError(f"invalid custom fruit OBB: {raw_obb}")
            obb[:4] *= self.obb_coordinate_scale
            cosine = np.cos(obb[4])
            sine = np.sin(obb[4])
            rotation_2d = np.array(
                [[cosine, -sine], [sine, cosine]], dtype=np.float64)
            sigma = rotation_2d @ np.diag(np.square(obb[2:4] * 0.5)) @ \
                rotation_2d.T
            instance["obb_gaussian"] = np.array(
                [obb[0], obb[1], sigma[0, 0], sigma[0, 1], sigma[1, 1]],
                dtype=np.float32,
            )
        return instances
