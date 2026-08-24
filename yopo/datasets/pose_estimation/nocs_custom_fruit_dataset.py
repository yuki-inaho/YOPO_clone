"""NOCS adapter for the single-class custom fruit dataset.

The custom annotations use class ID 1 (mapped to label 0), but label 0 means
``bottle`` in stock NOCS.  Stock ``NOCSDataset`` therefore applies its bottle
symmetry canonicalization to fruit rotations.  The adapter keeps the proven
NOCS parsing implementation while making the custom symmetry contract explicit.
"""

from __future__ import annotations

from typing import List, Optional

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

    def __init__(self, sym_ids: Optional[List[int]] = None, **kwargs) -> None:
        self._custom_sym_ids = [] if sym_ids is None else list(sym_ids)
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
            return super()._parse_instance_info(gt_info, intrinsic)
        finally:
            self.sym_ids = inherited_sym_ids
