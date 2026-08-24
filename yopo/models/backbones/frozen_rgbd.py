"""Frozen-RGB variant of the user-owned dual RGB-D backbone."""

from __future__ import annotations

from yopo.registry import MODELS

from .dual_rgbd import RGBDDualBackbone


@MODELS.register_module()
class FrozenRGBDDualBackbone(RGBDDualBackbone):
    """Keep a transferred RGB encoder fixed while training depth and fusion.

    The freeze happens during model construction, before the optimizer wrapper
    is built.  This lets mmengine omit every RGB parameter from optimizer
    groups instead of merely suppressing its gradient after registration.
    """

    def __init__(self, *args, freeze_rgb: bool = True, **kwargs) -> None:
        self.freeze_rgb = freeze_rgb
        super().__init__(*args, **kwargs)
        if self.freeze_rgb:
            self._freeze_rgb_branch()

    def _freeze_rgb_branch(self) -> None:
        self.rgb_backbone.requires_grad_(False)
        self.rgb_backbone.eval()

    def train(self, mode: bool = True):
        super().train(mode)
        if self.freeze_rgb:
            # Parent `train()` recursively toggles all submodules; restore
            # eval mode so transferred normalization/statistical layers stay
            # fixed as well as their parameters.
            self.rgb_backbone.eval()
        return self
