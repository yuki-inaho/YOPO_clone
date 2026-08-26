"""Small config-driven composition point for multi-scale detector features."""

from __future__ import annotations

import math
from collections.abc import Sequence

import torch
import torch.nn as nn
import torch.nn.functional as F
from mmengine.model import BaseModule
from torch import Tensor

from yopo.registry import MODELS
from yopo.utils import OptMultiConfig


class _RefineBlock(nn.Sequential):
    """Cheap spatial/channel mixing that preserves the tensor shape."""

    def __init__(self, channels: int) -> None:
        groups = math.gcd(channels, 32)
        super().__init__(
            nn.Conv2d(
                channels,
                channels,
                kernel_size=3,
                padding=1,
                groups=channels,
                bias=False,
            ),
            nn.Conv2d(channels, channels, kernel_size=1, bias=False),
            nn.GroupNorm(groups, channels),
            nn.SiLU(inplace=True),
        )


@MODELS.register_module()
class ResidualPyramidRefiner(BaseModule):
    """Top-down/bottom-up pyramid refinement with an identity start.

    The resize target is always taken from the neighbouring tensor. This is
    important for the 736x512 input, whose pyramid contains odd spatial sizes.
    """

    def __init__(
        self,
        channels: int,
        num_levels: int,
        beta_init: float = 0.0,
        init_cfg: OptMultiConfig = None,
    ) -> None:
        super().__init__(init_cfg=init_cfg)
        if channels <= 0:
            raise ValueError(f"channels must be positive, got {channels}")
        if num_levels < 2:
            raise ValueError(f"num_levels must be at least 2, got {num_levels}")
        self.channels = int(channels)
        self.num_levels = int(num_levels)
        self.top_down_blocks = nn.ModuleList(
            _RefineBlock(self.channels) for _ in range(self.num_levels - 1)
        )
        self.bottom_up_blocks = nn.ModuleList(
            _RefineBlock(self.channels) for _ in range(self.num_levels - 1)
        )
        self.output_blocks = nn.ModuleList(
            _RefineBlock(self.channels) for _ in range(self.num_levels)
        )
        self.beta = nn.Parameter(
            torch.full((self.num_levels,), float(beta_init), dtype=torch.float32)
        )

    def _validate(self, inputs: Sequence[Tensor]) -> None:
        if len(inputs) != self.num_levels:
            raise ValueError(
                f"expected {self.num_levels} feature levels, got {len(inputs)}"
            )
        for level, feature in enumerate(inputs):
            if feature.ndim != 4:
                raise ValueError(
                    f"level {level} must be NCHW, got shape {tuple(feature.shape)}"
                )
            if feature.shape[1] != self.channels:
                raise ValueError(
                    f"level {level} channels must be {self.channels}, "
                    f"got {feature.shape[1]}"
                )

    def forward(self, inputs: Sequence[Tensor]) -> tuple[Tensor, ...]:
        self._validate(inputs)
        source = tuple(inputs)

        top_down = list(source)
        for level in range(self.num_levels - 2, -1, -1):
            upsampled = F.interpolate(
                top_down[level + 1],
                size=top_down[level].shape[-2:],
                mode="nearest",
            )
            top_down[level] = top_down[level] + self.top_down_blocks[level](
                upsampled
            )

        bottom_up = [top_down[0]]
        for level in range(1, self.num_levels):
            downsampled = F.interpolate(
                bottom_up[-1],
                size=top_down[level].shape[-2:],
                mode="area",
            )
            bottom_up.append(
                top_down[level]
                + self.bottom_up_blocks[level - 1](downsampled)
            )

        outputs = []
        for level, (original, feature) in enumerate(zip(source, bottom_up)):
            delta = self.output_blocks[level](feature)
            scale = self.beta[level].to(dtype=original.dtype)
            outputs.append(original + scale * delta)
        return tuple(outputs)


@MODELS.register_module()
class ComposablePyramidNeck(BaseModule):
    """Apply one mapper followed by an optional feature-pyramid refiner."""

    def __init__(
        self,
        mapper: dict,
        refiner: dict | None = None,
        init_cfg: OptMultiConfig = None,
    ) -> None:
        super().__init__(init_cfg=init_cfg)
        self.mapper = MODELS.build(mapper)
        self.refiner = MODELS.build(refiner) if refiner is not None else None

    def forward(self, inputs: Sequence[Tensor]) -> tuple[Tensor, ...]:
        mapped = tuple(self.mapper(tuple(inputs)))
        if self.refiner is None:
            return mapped

        refined = tuple(self.refiner(mapped))
        if len(refined) != len(mapped):
            raise ValueError(
                "refiner changed the number of feature levels: "
                f"{len(mapped)} -> {len(refined)}"
            )
        for level, (before, after) in enumerate(zip(mapped, refined)):
            if after.shape != before.shape:
                raise ValueError(
                    f"refiner changed level {level} shape: "
                    f"{tuple(before.shape)} -> {tuple(after.shape)}"
                )
        return refined
