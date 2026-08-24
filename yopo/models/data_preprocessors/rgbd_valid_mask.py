"""Five-channel RGB-D-plus-validity-mask data preprocessor."""

from __future__ import annotations

from numbers import Number
from typing import List, Optional, Sequence, Union

import torch
import torch.nn as nn
from mmengine.model import ImgDataPreprocessor

from yopo.registry import MODELS

from .data_preprocessor import DetDataPreprocessor


@MODELS.register_module()
class RGBDValidMaskDataPreprocessor(DetDataPreprocessor):
    """Normalize RGB/depth while retaining an explicit fifth validity channel."""

    def __init__(
        self,
        mean: Sequence[Number],
        std: Sequence[Number],
        pad_size_divisor: int = 1,
        pad_value: Union[float, int] = 0,
        pad_mask: bool = False,
        mask_pad_value: int = 0,
        pad_seg: bool = False,
        seg_pad_value: int = 255,
        bgr_to_rgb: bool = False,
        rgb_to_bgr: bool = False,
        boxtype2tensor: bool = True,
        non_blocking: Optional[bool] = False,
        batch_augments: Optional[List[dict]] = None,
    ) -> None:
        if len(mean) != 5 or len(std) != 5:
            raise ValueError("RGBDValidMaskDataPreprocessor requires 5-value mean/std")
        if bgr_to_rgb or rgb_to_bgr:
            raise ValueError(
                "RGBDValidMaskDataPreprocessor cannot permute RGB channels "
                "without corrupting the depth/mask layout"
            )

        # Bypass ImgDataPreprocessor's RGB/grayscale channel-count assertion,
        # exactly as the project 4-channel preprocessor does, then install
        # five-channel normalization buffers explicitly.
        ImgDataPreprocessor.__init__(
            self,
            mean=None,
            std=None,
            pad_size_divisor=pad_size_divisor,
            pad_value=pad_value,
            bgr_to_rgb=False,
            rgb_to_bgr=False,
            non_blocking=non_blocking,
        )
        self._enable_normalize = True
        self.register_buffer(
            "mean", torch.tensor(list(mean), dtype=torch.float32).view(-1, 1, 1), False
        )
        self.register_buffer(
            "std", torch.tensor(list(std), dtype=torch.float32).view(-1, 1, 1), False
        )
        self._channel_conversion = False
        self.batch_augments = (
            nn.ModuleList([MODELS.build(augment) for augment in batch_augments])
            if batch_augments is not None
            else None
        )
        self.pad_mask = pad_mask
        self.mask_pad_value = mask_pad_value
        self.pad_seg = pad_seg
        self.seg_pad_value = seg_pad_value
        self.boxtype2tensor = boxtype2tensor
