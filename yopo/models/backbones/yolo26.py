"""Official YOLO26 n/s/m layers 0--10 used as YOPO's RGB backbone."""

from __future__ import annotations

import torch
import torch.nn as nn
from mmengine.model import BaseModule

from yopo.registry import MODELS


class _ConvNormAct(nn.Module):
    """JAX-compatible bias-free Conv, frozen-stat BN, and optional SiLU."""

    def __init__(
        self,
        in_channels: int,
        out_channels: int,
        kernel_size: int = 1,
        *,
        stride: int = 1,
        groups: int = 1,
        activate: bool = True,
    ) -> None:
        super().__init__()
        self.conv = nn.Conv2d(
            in_channels,
            out_channels,
            kernel_size,
            stride=stride,
            padding=kernel_size // 2,
            groups=groups,
            bias=False,
        )
        self.norm = nn.BatchNorm2d(
            out_channels,
            eps=1e-3,
            momentum=0.03,
            affine=True,
            track_running_stats=True,
        )
        self.act = nn.SiLU() if activate else nn.Identity()

    def forward(self, inputs: torch.Tensor) -> torch.Tensor:
        return self.act(self.norm(self.conv(inputs)))


class _Bottleneck(nn.Module):
    def __init__(
        self,
        in_channels: int,
        out_channels: int,
        *,
        expansion: float = 0.5,
    ) -> None:
        super().__init__()
        hidden = int(out_channels * expansion)
        self.cv1 = _ConvNormAct(in_channels, hidden, 3)
        self.cv2 = _ConvNormAct(hidden, out_channels, 3)
        self.shortcut = in_channels == out_channels

    def forward(self, inputs: torch.Tensor) -> torch.Tensor:
        output = self.cv2(self.cv1(inputs))
        return inputs + output if self.shortcut else output


class _C3k(nn.Module):
    def __init__(
        self,
        in_channels: int,
        out_channels: int,
        *,
        repeats: int = 2,
        expansion: float = 0.5,
    ) -> None:
        super().__init__()
        hidden = int(out_channels * expansion)
        self.cv1 = _ConvNormAct(in_channels, hidden)
        self.cv2 = _ConvNormAct(in_channels, hidden)
        self.blocks = nn.ModuleList(
            _Bottleneck(hidden, hidden, expansion=1.0) for _ in range(repeats)
        )
        self.cv3 = _ConvNormAct(hidden * 2, out_channels)

    def forward(self, inputs: torch.Tensor) -> torch.Tensor:
        transformed = self.cv1(inputs)
        residual = self.cv2(inputs)
        for block in self.blocks:
            transformed = block(transformed)
        return self.cv3(torch.cat((transformed, residual), dim=1))


class _C3kBranch(nn.Module):
    """Retain the JAX ``block/c3k`` state-tree boundary."""

    def __init__(self, channels: int) -> None:
        super().__init__()
        self.c3k = _C3k(channels, channels)

    def forward(self, inputs: torch.Tensor) -> torch.Tensor:
        return self.c3k(inputs)


class _BottleneckBranch(nn.Module):
    """Retain the JAX ``block/bottleneck`` state-tree boundary."""

    def __init__(self, channels: int) -> None:
        super().__init__()
        self.bottleneck = _Bottleneck(channels, channels)

    def forward(self, inputs: torch.Tensor) -> torch.Tensor:
        return self.bottleneck(inputs)


class _C3k2(nn.Module):
    def __init__(
        self,
        in_channels: int,
        out_channels: int,
        *,
        c3k: bool,
        expansion: float,
    ) -> None:
        super().__init__()
        hidden = int(out_channels * expansion)
        self.cv1 = _ConvNormAct(in_channels, hidden * 2)
        self.block = _C3kBranch(hidden) if c3k else _BottleneckBranch(hidden)
        self.cv2 = _ConvNormAct(hidden * 3, out_channels)

    def forward(self, inputs: torch.Tensor) -> torch.Tensor:
        first, second = self.cv1(inputs).chunk(2, dim=1)
        transformed = self.block(second)
        return self.cv2(torch.cat((first, second, transformed), dim=1))


class _SPPF(nn.Module):
    def __init__(self, channels: int) -> None:
        super().__init__()
        hidden = channels // 2
        self.cv1 = _ConvNormAct(channels, hidden, activate=False)
        self.cv2 = _ConvNormAct(hidden * 4, channels)
        self.pool = nn.MaxPool2d(kernel_size=5, stride=1, padding=2)

    def forward(self, inputs: torch.Tensor) -> torch.Tensor:
        first = self.cv1(inputs)
        pooled1 = self.pool(first)
        pooled2 = self.pool(pooled1)
        pooled3 = self.pool(pooled2)
        return self.cv2(torch.cat((first, pooled1, pooled2, pooled3), dim=1)) + inputs


class _Attention(nn.Module):
    def __init__(self, channels: int) -> None:
        super().__init__()
        self.num_heads = max(channels // 64, 1)
        self.head_channels = channels // self.num_heads
        self.key_channels = int(self.head_channels * 0.5)
        packed_channels = channels + 2 * self.key_channels * self.num_heads
        self.qkv = _ConvNormAct(channels, packed_channels, activate=False)
        self.pe = _ConvNormAct(
            channels,
            channels,
            3,
            groups=channels,
            activate=False,
        )
        self.projection = _ConvNormAct(channels, channels, activate=False)

    def forward(self, inputs: torch.Tensor) -> torch.Tensor:
        batch, channels, height, width = inputs.shape
        points = height * width
        packed = self.qkv(inputs).reshape(
            batch,
            self.num_heads,
            self.head_channels + 2 * self.key_channels,
            points,
        )
        query = packed[:, :, : self.key_channels]
        key = packed[:, :, self.key_channels : 2 * self.key_channels]
        value = packed[:, :, 2 * self.key_channels :]
        weights = torch.softmax(
            torch.einsum(
                "bhdn,bhdm->bhnm",
                query * self.key_channels**-0.5,
                key,
            ),
            dim=-1,
        )
        attended = torch.einsum("bhdm,bhnm->bhdn", value, weights)
        attended = attended.reshape(batch, channels, height, width)
        value_image = value.reshape(batch, channels, height, width)
        return self.projection(attended + self.pe(value_image))


class _PSABlock(nn.Module):
    def __init__(self, channels: int) -> None:
        super().__init__()
        self.attention = _Attention(channels)
        self.ffn = nn.Sequential(
            _ConvNormAct(channels, channels * 2),
            _ConvNormAct(channels * 2, channels, activate=False),
        )

    def forward(self, inputs: torch.Tensor) -> torch.Tensor:
        attended = inputs + self.attention(inputs)
        return attended + self.ffn(attended)


class _C2PSA(nn.Module):
    def __init__(self, channels: int) -> None:
        super().__init__()
        hidden = channels // 2
        self.cv1 = _ConvNormAct(channels, hidden * 2)
        self.block = _PSABlock(hidden)
        self.cv2 = _ConvNormAct(hidden * 2, channels)

    def forward(self, inputs: torch.Tensor) -> torch.Tensor:
        first, second = self.cv1(inputs).chunk(2, dim=1)
        transformed = self.block(second)
        return self.cv2(torch.cat((first, transformed), dim=1))


_SCALE_CHANNELS = {
    "n": (16, 32, 64, 64, 128, 128, 128, 256, 256, 256, 256),
    "s": (32, 64, 128, 128, 256, 256, 256, 512, 512, 512, 512),
    "m": (64, 128, 256, 256, 512, 512, 512, 512, 512, 512, 512),
}


@MODELS.register_module()
class YOLO26Backbone(BaseModule):
    """Official YOLO26 n/s/m layers 0--10 with P3/P4/P5 feature taps.

    The public ``return_idx`` values are YOPO output-level identifiers and
    align with HGNetV2-B0. Actual YOLO graph taps remain explicit in
    ``out_layer_indices``.
    """

    def __init__(
        self,
        scale: str,
        in_channels: int = 3,
        return_idx: tuple[int, int, int] | list[int] = (1, 2, 3),
        init_cfg: dict | None = None,
    ) -> None:
        super().__init__(init_cfg=init_cfg)
        if in_channels != 3:
            raise ValueError("YOLO26Backbone requires exactly three RGB channels")
        if scale not in _SCALE_CHANNELS:
            raise ValueError("scale must be one of 'n', 's', or 'm'")
        indices = tuple(return_idx)
        if len(indices) != 3 or len(set(indices)) != 3:
            raise ValueError("return_idx must contain three unique output-level IDs")
        self.scale = scale
        self.return_idx = indices
        self.out_layer_indices = (4, 6, 10)
        channels = _SCALE_CHANNELS[scale]
        self._out_channels = {
            index: channels[layer]
            for index, layer in zip(indices, self.out_layer_indices)
        }
        early_c3k = scale == "m"
        self.layers = nn.ModuleDict(
            {
                "0": _ConvNormAct(3, channels[0], 3, stride=2),
                "1": _ConvNormAct(channels[0], channels[1], 3, stride=2),
                "2": _C3k2(channels[1], channels[2], c3k=early_c3k, expansion=0.25),
                "3": _ConvNormAct(channels[2], channels[3], 3, stride=2),
                "4": _C3k2(channels[3], channels[4], c3k=early_c3k, expansion=0.25),
                "5": _ConvNormAct(channels[4], channels[5], 3, stride=2),
                "6": _C3k2(channels[5], channels[6], c3k=True, expansion=0.5),
                "7": _ConvNormAct(channels[6], channels[7], 3, stride=2),
                "8": _C3k2(channels[7], channels[8], c3k=True, expansion=0.5),
                "9": _SPPF(channels[9]),
                "10": _C2PSA(channels[10]),
            }
        )
        self.train(self.training)

    def train(self, mode: bool = True) -> "YOLO26Backbone":
        """Keep BN statistics frozen while preserving affine gradients."""

        super().train(mode)
        for module in self.modules():
            if isinstance(module, nn.BatchNorm2d):
                module.eval()
        return self

    def forward(self, inputs: torch.Tensor) -> tuple[torch.Tensor, ...]:
        if inputs.ndim != 4 or inputs.shape[1] != 3:
            raise ValueError(
                f"YOLO26Backbone expects NCHW RGB input, got {tuple(inputs.shape)}"
            )
        if inputs.shape[2] <= 0 or inputs.shape[3] <= 0:
            raise ValueError("input height and width must be positive")
        output = inputs
        features: list[torch.Tensor] = []
        for index, layer in self.layers.items():
            output = layer(output)
            if int(index) in self.out_layer_indices:
                features.append(output)
        return tuple(features)


@MODELS.register_module()
class YOLO26MBackbone(YOLO26Backbone):
    """Backward-compatible explicit alias for the original m-only class."""

    def __init__(
        self,
        in_channels: int = 3,
        return_idx: tuple[int, int, int] | list[int] = (1, 2, 3),
        init_cfg: dict | None = None,
    ) -> None:
        super().__init__(
            scale="m",
            in_channels=in_channels,
            return_idx=return_idx,
            init_cfg=init_cfg,
        )


__all__ = ["YOLO26Backbone", "YOLO26MBackbone"]
