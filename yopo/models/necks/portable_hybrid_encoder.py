"""Portable projection, AIFI, top-down FPN, and bottom-up PAN neck."""

from __future__ import annotations

from collections.abc import Sequence

import torch
import torch.nn as nn
import torch.nn.functional as F
from mmengine.model import BaseModule
from torch import Tensor

from yopo.registry import MODELS
from yopo.utils import OptMultiConfig


def position_encoding_2d(
    height: int,
    width: int,
    channels: int,
    *,
    device: torch.device,
    dtype: torch.dtype,
) -> Tensor:
    """Create the contract's row-major 2-D sine/cosine encoding."""

    if channels % 4:
        raise ValueError("position-encoding channels must be divisible by four")
    y, x = torch.meshgrid(
        torch.arange(height, device=device, dtype=torch.float32),
        torch.arange(width, device=device, dtype=torch.float32),
        indexing="ij",
    )
    quarter = channels // 4
    omega = 1.0 / (
        10000.0
        ** (torch.arange(quarter, device=device, dtype=torch.float32) / max(quarter, 1))
    )
    x_phase = x.reshape(-1, 1) * omega.reshape(1, -1)
    y_phase = y.reshape(-1, 1) * omega.reshape(1, -1)
    encoding = torch.cat(
        (x_phase.sin(), x_phase.cos(), y_phase.sin(), y_phase.cos()), dim=-1
    )
    return encoding.to(dtype=dtype)


class _AIFILayer(nn.Module):
    """Pre-normalized self-attention and GELU FFN matching the shared contract."""

    def __init__(self, hidden_dim: int, num_heads: int, ffn_dim: int) -> None:
        super().__init__()
        self.norm1 = nn.LayerNorm(hidden_dim, eps=1e-6)
        self.attention = nn.MultiheadAttention(
            hidden_dim, num_heads, dropout=0.0, bias=True, batch_first=True
        )
        self.norm2 = nn.LayerNorm(hidden_dim, eps=1e-6)
        self.ffn1 = nn.Linear(hidden_dim, ffn_dim, bias=True)
        self.ffn2 = nn.Linear(ffn_dim, hidden_dim, bias=True)

    def forward(self, tokens: Tensor) -> Tensor:
        normalized = self.norm1(tokens)
        attended, _ = self.attention(
            normalized, normalized, normalized, need_weights=False
        )
        tokens = tokens + attended
        normalized = self.norm2(tokens)
        return tokens + self.ffn2(F.gelu(self.ffn1(normalized), approximate="tanh"))


@MODELS.register_module()
class PortableHybridEncoderNeck(BaseModule):
    """Three shared feature levels plus one explicitly derived fourth level.

    Shared levels implement the exact v1 operation sequence: 1x1 projection,
    deepest-level AIFI, nearest-neighbour top-down concatenation, and
    zero-padded divide-by-four bottom-up pooling. The stride-64 fourth output
    is YOPO-only and intentionally excluded from checkpoint transfer.
    """

    def __init__(
        self,
        in_channels: Sequence[int],
        hidden_dim: int = 256,
        num_heads: int = 8,
        ffn_dim: int = 1024,
        num_aifi_layers: int = 1,
        num_outs: int = 4,
        init_cfg: OptMultiConfig = None,
    ) -> None:
        super().__init__(init_cfg=init_cfg)
        if len(in_channels) != 3 or min(in_channels) <= 0:
            raise ValueError("in_channels must contain three positive levels")
        if hidden_dim <= 0 or hidden_dim % num_heads:
            raise ValueError("hidden_dim must be positive and divisible by num_heads")
        if ffn_dim <= 0 or num_aifi_layers < 0:
            raise ValueError(
                "ffn_dim must be positive and AIFI layer count non-negative"
            )
        if num_outs not in (3, 4):
            raise ValueError("portable neck requires three or four outputs")
        self.in_channels = tuple(int(value) for value in in_channels)
        self.hidden_dim = int(hidden_dim)
        self.num_heads = int(num_heads)
        self.ffn_dim = int(ffn_dim)
        self.num_aifi_layers = int(num_aifi_layers)
        self.num_outs = int(num_outs)

        self.projections = nn.ModuleList(
            nn.Conv2d(channels, self.hidden_dim, 1, bias=False)
            for channels in self.in_channels
        )
        self.aifi = nn.ModuleList(
            _AIFILayer(self.hidden_dim, self.num_heads, self.ffn_dim)
            for _ in range(self.num_aifi_layers)
        )
        self.lateral = nn.ModuleList(
            nn.Conv2d(self.hidden_dim * 2, self.hidden_dim, 3, padding=1, bias=False)
            for _ in range(2)
        )
        self.pan = nn.ModuleList(
            nn.Conv2d(self.hidden_dim * 2, self.hidden_dim, 3, padding=1, bias=False)
            for _ in range(2)
        )
        self.derived_p6 = (
            nn.Conv2d(
                self.hidden_dim, self.hidden_dim, 3, stride=2, padding=1, bias=False
            )
            if num_outs == 4
            else None
        )

    def _validate(self, inputs: Sequence[Tensor]) -> None:
        if len(inputs) != 3:
            raise ValueError(f"expected three feature levels, got {len(inputs)}")
        previous_size: tuple[int, int] | None = None
        for level, (feature, channels) in enumerate(zip(inputs, self.in_channels)):
            if feature.ndim != 4:
                raise ValueError(
                    f"level {level} must be NCHW, got {tuple(feature.shape)}"
                )
            if feature.shape[1] != channels:
                raise ValueError(
                    f"level {level} channels must be {channels}, got {feature.shape[1]}"
                )
            size = (int(feature.shape[-2]), int(feature.shape[-1]))
            if previous_size is not None and (
                size[0] > previous_size[0] or size[1] > previous_size[1]
            ):
                raise ValueError(
                    "feature spatial sizes must be non-increasing by level"
                )
            previous_size = size

    @staticmethod
    def same_divide4_pool(source: Tensor) -> Tensor:
        """Match a 2x2 stride-2 SAME sum-pool divided by four, including borders."""

        pad_height = source.shape[-2] % 2
        pad_width = source.shape[-1] % 2
        if pad_height or pad_width:
            source = F.pad(source, (0, pad_width, 0, pad_height))
        return F.avg_pool2d(source, kernel_size=2, stride=2)

    def forward_shared(self, inputs: Sequence[Tensor]) -> tuple[Tensor, Tensor, Tensor]:
        """Return only the three contract-shared stride-8/16/32 levels."""

        self._validate(inputs)
        projected = [layer(feature) for layer, feature in zip(self.projections, inputs)]

        deepest = projected[-1]
        batch, channels, height, width = deepest.shape
        tokens = deepest.flatten(2).transpose(1, 2)
        tokens = tokens + position_encoding_2d(
            height,
            width,
            channels,
            device=tokens.device,
            dtype=tokens.dtype,
        ).unsqueeze(0)
        for layer in self.aifi:
            tokens = layer(tokens)
        projected[-1] = tokens.transpose(1, 2).reshape(batch, channels, height, width)

        top_down = list(projected)
        for lateral_index, level in enumerate(range(len(top_down) - 2, -1, -1)):
            upsampled = F.interpolate(
                top_down[level + 1],
                size=top_down[level].shape[-2:],
                mode="nearest-exact",
            )
            top_down[level] = F.silu(
                self.lateral[lateral_index](
                    torch.cat((top_down[level], upsampled), dim=1)
                )
            )

        outputs = list(top_down)
        for level in range(1, len(outputs)):
            pooled = self.same_divide4_pool(outputs[level - 1])
            pooled = F.interpolate(
                pooled, size=outputs[level].shape[-2:], mode="nearest-exact"
            )
            outputs[level] = F.silu(
                self.pan[level - 1](torch.cat((outputs[level], pooled), dim=1))
            )
        return outputs[0], outputs[1], outputs[2]

    def forward(self, inputs: Sequence[Tensor]) -> tuple[Tensor, ...]:
        shared = self.forward_shared(inputs)
        if self.derived_p6 is None:
            return shared
        return (*shared, self.derived_p6(shared[-1]))
