"""Query-aligned depth context and CoP stage fusion modules."""

from typing import Sequence

import torch
import torch.nn.functional as F
from torch import Tensor, nn


class MultiScaleDepthQuerySampler(nn.Module):
    """Sample query-aligned context from pre-fusion depth feature maps.

    The input boxes use normalized ``(cx, cy, w, h)`` coordinates. A regular
    ROI grid is sampled from every depth scale and pooled before the level
    features are concatenated and projected back to ``embed_dims``.
    """

    def __init__(self,
                 embed_dims: int = 256,
                 num_levels: int = 3,
                 roi_size: int = 3,
                 vectorize_layers: bool = True) -> None:
        super().__init__()
        if roi_size < 1 or roi_size % 2 == 0:
            raise ValueError(
                f'roi_size must be a positive odd integer, got {roi_size}')
        if num_levels < 1:
            raise ValueError(f'num_levels must be positive, got {num_levels}')
        self.embed_dims = embed_dims
        self.num_levels = num_levels
        self.roi_size = roi_size
        self.vectorize_layers = bool(vectorize_layers)
        self.output_projection = nn.Sequential(
            nn.Linear(embed_dims * num_levels, embed_dims),
            nn.LayerNorm(embed_dims),
            nn.GELU(),
        )

    def _sample_level(self, feature: Tensor, boxes: Tensor) -> Tensor:
        if feature.ndim != 4:
            raise ValueError(
                'depth feature must have shape [B,C,H,W], got '
                f'{tuple(feature.shape)}')
        if boxes.ndim != 3 or boxes.shape[-1] != 4:
            raise ValueError(
                'boxes must have shape [B,Q,4], got '
                f'{tuple(boxes.shape)}')
        batch_size, channels, _, _ = feature.shape
        if channels != self.embed_dims:
            raise ValueError(
                f'expected {self.embed_dims} depth channels, got {channels}')
        if boxes.shape[0] != batch_size:
            raise ValueError(
                'depth feature and boxes batch sizes differ: '
                f'{batch_size} vs {boxes.shape[0]}')

        centers = boxes[..., :2]
        sizes = boxes[..., 2:].clamp_min(1e-6)
        offsets = torch.linspace(
            -0.5,
            0.5,
            self.roi_size,
            device=boxes.device,
            dtype=boxes.dtype,
        )
        offset_y, offset_x = torch.meshgrid(offsets, offsets, indexing='ij')
        unit_grid = torch.stack((offset_x, offset_y), dim=-1)
        grid = centers[:, :, None, None, :] + (
            unit_grid[None, None, :, :, :] *
            sizes[:, :, None, None, :])
        grid = grid.clamp(0, 1).mul(2).sub(1)

        num_queries = boxes.shape[1]
        grid = grid.reshape(
            batch_size, num_queries * self.roi_size, self.roi_size, 2)
        samples = F.grid_sample(
            feature,
            grid,
            mode='bilinear',
            padding_mode='border',
            align_corners=False,
        )
        samples = samples.reshape(
            batch_size,
            channels,
            num_queries,
            self.roi_size,
            self.roi_size,
        )
        return samples.mean(dim=(-1, -2)).transpose(1, 2)

    def forward(self, depth_features: Sequence[Tensor], boxes: Tensor) -> Tensor:
        """Return one layer of depth contexts with shape ``[B,Q,C]``."""
        if len(depth_features) != self.num_levels:
            raise ValueError(
                f'expected {self.num_levels} depth levels, '
                f'got {len(depth_features)}')
        sampled_levels = [
            self._sample_level(feature, boxes) for feature in depth_features
        ]
        return self.output_projection(torch.cat(sampled_levels, dim=-1))

    def forward_layers(self, depth_features: Sequence[Tensor],
                       boxes: Tensor) -> Tensor:
        """Return all decoder-layer contexts with shape ``[L,B,Q,C]``."""
        if boxes.ndim != 4 or boxes.shape[-1] != 4:
            raise ValueError(
                'layer boxes must have shape [L,B,Q,4], got '
                f'{tuple(boxes.shape)}')
        if not self.vectorize_layers:
            return torch.stack([
                self(depth_features, layer_boxes) for layer_boxes in boxes
            ])
        num_layers, batch_size, num_queries, _ = boxes.shape
        # Grid sampling is independent for each query. Fold the decoder layer
        # axis into the query axis so every depth level needs one grid_sample
        # call instead of one call per level and decoder layer. The feature
        # maps stay [B,C,H,W] and are therefore never replicated L times.
        flattened_boxes = boxes.permute(1, 0, 2, 3).reshape(
            batch_size, num_layers * num_queries, 4)
        contexts = self(depth_features, flattened_boxes)
        return contexts.reshape(
            batch_size, num_layers, num_queries, self.embed_dims
        ).permute(1, 0, 2, 3).contiguous()


class CoPStageFusion(nn.Module):
    """Prepare one CoP stage input using a configurable dense skip."""

    VALID_MODES = ('residual', 'query_dense', 'depth_dense')

    def __init__(self, embed_dims: int = 256, mode: str = 'residual') -> None:
        super().__init__()
        if mode not in self.VALID_MODES:
            raise ValueError(
                f'mode must be one of {self.VALID_MODES}, got {mode!r}')
        self.embed_dims = embed_dims
        self.mode = mode
        self.last_dense_input_shape = None
        if mode == 'query_dense':
            input_dims = embed_dims * 2
        elif mode == 'depth_dense':
            input_dims = embed_dims * 3
        else:
            input_dims = None
        self.projection = None if input_dims is None else nn.Sequential(
            nn.Linear(input_dims, embed_dims),
            nn.LayerNorm(embed_dims),
            nn.GELU(),
        )

    def forward(self,
                current_chain: Tensor,
                original_query: Tensor,
                depth_query: Tensor = None) -> Tensor:
        if current_chain.shape != original_query.shape:
            raise ValueError(
                'current_chain and original_query shapes differ: '
                f'{tuple(current_chain.shape)} vs '
                f'{tuple(original_query.shape)}')
        if self.mode == 'residual':
            self.last_dense_input_shape = None
            return current_chain
        inputs = [current_chain, original_query]
        if self.mode == 'depth_dense':
            if depth_query is None:
                raise ValueError('depth_query is required in depth_dense mode')
            if depth_query.shape != current_chain.shape:
                raise ValueError(
                    'depth_query shape must match current_chain, got '
                    f'{tuple(depth_query.shape)} vs '
                    f'{tuple(current_chain.shape)}')
            inputs.append(depth_query)
        dense_input = torch.cat(inputs, dim=-1)
        self.last_dense_input_shape = tuple(dense_input.shape)
        return self.projection(dense_input)
