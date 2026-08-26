"""Box-only oriented contrastive denoising primitives.

This module deliberately does not depend on :class:`CdnQueryGenerator`.  It
only creates geometry and metadata that a future DINO integration can embed,
pad, and concatenate with matching queries.

The input and output boxes are flat, per-image tensors with normalized
coordinates in ``[0, 1]``.  ``box_format`` selects either ``cxcywh`` or
``xyxy`` and the output preserves that format.  Optional angles are separate
from the boxes because they are copied exactly and are never noised.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Literal

import torch
import torch.nn as nn
from torch import Tensor

from yopo.registry import MODELS

BoxFormat = Literal["cxcywh", "xyxy"]


@dataclass(frozen=True)
class BoxOnlyOCDQueries:
    """Generated positive/negative OCD query geometry and provenance.

    Query order is interleaved per source GT: ``positive_0, negative_0,
    positive_1, negative_1, ...``.  Consequently ``pair_query_indices[i]``
    contains the positive and negative query indices derived from source GT
    ``i``.
    """

    boxes: Tensor
    labels: Tensor
    angles: Tensor | None
    query_indices: Tensor
    source_box_indices: Tensor
    source_label_indices: Tensor
    positive_mask: Tensor
    pair_query_indices: Tensor

    @property
    def positive_query_indices(self) -> Tensor:
        """Indices of the small-noise positive queries."""
        return self.query_indices[self.positive_mask]

    @property
    def negative_query_indices(self) -> Tensor:
        """Indices of the large-noise negative queries."""
        return self.query_indices[~self.positive_mask]


def _boxes_to_xyxy(boxes: Tensor, box_format: BoxFormat) -> Tensor:
    if box_format == "xyxy":
        return boxes
    center = boxes[:, :2]
    half_size = boxes[:, 2:] * 0.5
    return torch.cat((center - half_size, center + half_size), dim=-1)


def _boxes_from_xyxy(boxes: Tensor, box_format: BoxFormat) -> Tensor:
    if box_format == "xyxy":
        return boxes
    center = (boxes[:, :2] + boxes[:, 2:]) * 0.5
    size = boxes[:, 2:] - boxes[:, :2]
    return torch.cat((center, size), dim=-1)


def _clip_and_enforce_min_size(boxes_xyxy: Tensor, min_size: float) -> Tensor:
    """Canonicalize, clip, and repair normalized xyxy boxes."""
    lower = torch.minimum(boxes_xyxy[:, :2], boxes_xyxy[:, 2:]).clamp(0, 1)
    upper = torch.maximum(boxes_xyxy[:, :2], boxes_xyxy[:, 2:]).clamp(0, 1)

    size = (upper - lower).clamp(min=min_size, max=1.0)
    half_size = size * 0.5
    center = (lower + upper) * 0.5
    center = torch.maximum(center, half_size)
    center = torch.minimum(center, 1.0 - half_size)
    return torch.cat((center - half_size, center + half_size), dim=-1)


def _sample_signed_magnitudes(
    shape: torch.Size,
    *,
    low: float,
    high: float,
    reference: Tensor,
    generator: torch.Generator | None,
) -> Tensor:
    magnitudes = torch.rand(
        shape,
        dtype=reference.dtype,
        device=reference.device,
        generator=generator,
    )
    magnitudes = magnitudes * (high - low) + low
    signs = torch.rand(
        shape,
        dtype=reference.dtype,
        device=reference.device,
        generator=generator,
    )
    signs = torch.where(signs < 0.5, -torch.ones_like(signs),
                        torch.ones_like(signs))
    return signs * magnitudes


def _validate_inputs(
    boxes: Tensor,
    labels: Tensor,
    angles: Tensor | None,
    box_format: str,
) -> Tensor:
    if box_format not in {"cxcywh", "xyxy"}:
        raise ValueError(
            f"box_format must be 'cxcywh' or 'xyxy', got {box_format!r}")
    if boxes.ndim != 2 or boxes.shape[-1] != 4:
        raise ValueError(f"boxes must have shape [N, 4], got {boxes.shape}")
    if not boxes.is_floating_point():
        raise TypeError("boxes must use a floating-point dtype")
    if labels.ndim != 1 or labels.shape[0] != boxes.shape[0]:
        raise ValueError(
            "labels must have shape [N] aligned with boxes; "
            f"got boxes={boxes.shape}, labels={labels.shape}")
    if labels.is_floating_point() or labels.dtype == torch.bool:
        raise TypeError("labels must use an integer dtype")
    if labels.device != boxes.device:
        raise ValueError("boxes and labels must be on the same device")
    if angles is not None:
        if angles.ndim < 1 or angles.shape[0] != boxes.shape[0]:
            raise ValueError(
                "angles must have first dimension N aligned with boxes; "
                f"got boxes={boxes.shape}, angles={angles.shape}")
        if angles.device != boxes.device:
            raise ValueError("boxes and angles must be on the same device")
        if not angles.is_floating_point():
            raise TypeError("angles must use a floating-point dtype")
        if not bool(torch.isfinite(angles).all()):
            raise ValueError("angles must contain only finite values")
    if not bool(torch.isfinite(boxes).all()):
        raise ValueError("boxes must contain only finite values")

    boxes_xyxy = _boxes_to_xyxy(boxes, box_format)  # type: ignore[arg-type]
    if bool(((boxes_xyxy < 0) | (boxes_xyxy > 1)).any()):
        raise ValueError(
            "boxes must describe normalized geometry entirely within [0, 1]")
    if bool((boxes_xyxy[:, 2:] <= boxes_xyxy[:, :2]).any()):
        raise ValueError("boxes must have strictly positive width and height")
    return boxes_xyxy


@torch.no_grad()
def generate_box_only_ocd_queries(
    boxes: Tensor,
    labels: Tensor,
    *,
    box_format: BoxFormat = "cxcywh",
    angles: Tensor | None = None,
    positive_noise_scale: float = 0.05,
    negative_noise_scale: tuple[float, float] = (0.2, 0.4),
    min_size: float = 1e-4,
    seed: int | None = None,
    generator: torch.Generator | None = None,
) -> BoxOnlyOCDQueries:
    """Create one small-noise positive and one large-noise negative per GT.

    Noise is applied only to the normalized axis-aligned ``xyxy`` envelope.
    Each edge displacement is relative to the source width or height.  Before
    clipping, positive magnitudes lie in ``[0, positive_noise_scale)`` and
    negative magnitudes lie in the inclusive-lower interval configured by
    ``negative_noise_scale``.  Output validity takes priority at image borders:
    boxes are clipped to ``[0, 1]`` and repaired to at least ``min_size``.

    Args:
        boxes: Per-image normalized boxes with shape ``[N, 4]``.
        labels: Integer class labels with shape ``[N]``.
        box_format: Input and output contract, ``cxcywh`` or ``xyxy``.
        angles: Optional orientation values with first dimension ``N``.  They
            are repeated for each pair without any perturbation or wrapping.
        positive_noise_scale: Maximum relative edge noise for positives.  It
            must be in ``[0, 0.5]``.
        negative_noise_scale: Minimum and maximum relative edge-noise
            magnitudes for negatives.  Its minimum must not be smaller than
            ``positive_noise_scale``.
        min_size: Minimum normalized output width and height.
        seed: Optional local seed.  It does not alter PyTorch's global RNG.
        generator: Optional caller-owned generator.  ``seed`` and
            ``generator`` are mutually exclusive.

    Returns:
        :class:`BoxOnlyOCDQueries` with an exact positive:negative ratio of
        1:1 and explicit query/source-label provenance.
    """
    boxes_xyxy = _validate_inputs(boxes, labels, angles, box_format)

    if not 0.0 <= positive_noise_scale <= 0.5:
        raise ValueError("positive_noise_scale must be in [0, 0.5]")
    if len(negative_noise_scale) != 2:
        raise ValueError("negative_noise_scale must contain (minimum, maximum)")
    negative_min, negative_max = negative_noise_scale
    if not positive_noise_scale <= negative_min <= negative_max:
        raise ValueError(
            "negative noise must satisfy positive_scale <= minimum <= maximum")
    if not 0.0 < min_size <= 1.0:
        raise ValueError("min_size must be in (0, 1]")
    if seed is not None and generator is not None:
        raise ValueError("seed and generator are mutually exclusive")
    if seed is not None:
        if isinstance(seed, bool) or not isinstance(seed, int) or seed < 0:
            raise ValueError("seed must be a non-negative integer")
        generator = torch.Generator(device=boxes.device).manual_seed(seed)

    num_boxes = boxes.shape[0]
    query_indices = torch.arange(
        num_boxes * 2, dtype=torch.long, device=boxes.device)
    source_indices = torch.arange(
        num_boxes, dtype=torch.long, device=boxes.device).repeat_interleave(2)
    positive_mask = torch.zeros(
        num_boxes * 2, dtype=torch.bool, device=boxes.device)
    positive_mask[0::2] = True
    pair_query_indices = query_indices.reshape(num_boxes, 2)

    if num_boxes == 0:
        output_boxes = boxes.new_empty((0, 4))
    else:
        size = boxes_xyxy[:, 2:] - boxes_xyxy[:, :2]
        edge_scale = size.repeat(1, 2)
        positive_noise = _sample_signed_magnitudes(
            boxes_xyxy.shape,
            low=0.0,
            high=positive_noise_scale,
            reference=boxes,
            generator=generator,
        )
        negative_noise = _sample_signed_magnitudes(
            boxes_xyxy.shape,
            low=negative_min,
            high=negative_max,
            reference=boxes,
            generator=generator,
        )
        paired_xyxy = torch.stack(
            (boxes_xyxy + positive_noise * edge_scale,
             boxes_xyxy + negative_noise * edge_scale),
            dim=1,
        ).reshape(num_boxes * 2, 4)
        paired_xyxy = _clip_and_enforce_min_size(paired_xyxy, min_size)
        output_boxes = _boxes_from_xyxy(paired_xyxy, box_format)

    return BoxOnlyOCDQueries(
        boxes=output_boxes,
        labels=labels.repeat_interleave(2),
        angles=(None if angles is None else angles.repeat_interleave(2, dim=0)),
        query_indices=query_indices,
        source_box_indices=source_indices,
        source_label_indices=source_indices.clone(),
        positive_mask=positive_mask,
        pair_query_indices=pair_query_indices,
    )


@MODELS.register_module()
class BoxOnlyOCDNoise(nn.Module):
    """Config-buildable box-noise strategy for DINO denoising queries.

    The returned normalized ``xyxy`` boxes preserve DINO's required ordering
    within every group: all positives followed by all negatives.  Only box
    geometry is returned, so OBB angles and 3D pose attributes cannot be
    perturbed by this component.
    """

    def __init__(
        self,
        positive_noise_scale: float = 0.05,
        negative_noise_scale: tuple[float, float] = (0.2, 0.4),
        min_size: float = 1e-4,
    ) -> None:
        super().__init__()
        # Reuse the pure helper's validation on an empty, valid input.  This
        # keeps one source of truth for the public noise contract.
        generate_box_only_ocd_queries(
            torch.empty((0, 4)),
            torch.empty((0,), dtype=torch.long),
            box_format="xyxy",
            positive_noise_scale=positive_noise_scale,
            negative_noise_scale=negative_noise_scale,
            min_size=min_size,
        )
        self.positive_noise_scale = float(positive_noise_scale)
        self.negative_noise_scale = tuple(
            float(value) for value in negative_noise_scale)
        self.min_size = float(min_size)

    @torch.no_grad()
    def forward(self, boxes_xyxy: Tensor, num_groups: int) -> Tensor:
        """Generate grouped positive/negative boxes for concatenated GTs."""
        if isinstance(num_groups, bool) or not isinstance(num_groups, int) \
                or num_groups < 1:
            raise ValueError("num_groups must be a positive integer")
        labels = torch.zeros(
            len(boxes_xyxy), dtype=torch.long, device=boxes_xyxy.device)
        grouped_boxes = []
        for _ in range(num_groups):
            queries = generate_box_only_ocd_queries(
                boxes_xyxy,
                labels,
                box_format="xyxy",
                positive_noise_scale=self.positive_noise_scale,
                negative_noise_scale=self.negative_noise_scale,
                min_size=self.min_size,
            )
            grouped_boxes.append(torch.cat(
                (queries.boxes[queries.positive_mask],
                 queries.boxes[~queries.positive_mask]),
                dim=0,
            ))
        return torch.cat(grouped_boxes, dim=0)


__all__ = [
    "BoxOnlyOCDNoise", "BoxOnlyOCDQueries", "generate_box_only_ocd_queries"
]
