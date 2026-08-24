"""Strict, auditable checkpoint selection for RGB encoder transfer."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Mapping, Tuple

import torch


SOURCE_RGB_PREFIX = "backbone."
TARGET_RGB_PREFIX = "backbone.rgb_backbone."
_IGNORED_SOURCE_SUFFIX = "num_batches_tracked"


@dataclass(frozen=True)
class RGBBackboneTransferSelection:
    """Exact subset selected for loading a 2D RGB backbone into RGB-D."""

    state_dict: dict[str, torch.Tensor]
    ignored_source_keys: Tuple[str, ...]
    missing_target_keys: Tuple[str, ...]


def build_rgb_backbone_transfer_state(
    source_state_dict: Mapping[str, torch.Tensor],
    target_state_dict: Mapping[str, torch.Tensor],
) -> RGBBackboneTransferSelection:
    """Map only shape-identical 2D HGNet RGB weights into a dual RGB-D model.

    Non-backbone 2D tensors are intentionally outside this function's scope.
    BatchNorm ``num_batches_tracked`` buffers are the sole allowed source-only
    keys because the target frozen-normalization implementation does not store
    them.  Every other source RGB key and every target RGB key is audited.
    """
    selected: dict[str, torch.Tensor] = {}
    ignored: list[str] = []
    unknown: list[str] = []
    mismatches: list[str] = []

    for source_key, source_value in source_state_dict.items():
        if not source_key.startswith(SOURCE_RGB_PREFIX):
            continue

        suffix = source_key.removeprefix(SOURCE_RGB_PREFIX)
        target_key = TARGET_RGB_PREFIX + suffix
        if target_key not in target_state_dict:
            if suffix.endswith(_IGNORED_SOURCE_SUFFIX):
                ignored.append(source_key)
            else:
                unknown.append(source_key)
            continue

        target_value = target_state_dict[target_key]
        if tuple(source_value.shape) != tuple(target_value.shape):
            mismatches.append(
                f"{source_key}: source={tuple(source_value.shape)}, "
                f"target={tuple(target_value.shape)}"
            )
            continue
        selected[target_key] = source_value

    if unknown:
        raise ValueError(
            "unknown RGB-backbone source keys: " + ", ".join(sorted(unknown))
        )
    if mismatches:
        raise ValueError("RGB-backbone shape mismatches: " + "; ".join(sorted(mismatches)))

    target_rgb_keys = {
        key for key in target_state_dict if key.startswith(TARGET_RGB_PREFIX)
    }
    missing = tuple(sorted(target_rgb_keys.difference(selected)))
    if missing:
        raise ValueError("missing target RGB-backbone keys: " + ", ".join(missing))

    return RGBBackboneTransferSelection(
        state_dict=selected,
        ignored_source_keys=tuple(sorted(ignored)),
        missing_target_keys=missing,
    )
