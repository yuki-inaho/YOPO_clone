"""Auditable weight transplantation into the compact B1+B0 YOPO topology."""

from __future__ import annotations

import json
import re
from collections import Counter
from collections.abc import Mapping
from dataclasses import dataclass
from pathlib import Path

import torch
from torch import Tensor

from .group_fisher import select_top_channels

_FFN_FIRST = re.compile(
    r"^(encoder|decoder)\.layers\.(\d+)\.ffn\.layers\.0\.0\.weight$"
)
_INDEXED_HEAD = re.compile(r"^(bbox_head\.[^.]+)\.(\d+)(\..+)$")
_REINITIALIZED_PREFIXES = (
    "backbone.rgb_backbone.",
    "backbone.depth_adapters.",
    "neck.",
)


@dataclass(frozen=True, slots=True)
class CompactTransferResult:
    state_dict: Mapping[str, Tensor]
    report: Mapping[str, object]


def load_pruning_choices(
    path: str | Path | None,
    *,
    allow_smoke: bool = False,
) -> dict[str, tuple[int, ...]]:
    if path is None:
        return {}
    source = Path(path)
    payload = json.loads(source.read_text(encoding="utf-8"))
    metadata = payload.get("metadata", {})
    if isinstance(metadata, dict) and metadata:
        calibration = metadata.get("calibration")
        purpose = metadata.get("purpose")
        estimator = metadata.get("estimator")
        if calibration not in ("synthetic_smoke_only", "real_train_data"):
            raise ValueError("pruning plan metadata has an unknown calibration")
        if purpose not in ("smoke", "selection"):
            raise ValueError("pruning plan metadata must declare its purpose")
        if estimator not in ("group_fisher", "group_taylor"):
            raise ValueError("pruning plan metadata has an unknown estimator")
        if calibration == "synthetic_smoke_only" and purpose != "smoke":
            raise ValueError("synthetic calibration cannot be a selection plan")
        if purpose == "smoke" and not allow_smoke:
            raise ValueError(
                "smoke pruning plans require explicit allow_smoke=True"
            )
    raw = payload.get("choices", payload)
    if not isinstance(raw, dict):
        raise ValueError("pruning choices must be a mapping")
    choices: dict[str, tuple[int, ...]] = {}
    for name, indices in raw.items():
        if not isinstance(name, str) or not isinstance(indices, list):
            raise ValueError("pruning choices must map names to index lists")
        choices[name] = tuple(int(index) for index in indices)
    return choices


def _head_maxima(state: Mapping[str, Tensor]) -> dict[str, int]:
    maxima: dict[str, int] = {}
    for key in state:
        match = _INDEXED_HEAD.match(key)
        if match:
            maxima[match.group(1)] = max(
                maxima.get(match.group(1), -1), int(match.group(2))
            )
    return maxima


def _ffn_keys(first_weight: str) -> tuple[str, str, str, str]:
    prefix = first_weight.removesuffix(".layers.0.0.weight")
    return (
        first_weight,
        f"{prefix}.layers.0.0.bias",
        f"{prefix}.layers.1.weight",
        f"{prefix}.layers.1.bias",
    )


def _group_l2_indices(
    source: Mapping[str, Tensor],
    first_weight: str,
    remaining: int,
) -> tuple[int, ...]:
    first_key, first_bias_key, second_key, _ = _ffn_keys(first_weight)
    first = source[first_key].float()
    second = source[second_key].float()
    if first.ndim != 2 or second.ndim != 2 or first.shape[0] != second.shape[1]:
        raise ValueError(f"unsupported FFN tensors for {first_weight}")
    squared = first.square().sum(dim=1) + second.square().sum(dim=0)
    if first_bias_key in source:
        squared = squared + source[first_bias_key].float().square()
    return select_top_channels(squared.sqrt(), remaining)


def _validate_choice(indices: tuple[int, ...], source_width: int, target_width: int) -> None:
    if len(indices) != target_width:
        raise ValueError(
            f"pruning choice keeps {len(indices)} channels, expected {target_width}"
        )
    if tuple(sorted(set(indices))) != indices:
        raise ValueError("pruning choice indices must be sorted and unique")
    if any(index < 0 or index >= source_width for index in indices):
        raise ValueError("pruning choice contains an out-of-range channel")


def build_compact_yopo_transfer_state(
    source_state: Mapping[str, Tensor],
    target_state: Mapping[str, Tensor],
    *,
    pruning_choices: Mapping[str, tuple[int, ...]] | None = None,
) -> CompactTransferResult:
    """Select, remap, and slice weights for the known compact architecture.

    RGB B1, its residual adapters, and the neck are deliberately reinitialized.
    Transformer FFN hidden axes are physically selected as coupled first-output
    and second-input channels.  If no calibrated plan is supplied, deterministic
    group L2 is used as a data-free smoke baseline.
    """

    choices = dict(pruning_choices or {})
    selected: dict[str, Tensor] = {}
    loaded_groups: Counter[str] = Counter()
    exact_keys: list[str] = []
    remapped_keys: list[dict[str, str]] = []
    sliced_groups: list[dict[str, object]] = []
    excluded_keys: list[str] = []
    shape_mismatches: list[dict[str, object]] = []

    source_head_max = _head_maxima(source_state)
    target_head_max = _head_maxima(target_state)

    for target_key, target_value in target_state.items():
        if target_key.startswith(_REINITIALIZED_PREFIXES):
            excluded_keys.append(target_key)
            continue
        head_match = _INDEXED_HEAD.match(target_key)
        source_key = target_key
        if head_match:
            prefix, index_text, suffix = head_match.groups()
            index = int(index_text)
            if (
                prefix in source_head_max
                and prefix in target_head_max
                and index == target_head_max[prefix]
                and source_head_max[prefix] > target_head_max[prefix]
            ):
                source_key = f"{prefix}.{source_head_max[prefix]}{suffix}"
        if source_key not in source_state:
            continue
        source_value = source_state[source_key]
        if source_value.shape == target_value.shape:
            selected[target_key] = source_value
            loaded_groups[target_key.split(".", 1)[0]] += 1
            if source_key == target_key:
                exact_keys.append(target_key)
            else:
                remapped_keys.append({"source": source_key, "target": target_key})

    for target_key, target_value in target_state.items():
        match = _FFN_FIRST.match(target_key)
        if not match:
            continue
        source_keys = _ffn_keys(target_key)
        if any(key not in source_state for key in source_keys):
            raise KeyError(f"source checkpoint misses FFN group for {target_key}")
        target_keys = source_keys
        if any(key not in target_state for key in target_keys):
            raise KeyError(f"target model misses FFN group for {target_key}")
        first_source = source_state[source_keys[0]]
        first_target = target_state[target_keys[0]]
        if first_source.shape == first_target.shape:
            continue
        source_width = int(first_source.shape[0])
        target_width = int(first_target.shape[0])
        unit_name = f"{match.group(1)}.layers.{match.group(2)}.ffn.hidden"
        indices = choices.get(unit_name)
        selection = "importance_plan"
        if indices is None:
            if choices:
                raise KeyError(
                    f"pruning plan misses required target group {unit_name!r}"
                )
            indices = _group_l2_indices(source_state, target_key, target_width)
            selection = "group_l2"
        _validate_choice(indices, source_width, target_width)
        index = torch.tensor(indices, dtype=torch.long)
        sliced = {
            target_keys[0]: source_state[source_keys[0]].index_select(0, index),
            target_keys[1]: source_state[source_keys[1]].index_select(0, index),
            target_keys[2]: source_state[source_keys[2]].index_select(1, index),
            target_keys[3]: source_state[source_keys[3]],
        }
        for key, value in sliced.items():
            if value.shape != target_state[key].shape:
                raise ValueError(
                    f"sliced tensor {key} has shape {tuple(value.shape)}, "
                    f"expected {tuple(target_state[key].shape)}"
                )
            selected[key] = value
        sliced_groups.append(
            {
                "unit": unit_name,
                "selection": selection,
                "source_channels": source_width,
                "target_channels": target_width,
                "kept_indices": list(indices),
            }
        )

    for key, target_value in target_state.items():
        if key in selected or key.startswith(_REINITIALIZED_PREFIXES):
            continue
        if key in source_state and source_state[key].shape != target_value.shape:
            shape_mismatches.append(
                {
                    "key": key,
                    "source_shape": list(source_state[key].shape),
                    "target_shape": list(target_value.shape),
                }
            )

    unexpected = sorted(set(selected) - set(target_state))
    if unexpected:
        raise RuntimeError(f"transfer produced unexpected target keys: {unexpected}")
    for key, value in selected.items():
        if value.shape != target_state[key].shape:
            raise RuntimeError(f"transfer produced a shape mismatch for {key}")

    report = {
        "schema_version": 1,
        "selection_default": "importance_plan" if choices else "group_l2",
        "loaded_key_count": len(selected),
        "loaded_tensor_elements": sum(value.numel() for value in selected.values()),
        "loaded_groups": dict(sorted(loaded_groups.items())),
        "exact_key_count": len(exact_keys),
        "remapped_prediction_keys": remapped_keys,
        "sliced_ffn_groups": sliced_groups,
        "reinitialized_prefixes": list(_REINITIALIZED_PREFIXES),
        "reinitialized_key_count": len(excluded_keys),
        "shape_mismatches_not_loaded": shape_mismatches,
        "target_only_keys": sorted(set(target_state) - set(selected)),
        "source_only_keys": sorted(set(source_state) - set(target_state)),
    }
    return CompactTransferResult(state_dict=selected, report=report)
