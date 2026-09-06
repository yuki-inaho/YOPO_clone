"""Strict Rotated JAX YOLO26 n/s/m backbone transfer into YOPO."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Mapping

import numpy as np
import torch
from torch import Tensor


@dataclass(frozen=True, slots=True)
class RotatedYOLO26TransferReport:
    """Exact ownership report for one Rotated-to-YOPO backbone transfer."""

    mapped: tuple[str, ...]
    missing: tuple[str, ...]
    shape_errors: tuple[str, ...]
    nonfinite: tuple[str, ...]
    excluded_target: tuple[str, ...]
    excluded_source: tuple[str, ...]
    ignored_non_backbone: tuple[str, ...]

    @property
    def ok(self) -> bool:
        return not (
            self.missing or self.shape_errors or self.nonfinite or self.excluded_source
        )

    def to_dict(self) -> dict[str, Any]:
        return {
            "ok": self.ok,
            "mapped": list(self.mapped),
            "missing": list(self.missing),
            "shape_errors": list(self.shape_errors),
            "nonfinite": list(self.nonfinite),
            "excluded_target": list(self.excluded_target),
            "excluded_source": list(self.excluded_source),
            "ignored_non_backbone": list(self.ignored_non_backbone),
        }


def _rotated_module_path(relative: str) -> str:
    relative = relative.removeprefix("layers.")
    layer, separator, remainder = relative.partition(".")
    if not separator:
        return f"yolo26/model/{layer}"
    if layer == "10":
        remainder = remainder.replace("block.attention.projection", "m.0.attn.proj")
        remainder = remainder.replace("block.attention", "m.0.attn")
        remainder = remainder.replace("block.ffn", "m.0.ffn")
    else:
        remainder = remainder.replace("block.bottleneck", "m.0.bottleneck")
        remainder = remainder.replace("block.c3k.blocks", "m.0.c3k.m")
        remainder = remainder.replace("block.c3k", "m.0.c3k")
    return f"yolo26/model/{layer}/{remainder.replace('.', '/')}"


def _source_contract(
    target_key: str,
    *,
    target_prefix: str,
    weights: str,
) -> tuple[str, bool] | None:
    relative = target_key.removeprefix(target_prefix)
    if relative.endswith(".conv.weight"):
        module = _rotated_module_path(relative.removesuffix(".conv.weight"))
        return f"{weights}/{module}/conv/kernel", True
    if ".norm." not in relative or relative.endswith("num_batches_tracked"):
        return None
    module, field = relative.rsplit(".norm.", 1)
    source_field = {
        "weight": (weights, "scale"),
        "bias": (weights, "bias"),
        "running_mean": ("batch_stats", "mean"),
        "running_var": ("batch_stats", "var"),
    }.get(field)
    if source_field is None:
        return None
    collection, field = source_field
    return f"{collection}/{_rotated_module_path(module)}/norm/{field}", False


def _is_selected_backbone_key(key: str, *, weights: str) -> bool:
    prefixes = (f"{weights}/yolo26/model/", "batch_stats/yolo26/model/")
    prefix = next(
        (candidate for candidate in prefixes if key.startswith(candidate)), None
    )
    if prefix is None:
        return False
    layer = key.removeprefix(prefix).partition("/")[0]
    return layer.isdigit() and 0 <= int(layer) <= 10


def _is_any_selected_collection_key(key: str) -> bool:
    return key.startswith(
        ("params/yolo26/model/", "ema/yolo26/model/", "batch_stats/yolo26/model/")
    )


def convert_rotated_yolo26_backbone_arrays(
    source_arrays: Mapping[str, np.ndarray],
    target_state: Mapping[str, Tensor],
    *,
    weights: str = "params",
    target_prefix: str = "backbone.rgb_backbone.",
    strict: bool = True,
) -> tuple[dict[str, Tensor], RotatedYOLO26TransferReport]:
    """Map every and only same-scale layers 0--10 leaf, rejecting drift."""

    if weights not in {"params", "ema"}:
        raise ValueError("weights must be 'params' or 'ema'")
    target_keys = sorted(key for key in target_state if key.startswith(target_prefix))
    if not target_keys:
        raise ValueError(f"target has no keys under {target_prefix!r}")

    converted: dict[str, Tensor] = {}
    mapped: list[str] = []
    missing: list[str] = []
    shape_errors: list[str] = []
    nonfinite: list[str] = []
    excluded_target: list[str] = []
    consumed: set[str] = set()

    for target_key in target_keys:
        target = target_state[target_key]
        contract = _source_contract(
            target_key, target_prefix=target_prefix, weights=weights
        )
        if contract is None:
            if target_key.endswith("num_batches_tracked"):
                excluded_target.append(target_key)
            else:
                missing.append(f"unsupported target leaf {target_key}")
            continue
        source_key, transpose = contract
        if source_key not in source_arrays:
            missing.append(f"{target_key} <- {source_key}")
            continue
        array = np.asarray(source_arrays[source_key])
        if not np.issubdtype(array.dtype, np.floating) or not np.isfinite(array).all():
            nonfinite.append(source_key)
            continue
        if transpose:
            if array.ndim != 4:
                shape_errors.append(
                    f"{source_key}: expected HWIO rank 4, got {array.shape}"
                )
                continue
            array = array.transpose(3, 2, 0, 1)
        if tuple(array.shape) != tuple(target.shape):
            shape_errors.append(
                f"{target_key}: converted shape {tuple(array.shape)} "
                f"!= target {tuple(target.shape)}"
            )
            continue
        converted[target_key] = torch.as_tensor(
            np.ascontiguousarray(array), dtype=target.dtype, device="cpu"
        )
        consumed.add(source_key)
        mapped.append(f"{source_key} -> {target_key}")

    excluded_source = sorted(
        key
        for key in source_arrays
        if _is_selected_backbone_key(key, weights=weights) and key not in consumed
    )
    ignored_non_backbone = sorted(
        key
        for key in source_arrays
        if _is_any_selected_collection_key(key)
        and not _is_selected_backbone_key(key, weights=weights)
    )
    report = RotatedYOLO26TransferReport(
        mapped=tuple(mapped),
        missing=tuple(missing),
        shape_errors=tuple(shape_errors),
        nonfinite=tuple(nonfinite),
        excluded_target=tuple(excluded_target),
        excluded_source=tuple(excluded_source),
        ignored_non_backbone=tuple(ignored_non_backbone),
    )
    if strict and not report.ok:
        raise ValueError(
            "strict Rotated YOLO26 transfer failed: "
            f"missing={report.missing[:3]}, "
            f"shape={report.shape_errors[:3]}, "
            f"nonfinite={report.nonfinite[:3]}, "
            f"extra={report.excluded_source[:3]}"
        )
    return converted, report


__all__ = [
    "RotatedYOLO26TransferReport",
    "convert_rotated_yolo26_backbone_arrays",
]
