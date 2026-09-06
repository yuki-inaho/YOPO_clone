"""Strict JAX HWIO to PyTorch OIHW transfer for YOLO26 layers 0--10."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Mapping

import numpy as np
import torch
from torch import Tensor


@dataclass(frozen=True, slots=True)
class JAXYOLO26TransferReport:
    mapped: tuple[str, ...]
    missing: tuple[str, ...]
    shape_errors: tuple[str, ...]
    excluded_target: tuple[str, ...]
    excluded_source: tuple[str, ...]

    @property
    def ok(self) -> bool:
        return not self.missing and not self.shape_errors and not self.excluded_source

    def to_dict(self) -> dict[str, Any]:
        return {
            "ok": self.ok,
            "mapped": list(self.mapped),
            "missing": list(self.missing),
            "shape_errors": list(self.shape_errors),
            "excluded_target": list(self.excluded_target),
            "excluded_source": list(self.excluded_source),
        }


def _source_contract(
    target_key: str,
    *,
    target_prefix: str,
    weights: str,
) -> tuple[str, bool] | None:
    relative = target_key.removeprefix(target_prefix)
    if relative.startswith("layers."):
        relative = relative.removeprefix("layers.")
    if relative.endswith(".conv.weight"):
        path = relative.removesuffix(".conv.weight").replace(".", "/")
        return f"{weights}::backbone/{path}/conv/kernel", True
    if ".norm." not in relative or relative.endswith("num_batches_tracked"):
        return None
    path, field = relative.rsplit(".norm.", 1)
    if field not in {"weight", "bias", "running_mean", "running_var"}:
        return None
    return f"{weights}::backbone/{path.replace('.', '/')}/norm/{field}", False


def convert_jax_yolo26_backbone_arrays(
    source_arrays: Mapping[str, np.ndarray],
    target_state: Mapping[str, Tensor],
    *,
    weights: str = "ema",
    target_prefix: str = "backbone.rgb_backbone.",
    strict: bool = True,
) -> tuple[dict[str, Tensor], JAXYOLO26TransferReport]:
    """Map every and only one explicit YOLO26 backbone, rejecting drift."""

    if weights not in {"params", "ema"}:
        raise ValueError("weights must be 'params' or 'ema'")
    target_keys = sorted(key for key in target_state if key.startswith(target_prefix))
    if not target_keys:
        raise ValueError(f"target has no keys under {target_prefix!r}")

    converted: dict[str, Tensor] = {}
    mapped: list[str] = []
    missing: list[str] = []
    shape_errors: list[str] = []
    excluded_target: list[str] = []
    consumed: set[str] = set()

    for target_key in target_keys:
        target = target_state[target_key]
        contract = _source_contract(
            target_key,
            target_prefix=target_prefix,
            weights=weights,
        )
        if contract is None:
            if target_key.endswith("num_batches_tracked"):
                excluded_target.append(target_key)
                continue
            missing.append(f"unsupported target leaf {target_key}")
            continue
        source_key, transpose = contract
        if source_key not in source_arrays:
            missing.append(f"{target_key} <- {source_key}")
            continue
        array = np.asarray(source_arrays[source_key])
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

    source_prefix = f"{weights}::backbone/"
    extra = sorted(
        key
        for key in source_arrays
        if key.startswith(source_prefix) and key not in consumed
    )
    report = JAXYOLO26TransferReport(
        mapped=tuple(mapped),
        missing=tuple(missing),
        shape_errors=tuple(shape_errors),
        excluded_target=tuple(excluded_target),
        excluded_source=tuple(extra),
    )
    if strict and not report.ok:
        raise ValueError(
            "strict JAX YOLO26 transfer failed: "
            f"missing={report.missing[:3]}, "
            f"shape={report.shape_errors[:3]}, "
            f"extra={report.excluded_source[:3]}"
        )
    return converted, report


__all__ = ["JAXYOLO26TransferReport", "convert_jax_yolo26_backbone_arrays"]
