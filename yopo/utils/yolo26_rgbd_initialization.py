"""Audited stage-8 reuse contract for YOLO26m RGB-D initialization."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Mapping

import torch


_REUSED_PREFIXES = (
    "backbone.depth_backbone.",
    "encoder.",
    "decoder.",
    "bbox_head.",
    "query_embedding.",
    "memory_trans_fc.",
    "memory_trans_norm.",
    "dn_query_generator.",
)
_REUSED_EXACT = {"level_embed"}


def _is_reused(key: str) -> bool:
    if key in _REUSED_EXACT:
        return True
    if key.startswith("neck."):
        return not key.startswith("neck.projections.")
    return key.startswith(_REUSED_PREFIXES)


@dataclass(frozen=True, slots=True)
class Stage8ReuseReport:
    mapped: tuple[str, ...]
    missing: tuple[str, ...]
    shape_errors: tuple[str, ...]
    fresh_target: tuple[str, ...]
    excluded_source: tuple[str, ...]
    unexpected_source: tuple[str, ...]

    @property
    def ok(self) -> bool:
        return not self.missing and not self.shape_errors and not self.unexpected_source

    def to_dict(self) -> dict[str, Any]:
        return {
            "ok": self.ok,
            "mapped": list(self.mapped),
            "missing": list(self.missing),
            "shape_errors": list(self.shape_errors),
            "fresh_target": list(self.fresh_target),
            "excluded_source": list(self.excluded_source),
            "unexpected_source": list(self.unexpected_source),
        }


def select_stage8_reuse_state(
    source_state: Mapping[str, torch.Tensor],
    target_state: Mapping[str, torch.Tensor],
    *,
    strict: bool = True,
) -> tuple[dict[str, torch.Tensor], Stage8ReuseReport]:
    """Select every allowlisted same-shape YOPO leaf and nothing else."""

    selected: dict[str, torch.Tensor] = {}
    missing: list[str] = []
    shape_errors: list[str] = []
    mapped: list[str] = []

    for target_key, target_value in target_state.items():
        if not _is_reused(target_key):
            continue
        if target_key not in source_state:
            missing.append(target_key)
            continue
        source_value = source_state[target_key]
        if tuple(source_value.shape) != tuple(target_value.shape):
            shape_errors.append(
                f"{target_key}: source={tuple(source_value.shape)}, "
                f"target={tuple(target_value.shape)}"
            )
            continue
        selected[target_key] = source_value.detach().cpu()
        mapped.append(target_key)

    unexpected = sorted(
        key for key in source_state if _is_reused(key) and key not in target_state
    )
    report = Stage8ReuseReport(
        mapped=tuple(sorted(mapped)),
        missing=tuple(sorted(missing)),
        shape_errors=tuple(sorted(shape_errors)),
        fresh_target=tuple(sorted(set(target_state).difference(selected))),
        excluded_source=tuple(sorted(key for key in source_state if not _is_reused(key))),
        unexpected_source=tuple(unexpected),
    )
    if strict and not report.ok:
        raise ValueError(
            "strict stage8 reuse failed: "
            f"missing={report.missing[:3]}, "
            f"shape={report.shape_errors[:3]}, "
            f"unexpected={report.unexpected_source[:3]}"
        )
    return selected, report


__all__ = ["Stage8ReuseReport", "select_stage8_reuse_state"]
