"""Strict raw DEIM backbone and encoder transfer into one YOLO26 feature branch."""

import numpy as np
import torch

from yopo.utils.jax_feature_transfer import convert_jax_feature_arrays
from yopo.utils.jax_yolo26_transfer import convert_jax_yolo26_backbone_arrays


def convert_raw_feature_branch(source, branch):
    """Map all raw feature leaves; sum only a depth branch's RGB stem kernel."""
    groups = {}
    for group in ("backbone", "encoder"):
        groups[group] = {
            key: np.asarray(source[key])
            for key in source
            if key.startswith(f"params::{group}/")
        }
        if not groups[group]:
            raise ValueError(f"missing raw {group} collection")
        for key, value in groups[group].items():
            if not np.isfinite(value).all():
                raise ValueError(f"nonfinite raw feature leaf: {key}")
    stem_key = "params::backbone/0/conv/kernel"
    if stem_key not in groups["backbone"]:
        raise ValueError("missing raw stem")
    stem = groups["backbone"][stem_key]
    if stem.ndim != 4 or stem.shape[2] != 3:
        raise ValueError("source stem must have HWIO shape with three input channels")
    if branch.backbone.in_channels == 1:
        groups["backbone"][stem_key] = stem.sum(axis=2, keepdims=True)
    backbone, backbone_report = convert_jax_yolo26_backbone_arrays(
        groups["backbone"],
        {
            "backbone." + key: value
            for key, value in branch.backbone.state_dict().items()
        },
        weights="params",
        target_prefix="backbone.",
        strict=True,
    )
    encoder, encoder_report = convert_jax_feature_arrays(
        groups["encoder"],
        {"neck." + key: value for key, value in branch.encoder.state_dict().items()},
        weights="params",
        strict=True,
    )
    if encoder_report.excluded_source or encoder_report.excluded_target:
        raise ValueError("extra or unmapped encoder leaves")
    state = {
        **backbone,
        **{
            "encoder." + key.removeprefix("neck."): value
            for key, value in encoder.items()
        },
    }
    # BN counters have no JAX equivalent. Every other destination must be mapped.
    expected = {
        key for key in branch.state_dict() if not key.endswith("num_batches_tracked")
    }
    if set(state) != expected:
        raise ValueError("raw feature destination coverage mismatch")
    if len(backbone_report.mapped) != 200 or len(groups["encoder"]) != 23:
        raise ValueError("expected n/s backbone 200 and encoder 23 source leaves")
    if any(not torch.isfinite(value).all() for value in state.values()):
        raise ValueError("nonfinite converted feature leaf")
    return state, {
        "weights": "params",
        "scale": branch.scale,
        "backbone": backbone_report.to_dict(),
        "encoder": encoder_report.to_dict(),
        "encoder_source_leaves": len(groups["encoder"]),
        "depth_stem": "sum_input_channels"
        if branch.backbone.in_channels == 1
        else "unchanged",
    }
