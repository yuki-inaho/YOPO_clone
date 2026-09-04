"""Strict DEIMv2-JAX RGB-D backbone/feature-stack to YOPO weight mapping."""

from __future__ import annotations

from dataclasses import dataclass
import re
from typing import Any, Mapping

import numpy as np
import torch
from torch import Tensor


@dataclass(frozen=True, slots=True)
class JAXFeatureTransferReport:
    mapped: tuple[str, ...]
    missing: tuple[str, ...]
    shape_errors: tuple[str, ...]
    excluded_target: tuple[str, ...]
    excluded_source: tuple[str, ...]

    @property
    def ok(self) -> bool:
        return not self.missing and not self.shape_errors

    def to_dict(self) -> dict[str, Any]:
        return {
            'ok': self.ok,
            'mapped': list(self.mapped),
            'missing': list(self.missing),
            'shape_errors': list(self.shape_errors),
            'excluded_target': list(self.excluded_target),
            'excluded_source': list(self.excluded_source),
        }


def _backbone_source_key(target_key: str, weights: str) -> tuple[str, bool] | None:
    branches = {
        'backbone.rgb_backbone.': 'backbone/params/backbone/',
        'backbone.depth_backbone.': 'backbone/depth_params/backbone/',
    }
    branch_prefix = next((prefix for prefix in branches if target_key.startswith(prefix)), None)
    if branch_prefix is None:
        return None
    relative = target_key[len(branch_prefix):]
    collection = branches[branch_prefix]
    transpose = False
    if relative.endswith('.conv.weight'):
        relative = relative[:-len('.conv.weight')] + '/conv/kernel'
        transpose = True
    elif '.bn.' in relative:
        path, field = relative.rsplit('.bn.', 1)
        if field == 'weight':
            relative = path + '/norm/scale'
        elif field == 'bias':
            relative = path + '/norm/bias'
        elif field == 'running_mean':
            collection = collection.replace('/params/', '/batch_stats/').replace(
                '/depth_params/', '/depth_batch_stats/'
            )
            relative = path + '/norm/mean'
        elif field == 'running_var':
            collection = collection.replace('/params/', '/batch_stats/').replace(
                '/depth_params/', '/depth_batch_stats/'
            )
            relative = path + '/norm/var'
        else:
            return None
    elif '.lab.' in relative:
        path, field = relative.rsplit('.lab.', 1)
        if field not in {'scale', 'bias'}:
            return None
        relative = f'{path}/lab/{field}'
    else:
        return None
    return f'{weights}::{collection}{relative.replace(".", "/")}', transpose


def _single_source_key(target_key: str, weights: str) -> tuple[str, str] | None:
    backbone = _backbone_source_key(target_key, weights)
    if backbone is not None:
        source, transpose = backbone
        return source, 'conv' if transpose else 'identity'
    if target_key == 'backbone.depth_beta':
        return f'{weights}::backbone/fusion_beta', 'identity'
    adapter = re.fullmatch(r'backbone\.depth_adapters\.(\d+)\.weight', target_key)
    if adapter:
        return (
            f'{weights}::backbone/fusion_params/adapters/{adapter.group(1)}/kernel',
            'conv',
        )
    conv = re.fullmatch(r'neck\.(projections|lateral|pan)\.(\d+)\.weight', target_key)
    if conv:
        return f'{weights}::encoder/{conv.group(1)}/{conv.group(2)}/kernel', 'conv'
    norm = re.fullmatch(r'neck\.aifi\.(\d+)\.(norm1|norm2)\.(weight|bias)', target_key)
    if norm:
        return (
            f'{weights}::encoder/aifi/{norm.group(1)}/{norm.group(2)}/{norm.group(3)}',
            'identity',
        )
    ffn = re.fullmatch(r'neck\.aifi\.(\d+)\.(ffn1|ffn2)\.(weight|bias)', target_key)
    if ffn:
        operation = 'dense' if ffn.group(3) == 'weight' else 'identity'
        source_field = 'kernel' if ffn.group(3) == 'weight' else 'bias'
        return (
            f'{weights}::encoder/aifi/{ffn.group(1)}/{ffn.group(2)}/{source_field}',
            operation,
        )
    output = re.fullmatch(
        r'neck\.aifi\.(\d+)\.attention\.out_proj\.(weight|bias)', target_key
    )
    if output:
        source_field = 'kernel' if output.group(2) == 'weight' else 'bias'
        operation = 'dense' if output.group(2) == 'weight' else 'identity'
        return (
            f'{weights}::encoder/aifi/{output.group(1)}/attention/output/{source_field}',
            operation,
        )
    return None


def _qkv_source_keys(target_key: str, weights: str) -> tuple[tuple[str, ...], str] | None:
    match = re.fullmatch(
        r'neck\.aifi\.(\d+)\.attention\.in_proj_(weight|bias)', target_key
    )
    if not match:
        return None
    field = 'kernel' if match.group(2) == 'weight' else 'bias'
    keys = tuple(
        f'{weights}::encoder/aifi/{match.group(1)}/attention/{name}/{field}'
        for name in ('query', 'key', 'value')
    )
    return keys, 'qkv_dense' if field == 'kernel' else 'qkv_identity'


def _convert(values: tuple[np.ndarray, ...], operation: str) -> np.ndarray:
    if operation == 'identity':
        return values[0]
    if operation == 'conv':
        return values[0].transpose(3, 2, 0, 1)
    if operation == 'dense':
        return values[0].T
    if operation == 'qkv_dense':
        return np.concatenate([value.T for value in values], axis=0)
    if operation == 'qkv_identity':
        return np.concatenate(values, axis=0)
    raise AssertionError(f'unsupported transfer operation {operation}')


def convert_jax_feature_arrays(
    source_arrays: Mapping[str, np.ndarray],
    target_state: Mapping[str, Tensor],
    *,
    weights: str = 'ema',
    strict: bool = True,
) -> tuple[dict[str, Tensor], JAXFeatureTransferReport]:
    """Convert every contract-shared target leaf and reject silent omissions."""

    if weights not in {'params', 'ema'}:
        raise ValueError("weights must be 'params' or 'ema'")
    converted: dict[str, Tensor] = {}
    mapped: list[str] = []
    missing: list[str] = []
    shape_errors: list[str] = []
    excluded_target: list[str] = []
    consumed: set[str] = set()

    for target_key, target in target_state.items():
        if target_key.startswith('neck.derived_p6.') or target_key.endswith('num_batches_tracked'):
            excluded_target.append(target_key)
            continue
        qkv = _qkv_source_keys(target_key, weights)
        single = _single_source_key(target_key, weights)
        if qkv is not None:
            source_keys, operation = qkv
        elif single is not None:
            source_key, operation = single
            source_keys = (source_key,)
        else:
            # Task heads, YOPO deformable encoder/decoder, and all other leaves
            # are outside the shared feature contract.
            continue
        absent = [key for key in source_keys if key not in source_arrays]
        if absent:
            missing.append(f'{target_key} <- {absent}')
            continue
        array = _convert(tuple(np.asarray(source_arrays[key]) for key in source_keys), operation)
        if tuple(array.shape) != tuple(target.shape):
            shape_errors.append(
                f'{target_key}: converted shape {tuple(array.shape)} != target {tuple(target.shape)}'
            )
            continue
        converted[target_key] = torch.as_tensor(
            np.ascontiguousarray(array), dtype=target.dtype, device='cpu'
        )
        consumed.update(source_keys)
        mapped.append(f'{" + ".join(source_keys)} -> {target_key}')

    selected_prefix = f'{weights}::'
    excluded_source = sorted(
        key
        for key in source_arrays
        if key.startswith(selected_prefix) and key not in consumed
    )
    report = JAXFeatureTransferReport(
        mapped=tuple(mapped),
        missing=tuple(missing),
        shape_errors=tuple(shape_errors),
        excluded_target=tuple(sorted(excluded_target)),
        excluded_source=tuple(excluded_source),
    )
    if strict and not report.ok:
        raise ValueError(
            'strict JAX feature transfer failed: '
            f'missing={report.missing[:3]}, shape_errors={report.shape_errors[:3]}'
        )
    return converted, report
