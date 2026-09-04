"""Stable loading and validation for the shared RGB-D feature contract."""

from __future__ import annotations

import hashlib
import json
from pathlib import Path
from typing import Any


def feature_contract_fingerprint(payload: dict[str, Any]) -> str:
    canonical_payload = dict(payload)
    canonical_payload.pop('fingerprint_sha256', None)
    canonical = json.dumps(
        canonical_payload,
        sort_keys=True,
        separators=(',', ':'),
        ensure_ascii=True,
    ).encode('utf-8')
    return hashlib.sha256(canonical).hexdigest()


def _require_equal(actual: Any, expected: Any, label: str) -> None:
    if actual != expected:
        raise ValueError(
            f'feature contract {label} must be {expected!r}; received {actual!r}'
        )


def load_feature_contract(path: str | Path) -> dict[str, Any]:
    payload = json.loads(Path(path).read_text(encoding='utf-8'))
    if not isinstance(payload, dict):
        raise ValueError('feature contract root must be an object')
    _require_equal(payload.get('schema_version'), 1, 'schema_version')
    _require_equal(payload.get('contract_name'), 'rgbd_hgnetv2_hybrid_pafpn', 'name')
    _require_equal(
        payload.get('fingerprint_sha256'),
        feature_contract_fingerprint(payload),
        'fingerprint',
    )
    return payload


def validate_neck_feature_contract(neck: Any, payload: dict[str, Any]) -> None:
    """Validate the public portable-neck shape and AIFI attributes."""

    stack = payload['feature_stack']
    levels = payload['backbone']['levels']
    _require_equal(list(neck.in_channels), [level['fused_channels'] for level in levels], 'neck inputs')
    _require_equal(neck.hidden_dim, stack['projection_channels'], 'projection channels')
    _require_equal(neck.num_heads, stack['aifi_heads'], 'AIFI heads')
    _require_equal(neck.ffn_dim, stack['aifi_ffn_channels'], 'AIFI FFN channels')
    _require_equal(neck.num_aifi_layers, stack['aifi_layers'], 'AIFI layers')
    _require_equal(neck.num_outs, 4, 'output count')
