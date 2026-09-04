"""Prepare a YOPO CoP checkpoint for metric-depth residual anchoring.

The ordinary YOPO CoP z heads emit absolute depth in metres.  The structural
sensor-depth anchor changes that contract to ``sensor_depth + residual``.  A
weights-only transfer therefore has to zero the CoP z output heads before the
first anchored update; all RGB/OBB/ellipse/shape features remain untouched.
"""

from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path
from typing import Any

import torch


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _tensor_sha256(value: torch.Tensor) -> str:
    array = value.detach().cpu().contiguous().numpy()
    return hashlib.sha256(array.tobytes()).hexdigest()


def prepare_checkpoint(source: Path, output: Path) -> tuple[Path, Path]:
    checkpoint = torch.load(source, map_location="cpu", weights_only=False)
    if not isinstance(checkpoint, dict) or "state_dict" not in checkpoint:
        raise ValueError("YOPO checkpoint must be a mapping with state_dict")
    state_dict = checkpoint["state_dict"]
    if not isinstance(state_dict, dict):
        raise ValueError("YOPO checkpoint state_dict must be a mapping")

    keys = sorted(
        key for key in state_dict
        if key.startswith("bbox_head.cop_z_out.")
        and (key.endswith(".weight") or key.endswith(".bias"))
    )
    if not keys:
        raise ValueError("no bbox_head.cop_z_out weight/bias keys found")

    before = {}
    for key in keys:
        value = state_dict[key]
        if not isinstance(value, torch.Tensor):
            raise TypeError(f"{key} is not a tensor")
        if key.endswith(".weight") and value.ndim != 2:
            raise ValueError(f"unexpected CoP z weight shape for {key}: {tuple(value.shape)}")
        if key.endswith(".bias") and value.ndim != 1:
            raise ValueError(f"unexpected CoP z bias shape for {key}: {tuple(value.shape)}")
        before[key] = _tensor_sha256(value)
        state_dict[key] = torch.zeros_like(value)

    prepared = dict(checkpoint)
    prepared["state_dict"] = state_dict
    meta = dict(prepared.get("meta") or {})
    meta.update({
        "format": "yopo_sensor_depth_residual_init_v1",
        "source_checkpoint": str(source.resolve()),
        "source_checkpoint_sha256": _sha256(source),
        "reset_contract": "bbox_head.cop_z_out.* => zero residual output",
        "reset_keys": keys,
        "source_tensor_sha256": before,
    })
    prepared["meta"] = meta
    # This is intentionally a weights-only initialization.  Carrying a stale
    # optimizer/EMA state would make the transition look like a resume.
    for key in ("optimizer", "param_schedulers", "message_hub"):
        prepared.pop(key, None)

    output.parent.mkdir(parents=True, exist_ok=True)
    torch.save(prepared, output)
    report: dict[str, Any] = {
        "format": "yopo_sensor_depth_residual_init_v1",
        "source_checkpoint": str(source.resolve()),
        "source_checkpoint_sha256": _sha256(source),
        "output_checkpoint": str(output.resolve()),
        "output_checkpoint_sha256": _sha256(output),
        "reset_keys": keys,
        "source_tensor_sha256": before,
        "output_tensor_sha256": {
            key: _tensor_sha256(state_dict[key]) for key in keys
        },
    }
    report_path = output.with_suffix(output.suffix + ".json")
    report_path.write_text(
        json.dumps(report, indent=2, sort_keys=True), encoding="utf-8")
    return output, report_path


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--source", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    output, report = prepare_checkpoint(
        args.source.expanduser().resolve(), args.output.expanduser().resolve())
    print(output)
    print(report)


if __name__ == "__main__":
    main()
