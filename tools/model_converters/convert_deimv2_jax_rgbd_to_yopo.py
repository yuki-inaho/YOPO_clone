"""Convert a portable DEIMv2-JAX RGB-D feature stack into a YOPO partial checkpoint."""

from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path
from typing import Any

from mmengine.config import Config
from mmengine.utils import import_modules_from_strings
import numpy as np
import torch

from yopo.registry import MODELS
from yopo.utils.jax_feature_transfer import convert_jax_feature_arrays


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open('rb') as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b''):
            digest.update(chunk)
    return digest.hexdigest()


def _target_feature_state(config: Config) -> dict[str, torch.Tensor]:
    custom_imports = config.get('custom_imports')
    if custom_imports:
        import_modules_from_strings(**custom_imports)
    backbone = MODELS.build(config.model.backbone)
    neck = MODELS.build(config.model.neck)
    return {
        **{f'backbone.{key}': value for key, value in backbone.state_dict().items()},
        **{f'neck.{key}': value for key, value in neck.state_dict().items()},
    }


def convert_checkpoint(
    source_checkpoint: Path,
    target_config: Path,
    output: Path,
    *,
    weights: str,
) -> tuple[Path, Path]:
    arrays_path = source_checkpoint / 'arrays.npz'
    manifest_path = source_checkpoint / 'manifest.json'
    if not arrays_path.is_file() or not manifest_path.is_file():
        raise FileNotFoundError(
            f'portable checkpoint requires arrays.npz and manifest.json: {source_checkpoint}'
        )
    config = Config.fromfile(str(target_config))
    target_state = _target_feature_state(config)
    with np.load(arrays_path, allow_pickle=False) as source_arrays:
        converted, transfer_report = convert_jax_feature_arrays(
            source_arrays,
            target_state,
            weights=weights,
            strict=True,
        )

    source_manifest: dict[str, Any] = json.loads(manifest_path.read_text(encoding='utf-8'))
    metadata = {
        'format': 'deimv2_jax_rgbd_to_yopo_feature_v1',
        'source_checkpoint': str(source_checkpoint.resolve()),
        'source_arrays_sha256': _sha256(arrays_path),
        'source_manifest_sha256': _sha256(manifest_path),
        'source_step': source_manifest.get('step', source_manifest.get('metadata', {}).get('step')),
        'source_weights': weights,
        'target_config': str(target_config.resolve()),
        'transfer_report': transfer_report.to_dict(),
    }
    output.parent.mkdir(parents=True, exist_ok=True)
    torch.save({'state_dict': converted, 'meta': metadata}, output)
    metadata['output_sha256'] = _sha256(output)
    report_path = output.with_suffix(output.suffix + '.json')
    report_path.write_text(json.dumps(metadata, indent=2, sort_keys=True), encoding='utf-8')
    return output, report_path


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--source-checkpoint', type=Path, required=True)
    parser.add_argument('--target-config', type=Path, required=True)
    parser.add_argument('--output', type=Path, required=True)
    parser.add_argument('--weights', choices=('params', 'ema'), default='ema')
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    output, report = convert_checkpoint(
        args.source_checkpoint.expanduser().resolve(),
        args.target_config.expanduser().resolve(),
        args.output.expanduser().resolve(),
        weights=args.weights,
    )
    print(output)
    print(report)


if __name__ == '__main__':
    main()
