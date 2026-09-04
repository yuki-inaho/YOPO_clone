"""Compare one JAX RGB-D feature fixture with the mapped YOPO implementation."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

from mmengine.config import Config
from mmengine.utils import import_modules_from_strings
import numpy as np
import torch
import torch.nn as nn

from yopo.registry import MODELS
from yopo.models.necks.portable_hybrid_encoder import position_encoding_2d
from yopo.utils.feature_contract import load_feature_contract
from yopo.utils.jax_feature_transfer import convert_jax_feature_arrays


class _SharedFeatureModel(nn.Module):
    def __init__(self, backbone: nn.Module, neck: nn.Module) -> None:
        super().__init__()
        self.backbone = backbone
        self.neck = neck

    def forward(self, images: torch.Tensor):
        backbone_features = tuple(self.backbone(images))
        shared = self.neck.forward_shared(backbone_features)
        return backbone_features, shared


def _compare(actual: np.ndarray, expected_nhwc: np.ndarray, atol: float, rtol: float):
    expected = expected_nhwc.transpose(0, 3, 1, 2)
    difference = np.abs(actual - expected)
    return {
        'shape_nchw': list(actual.shape),
        'max_abs_error': float(difference.max(initial=0.0)),
        'mean_abs_error': float(difference.mean()),
        'finite': bool(np.isfinite(actual).all()),
        'within_tolerance': bool(np.allclose(actual, expected, atol=atol, rtol=rtol)),
    }


def _compare_same(actual: np.ndarray, expected: np.ndarray, atol: float, rtol: float):
    difference = np.abs(actual - expected)
    return {
        'shape': list(actual.shape),
        'max_abs_error': float(difference.max(initial=0.0)),
        'mean_abs_error': float(difference.mean()),
        'finite': bool(np.isfinite(actual).all()),
        'within_tolerance': bool(np.allclose(actual, expected, atol=atol, rtol=rtol)),
    }


def check_parity(
    fixture_dir: Path,
    target_config: Path,
    report_path: Path,
    *,
    atol: float,
    rtol: float,
) -> dict:
    fixture_manifest = json.loads(
        (fixture_dir / 'manifest.json').read_text(encoding='utf-8')
    )
    target_contract = load_feature_contract(
        target_config.parents[2] / 'contracts' / 'rgbd_feature_stack_v1.json'
    )
    if (
        fixture_manifest['contract_fingerprint_sha256']
        != target_contract['fingerprint_sha256']
    ):
        raise ValueError('fixture and target feature-contract fingerprints differ')
    config = Config.fromfile(str(target_config))
    if config.get('custom_imports'):
        import_modules_from_strings(**config.custom_imports)
    model = _SharedFeatureModel(
        MODELS.build(config.model.backbone), MODELS.build(config.model.neck)
    ).eval()
    target_state = model.state_dict()
    with np.load(fixture_dir / 'arrays.npz', allow_pickle=False) as source:
        converted, transfer = convert_jax_feature_arrays(
            source, target_state, weights='params', strict=True
        )
    incompatible = model.load_state_dict(converted, strict=False)
    allowed_missing = sorted(key for key in target_state if key.startswith('neck.derived_p6.'))
    if sorted(incompatible.missing_keys) != allowed_missing or incompatible.unexpected_keys:
        raise ValueError(
            'mapped state did not match the strict target allowlist: '
            f'missing={incompatible.missing_keys}, unexpected={incompatible.unexpected_keys}'
        )

    with np.load(fixture_dir / 'fixture.npz', allow_pickle=False) as fixture:
        images = torch.from_numpy(fixture['images'].transpose(0, 3, 1, 2).copy())
        with torch.no_grad():
            backbone_features, shared = model(images)
            projected = tuple(
                layer(feature)
                for layer, feature in zip(model.neck.projections, backbone_features)
            )
            deepest = projected[-1]
            batch, channels, height, width = deepest.shape
            positioned_tokens = deepest.flatten(2).transpose(1, 2)
            positioned_tokens = positioned_tokens + position_encoding_2d(
                height,
                width,
                channels,
                device=positioned_tokens.device,
                dtype=positioned_tokens.dtype,
            ).unsqueeze(0)
            aifi_tokens = positioned_tokens
            for layer in model.neck.aifi:
                aifi_tokens = layer(aifi_tokens)
            top_down = list(projected)
            top_down[-1] = aifi_tokens.transpose(1, 2).reshape(
                batch, channels, height, width
            )
            for lateral_index, level in enumerate(range(len(top_down) - 2, -1, -1)):
                upsampled = torch.nn.functional.interpolate(
                    top_down[level + 1],
                    size=top_down[level].shape[-2:],
                    mode='nearest-exact',
                )
                top_down[level] = torch.nn.functional.silu(
                    model.neck.lateral[lateral_index](
                        torch.cat((top_down[level], upsampled), dim=1)
                    )
                )
        backbone_results = [
            _compare(value.numpy(), fixture[f'backbone_{index}'], atol, rtol)
            for index, value in enumerate(backbone_features)
        ]
        shared_results = [
            _compare(value.numpy(), fixture[f'shared_{index}'], atol, rtol)
            for index, value in enumerate(shared)
        ]
        projected_results = [
            _compare(value.numpy(), fixture[f'projected_{index}'], atol, rtol)
            for index, value in enumerate(projected)
        ]
        positioned_result = _compare_same(
            positioned_tokens.numpy(), fixture['positioned_tokens'], atol, rtol
        )
        aifi_result = _compare_same(aifi_tokens.numpy(), fixture['aifi_tokens'], atol, rtol)
        top_down_results = [
            _compare(value.numpy(), fixture[f'top_down_{index}'], atol, rtol)
            for index, value in enumerate(top_down)
        ]

    passed = all(
        result['finite'] and result['within_tolerance']
        for result in (*backbone_results, *shared_results)
    )
    report = {
        'format': 'deimv2_jax_yopo_feature_parity_v1',
        'passed': passed,
        'atol': atol,
        'rtol': rtol,
        'fixture_dir': str(fixture_dir.resolve()),
        'target_config': str(target_config.resolve()),
        'contract_fingerprint_sha256': target_contract['fingerprint_sha256'],
        'mapped_tensors': len(transfer.mapped),
        'excluded_target': list(transfer.excluded_target),
        'backbone_levels': backbone_results,
        'projected_levels': projected_results,
        'positioned_tokens': positioned_result,
        'aifi_tokens': aifi_result,
        'top_down_levels': top_down_results,
        'shared_levels': shared_results,
    }
    report_path.parent.mkdir(parents=True, exist_ok=True)
    report_path.write_text(json.dumps(report, indent=2, sort_keys=True), encoding='utf-8')
    if not passed:
        raise RuntimeError(f'feature parity exceeded tolerance; see {report_path}')
    return report


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--fixture-dir', type=Path, required=True)
    parser.add_argument('--target-config', type=Path, required=True)
    parser.add_argument('--report', type=Path, required=True)
    parser.add_argument('--atol', type=float, default=2e-4)
    parser.add_argument('--rtol', type=float, default=2e-4)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    report = check_parity(
        args.fixture_dir.expanduser().resolve(),
        args.target_config.expanduser().resolve(),
        args.report.expanduser().resolve(),
        atol=args.atol,
        rtol=args.rtol,
    )
    print(json.dumps(report, indent=2, sort_keys=True))


if __name__ == '__main__':
    main()
