#!/usr/bin/env python
"""Export a YOPO checkpoint to a portable, fixed-shape ONNX graph."""

from __future__ import annotations

import argparse
import json
import os
from pathlib import Path
from typing import Dict

import numpy as np
import onnx
import torch
from torch import nn

from mmengine.config import Config
from yopo.deployment.onnx import (DEFAULT_INPUT_SIZE, ONNX_OUTPUT_NAMES,
                                  YopoOnnxWrapper, build_yopo_model)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description='Export YOPO to ONNX.')
    parser.add_argument(
        '--config', default='configs/yopo/nocs_yopo_real_camera_r50.py')
    parser.add_argument(
        '--checkpoint', default='checkpoints/nocs_yopo_real_camera_r50.pth')
    parser.add_argument(
        '--output', default='work_dirs/yopo_onnx/yopo_nocs_r50.onnx')
    parser.add_argument('--opset', type=int, default=17)
    parser.add_argument('--simplify', action='store_true')
    parser.add_argument('--no-verify', action='store_true')
    return parser.parse_args()


def _verify(wrapper: nn.Module, output: str, example: torch.Tensor) -> Dict:
    import onnxruntime as ort

    with torch.inference_mode():
        torch_outputs = [value.detach().cpu().numpy()
                         for value in wrapper(example)]
    session = ort.InferenceSession(output, providers=['CPUExecutionProvider'])
    ort_outputs = session.run(None, {'image': example.numpy()})
    max_abs = max(float(np.max(np.abs(torch_value - ort_value)))
                  for torch_value, ort_value in zip(torch_outputs, ort_outputs))
    mean_abs = float(np.mean([
        np.mean(np.abs(torch_value - ort_value))
        for torch_value, ort_value in zip(torch_outputs, ort_outputs)
    ]))
    allclose = all(np.allclose(torch_value, ort_value, rtol=2e-3, atol=2e-3)
                   for torch_value, ort_value in zip(torch_outputs, ort_outputs))
    if not allclose:
        raise RuntimeError(
            'ONNX Runtime verification failed: max_abs={}'.format(max_abs))
    return {
        'allclose': allclose,
        'max_abs': max_abs,
        'mean_abs': mean_abs,
        'providers': session.get_providers(),
    }


def main() -> None:
    args = parse_args()
    if not os.path.isfile(args.checkpoint):
        raise FileNotFoundError(args.checkpoint)
    output = Path(args.output)
    output.parent.mkdir(parents=True, exist_ok=True)

    # Export on CPU. MMDetection's CUDA deformable-attention kernel and the
    # ONNX expansion path do not produce numerically equivalent traces here;
    # the CPU implementation expands to standard ONNX operators and is what
    # the portable ONNX Runtime artifact is verified against.
    model = build_yopo_model(args.config, args.checkpoint, device='cpu')
    wrapper = YopoOnnxWrapper(model, DEFAULT_INPUT_SIZE).cpu().eval()
    example = torch.zeros(1, 3, DEFAULT_INPUT_SIZE[1], DEFAULT_INPUT_SIZE[0])
    with torch.inference_mode():
        eager_shapes = [list(value.shape) for value in wrapper(example)]
    torch.onnx.export(
        wrapper,
        example,
        str(output),
        opset_version=args.opset,
        input_names=['image'],
        output_names=list(ONNX_OUTPUT_NAMES),
        do_constant_folding=True,
        verbose=False)

    if args.simplify:
        from onnxsim import simplify
        onnx_model = onnx.load(str(output))
        onnx_model, success = simplify(onnx_model)
        if not success:
            raise RuntimeError('onnxsim reported unsuccessful simplification')
        onnx.save(onnx_model, str(output))

    onnx_model = onnx.load(str(output))
    onnx.checker.check_model(onnx_model)
    verification = None if args.no_verify else _verify(wrapper, str(output), example)

    cfg = Config.fromfile(args.config)
    metadata = {
        'format': 'yopo-onnx-raw-plus-python-postprocess',
        'config': args.config,
        'checkpoint': args.checkpoint,
        'opset': args.opset,
        'input': {
            'name': 'image',
            'shape': [1, 3, DEFAULT_INPUT_SIZE[1], DEFAULT_INPUT_SIZE[0]],
            'dtype': 'float32',
            'layout': 'NCHW',
            'color': 'BGR',
            'range': '0..255',
        },
        'outputs': [{
            'name': name,
            'eager_shape': shape,
        } for name, shape in zip(ONNX_OUTPUT_NAMES, eager_shapes)],
        'postprocess': {
            'num_classes': int(cfg.model.bbox_head.num_classes),
            'max_per_img': int(cfg.model.test_cfg.get('max_per_img', 300)),
            'classwise_rotation': bool(cfg.model.bbox_head.get(
                'classwise_rotation', True)),
            'classwise_sizes': bool(cfg.model.bbox_head.get(
                'classwise_sizes', True)),
            'use_log_z': bool(cfg.model.bbox_head.get('use_log_z', False)),
            'implementation': 'yopo.deployment.onnx.postprocess_pose_outputs',
        },
        'export_device': 'cpu',
        'onnxruntime_verification': verification,
    }
    sidecar = output.with_suffix('.json')
    sidecar.write_text(json.dumps(metadata, indent=2), encoding='utf-8')
    print('wrote {} ({} bytes)'.format(output, output.stat().st_size))
    print('wrote {}'.format(sidecar))
    if verification is not None:
        print('verified ONNX Runtime: max_abs={:.6g}, mean_abs={:.6g}'.format(
            verification['max_abs'], verification['mean_abs']))


if __name__ == '__main__':
    main()
