#!/usr/bin/env python
"""Run the released YOPO checkpoint on representative tomato RGB-D scenes."""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import time
from pathlib import Path
from typing import Dict, List

import numpy as np
import torch
from mmengine.config import Config
from mmengine.dataset import Compose, pseudo_collate
from mmengine.registry import init_default_scope

from yopo.deployment.onnx import build_yopo_model
from yopo.deployment.tomato import (discover_rgbd_samples,
                                    inspect_rgbd_sample)
import yopo.datasets  # noqa: F401


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description='Run YOPO on one RGB-D pair from each tomato scene.')
    parser.add_argument(
        '--data-root',
        default='/home/inaho-omen/data/2026_tomato/standard')
    parser.add_argument(
        '--config', default='configs/yopo/nocs_yopo_real_camera_r50.py')
    parser.add_argument(
        '--checkpoint', default='checkpoints/nocs_yopo_real_camera_r50.pth')
    parser.add_argument('--output', default='work_dirs/tomato_inference.json')
    parser.add_argument('--max-scenes', type=int, default=10)
    parser.add_argument('--score-thr', type=float, default=0.3)
    parser.add_argument('--device', default='auto', choices=('auto', 'cpu', 'cuda'))
    return parser.parse_args()


def sha256(path: str) -> str:
    digest = hashlib.sha256()
    with open(path, 'rb') as stream:
        for block in iter(lambda: stream.read(8 * 1024 * 1024), b''):
            digest.update(block)
    return digest.hexdigest()


def _prediction_summary(instance, score_thr: float) -> Dict:
    scores = instance.scores.detach().cpu().numpy().astype(np.float32)
    order = np.argsort(-scores, kind='stable')
    top = order[:5]
    return {
        'num_predictions': int(scores.size),
        'num_score_ge_threshold': int(np.count_nonzero(scores >= score_thr)),
        'score_max': float(scores.max()) if scores.size else None,
        'score_mean': float(scores.mean()) if scores.size else None,
        'top_predictions': [{
            'score': float(scores[index]),
            'label': int(instance.labels[index].item()),
            'bbox': [float(v) for v in instance.bboxes[index].detach().cpu()],
            'translation': [
                float(v) for v in instance.translations[index].detach().cpu()
            ],
            'size': [float(v) for v in instance.sizes[index].detach().cpu()],
        } for index in top],
    }


def main() -> None:
    args = parse_args()
    if args.device == 'auto':
        device = 'cuda:0' if torch.cuda.is_available() else 'cpu'
    elif args.device == 'cuda':
        device = 'cuda:0'
    else:
        device = 'cpu'
    if device.startswith('cuda') and not torch.cuda.is_available():
        raise RuntimeError('CUDA requested but torch.cuda.is_available() is false')
    if not os.path.isfile(args.checkpoint):
        raise FileNotFoundError(args.checkpoint)

    samples = [inspect_rgbd_sample(sample) for sample in
               discover_rgbd_samples(args.data_root, args.max_scenes)]
    # Loading the config here also verifies that the intended test resolution
    # remains the 640x480 pipeline used by this smoke test.
    cfg = Config.fromfile(args.config)
    assert tuple(cfg.test_dataloader.dataset.pipeline[1].scale) == (640, 480)
    init_default_scope('yopo')
    pipeline = Compose([
        dict(type='LoadImageFromFile'),
        dict(type='Resize', scale=(640, 480), keep_ratio=True),
        dict(type='Pack9DPoseInputs', meta_keys=(
            'img_path', 'ori_shape', 'img_shape', 'scale_factor', 'intrinsic')),
    ])
    model = build_yopo_model(args.config, args.checkpoint, device=device)

    records = []
    for sample in samples:
        item = pipeline({
            'img_path': sample['rgb_path'],
            'intrinsic': sample['rgb_intrinsic'],
        })
        batch = pseudo_collate([item])
        start = time.perf_counter()
        with torch.inference_mode():
            data_sample = model.test_step(batch)[0]
        elapsed_ms = (time.perf_counter() - start) * 1000.0
        record = dict(sample)
        record['device'] = device
        record['elapsed_ms'] = elapsed_ms
        record['model_input'] = 'RGB only; paired depth validated but not fed to released YOPO'
        record['prediction'] = _prediction_summary(
            data_sample.pred_instances, args.score_thr)
        records.append(record)
        print('{}: {:.1f} ms, max_score={:.6f}, predictions={}'.format(
            sample['scene'], elapsed_ms, record['prediction']['score_max'],
            record['prediction']['num_predictions']))

    output = Path(args.output)
    output.parent.mkdir(parents=True, exist_ok=True)
    payload = {
        'format': 'yopo-tomato-rgbd-smoke-test',
        'config': args.config,
        'checkpoint': args.checkpoint,
        'checkpoint_sha256': sha256(args.checkpoint),
        'data_root': args.data_root,
        'requested_scenes': args.max_scenes,
        'processed_scenes': len(records),
        'score_threshold': args.score_thr,
        'device': device,
        'notes': [
            'YOPO release checkpoints are RGB-only NOCS pose models.',
            'Depth files are paired and validated for shape/dtype/valid pixels; depth is not concatenated.',
            'The tomato data is out-of-domain for the NOCS checkpoint, so this is a runtime smoke test, not an accuracy evaluation.',
        ],
        'scenes': records,
    }
    output.write_text(json.dumps(payload, ensure_ascii=False, indent=2),
                      encoding='utf-8')
    print('wrote {}'.format(output))


if __name__ == '__main__':
    main()
