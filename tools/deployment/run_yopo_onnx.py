#!/usr/bin/env python
"""Run the exported YOPO ONNX artifact on standard tomato RGB-D data."""

from __future__ import annotations

import argparse
import hashlib
import json
import time
from pathlib import Path

import cv2
import numpy as np
import onnxruntime as ort

from yopo.deployment.onnx import postprocess_pose_outputs
from yopo.deployment.tomato import (discover_rgbd_samples,
                                    inspect_rgbd_sample)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description='Run YOPO ONNX on tomato RGB-D scenes.')
    parser.add_argument(
        '--model', default='work_dirs/yopo_onnx/yopo_nocs_r50.onnx')
    parser.add_argument(
        '--data-root', default='/home/inaho-omen/data/2026_tomato/standard')
    parser.add_argument('--output', default='work_dirs/tomato_onnx_inference.json')
    parser.add_argument('--max-scenes', type=int, default=10)
    parser.add_argument('--score-thr', type=float, default=0.3)
    parser.add_argument('--providers', nargs='+', default=['CPUExecutionProvider'])
    return parser.parse_args()


def sha256(path: str) -> str:
    digest = hashlib.sha256()
    with open(path, 'rb') as stream:
        for block in iter(lambda: stream.read(8 * 1024 * 1024), b''):
            digest.update(block)
    return digest.hexdigest()


def main() -> None:
    args = parse_args()
    model_path = Path(args.model)
    metadata_path = model_path.with_suffix('.json')
    metadata = json.loads(metadata_path.read_text(encoding='utf-8')) \
        if metadata_path.is_file() else {}
    if not model_path.is_file():
        raise FileNotFoundError(args.model)
    session = ort.InferenceSession(args.model, providers=args.providers)
    input_info = session.get_inputs()[0]
    shape = input_info.shape
    if (shape[0] != 1 or shape[1] != 3 or
            not isinstance(shape[2], int) or not isinstance(shape[3], int)):
        raise ValueError('expected static batch-1 NCHW ONNX input, got {}'.format(
            shape))
    input_height, input_width = int(shape[2]), int(shape[3])
    post = metadata.get('postprocess', {})
    samples = [inspect_rgbd_sample(sample) for sample in
               discover_rgbd_samples(args.data_root, args.max_scenes)]

    records = []
    for sample in samples:
        image = cv2.imread(sample['rgb_path'], cv2.IMREAD_COLOR)
        original_shape = image.shape[:2]
        image = cv2.resize(image, (input_width, input_height),
                           interpolation=cv2.INTER_LINEAR)
        tensor = np.ascontiguousarray(
            image.transpose(2, 0, 1)[None], dtype=np.float32)
        start = time.perf_counter()
        outputs = session.run(None, {input_info.name: tensor})
        elapsed_ms = (time.perf_counter() - start) * 1000.0
        prediction = postprocess_pose_outputs(
            outputs,
            intrinsic=sample['rgb_intrinsic'],
            original_shape=original_shape,
            input_size=(input_width, input_height),
            num_classes=int(post.get('num_classes', 6)),
            max_per_img=int(post.get('max_per_img', 300)),
            classwise_rotation=bool(post.get('classwise_rotation', True)),
            classwise_sizes=bool(post.get('classwise_sizes', True)),
            use_log_z=bool(post.get('use_log_z', False)))
        scores = prediction['scores']
        top = np.argsort(-scores, kind='stable')[:5]
        record = dict(sample)
        record.update({
            'elapsed_ms': elapsed_ms,
            'model_input': 'RGB only; paired depth validated but not fed to released YOPO',
            'num_predictions': int(scores.size),
            'num_score_ge_threshold': int(np.count_nonzero(scores >= args.score_thr)),
            'score_max': float(scores.max()) if scores.size else None,
            'score_mean': float(scores.mean()) if scores.size else None,
            'top_predictions': [{
                'score': float(scores[index]),
                'label': int(prediction['labels'][index]),
                'bbox': prediction['bboxes'][index].tolist(),
                'translation': prediction['translations'][index].tolist(),
                'size': prediction['sizes'][index].tolist(),
            } for index in top],
        })
        records.append(record)
        print('{}: {:.1f} ms, max_score={:.6f}, predictions={}'.format(
            sample['scene'], elapsed_ms, record['score_max'],
            record['num_predictions']))

    output = Path(args.output)
    output.parent.mkdir(parents=True, exist_ok=True)
    payload = {
        'format': 'yopo-onnx-tomato-rgbd-smoke-test',
        'model': args.model,
        'model_sha256': sha256(args.model),
        'providers': session.get_providers(),
        'data_root': args.data_root,
        'input_shape': [1, 3, input_height, input_width],
        'requested_scenes': args.max_scenes,
        'processed_scenes': len(records),
        'score_threshold': args.score_thr,
        'scenes': records,
    }
    output.write_text(json.dumps(payload, ensure_ascii=False, indent=2),
                      encoding='utf-8')
    print('wrote {}'.format(output))


if __name__ == '__main__':
    main()
