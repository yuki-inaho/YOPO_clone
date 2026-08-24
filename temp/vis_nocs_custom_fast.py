"""Lightweight visualization of GT vs predicted 9D pose (custom val).

Direct OpenCV drawing (bbox + projected 3D axes + center) — no mmengine
visualizer so it is fast and has no backend deps.

Usage:
  .venv/bin/python temp/vis_nocs_custom_fast.py \
      --out-dir work_dirs/cop_finetune/vis \
      --max-imgs 10 --score-thr 0.4
"""
import argparse
import os

import numpy as np
import cv2

from mmengine.config import Config
from mmengine.registry import init_default_scope


def _K(intrinsic):
    if len(intrinsic) == 4:
        fx, fy, cx, cy = intrinsic
        return np.array([[fx, 0, cx], [0, fy, cy], [0, 0, 1]], np.float64)
    return np.array(intrinsic, np.float64).reshape(3, 3)


def _proj(pts3d, K):
    # pts3d: (N,3) in camera coords -> (N,2) pixels
    p = pts3d @ K.T
    p = p[:, :2] / p[:, 2:3]
    return p


def _draw_boxes(img, boxes, color, label_texts=None):
    for i, b in enumerate(boxes):
        x1, y1, x2, y2 = [int(v) for v in b]
        cv2.rectangle(img, (x1, y1), (x2, y2), color, 2)
        if label_texts and label_texts[i]:
            t = label_texts[i]
            cv2.putText(img, t, (x1, max(y1 - 4, 10)),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.5, color, 1)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--config',
                        default='configs/yopo/test_nocs_custom_rgbd_deim_cop.py')
    parser.add_argument('--checkpoint',
                        default='work_dirs/cop_finetune/epoch_50.pth')
    parser.add_argument('--out-dir', default='work_dirs/cop_finetune/vis')
    parser.add_argument('--max-imgs', type=int, default=10)
    parser.add_argument('--score-thr', type=float, default=0.4)
    args = parser.parse_args()

    init_default_scope('yopo')
    import yopo  # noqa: F401
    from yopo.registry import DATASETS, MODELS
    from torch.utils.data import DataLoader
    from mmengine.dataset import pseudo_collate
    import torch

    cfg = Config.fromfile(args.config)
    cfg.load_from = args.checkpoint
    model = MODELS.build(cfg.model)
    dataset = DATASETS.build(cfg.val_dataloader.dataset)
    loader = DataLoader(dataset, batch_size=1, collate_fn=pseudo_collate,
                        shuffle=False, num_workers=0)

    os.makedirs(args.out_dir, exist_ok=True)
    model.cuda().eval()

    axis_len = 0.05  # meters, ~object half-size
    n = 0
    with torch.no_grad():
        for data_batch in loader:
            if n >= args.max_imgs:
                break
            data_samples = model.test_step(data_batch)
            ds = data_samples[0]
            img_path = ds.img_path
            K = _K(ds.intrinsic)

            img = cv2.imread(img_path)  # BGR
            if img is None:
                print('WARN: cannot read', img_path)
                n += 1
                continue

            # GT
            gt = ds.gt_instances
            if len(gt) and hasattr(gt, 'bboxes'):
                boxes = np.asarray(gt.bboxes.cpu())
                _draw_boxes(img, boxes, (0, 200, 0))  # green
                Ts = np.asarray(gt.T.cpu())
                sizes = np.asarray(gt.sizes.cpu()) if hasattr(gt, 'sizes') else None
                for k, T in enumerate(Ts):
                    T = np.asarray(T, np.float64)
                    s = sizes[k] if sizes is not None else np.array([0.05, 0.05, 0.05])
                    local = np.array([[0, 0, 0], [s[0], 0, 0], [0, s[1], 0], [0, 0, s[2]]])
                    world = local @ T[:3, :3].T + T[:3, 3]
                    pts = _proj(world, K).astype(int)
                    o = pts[0]
                    for c, j in zip([(0, 255, 0), (0, 255, 0), (0, 255, 0)], [1, 2, 3]):
                        cv2.line(img, tuple(o), tuple(pts[j]), c, 2)

            # Pred
            if hasattr(ds, 'pred_instances') and len(ds.pred_instances):
                pred = ds.pred_instances
                score = np.asarray(pred.scores.cpu())
                ok = score >= args.score_thr
                boxes = np.asarray(pred.bboxes.cpu())[ok]
                _draw_boxes(img, boxes, (0, 0, 255),
                            [f'{s:.2f}' for s in np.asarray(score)[ok]])  # red
                Ts = np.asarray(pred.T.cpu())[ok]
                sizes = np.asarray(pred.sizes.cpu())[ok]
                for k, T in enumerate(Ts):
                    T = np.asarray(T, np.float64)
                    s = sizes[k]
                    local = np.array([[0, 0, 0], [s[0], 0, 0], [0, s[1], 0], [0, 0, s[2]]])
                    world = local @ T[:3, :3].T + T[:3, 3]
                    pts = _proj(world, K).astype(int)
                    o = pts[0]
                    for j in [1, 2, 3]:
                        cv2.line(img, tuple(o), tuple(pts[j]), (255, 0, 0), 2)

            frame = os.path.basename(img_path).replace('_color.png', '')
            out = os.path.join(args.out_dir, f'{frame}_gt_pred.png')
            cv2.imwrite(out, img)
            print('saved', out)
            n += 1
    print('done')


if __name__ == '__main__':
    main()
