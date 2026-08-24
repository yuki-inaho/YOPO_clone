"""Stage 2: draw GT vs predicted 9D pose from a dumped pickle (no inference).

Reads the file produced by dump_nocs_custom_infer.py and renders each frame
with GT (green bbox + axis) and predictions above the score threshold
(red bbox + axis). Pure CPU + OpenCV, fast.

Usage:
  .venv/bin/python temp/draw_nocs_custom_results.py \
      --in work_dirs/cop_finetune/cop_val_results.pkl \
      --out-dir work_dirs/cop_finetune/vis \
      --score-thr 0.4 [--max-imgs 10]
"""
import argparse
import os
import pickle

import numpy as np
import cv2


def _K(intrinsic):
    if len(intrinsic) == 4:
        fx, fy, cx, cy = intrinsic
        return np.array([[fx, 0, cx], [0, fy, cy], [0, 0, 1]], np.float64)
    return np.array(intrinsic, np.float64).reshape(3, 3)


def _proj(pts3d, K):
    p = pts3d @ K.T
    p = p[:, :2] / p[:, 2:3]
    return p


def _draw_boxes(img, boxes, color, texts=None):
    for i, b in enumerate(boxes):
        x1, y1, x2, y2 = [int(v) for v in b]
        cv2.rectangle(img, (x1, y1), (x2, y2), color, 2)
        if texts and texts[i]:
            cv2.putText(img, texts[i], (x1, max(y1 - 4, 10)),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.5, color, 1)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--input',
                        default='work_dirs/cop_finetune/cop_val_results.pkl')
    parser.add_argument('--out-dir', default='work_dirs/cop_finetune/vis')
    parser.add_argument('--score-thr', type=float, default=0.4)
    parser.add_argument('--max-imgs', type=int, default=10)
    parser.add_argument('--box-only', action='store_true',
                        help='draw only 2D boxes (skip 3D axes)')
    args = parser.parse_args()

    with open(args.input, 'rb') as f:
        records = pickle.load(f)
    os.makedirs(args.out_dir, exist_ok=True)

    for rec in records[:args.max_imgs]:
        img = cv2.imread(rec['img_path'])
        if img is None:
            print('WARN cannot read', rec['img_path'])
            continue
        K = _K(rec['intrinsic'])

        # GT (green)
        gt = rec['gt']
        boxes = gt['bboxes']
        texts = [f'gt' for _ in range(len(boxes))]
        _draw_boxes(img, boxes, (0, 200, 0))
        if not args.box_only and gt['T'] is not None and gt['sizes'] is not None:
            for k, T in enumerate(gt['T']):
                T = np.asarray(T, np.float64)
                s = np.asarray(gt['sizes'][k], np.float64)
                local = np.array([[0, 0, 0], [s[0], 0, 0],
                                  [0, s[1], 0], [0, 0, s[2]]])
                world = local @ T[:3, :3].T + T[:3, 3]
                pts = _proj(world, K).astype(int)
                o = pts[0]
                for j in [1, 2, 3]:
                    cv2.line(img, tuple(o), tuple(pts[j]), (0, 220, 0), 2)

        # Pred (red), above threshold
        pred = rec['pred']
        scores = np.asarray(pred['scores'])
        ok = scores >= args.score_thr
        pboxes = np.asarray(pred['bboxes'])[ok]
        ptexts = [f'{s:.2f}' for s in scores[ok]]
        _draw_boxes(img, pboxes, (0, 0, 255), ptexts)
        if not args.box_only and pred['T'] is not None and pred['sizes'] is not None:
            Ts = np.asarray(pred['T'])[ok]
            sizes = np.asarray(pred['sizes'])[ok]
            for k, T in enumerate(Ts):
                T = np.asarray(T, np.float64)
                s = np.asarray(sizes[k], np.float64)
                local = np.array([[0, 0, 0], [s[0], 0, 0],
                                  [0, s[1], 0], [0, 0, s[2]]])
                world = local @ T[:3, :3].T + T[:3, 3]
                pts = _proj(world, K).astype(int)
                o = pts[0]
                for j in [1, 2, 3]:
                    cv2.line(img, tuple(o), tuple(pts[j]), (0, 0, 255), 2)

        frame = os.path.basename(rec['img_path']).replace('_color.png', '')
        out = os.path.join(args.out_dir, f'{frame}_gt_pred.png')
        cv2.imwrite(out, img)
        print('saved', out)
    print('done')


if __name__ == '__main__':
    main()
