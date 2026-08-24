"""Stage 1: run inference on the custom val split and dump
(predictions + ground truth + meta) to a single pickle file.

No drawing here — pure GPU inference + save. Fast (~seconds for 50 frames).

Usage:
  .venv/bin/python temp/dump_nocs_custom_infer.py \
      --out work_dirs/cop_finetune/cop_val_results.pkl [--max-imgs 50]
"""
import argparse
import os
import pickle

import numpy as np
import torch

from mmengine.config import Config
from mmengine.registry import init_default_scope


def _t(x):
    return x.cpu().numpy() if isinstance(x, torch.Tensor) else np.asarray(x)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--config',
                        default='configs/yopo/test_nocs_custom_rgbd_deim_cop.py')
    parser.add_argument('--checkpoint',
                        default='work_dirs/cop_finetune/epoch_50.pth')
    parser.add_argument('--out',
                        default='work_dirs/cop_finetune/cop_val_results.pkl')
    parser.add_argument('--max-imgs', type=int, default=50)
    args = parser.parse_args()

    init_default_scope('yopo')
    import yopo  # noqa: F401
    from yopo.registry import DATASETS, MODELS
    from torch.utils.data import DataLoader
    from mmengine.dataset import pseudo_collate

    cfg = Config.fromfile(args.config)
    cfg.load_from = args.checkpoint
    model = MODELS.build(cfg.model).cuda().eval()
    dataset = DATASETS.build(cfg.val_dataloader.dataset)
    loader = DataLoader(dataset, batch_size=1, collate_fn=pseudo_collate,
                        shuffle=False, num_workers=0)

    records = []
    with torch.no_grad():
        for idx, data_batch in enumerate(loader):
            if idx >= args.max_imgs:
                break
            data_samples = model.test_step(data_batch)
            ds = data_samples[0]

            rec = {
                'img_path': ds.img_path,
                'img_id': ds.img_id,
                'intrinsic': list(ds.intrinsic),
            }
            # Ground truth
            gt = ds.gt_instances
            rec['gt'] = {
                'bboxes': _t(gt.bboxes),
                'labels': _t(gt.labels),
                'T': _t(gt.T),
                'sizes': _t(gt.sizes) if hasattr(gt, 'sizes') else None,
                'centers_2d': _t(gt.centers_2d) if hasattr(gt, 'centers_2d') else None,
            }
            # Predictions
            pred = ds.pred_instances
            rec['pred'] = {
                'bboxes': _t(pred.bboxes),
                'labels': _t(pred.labels),
                'scores': _t(pred.scores),
                'T': _t(pred.T),
                'sizes': _t(pred.sizes),
                'translations': _t(pred.translations) if hasattr(pred, 'translations') else None,
                'rotations': _t(pred.rotations) if hasattr(pred, 'rotations') else None,
            }
            records.append(rec)
            print(f'[{idx}] {rec["img_path"]}  gt={len(rec["gt"]["labels"])} '
                  f'pred={len(rec["pred"]["scores"])}')

    os.makedirs(os.path.dirname(args.out), exist_ok=True)
    with open(args.out, 'wb') as f:
        pickle.dump(records, f)
    print(f'saved {len(records)} records -> {args.out}')


if __name__ == '__main__':
    main()
