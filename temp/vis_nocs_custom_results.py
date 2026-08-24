"""Visualize GT vs predicted 9D pose on the custom val split.

Loads the trained CoP checkpoint, runs predict on the val set, and renders
GT (left) vs prediction (right) with PoseLocalVisualizer, saving PNGs to
--out-dir (default work_dirs/cop_finetune/vis).

Usage:
  .venv/bin/python temp/vis_nocs_custom_results.py \
      --out-dir work_dirs/cop_finetune/vis \
      --max-imgs 8 \
      --score-thr 0.3
"""
import argparse
import os

import numpy as np

from mmengine.config import Config
from mmengine.registry import init_default_scope


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--config',
                        default='configs/yopo/test_nocs_custom_rgbd_deim_cop.py')
    parser.add_argument('--checkpoint',
                        default='work_dirs/cop_finetune/epoch_50.pth')
    parser.add_argument('--out-dir',
                        default='work_dirs/cop_finetune/vis')
    parser.add_argument('--max-imgs', type=int, default=8)
    parser.add_argument('--score-thr', type=float, default=0.3)
    args = parser.parse_args()

    init_default_scope('yopo')
    import yopo  # noqa: F401  (registers modules)

    from yopo.registry import (DATASETS, MODELS, TRANSFORMS, VISUALIZERS)

    cfg = Config.fromfile(args.config)
    cfg.load_from = args.checkpoint

    # Build model + val dataset
    model = MODELS.build(cfg.model)
    dataset = DATASETS.build(cfg.val_dataloader.dataset)

    # Build visualizer
    vis = VISUALIZERS.build(cfg.visualizer)
    vis.dataset_meta = dict(
        classes=('fruit',), palette=[(220, 20, 60)])

    os.makedirs(args.out_dir, exist_ok=True)

    import torch

    # iterate over dataset, construct a batch, run predict
    from mmengine.registry import DATA_SAMPLERS
    from torch.utils.data import DataLoader
    from mmengine.dataset import pseudo_collate
    loader = DataLoader(dataset, batch_size=1, collate_fn=pseudo_collate,
                        shuffle=False, num_workers=0)

    model.cuda().eval()
    n = 0
    with torch.no_grad():
        for data_batch in loader:
            if n >= args.max_imgs:
                break
            data_samples = model.test_step(data_batch)
            for ds in data_samples:
                img = ds.img_path
                ori = ds.ori_shape
                import mmcv
                # give a unique, extension-friendly img_id for the output path
                frame = os.path.basename(img).replace('_color.png', '')
                ds.set_metainfo({'img_id': f'{frame}_gt_pred'})
                image = mmcv.imread(img, channel_order='rgb').astype(np.float32)
                vis.add_datasample(
                    name=ds.img_id,
                    image=image,
                    data_sample=ds,
                    draw_gt=True,
                    draw_pred=True,
                    out_file=os.path.join(args.out_dir, 'tmp.png'),
                    pred_score_thr=args.score_thr,
                )
                print('saved', ds.img_id)
            n += 1
    print(f'done. images in {args.out_dir}')


if __name__ == '__main__':
    main()
