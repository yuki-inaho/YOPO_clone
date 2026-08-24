# Copyright (c) OpenMMLab. All rights reserved.
import glob
import os.path as osp

import cv2
import numpy as np

from yopo.registry import DATASETS
from .base_det_dataset import BaseDetDataset


def qbox2rbox_np(points):
    """Convert a quadrilateral (8 values) to a rotated box (cx, cy, w, h, t).

    Args:
        points (np.ndarray): Shape (8,). 8 polygon vertex coordinates.

    Returns:
        np.ndarray: Shape (5,) rotated box in <cx, cy, w, h, t> (radian).
    """
    pts = points.reshape(4, 2).astype(np.float32)
    (x, y), (w, h), angle = cv2.minAreaRect(pts)
    if h > w:
        w, h = h, w
        angle += 90
    return np.array([x, y, w, h, np.deg2rad(angle)], dtype=np.float32)


@DATASETS.register_module()
class DOTATomatoDataset(BaseDetDataset):
    """DOTA-format tomato OBB dataset.

    Reads the DOTA-style annotation layout: ``ann_file`` is a directory of
    ``*.txt`` files, one per image. Each line in a txt file is
    ``x1 y1 x2 y2 x3 y3 x4 y4 class difficulty``. The 8-vertex polygon (qbox)
    is converted on load into a 5-dim rotated box (rbox, <cx, cy, w, h, rad>).

    Args:
        img_shape (tuple[int]): The shape of images (h, w). Defaults to
            (512, 736).
        diff_thr (int): Difficulty threshold; instances with difficulty
            larger than this are ignored. Defaults to 100.
    """

    METAINFO = {"classes": ("stem",), "palette": [(255, 255, 0)]}

    def __init__(self, img_shape=(512, 736), diff_thr=100, **kwargs):
        self.img_shape = img_shape
        self.diff_thr = diff_thr
        super().__init__(**kwargs)

    def load_data_list(self):
        cls_map = {c: i for i, c in enumerate(self.metainfo["classes"])}
        data_list = []
        txt_files = sorted(glob.glob(osp.join(self.ann_file, "*.txt")))
        for txt_file in txt_files:
            img_id = osp.split(txt_file)[1][:-4]
            img_name = img_id + ".jpg"
            img_path = osp.join(self.data_prefix["img_path"], img_name)
            if not osp.exists(img_path):
                continue
            instances = []
            with open(txt_file) as f:
                for line in f:
                    parts = line.split()
                    if len(parts) < 9:
                        continue
                    bbox_q = [float(i) for i in parts[:8]]
                    cls_name = parts[8]
                    difficulty = int(parts[9]) if len(parts) > 9 else 0
                    if cls_name not in cls_map:
                        continue
                    instances.append({
                        "bbox": qbox2rbox_np(np.asarray(bbox_q)),
                        "bbox_label": cls_map[cls_name],
                        "ignore_flag": int(difficulty > self.diff_thr)
                    })
            data_list.append({
                "img_id": img_id,
                "file_name": img_name,
                "img_path": img_path,
                "height": self.img_shape[0],
                "width": self.img_shape[1],
                "instances": instances,
            })
        return data_list

    def filter_data(self):
        if self.test_mode:
            return self.data_list
        filter_empty_gt = (self.filter_cfg.get("filter_empty_gt", False)
                           if self.filter_cfg is not None else False)
        valid_data_infos = []
        for data_info in self.data_list:
            if filter_empty_gt and len(data_info["instances"]) == 0:
                continue
            valid_data_infos.append(data_info)
        return valid_data_infos

    def get_cat_ids(self, idx):
        data_info = self.get_data_info(idx)
        return [inst["bbox_label"] for inst in data_info["instances"]]