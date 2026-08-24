# Copyright (c) OpenMMLab. All rights reserved.
import json
import os
import os.path as osp
from typing import List, Optional, Union
import pickle

import numpy as np
import torch
from mmengine.fileio import get, get_local_path, list_from_file

from yopo.registry import DATASETS
from ..base_det_dataset import BaseDetDataset

from .nocs_utils import get_bbox


@DATASETS.register_module()
class NOCSDataset(BaseDetDataset):
    """YCB-Video in BOP Challenge dataset for 6D Pose Estimation.

    Args:
        data_root (str, optional): The root directory for ``data_prefix`` and
            ``ann_file``. Defaults to ''.
        data_prefix (dict): Prefix for training data. Defaults to
            dict(img_path='').
        img_subdir (str): Subdir where images are stored. Default: JPEGImages.
        ann_subdir (str): Subdir where annotations are. Default: Annotations.
        backend_args (dict, optional): Arguments to instantiate the
            corresponding backend. Defaults to None.
    """

    METAINFO = {
        "classes": ("bottle", "bowl", "camera", "can", "laptop", "mug"),
        "palette": [
            (220, 20, 60),
            (119, 11, 32),
            (0, 0, 142),
            (0, 0, 230),
            (106, 0, 228),
            (0, 60, 100),
        ],
    }

    SPLIT_INFO = dict(
        camera_train=dict(
            img_path="camera/train_list.txt",
            model_path=None,  # 'obj_models/camera_train.pkl',
            intrinsic=[577.5, 577.5, 319.5, 239.5],
        ),
        real_train=dict(
            img_path="real/train_list.txt",
            model_path=None,  # 'obj_models/real_train.pkl',
            intrinsic=[591.0125, 590.16775, 322.525, 244.11084],
        ),
        camera_val=dict(
            img_path="camera/val_list.txt",
            model_path=None,  # 'obj_models/camera_val.pkl',
            label_path="segmentation_results/CAMERA25",
            intrinsic=[577.5, 577.5, 319.5, 239.5],
        ),
        real_test=dict(
            img_path="real/test_list.txt",
            model_path=None,  # 'obj_models/real_test.pkl',
            label_path="segmentation_results/REAL275",
            intrinsic=[591.0125, 590.16775, 322.525, 244.11084],
        ),
        custom_val=dict(
            # Custom RGB-D dataset: same train-style _label.pkl layout as
            # real_train, but evaluated on the val/test image list.
            img_path="real/test_list.txt",
            model_path=None,
            label_path="real/",
            intrinsic=[591.0125, 590.16775, 322.525, 244.11084],
        ),
        overfit=dict(
            img_path="real/test_scene_1_list.txt",
            model_path=None,  # 'obj_models/overfit.pkl',
            intrinsic=[591.0125, 590.16775, 322.525, 244.11084],
        ),
    )
    IMG_SHAPE = (480, 640)  # (height, width)

    def __init__(
        self,
        split: str = "real_train",
        num_sample_points: int = 1000,
        use_cuboid_as_bbox: bool = False,
        use_log_z: bool = False,
        intrinsic: Optional[List[float]] = None,
        **kwargs,
    ) -> None:
        assert split in self.SPLIT_INFO, (
            f"Invalid data_type: {split}. "
            f"Available types: {list(self.SPLIT_INFO.keys())}"
        )

        self.split = split
        # Allow overriding the (normally hard-coded) per-split intrinsic from
        # the config. Used by the custom RGB-D dataset whose camera intrinsics
        # differ from the stock NOCS/REAL275 values.
        if intrinsic is not None:
            self.SPLIT_INFO[self.split]["intrinsic"] = list(intrinsic)
        self.num_sample_points = num_sample_points
        self.use_cuboid_as_bbox = use_cuboid_as_bbox
        self.use_log_z = use_log_z

        self.xmap = np.array(
            [[i for i in range(self.IMG_SHAPE[1])] for j in range(self.IMG_SHAPE[0])]
        )
        self.ymap = np.array(
            [[j for i in range(self.IMG_SHAPE[1])] for j in range(self.IMG_SHAPE[0])]
        )
        self.sym_ids = [0, 1, 3]  # 0-indexed
        self.norm_scale = 1000.0  # normalization scale
        super().__init__(**kwargs)

    def load_data_list(self) -> List[dict]:
        """Load annotation from YCB-Video in BOP Challenge dataset.

        The folder structure is as follows:

        data_root/
            sub_data_root/
                ├── 000001/
                │   ├── rgb/
                │   ├── scene_camera.json
                │   ├── scene_gt.json
                │   └── scene_gt_info.json
                ├── 000002/
                │   ├── rgb/
                │   ├── scene_camera.json
                │   ├── scene_gt.json
                │   └── scene_gt_info.json
            ...

        """
        self.cat2label = {cat: i for i, cat in enumerate(self._metainfo["classes"])}

        dataset_info = self.SPLIT_INFO[self.split]
        img_path = dataset_info["img_path"]
        model_path = dataset_info["model_path"]
        intrinsic = dataset_info["intrinsic"]

        self.img_list = [
            osp.join(self.data_root, img_path.split("/")[0], line.rstrip("\n"))
            for line in open(osp.join(self.data_root, img_path), "r")
        ]
        self.img_index = np.arange(len(self.img_list))

        self.models = dict()
        if self.SPLIT_INFO[self.split]["model_path"] is not None:
            with open(os.path.join(self.data_root, model_path), "rb") as f:
                self.models.update(pickle.load(f))

        data_list = []
        for img_file in self.img_list:
            file_name = "/".join(img_file.split("/")[1:])

            frame_id = os.path.basename(file_name).split(".")[0][:4]
            scene = file_name.split("/")[-2]
            img_id = scene + "/" + frame_id

            # read label_info
            if "train" in self.split or self.split.startswith("custom"):
                label_file = osp.join(img_file + "_label.pkl")
            else:
                label_path = dataset_info.get("label_path", None)
                if label_path is None:
                    raise ValueError(f"No label_path found for split {self.split}")
                split = self.split.split("_")[-1]
                label_file = osp.join(
                    self.data_root,
                    label_path,
                    f"results_{split}_{scene}_{frame_id}.pkl",
                )

            with open(label_file, "rb") as f:
                gt_info = pickle.load(f)

            raw_img_info = {}
            raw_img_info["img_id"] = img_id
            raw_img_info["file_name"] = file_name
            raw_img_info["img_path"] = img_file + "_color.png"
            raw_img_info["intrinsic"] = intrinsic

            parsed_data_info = self.parse_data_info(raw_img_info, gt_info)
            if parsed_data_info is None:
                continue
            data_list.append(parsed_data_info)

        return data_list

    def parse_data_info(
        self,
        img_info: dict,
        gt_info: dict,
    ) -> Union[dict, List[dict]]:
        """Parse raw annotation to target format.

        Args:

        Returns:
            Union[dict, List[dict]]: Parsed annotation.
        """
        data_info = {}
        data_info["img_path"] = img_info["img_path"]

        if self.split.startswith("camera"):
            # Original NOCS camera dataset only has depth maps for the instances.
            depth_path = img_info["img_path"].replace("/camera", "/camera_full_depths")
            depth_path = depth_path.replace("color.png", "composed.png")
        else:
            depth_path = img_info["img_path"].replace("color.png", "depth.png")
        if not osp.exists(depth_path):
            return None
        data_info["depth_path"] = depth_path

        data_info["img_id"] = img_info["img_id"]
        data_info["intrinsic"] = img_info["intrinsic"]

        data_info["height"] = self.IMG_SHAPE[0]
        data_info["width"] = self.IMG_SHAPE[1]

        data_info["instances"] = self._parse_instance_info(
            gt_info, data_info["intrinsic"]
        )

        return data_info

    def _parse_instance_info(self, gt_info: dict, intrinsic: list[float]) -> List[dict]:
        """parse instance information.

        Args:
            raw_ann_info (dict): Raw annotation information.
        Returns:
            List[dict]: List of instances.
        """
        is_train = "train" in self.split or self.split.startswith("custom")

        if is_train:
            class_ids = gt_info["class_ids"]
        else:
            class_ids = gt_info["gt_class_ids"]

        instances = []
        for idx in range(len(class_ids)):
            instance = {}

            if is_train:
                class_id = gt_info["class_ids"][idx] - 1  # 1-indexed to 0-indexed
                # rmin, rmax, cmin, cmax = get_bbox(gt_info['bboxes'][idx])
                y1, x1, y2, x2 = gt_info["bboxes"][idx]
                instance_id = gt_info["instance_ids"][idx]

                translation = gt_info["translations"][idx]
                rotation = gt_info["rotations"][idx]
                # size: 3D size of NOCS model, scales: scale_factor for NOCS model
                if "sizes" in gt_info:  # for train
                    size = gt_info["sizes"][idx]
                elif "size" in gt_info:
                    size = gt_info["size"][idx]
                    raise RuntimeError(
                        "Found size in gt_info, but it is not used for training."
                    )
                else:
                    raise ValueError(
                        f"No size information found in gt_info: {gt_info.keys()}"
                    )
                size = size * gt_info["scales"][idx]

                T = np.eye(4, dtype=np.float32)
                T[:3, :3] = rotation  # * gt_info['scales'][idx]
                T[:3, 3] = translation
                gt_handle_visibility = 1
            else:
                class_id = gt_info["gt_class_ids"][idx] - 1  # 1-indexed to 0-indexed
                y1, x1, y2, x2 = gt_info["gt_bboxes"][idx]
                instance_id = idx

                T = gt_info["gt_RTs"][idx]
                translation = T[:3, 3].tolist()
                rotation = T[:3, :3]
                size = gt_info["gt_scales"][idx]
                gt_handle_visibility = gt_info["gt_handle_visibility"][idx]

            bbox = [x1, y1, x2, y2]  # x1, y1, x2, y2

            if len(intrinsic) == 4:
                # intrinsic is [fx, fy, cx, cy]
                K = np.array(
                    [
                        [intrinsic[0], 0, intrinsic[2]],
                        [0, intrinsic[1], intrinsic[3]],
                        [0, 0, 1],
                    ],
                    dtype=np.float32,
                )
            elif len(intrinsic) == 9:
                K = np.array(intrinsic).reshape(3, 3)
            else:
                raise ValueError(f"Invalid intrinsic shape: {len(intrinsic)}")

            # Calculate 2D center projection

            # center_2d = K @ np.array(translation) / translation[2]  # normalize by z
            # center_2d = center_2d[:2].tolist()  # only x, y
            fx, fy, cx, cy = intrinsic
            if translation[2] > 0:  # Avoid division by zero
                center_2d_x = fx * translation[0] / translation[2] + cx
                center_2d_y = fy * translation[1] / translation[2] + cy
                center_2d = [center_2d_x, center_2d_y]
            else:
                center_2d = [cx, cy]  # Default to image center

            # symmetry handling
            if class_id in self.sym_ids:
                theta_x = rotation[0, 0] + rotation[2, 2]
                theta_y = rotation[0, 2] - rotation[2, 0]
                r_norm = np.sqrt(theta_x**2 + theta_y**2)
                s_map = np.array(
                    [
                        [theta_x / r_norm, 0.0, -theta_y / r_norm],
                        [0.0, 1.0, 0.0],
                        [theta_y / r_norm, 0.0, theta_x / r_norm],
                    ]
                )
                rotation = rotation @ s_map

            if self.use_cuboid_as_bbox:
                width, height, depth = size
                corners_3d = np.array(
                    [
                        [-width / 2, -height / 2, -depth / 2],
                        [width / 2, -height / 2, -depth / 2],
                        [width / 2, height / 2, -depth / 2],
                        [-width / 2, height / 2, -depth / 2],
                        [-width / 2, -height / 2, depth / 2],
                        [width / 2, -height / 2, depth / 2],
                        [width / 2, height / 2, depth / 2],
                        [-width / 2, height / 2, depth / 2],
                    ]
                )
                corners_3d_cam = (T[:3, :3] @ corners_3d.T + T[:3, 3, None]).T
                projected_corners = (K @ corners_3d_cam.T).T
                projected_corners_2d = (
                    projected_corners[:, :2] / projected_corners[:, 2:]
                )
                projected_corners_2d = projected_corners_2d.reshape(8, 2)

                x1 = projected_corners_2d[:, 0].min()
                y1 = projected_corners_2d[:, 1].min()
                x2 = projected_corners_2d[:, 0].max()
                y2 = projected_corners_2d[:, 1].max()
                bbox = [x1, y1, x2, y2]

            rotation = rotation.flatten()
            r_6d_rep = [rotation[i] for i in [0, 3, 6, 1, 4, 7]]

            instance["bbox"] = bbox
            instance["bbox_label"] = class_id
            instance["instance_id"] = instance_id

            instance["translation"] = translation
            instance["rotation"] = r_6d_rep
            instance["size"] = size
            instance["gt_handle_visibility"] = gt_handle_visibility

            instance["center_2d"] = center_2d
            if self.use_log_z:
                # use log scale for z
                instance["z"] = np.log(translation[2])
            else:
                instance["z"] = translation[
                    2
                ]  # add this line to handle the case when use_log_z is False

            instance["T"] = T
            instance["ignore_flag"] = 0

            instances.append(instance)
        return instances
