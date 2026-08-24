from pathlib import Path

import torch
from mmengine.config import Config
from mmengine.registry import init_default_scope

from yopo.registry import DATASETS
from yopo.utils import register_all_modules


REPO_ROOT = Path(__file__).resolve().parents[1]
CONFIG_PATH = REPO_ROOT / "configs/yopo/nocs_custom_fruit_rgbd_3dbbox_amp_20ep.py"


def test_every_train_pipeline_pose_target_is_finite():
    """RandomFlip must preserve finite custom-fruit 3D supervision."""
    register_all_modules()
    init_default_scope("yopo")
    cfg = Config.fromfile(CONFIG_PATH)
    dataset = DATASETS.build(cfg.train_dataloader.dataset)

    fields = ("bboxes", "centers_2d", "z", "rotations", "sizes")
    for index in range(len(dataset)):
        sample = dataset[index]["data_samples"].gt_instances
        for field in fields:
            value = getattr(sample, field)
            value = value.tensor if hasattr(value, "tensor") else value
            assert torch.isfinite(value).all(), (index, field)
