"""Contract tests for safe 2D OBB to RGB-D RGB-backbone transfer."""

from __future__ import annotations

import pytest
import torch
from mmengine.config import Config

from yopo.registry import MODELS
from yopo.utils import register_all_modules


CHECKPOINT_PATH = (
    "work_dirs/rddetr_tomato_riou_linear_ft20/"
    "best_rbbox_mAP_50_epoch_20.pth"
)
TARGET_CONFIG_PATH = "configs/yopo/nocs_custom_real_hgnetv2_rgbd_deim_cop_dual.py"


def _source_and_target_state_dicts():
    register_all_modules()
    source = torch.load(CHECKPOINT_PATH, map_location="cpu", weights_only=False)[
        "state_dict"
    ]
    config = Config.fromfile(TARGET_CONFIG_PATH)
    target = MODELS.build(config.model).state_dict()
    return source, target


def test_partial_transfer_schema():
    """Only exact RGB B2 tensors may cross the 2D/3D task boundary."""
    from yopo.utils.partial_checkpoint import build_rgb_backbone_transfer_state

    source, target = _source_and_target_state_dicts()
    selection = build_rgb_backbone_transfer_state(source, target)

    assert len(selection.state_dict) == 300
    assert selection.missing_target_keys == ()
    assert len(selection.ignored_source_keys) == 60
    assert all(key.endswith("num_batches_tracked") for key in selection.ignored_source_keys)
    assert all(key.startswith("backbone.rgb_backbone.") for key in selection.state_dict)

    unknown_key_source = dict(source)
    unknown_key_source["backbone.unexpected.weight"] = torch.zeros(1)
    with pytest.raises(ValueError, match="unknown RGB-backbone source keys"):
        build_rgb_backbone_transfer_state(unknown_key_source, target)

    mismatch_source = dict(source)
    mismatch_source["backbone.stem.stem1.conv.weight"] = torch.zeros(1)
    with pytest.raises(ValueError, match="RGB-backbone shape mismatches"):
        build_rgb_backbone_transfer_state(mismatch_source, target)
