from pathlib import Path

import torch
from mmengine.config import Config

from yopo.utils.partial_checkpoint import build_rgb_backbone_transfer_state


CONFIG_ROOT = "configs/yopo"
FULL_3D_CHECKPOINT = Path(
    "work_dirs/rgbd3d_amp26_lr2e4_20ep_eval/best_3d_iou_0.50_epoch_14.pth"
)
RGB_2D_CHECKPOINT = Path(
    "work_dirs/rddetr_tomato_riou_linear_ft20/best_rbbox_mAP_50_epoch_20.pth"
)


def test_long_finetune_is_fresh_and_preserves_the_full_training_contract():
    cfg = Config.fromfile(f"{CONFIG_ROOT}/nocs_custom_fruit_rgbd_3dbbox_long_finetune.py")

    assert cfg.load_from == str(FULL_3D_CHECKPOINT)
    assert cfg.resume is False
    assert cfg.train_cfg.max_epochs == 80
    assert cfg.train_cfg.val_interval == 5
    assert cfg.train_dataloader.batch_size == 26
    assert cfg.model.backbone.freeze_rgb is True
    assert cfg.model.test_cfg.max_per_img == 100
    assert cfg.optim_wrapper.type == "AmpScheduleFreeOptimWrapper"
    assert cfg.optim_wrapper.optimizer.lr == 5e-5
    assert cfg.default_hooks.checkpoint.save_best == "3d_iou_0.50"
    assert cfg.default_hooks.checkpoint.save_last is False
    assert cfg.default_hooks.checkpoint.save_optimizer is False
    assert all(hook["type"] != "ThroughputBenchmarkHook" for hook in cfg.custom_hooks)


def test_frozen_rgb_weights_in_the_full_best_match_the_strict_2d_source():
    full_checkpoint = torch.load(FULL_3D_CHECKPOINT, map_location="cpu", weights_only=False)
    rgb_checkpoint = torch.load(RGB_2D_CHECKPOINT, map_location="cpu", weights_only=False)
    selection = build_rgb_backbone_transfer_state(
        rgb_checkpoint["state_dict"], full_checkpoint["state_dict"]
    )

    assert len(selection.state_dict) == 300
    assert selection.missing_target_keys == ()
    for target_key, source_value in selection.state_dict.items():
        assert torch.equal(full_checkpoint["state_dict"][target_key], source_value)


def test_lr1e5_candidate_changes_only_the_fresh_optimizer_learning_rate():
    baseline = Config.fromfile(f"{CONFIG_ROOT}/nocs_custom_fruit_rgbd_3dbbox_long_finetune.py")
    candidate = Config.fromfile(
        f"{CONFIG_ROOT}/nocs_custom_fruit_rgbd_3dbbox_long_finetune_lr1e5.py"
    )

    assert baseline.optim_wrapper.optimizer.lr == 5e-5
    assert candidate.optim_wrapper.optimizer.lr == 1e-5
    assert candidate.load_from == baseline.load_from
    assert candidate.resume is False
    assert candidate.train_cfg == baseline.train_cfg
    assert candidate.param_scheduler == baseline.param_scheduler
    assert candidate.train_dataloader == baseline.train_dataloader
    assert candidate.model == baseline.model
    assert candidate.default_hooks == baseline.default_hooks


def test_lr1e6_candidate_changes_only_the_fresh_optimizer_learning_rate():
    baseline = Config.fromfile(f"{CONFIG_ROOT}/nocs_custom_fruit_rgbd_3dbbox_long_finetune.py")
    candidate = Config.fromfile(
        f"{CONFIG_ROOT}/nocs_custom_fruit_rgbd_3dbbox_long_finetune_lr1e6.py"
    )

    assert baseline.optim_wrapper.optimizer.lr == 5e-5
    assert candidate.optim_wrapper.optimizer.lr == 1e-6
    assert candidate.load_from == baseline.load_from
    assert candidate.resume is False
    assert candidate.train_cfg == baseline.train_cfg
    assert candidate.param_scheduler == baseline.param_scheduler
    assert candidate.train_dataloader == baseline.train_dataloader
    assert candidate.model == baseline.model
    assert candidate.default_hooks == baseline.default_hooks
