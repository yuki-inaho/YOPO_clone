"""Integration contract for strict 2D RGB transfer into the 3D model."""

from __future__ import annotations

import json
from types import SimpleNamespace

import torch
from mmengine.config import Config
from mmengine.dataset import pseudo_collate
from mmengine.optim import build_optim_wrapper
from mmengine.utils import import_modules_from_strings

from yopo.registry import DATASETS, MODELS
from yopo.utils import register_all_modules


CONFIG_PATH = "configs/yopo/nocs_custom_fruit_rgbd_3dbbox_transfer.py"


def _build_model_and_config():
    register_all_modules()
    config = Config.fromfile(CONFIG_PATH)
    import_modules_from_strings(**config.custom_imports)
    model = MODELS.build(config.model)
    return model, config


def test_rgbd_3dbbox_transfer_freeze(tmp_path):
    """Load exact RGB weights, omit frozen RGB from optimizer, backprop one batch."""
    from yopo.engine.hooks.rgb_backbone_transfer import RGBBackboneTransferHook

    model, config = _build_model_and_config()
    runner = SimpleNamespace(model=model, work_dir=str(tmp_path))
    hook = RGBBackboneTransferHook(
        checkpoint=config.rgb_transfer_checkpoint,
        report_filename="partial_transfer_report.json",
    )
    hook.before_train(runner)

    report = json.loads((tmp_path / "partial_transfer_report.json").read_text())
    assert report["loaded_key_count"] == 300
    assert report["unexpected_keys"] == []
    assert report["missing_target_rgb_keys"] == []
    assert all(not parameter.requires_grad for parameter in model.backbone.rgb_backbone.parameters())

    optim_wrapper = build_optim_wrapper(
        model,
        config.optim_wrapper,
    )
    optimized_ids = {
        id(parameter)
        for group in optim_wrapper.optimizer.param_groups
        for parameter in group["params"]
    }
    assert not any(id(parameter) in optimized_ids for parameter in model.backbone.rgb_backbone.parameters())

    dataset = DATASETS.build(config.val_dataloader.dataset)
    packed = next(
        dataset[index]
        for index in range(len(dataset))
        if len(dataset[index]["data_samples"].gt_instances) > 0
    )
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model.to(device).train()
    batch = model.data_preprocessor(pseudo_collate([packed]), training=True)
    batch["inputs"] = batch["inputs"].to(device)
    batch["data_samples"] = [sample.to(device) for sample in batch["data_samples"]]
    losses = model.loss(batch["inputs"], batch["data_samples"])
    total_loss = sum(value for key, value in losses.items() if "loss" in key)
    assert torch.isfinite(total_loss)
    total_loss.backward()
    assert any(parameter.grad is not None for parameter in model.backbone.depth_backbone.parameters())
    assert any(parameter.grad is not None for parameter in model.backbone.fuse.parameters())
    assert any(parameter.grad is not None for parameter in model.bbox_head.parameters())
