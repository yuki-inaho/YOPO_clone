from pathlib import Path

from mmengine.config import Config


CONFIG_ROOT = "configs/yopo/"


def test_rgb_transfer_hook_factory_preserves_both_config_contracts():
    transfer = Config.fromfile(CONFIG_ROOT + "nocs_custom_fruit_rgbd_3dbbox_transfer.py")
    benchmark = Config.fromfile(
        CONFIG_ROOT + "nocs_custom_fruit_rgbd_3dbbox_throughput_benchmark.py"
    )

    expected_transfer = dict(
        type="RGBBackboneTransferHook",
        checkpoint=(
            "work_dirs/rddetr_tomato_riou_linear_ft20/"
            "best_rbbox_mAP_50_epoch_20.pth"
        ),
        report_filename="partial_transfer_report.json",
    )
    assert transfer.custom_hooks == [expected_transfer]
    assert benchmark.custom_hooks[0] == expected_transfer
    assert benchmark.custom_hooks[1]["type"] == "ThroughputBenchmarkHook"
    assert expected_transfer["report_filename"] == "partial_transfer_report.json"


def test_refactored_configs_keep_the_training_contracts():
    amp20 = Config.fromfile(CONFIG_ROOT + "nocs_custom_fruit_rgbd_3dbbox_amp_20ep.py")
    capacity = Config.fromfile(
        CONFIG_ROOT + "nocs_custom_fruit_rgbd_3dbbox_amp_batch26_capacity.py"
    )
    smoke = Config.fromfile(CONFIG_ROOT + "nocs_custom_fruit_rgbd_3dbbox_amp_smoke.py")

    assert amp20.train_dataloader.batch_size == 26
    assert amp20.optim_wrapper.type == "AmpScheduleFreeOptimWrapper"
    assert amp20.optim_wrapper.optimizer.lr == 2e-4
    assert amp20.model.test_cfg.max_per_img == 100
    assert amp20.val_evaluator.type == "NOCSMetric"
    assert capacity.train_dataloader.batch_size == 26
    assert capacity.train_cfg.type == "IterBasedTrainLoop"
    assert capacity.train_cfg.max_iters == 4
    assert smoke.train_dataloader.batch_size == 1
    assert smoke.train_cfg.type == "IterBasedTrainLoop"
    assert smoke.train_cfg.max_iters == 1


def test_child_configs_inherit_the_parent_load_and_resume_defaults():
    config_root = Path(CONFIG_ROOT)
    for filename in (
        "nocs_custom_fruit_rgbd_3dbbox_amp_smoke.py",
        "nocs_custom_fruit_rgbd_3dbbox_amp_batch26_capacity.py",
        "nocs_custom_fruit_rgbd_3dbbox_amp_20ep.py",
        "nocs_custom_fruit_rgbd_3dbbox_throughput_benchmark.py",
    ):
        source = (config_root / filename).read_text(encoding="utf-8")
        assert "load_from = None" not in source
        assert "resume = False" not in source


def test_interval5_speed_config_only_changes_validation_schedule():
    baseline = Config.fromfile(CONFIG_ROOT + "nocs_custom_fruit_rgbd_3dbbox_amp_20ep.py")
    speed = Config.fromfile(
        CONFIG_ROOT + "nocs_custom_fruit_rgbd_3dbbox_amp_interval5.py"
    )

    assert baseline.train_cfg.val_interval == 1
    assert speed.train_cfg.val_interval == 5
    assert speed.train_cfg.type == baseline.train_cfg.type
    assert speed.train_cfg.max_epochs == baseline.train_cfg.max_epochs
    assert speed.train_dataloader == baseline.train_dataloader
    assert speed.model == baseline.model
    assert speed.optim_wrapper == baseline.optim_wrapper
    assert speed.val_evaluator == baseline.val_evaluator
    assert speed.default_hooks == baseline.default_hooks
