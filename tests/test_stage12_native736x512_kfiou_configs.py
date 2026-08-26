"""Contracts for selected native-736x512 Stage-12 KFIoU runs."""

from pathlib import Path

from mmengine.config import Config
from mmengine.hooks import CheckpointHook, EarlyStoppingHook


CONFIG_ROOT = Path('configs/yopo')
PREFIX = (
    'nocs_fruits_detection_Jun30_2025_rgbd_3dbbox_'
    'stage12_native736x512_kfiou_'
)


def _load(suffix: str) -> Config:
    return Config.fromfile(CONFIG_ROOT / f'{PREFIX}{suffix}.py')


def _assert_selected_objective(config: Config) -> None:
    loss = config.model.bbox_head.loss_obb_aux
    assert loss.type == 'GaussianKFIoULoss'
    assert loss.loss_weight == 1.0
    assert loss.fail_on_invalid is True
    assert loss.eps == 1e-7
    assert config.optim_wrapper.optimizer.lr == 3e-6
    assert config.param_scheduler == []
    assert config.load_from is None
    assert config.resume is False


def test_selected_base_is_native_geometry_plus_exact_probe_objective() -> None:
    config = _load('base')

    _assert_selected_objective(config)
    assert config.train_dataloader.batch_size == 16
    assert config.model.data_preprocessor.pad_size_divisor == 1
    assert [step.type for step in config.train_dataloader.dataset.pipeline
            ].count('AssertIdentityImageGeometry') == 1
    assert [hook.type for hook in config.custom_hooks] == [
        'ScheduleFreeOptimizerModeHook']


def test_capacity_smoke_uses_selected_loss_without_val_or_checkpoint() -> None:
    config = _load('capacity_smoke')

    _assert_selected_objective(config)
    assert config.train_dataloader.batch_size == 16
    assert config.train_cfg.type == 'IterBasedTrainLoop'
    assert config.train_cfg.max_iters == 2
    assert config.val_dataloader is None
    assert config.val_cfg is None
    assert config.val_evaluator is None
    assert config.default_hooks.checkpoint is None


def test_one_epoch_gate_runs_corrected_validation_without_early_stop() -> None:
    config = _load('gate1')

    _assert_selected_objective(config)
    assert config.train_cfg.max_epochs == 1
    assert config.train_cfg.val_interval == 1
    assert config.val_dataloader is not None
    assert config.val_evaluator.type == 'NOCSMetric'
    assert config.default_hooks.checkpoint is None
    assert [hook.type for hook in config.custom_hooks] == [
        'ScheduleFreeOptimizerModeHook']


def test_full_schedule_hooks_and_checkpoint_schema_are_executable() -> None:
    config = _load('full')

    _assert_selected_objective(config)
    assert config.train_cfg.max_epochs == 50
    assert config.train_cfg.val_interval == 5

    hooks = {hook.type: hook for hook in config.custom_hooks}
    assert set(hooks) == {
        'ScheduleFreeOptimizerModeHook', 'EarlyStoppingHook'}
    early = hooks['EarlyStoppingHook']
    assert early.monitor == 'AP50_95'
    assert early.min_delta == 1e-3
    assert early.patience == 4
    EarlyStoppingHook(**{
        key: value for key, value in early.items() if key != 'type'})

    checkpoint = config.default_hooks.checkpoint
    assert checkpoint.interval == 5
    assert checkpoint.save_last is True
    assert checkpoint.save_optimizer is True
    assert checkpoint.max_keep_ckpts == 2
    assert checkpoint.save_best == [
        'AP50_95', 'AP75', '3d_iou_0.25']
    built = CheckpointHook(**{
        key: value for key, value in checkpoint.items()
        if key not in {'type', '_delete_'}
    })
    assert built.save_best == ['AP50_95', 'AP75', '3d_iou_0.25']
