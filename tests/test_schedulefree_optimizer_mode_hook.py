from pathlib import Path
from types import SimpleNamespace

import pytest
import torch
from mmengine.config import Config
from mmengine.hooks import CheckpointHook, EarlyStoppingHook, Hook

from yopo.engine.hooks import ScheduleFreeOptimizerModeHook


class _FakeScheduleFreeOptimizer:
    def __init__(self) -> None:
        self.mode = 'train'
        self.eval_weight = torch.tensor(3.0)
        self.train_weight = torch.tensor(1.0)
        self.weight = self.train_weight.clone()

    def eval(self) -> None:
        if self.mode == 'train':
            self.weight.copy_(self.eval_weight)
            self.mode = 'eval'

    def train(self) -> None:
        if self.mode == 'eval':
            self.weight.copy_(self.train_weight)
            self.mode = 'train'


def _runner(optimizer):
    return SimpleNamespace(
        optim_wrapper=SimpleNamespace(optimizer=optimizer))


def test_hook_uses_averaged_weights_for_whole_validation_lifecycle():
    optimizer = _FakeScheduleFreeOptimizer()
    runner = _runner(optimizer)
    hook = ScheduleFreeOptimizerModeHook()

    hook.before_val(runner)
    assert optimizer.mode == 'eval'
    assert optimizer.weight.item() == pytest.approx(3.0)

    # CheckpointHook and EarlyStoppingHook consume metrics in after_val_epoch;
    # restoration deliberately happens only at the later after_val boundary.
    assert ScheduleFreeOptimizerModeHook.after_val_epoch is Hook.after_val_epoch
    assert CheckpointHook.priority == 'VERY_LOW'
    assert EarlyStoppingHook.priority == 'LOWEST'
    assert optimizer.mode == 'eval'

    hook.after_val(runner)
    assert optimizer.mode == 'train'
    assert optimizer.weight.item() == pytest.approx(1.0)


def test_hook_mode_transitions_are_idempotent():
    optimizer = _FakeScheduleFreeOptimizer()
    runner = _runner(optimizer)
    hook = ScheduleFreeOptimizerModeHook()

    hook.before_val(runner)
    hook.before_val(runner)
    assert optimizer.mode == 'eval'
    assert optimizer.weight.item() == pytest.approx(3.0)

    hook.after_val(runner)
    hook.after_val(runner)
    assert optimizer.mode == 'train'
    assert optimizer.weight.item() == pytest.approx(1.0)


@pytest.mark.parametrize('runner', [SimpleNamespace(), _runner(object())])
def test_hook_fails_clearly_for_unsupported_runner_or_optimizer(runner):
    with pytest.raises(RuntimeError, match='ScheduleFreeOptimizerModeHook requires'):
        ScheduleFreeOptimizerModeHook().before_val(runner)


def test_stage10_full_config_enables_hook_before_early_stopping():
    config_path = (
        Path(__file__).parents[1]
        / 'configs/yopo/'
        'nocs_fruits_detection_Jun30_2025_rgbd_3dbbox_'
        'stage10_2d_anchor_mal_full.py'
    )
    config = Config.fromfile(config_path)

    assert [hook.type for hook in config.custom_hooks] == [
        'ScheduleFreeOptimizerModeHook',
        'EarlyStoppingHook',
    ]
