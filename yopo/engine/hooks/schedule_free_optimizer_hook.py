# Copyright (c) OpenMMLab. All rights reserved.
"""Hook for schedule-free optimizers."""

from typing import Any, Optional

from mmengine.hooks import Hook

from yopo.registry import HOOKS


@HOOKS.register_module()
class ScheduleFreeOptimizerHook(Hook):
    """Keep schedule-free optimizers in train/eval modes.

    The ``schedulefree`` optimizers and ``AutoMuonScheduleFreeOptimizer`` expose
    ``train`` and ``eval`` methods. MMEngine does not call these methods by
    default, so this hook switches modes around train, validation, testing, and
    checkpoint serialization.
    """

    priority = 'VERY_HIGH'

    def _optimizer(self, runner) -> Optional[Any]:
        optim_wrapper = getattr(runner, 'optim_wrapper', None)
        return getattr(optim_wrapper, 'optimizer', None)

    def _set_mode(self, runner, mode: str) -> None:
        optimizer = self._optimizer(runner)
        if optimizer is None:
            return
        mode_fn = getattr(optimizer, mode, None)
        if callable(mode_fn):
            mode_fn()

    def before_train(self, runner) -> None:
        self._set_mode(runner, 'train')

    def before_train_epoch(self, runner) -> None:
        self._set_mode(runner, 'train')

    def before_train_iter(self, runner, batch_idx: int, data_batch=None) -> None:
        self._set_mode(runner, 'train')

    def before_val(self, runner) -> None:
        self._set_mode(runner, 'eval')

    def after_val(self, runner, metrics=None) -> None:
        self._set_mode(runner, 'train')

    def before_test(self, runner) -> None:
        self._set_mode(runner, 'eval')

    def after_test(self, runner, metrics=None) -> None:
        self._set_mode(runner, 'train')

    def before_save_checkpoint(self, runner, checkpoint: dict) -> None:
        self._set_mode(runner, 'eval')
