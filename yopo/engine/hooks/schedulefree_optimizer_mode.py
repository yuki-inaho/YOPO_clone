"""Switch ScheduleFree optimizers to averaged weights during validation."""

from typing import Any

from mmengine.hooks import Hook

from yopo.registry import HOOKS


@HOOKS.register_module()
class ScheduleFreeOptimizerModeHook(Hook):
    """Use a ScheduleFree optimizer's evaluation weights for validation.

    The configured optimization wrapper must expose an ``optimizer`` whose
    public API provides callable ``eval()`` and ``train()`` methods.  Keeping
    the optimizer in eval mode until ``after_val`` is intentional: MMEngine's
    checkpoint and early-stopping hooks run in ``after_val_epoch`` and must
    therefore observe (and, for best checkpoints, save) the averaged weights.
    """

    @staticmethod
    def _optimizer(runner: Any) -> Any:
        wrapper = getattr(runner, 'optim_wrapper', None)
        optimizer = getattr(wrapper, 'optimizer', None)
        if optimizer is None:
            raise RuntimeError(
                'ScheduleFreeOptimizerModeHook requires '
                'runner.optim_wrapper.optimizer')
        if not callable(getattr(optimizer, 'eval', None)):
            raise RuntimeError(
                'ScheduleFreeOptimizerModeHook requires an optimizer with '
                'a callable eval() method')
        if not callable(getattr(optimizer, 'train', None)):
            raise RuntimeError(
                'ScheduleFreeOptimizerModeHook requires an optimizer with '
                'a callable train() method')
        return optimizer

    def before_val(self, runner: Any) -> None:
        self._optimizer(runner).eval()

    def after_val(self, runner: Any) -> None:
        self._optimizer(runner).train()
