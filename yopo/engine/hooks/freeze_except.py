"""Fail-closed parameter allowlist for controlled fine-tuning studies."""

from __future__ import annotations

import re
from collections.abc import Sequence

from mmengine.hooks import Hook
from mmengine.model import is_model_wrapper
from mmengine.runner import Runner

from yopo.registry import HOOKS


@HOOKS.register_module()
class FreezeExceptHook(Hook):
    """Freeze every parameter except names fully matched by an allowlist."""

    def __init__(self, trainable_patterns: Sequence[str]) -> None:
        if not trainable_patterns:
            raise ValueError('trainable_patterns must not be empty')
        self.patterns = tuple(re.compile(pattern)
                              for pattern in trainable_patterns)
        self.trainable_parameter_names: tuple[str, ...] = ()

    def before_train(self, runner: Runner) -> None:
        model = runner.model
        if is_model_wrapper(model):
            model = model.module

        matched = []
        for name, parameter in model.named_parameters():
            trainable = any(pattern.fullmatch(name)
                            for pattern in self.patterns)
            parameter.requires_grad_(trainable)
            if trainable:
                matched.append(name)
        if not matched:
            raise RuntimeError('FreezeExceptHook matched no parameters')
        self.trainable_parameter_names = tuple(matched)
        logger = getattr(runner, 'logger', None)
        if logger is not None:
            logger.info(
                'FreezeExceptHook trainable parameters (%d): %s',
                len(matched), ', '.join(matched))
