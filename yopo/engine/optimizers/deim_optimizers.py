# Copyright (c) OpenMMLab. All rights reserved.
"""DEIM-style optimizers adapted for YOPO (mmengine).

Ports ``AutoMuonWithAuxAdam`` and ``AdamWScheduleFreeOptimizer`` from the
DEIM training stack so they can be used from mmengine configs via
``optim_wrapper.optimizer``.

- ``AutoMuonWithAuxAdam``: Muon optimizer for matrix-like parameters
  (2D / 4D) and an AdamW-style update for the remaining (vector) parameters.
  See https://github.com/KellerJordan/Muon
- ``AdamWScheduleFreeOptimizer``: thin adapter around
  ``schedulefree.AdamWScheduleFree`` so it can be built through
  mmengine's ``OPTIMIZERS`` registry.
"""
from __future__ import annotations

import inspect
from collections.abc import Iterable
from typing import Any

import torch
from torch import Tensor
from torch.optim import Optimizer

from yopo.registry import OPTIMIZERS, OPTIM_WRAPPERS
from mmengine.optim import OptimWrapper


def _as_param_list(
        params: Iterable[Tensor] | Iterable[dict[str, Any]]) -> list[Tensor]:
    param_items = list(params)
    if not param_items:
        return []
    flat_params: list[Tensor] = []
    if isinstance(param_items[0], dict):
        for group in param_items:
            flat_params.extend(list(group['params']))
    else:
        flat_params.extend(param_items)
    return [p for p in flat_params if p.requires_grad]


@OPTIMIZERS.register_module()
class AutoMuonWithAuxAdam(Optimizer):
    """Muon for matrix-like parameters plus AdamW-style updates for the rest.

    Args:
        params (Iterable): Model parameters or parameter groups.
        lr (float): Learning rate for the Muon (matrix) group.
            Defaults to 0.01.
        weight_decay (float): Weight decay applied to the Muon group.
            Defaults to 0.01.
        momentum (float): Momentum for Muon. Defaults to 0.95.
        nesterov (bool): Whether to use Nesterov momentum in Muon.
            Defaults to True.
        ns_steps (int): Number of Newton-Schulz steps. Defaults to 5.
        adam_lr (float): Learning rate for the AdamW (vector) group.
            Defaults to 0.00025.
        adam_betas (tuple): Betas for the AdamW group. Defaults to (0.9, 0.95).
        adam_eps (float): Epsilon for the AdamW group. Defaults to 1e-10.
        adam_weight_decay (float, optional): Weight decay for the AdamW group;
            defaults to ``weight_decay``.
    """

    def __init__(
        self,
        params: Iterable[Tensor] | Iterable[dict[str, Any]],
        lr: float = 0.01,
        weight_decay: float = 0.01,
        momentum: float = 0.95,
        nesterov: bool = True,
        ns_steps: int = 5,
        adam_lr: float = 0.00025,
        adam_betas: tuple[float, float] = (0.9, 0.95),
        adam_eps: float = 1e-10,
        adam_weight_decay: float | None = None,
        **kwargs,
    ) -> None:
        trainable_params = _as_param_list(params)
        muon_params = [p for p in trainable_params if p.ndim in (2, 4)]
        adam_params = [p for p in trainable_params if p.ndim not in (2, 4)]

        if adam_weight_decay is None:
            adam_weight_decay = weight_decay

        param_groups: list[dict[str, Any]] = []
        if muon_params:
            param_groups.append(
                dict(
                    params=muon_params,
                    use_muon=True,
                    lr=lr,
                    weight_decay=weight_decay,
                    momentum=momentum,
                    nesterov=nesterov,
                    ns_steps=ns_steps,
                ))
        if adam_params:
            param_groups.append(
                dict(
                    params=adam_params,
                    use_muon=False,
                    lr=adam_lr,
                    weight_decay=adam_weight_decay,
                    betas=adam_betas,
                    eps=adam_eps,
                ))
        if not param_groups:
            raise ValueError('AutoMuonWithAuxAdam got no trainable parameters')

        defaults = dict(
            lr=lr,
            weight_decay=weight_decay,
            momentum=momentum,
            nesterov=nesterov,
            ns_steps=ns_steps,
            betas=adam_betas,
            eps=adam_eps,
        )
        super().__init__(param_groups, defaults)

    @torch.no_grad()
    def step(self, closure=None):  # type: ignore[override]
        loss = None
        if closure is not None:
            with torch.enable_grad():
                loss = closure()
        for group in self.param_groups:
            if group.get('use_muon', False):
                self._step_muon_group(group)
            else:
                self._step_adam_group(group)
        return loss

    def _step_muon_group(self, group: dict[str, Any]) -> None:
        try:
            from muon import muon_update
        except ImportError as exc:  # pragma: no cover
            raise ImportError(
                'AutoMuonWithAuxAdam requires muon-optimizer'
            ) from exc
        lr = group['lr']
        weight_decay = group['weight_decay']
        for param in group['params']:
            if param.grad is None:
                continue
            grad = param.grad
            if grad.is_sparse:
                raise RuntimeError(
                    'AutoMuonWithAuxAdam does not support sparse gradients')
            state = self.state[param]
            if len(state) == 0:
                state['momentum_buffer'] = torch.zeros_like(param)
            update = muon_update(
                grad,
                state['momentum_buffer'],
                beta=group['momentum'],
                ns_steps=group['ns_steps'],
                nesterov=group['nesterov'],
            )
            if weight_decay:
                param.mul_(1 - lr * weight_decay)
            param.add_(update.reshape_as(param), alpha=-lr)

    def _step_adam_group(self, group: dict[str, Any]) -> None:
        try:
            from muon import adam_update
        except ImportError as exc:  # pragma: no cover
            raise ImportError(
                'AutoMuonWithAuxAdam requires muon-optimizer') from exc
        lr = group['lr']
        weight_decay = group['weight_decay']
        betas = group['betas']
        eps = group['eps']
        for param in group['params']:
            if param.grad is None:
                continue
            grad = param.grad
            if grad.is_sparse:
                raise RuntimeError(
                    'AutoMuonWithAuxAdam does not support sparse gradients')
            state = self.state[param]
            if len(state) == 0:
                state['step'] = 0
                state['exp_avg'] = torch.zeros_like(param)
                state['exp_avg_sq'] = torch.zeros_like(param)
            state['step'] += 1
            update = adam_update(
                grad,
                state['exp_avg'],
                state['exp_avg_sq'],
                state['step'],
                betas,
                eps,
            )
            if weight_decay:
                param.mul_(1 - lr * weight_decay)
            param.add_(update, alpha=-lr)


@OPTIMIZERS.register_module()
class AdamWScheduleFreeOptimizer(Optimizer):
    """MMEngine registry adapter for ``schedulefree.AdamWScheduleFree``.

    The actual optimizer is built via ``__new__`` so that ``_yopo_src.pth`` /
    mmengine ``OPTIMIZERS.build`` returns a real ``AdamWScheduleFree`` instance
    (a subclass of ``torch.optim.Optimizer``).
    """

    def __new__(
        cls,
        params: Iterable[Tensor] | Iterable[dict[str, Any]],
        lr: float | Tensor = 0.0025,
        betas: tuple[float, float] = (0.9, 0.95),
        eps: float = 1e-8,
        weight_decay: float = 0.0,
        warmup_steps: int = 0,
        r: float = 0.0,
        weight_lr_power: float = 2.0,
        foreach: bool | None = True,
        **kwargs,
    ):
        try:
            import schedulefree
        except ImportError as exc:  # pragma: no cover
            raise ImportError(
                'AdamWScheduleFreeOptimizer requires schedulefree') from exc
        return schedulefree.AdamWScheduleFree(
            params=params,
            lr=lr,
            betas=betas,
            eps=eps,
            weight_decay=weight_decay,
            warmup_steps=warmup_steps,
            r=r,
            weight_lr_power=weight_lr_power,
            foreach=foreach,
        )

    def __init__(
        self,
        params: Iterable[Tensor] | Iterable[dict[str, Any]],
        lr: float | Tensor = 0.0025,
        betas: tuple[float, float] = (0.9, 0.95),
        eps: float = 1e-8,
        weight_decay: float = 0.0,
        warmup_steps: int = 0,
        r: float = 0.0,
        weight_lr_power: float = 2.0,
        foreach: bool | None = True,
        **kwargs,
    ) -> None:
        pass

    def _filter_kwargs(self, kwargs: dict[str, Any]) -> dict[str, Any]:
        signature = inspect.signature(type(self).__new__)
        allowed = set(signature.parameters) - {'cls', 'params', 'kwargs'}
        return {k: v for k, v in kwargs.items() if k in allowed}


@OPTIM_WRAPPERS.register_module()
class ScheduleFreeOptimWrapper(OptimWrapper):
    """OptimWrapper that keeps a ScheduleFree optimizer in train mode.

    ``schedulefree.AdamWScheduleFree`` requires an explicit ``optimizer.train()``
    call before the first ``step()``; without it the optimizer refuses to step
    (``"Optimizer was not in train mode when step was called"``). mmengine's
    default ``OptimWrapper`` never calls ``train()``/``eval()`` on the
    optimizer, so this wrapper lazily flips the optimizer into train mode right
    before every ``step()``.

    Note:
        AMP is intentionally NOT used here: with the custom 9D-pose model a
        fresh fp16 (AmpOptimWrapper / autocast) forward produced NaNs in the
        assigner costs while the fp32 path and AdamW isolation both ran 20
        epochs clean. Effective batch 24 (8 x 3 accumulation) fits well inside
        the 20 GB VRAM at fp32 (~8.7 GB measured), so no fp16 is needed.
    """

    def __init__(self, *args, **kwargs) -> None:
        super().__init__(*args, **kwargs)
        self._ensure_train_mode()

    @torch.no_grad()
    def _ensure_train_mode(self) -> None:
        opt = self.optimizer
        if hasattr(opt, 'train') and len(opt.param_groups) > 0:
            first = opt.param_groups[0]
            if isinstance(first, dict) and not first.get('train_mode', False):
                opt.train()

    def step(self, **kwargs) -> None:
        self._ensure_train_mode()
        super().step(**kwargs)
