# Copyright (c) OpenMMLab. All rights reserved.
"""Optimizer registry adapters for Muon and schedule-free AdamW."""

from typing import Any, Dict, Iterable, List, Optional, Tuple, Union

import torch
from torch import Tensor
from torch.optim import Optimizer

from yopo.registry import OPTIMIZERS


def _missing_dependency_error(package: str, install_hint: str) -> ImportError:
    return ImportError(
        f'{package} is required for this optimizer. Install it with '
        f'`{install_hint}` or install YOPO optional optimizer dependencies.')


def _load_muon_updates():
    try:
        from muon import adam_update, muon_update
    except ImportError as exc:
        raise _missing_dependency_error(
            'muon-optimizer',
            'uv pip install "muon-optimizer @ '
            'git+https://github.com/KellerJordan/Muon.git@'
            'f98f1cacc0263b04290753e32be8d498c1efc806"',
        ) from exc
    return adam_update, muon_update


def _as_param_list(params: Iterable[Union[Tensor, Dict[str, Any]]]
                   ) -> List[Tensor]:
    param_items = list(params)
    if not param_items:
        return []

    flat_params = []
    if isinstance(param_items[0], dict):
        for group in param_items:
            flat_params.extend(list(group['params']))  # type: ignore[index]
    else:
        flat_params.extend(param_items)
    return [param for param in flat_params if param.requires_grad]


@OPTIMIZERS.register_module()
class AutoMuonWithAuxAdam(Optimizer):
    """Muon optimizer adapter with AdamW-style auxiliary updates.

    Parameters with ndim 2 or 4 are optimized by Muon. Other trainable
    parameters, such as bias and normalization vectors, are optimized by the
    Adam update from the upstream Muon package. This adapter is intended for
    single-process OpenMMLab/MMEngine training paths and does not require an
    initialized distributed process group.
    """

    def __init__(self,
                 params: Iterable[Union[Tensor, Dict[str, Any]]],
                 lr: float = 0.01,
                 weight_decay: float = 0.01,
                 momentum: float = 0.95,
                 nesterov: bool = True,
                 ns_steps: int = 5,
                 adam_lr: float = 0.00025,
                 adam_betas: Tuple[float, float] = (0.9, 0.95),
                 adam_eps: float = 1e-10,
                 adam_weight_decay: Optional[float] = None) -> None:
        self._adam_update, self._muon_update = _load_muon_updates()

        trainable_params = _as_param_list(params)
        muon_params = [p for p in trainable_params if p.ndim in (2, 4)]
        adam_params = [p for p in trainable_params if p.ndim not in (2, 4)]

        if adam_weight_decay is None:
            adam_weight_decay = weight_decay

        param_groups = []
        if muon_params:
            param_groups.append(
                dict(
                    params=muon_params,
                    use_muon=True,
                    lr=lr,
                    weight_decay=weight_decay,
                    momentum=momentum,
                    nesterov=nesterov,
                    ns_steps=ns_steps))
        if adam_params:
            param_groups.append(
                dict(
                    params=adam_params,
                    use_muon=False,
                    lr=adam_lr,
                    weight_decay=adam_weight_decay,
                    betas=adam_betas,
                    eps=adam_eps))
        if not param_groups:
            raise ValueError('AutoMuonWithAuxAdam got no trainable parameters')

        defaults = dict(
            lr=lr,
            weight_decay=weight_decay,
            momentum=momentum,
            nesterov=nesterov,
            ns_steps=ns_steps,
            betas=adam_betas,
            eps=adam_eps)
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

    def _step_muon_group(self, group: Dict[str, Any]) -> None:
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

            update = self._muon_update(
                grad,
                state['momentum_buffer'],
                beta=group['momentum'],
                ns_steps=group['ns_steps'],
                nesterov=group['nesterov'])
            if weight_decay:
                param.mul_(1 - lr * weight_decay)
            param.add_(update.reshape_as(param), alpha=-lr)

    def _step_adam_group(self, group: Dict[str, Any]) -> None:
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
            update = self._adam_update(
                grad,
                state['exp_avg'],
                state['exp_avg_sq'],
                state['step'],
                betas,
                eps)
            if weight_decay:
                param.mul_(1 - lr * weight_decay)
            param.add_(update, alpha=-lr)


@OPTIMIZERS.register_module()
class AutoMuonScheduleFreeOptimizer(AutoMuonWithAuxAdam):
    """Muon adapter with schedule-free train/eval mode switching."""

    def __init__(self,
                 params: Iterable[Union[Tensor, Dict[str, Any]]],
                 lr: float = 0.01,
                 weight_decay: float = 0.01,
                 momentum: float = 0.95,
                 nesterov: bool = True,
                 ns_steps: int = 5,
                 adam_lr: float = 0.00025,
                 adam_betas: Tuple[float, float] = (0.9, 0.95),
                 adam_eps: float = 1e-10,
                 adam_weight_decay: Optional[float] = None,
                 schedule_momentum: float = 0.9) -> None:
        if not 0.0 < schedule_momentum < 1.0:
            raise ValueError(
                'schedule_momentum must be in the open interval (0, 1)')
        super().__init__(
            params=params,
            lr=lr,
            weight_decay=weight_decay,
            momentum=momentum,
            nesterov=nesterov,
            ns_steps=ns_steps,
            adam_lr=adam_lr,
            adam_betas=adam_betas,
            adam_eps=adam_eps,
            adam_weight_decay=adam_weight_decay)
        for group in self.param_groups:
            group['schedule_momentum'] = schedule_momentum
            group['train_mode'] = False

    @torch.no_grad()
    def train(self) -> None:
        for group in self.param_groups:
            if group['train_mode']:
                continue
            beta = group['schedule_momentum']
            for param in group['params']:
                state = self.state[param]
                if 'schedule_free_z' in state:
                    param.lerp_(
                        state['schedule_free_z'].to(param.device),
                        weight=1 - beta)
            group['train_mode'] = True

    @torch.no_grad()
    def eval(self) -> None:
        for group in self.param_groups:
            if not group['train_mode']:
                continue
            beta = group['schedule_momentum']
            for param in group['params']:
                state = self.state[param]
                if 'schedule_free_z' in state:
                    param.lerp_(
                        state['schedule_free_z'].to(param.device),
                        weight=1 - 1 / beta)
            group['train_mode'] = False

    @torch.no_grad()
    def step(self, closure=None):  # type: ignore[override]
        if not self.param_groups[0]['train_mode']:
            raise RuntimeError(
                'AutoMuonScheduleFreeOptimizer.step() requires train mode. '
                'Use ScheduleFreeOptimizerHook or call optimizer.train() '
                'before stepping.')
        loss = super().step(closure)
        for group in self.param_groups:
            beta = group['schedule_momentum']
            for param in group['params']:
                state = self.state[param]
                if 'schedule_free_z' not in state:
                    state['schedule_free_z'] = param.detach().clone(
                        memory_format=torch.preserve_format)
                else:
                    state['schedule_free_z'].lerp_(
                        param.detach(), weight=1 - beta)
        return loss


try:
    import schedulefree
except ImportError:

    class _AdamWScheduleFreeBase(Optimizer):

        def __init__(self, *args, **kwargs) -> None:
            raise _missing_dependency_error(
                'schedulefree',
                'uv pip install schedulefree==1.4.1')
else:
    _AdamWScheduleFreeBase = schedulefree.AdamWScheduleFree


@OPTIMIZERS.register_module()
class AdamWScheduleFreeOptimizer(_AdamWScheduleFreeBase):
    """Registry name for ``schedulefree.AdamWScheduleFree``."""
