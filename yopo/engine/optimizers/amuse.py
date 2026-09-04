# Copyright (c) 2026 AMUSE authors. Licensed under Apache-2.0.
"""AMUSE optimizer adapter for the YOPO/MMEngine training stack.

This is the PyTorch implementation from ``kjeiun/amuse`` adapted only at the
registry/parameter-group boundary needed by MMEngine.  Matrix-like YOPO
parameters use the AMUSE Muon path; vectors and other parameters use the
AMUSE AdamW fallback.  AMUSE is schedule-free and therefore requires an
explicit ``train()``/``eval()`` transition around validation.
"""

from __future__ import annotations

from collections.abc import Iterable
from typing import Any

import torch
from torch import Tensor
from torch.optim import Optimizer

from mmengine.optim import AmpOptimWrapper, OptimWrapper

from yopo.registry import OPTIMIZERS, OPTIM_WRAPPERS


def _source_groups(
    params: Iterable[Tensor] | Iterable[dict[str, Any]],
) -> list[dict[str, Any]]:
    items = list(params)
    if not items:
        return []
    if isinstance(items[0], dict):
        return [dict(group) for group in items]
    return [{"params": items}]


def _is_muon_parameter(parameter: Tensor) -> bool:
    # Match YOPO's existing AutoMuon split: convolution and linear kernels.
    return parameter.ndim in (2, 4)


@torch.no_grad()
def _zeropower_via_newton_schulz5(
    gradient: Tensor,
    steps: int,
) -> Tensor:
    if gradient.ndim < 2:
        raise ValueError("AMUSE Muon requires a matrix-like gradient")
    a, b, c = 3.4445, -4.7750, 2.0315
    work = gradient.bfloat16()
    transposed = False
    if work.size(-2) > work.size(-1):
        work = work.mT
        transposed = True
    work = work / (work.norm(dim=(-2, -1), keepdim=True) + 1e-7)
    for _ in range(steps):
        gram = work @ work.mT
        work = a * work + (b * gram + c * (gram @ gram)) @ work
    if transposed:
        work = work.mT
    return work


@torch.no_grad()
def _amuse_muon_update(
    gradient: Tensor,
    momentum_buffer: Tensor,
    *,
    momentum: float,
    steps: int,
    aux_update_type: str,
) -> Tensor:
    momentum_buffer.lerp_(gradient, 1.0 - momentum)
    update = gradient.lerp_(momentum_buffer, momentum)
    if update.ndim == 4:
        update = update.view(len(update), -1)
    update = _zeropower_via_newton_schulz5(update, steps)
    if aux_update_type == "adamw":
        # Scaling used by the official AMUSE AdamW-aux setting.
        update *= 0.2 * max(update.size(0), update.size(1)) ** 0.5
    elif aux_update_type == "sgd":
        update *= max(1.0, update.size(-2) / update.size(-1)) ** 0.5
    else:
        raise ValueError("AMUSE aux_update_type must be 'adamw' or 'sgd'")
    return update


@OPTIMIZERS.register_module()
class AmuseOptimizer(Optimizer):
    """AMUSE with MMEngine-compatible automatic parameter partitioning.

    ``DefaultOptimWrapperConstructor`` may pass one group per parameter when
    ``paramwise_cfg`` is active.  The adapter preserves each incoming LR and
    decay multiplier while splitting every group into Muon and auxiliary
    AMUSE groups.
    """

    def __init__(
        self,
        params: Iterable[Tensor] | Iterable[dict[str, Any]],
        lr: float = 2.0e-4,
        aux_lr: float | None = None,
        weight_decay: float = 1.0e-4,
        aux_weight_decay: float | None = None,
        beta1: float = 0.9,
        beta2: float = 0.999,
        eps: float = 1.0e-10,
        momentum: float = 0.95,
        ns_steps: int = 5,
        warmup_steps: int = 100,
        rho: float = 1.0,
        r: float = 0.0,
        weight_lr_power: float = 2.0,
        weight_decay_at_y: float = 0.0,
        aux_update_type: str = "adamw",
        **kwargs: Any,
    ) -> None:
        del kwargs
        if warmup_steps <= 0:
            raise ValueError("AmuseOptimizer requires warmup_steps > 0")
        if not 0.0 <= beta1 < 1.0:
            raise ValueError("AMUSE beta1 must be in [0, 1)")
        if not 0.0 <= beta2 < 1.0:
            raise ValueError("AMUSE beta2 must be in [0, 1)")
        if not 0.0 <= momentum < 1.0:
            raise ValueError("AMUSE momentum must be in [0, 1)")
        if ns_steps <= 0:
            raise ValueError("AMUSE ns_steps must be positive")
        if aux_update_type not in {"adamw", "sgd"}:
            raise ValueError("AMUSE aux_update_type must be 'adamw' or 'sgd'")

        self.weight_decay_at_y = float(weight_decay_at_y)
        self.beta1_init = float(beta1)
        self.weight_lr_power = float(weight_lr_power)
        self.warmup_steps = int(warmup_steps)
        self.rho = float(rho)
        self.r = float(r)
        self._train_mode = False

        aux_lr_value = float(lr if aux_lr is None else aux_lr)
        aux_wd_value = float(weight_decay if aux_weight_decay is None else aux_weight_decay)
        internal_groups: list[dict[str, Any]] = []
        for source in _source_groups(params):
            source_params = [p for p in source["params"] if p.requires_grad]
            if not source_params:
                continue
            source_lr = float(source.get("lr", lr))
            source_wd = float(source.get("weight_decay", weight_decay))
            lr_ratio = source_lr / float(lr) if float(lr) else 1.0
            source_aux_lr = aux_lr_value * lr_ratio
            source_aux_wd = aux_wd_value
            if "weight_decay" in source and aux_weight_decay is not None:
                source_aux_wd = float(aux_weight_decay) * (source_wd / float(weight_decay)) if weight_decay else source_wd
            muon_params = [p for p in source_params if _is_muon_parameter(p)]
            aux_params = [p for p in source_params if not _is_muon_parameter(p)]
            if muon_params:
                internal_groups.append(
                    {
                        "params": muon_params,
                        "use_muon": True,
                        "update_type": "muon",
                        "lr": source_lr,
                        "weight_decay": source_wd,
                        "momentum": momentum,
                        "aux_update_type": aux_update_type,
                    }
                )
            if aux_params:
                internal_groups.append(
                    {
                        "params": aux_params,
                        "use_muon": False,
                        "update_type": aux_update_type,
                        "lr": source_aux_lr,
                        "weight_decay": source_aux_wd,
                        "beta2": beta2,
                        "eps": eps,
                    }
                )
        if not internal_groups:
            raise ValueError("AmuseOptimizer got no trainable parameters")

        # MMEngine's OptimWrapper uses optimizer.defaults['lr'] for its
        # aggregate/base learning-rate log entry when parameter-wise groups
        # are present.
        super().__init__(internal_groups, defaults={"lr": float(lr)})
        for group in self.param_groups:
            group.setdefault("warmup_steps", self.warmup_steps)
            group.setdefault("k", 0)
            group.setdefault("weight_sum", 0.0)
            group.setdefault("beta1", self.beta1_init)
            group["base_lr"] = float(group["lr"])
            if group["update_type"] == "muon":
                group.setdefault("momentum", momentum)
                group.setdefault("aux_update_type", aux_update_type)
                for parameter in group["params"]:
                    self.state[parameter]["momentum_buffer"] = torch.zeros_like(parameter)
            elif group["update_type"] == "adamw":
                group.setdefault("beta2", beta2)
                group.setdefault("eps", eps)
                for parameter in group["params"]:
                    self.state[parameter]["exp_avg_sq"] = torch.zeros_like(parameter)
            elif group["update_type"] == "sgd":
                pass
            else:  # pragma: no cover - guarded by constructor validation
                raise ValueError(f"Unsupported AMUSE update type: {group['update_type']}")

    def _compute_beta1(self, group: dict[str, Any], t: int, checkpoint: float) -> float:
        if t <= self.warmup_steps:
            if t == self.warmup_steps:
                group["c_warmup"] = checkpoint
            return self.beta1_init
        c_warmup = float(group.get("c_warmup", 1.0 / self.warmup_steps))
        ratio = checkpoint * (1.0 - c_warmup) / max(c_warmup * (1.0 - checkpoint), 1e-12)
        return 1.0 - ratio**self.rho * (1.0 - self.beta1_init)

    @torch.no_grad()
    def train(self, mode: bool = True) -> "AmuseOptimizer":
        mode = bool(mode)
        if mode and not self._train_mode:
            for group in self.param_groups:
                beta1 = float(group.get("beta1", self.beta1_init))
                for parameter in group["params"]:
                    state = self.state[parameter]
                    if "z" in state:
                        parameter.add_(state["z"] - parameter, alpha=1.0 - beta1)
        elif not mode and self._train_mode:
            for group in self.param_groups:
                beta1 = float(group.get("beta1", self.beta1_init))
                for parameter in group["params"]:
                    state = self.state[parameter]
                    if "z" in state:
                        parameter.add_(state["z"] - parameter, alpha=1.0 - 1.0 / beta1)
        self._train_mode = mode
        return self

    def eval(self) -> "AmuseOptimizer":
        return self.train(False)

    @property
    def train_mode(self) -> bool:
        return self._train_mode

    def state_dict(self) -> dict[str, Any]:
        """Persist whether model parameters currently hold train or averaged weights."""
        state = super().state_dict()
        state['_amuse_train_mode'] = self._train_mode
        return state

    def load_state_dict(self, state_dict: dict[str, Any]) -> None:
        """Restore the mode that corresponds to the checkpointed model weights."""
        state = state_dict.copy()
        train_mode = bool(state.pop('_amuse_train_mode', False))
        super().load_state_dict(state)
        self._train_mode = train_mode

    def _z(self, parameter: Tensor) -> Tensor:
        state = self.state[parameter]
        if "z" not in state:
            state["z"] = parameter.detach().clone(memory_format=torch.preserve_format)
        return state["z"]

    @torch.no_grad()
    def step(self, closure=None):  # type: ignore[override]
        if not self._train_mode:
            raise RuntimeError("AmuseOptimizer.step() requires optimizer.train()")
        loss = None
        if closure is not None:
            with torch.enable_grad():
                loss = closure()

        for group in self.param_groups:
            t = int(group["k"]) + 1
            warmup = int(group.get("warmup_steps", self.warmup_steps))
            lr = float(group["base_lr"]) * min(1.0, t / warmup)
            group["lr"] = lr
            weight = (t**self.r) * (lr**self.weight_lr_power)
            weight_sum = float(group.get("weight_sum", 0.0)) + weight
            checkpoint = weight / weight_sum if weight_sum > 0.0 else 1.0
            beta1 = self._compute_beta1(group, t, checkpoint)
            group["weight_sum"] = weight_sum
            group["beta1"] = beta1
            update_type = group["update_type"]

            for parameter in group["params"]:
                if parameter.grad is None:
                    continue
                state = self.state[parameter]
                z = self._z(parameter)
                if self.weight_decay_at_y:
                    z.sub_(parameter, alpha=lr * self.weight_decay_at_y)
                    parameter.sub_(parameter, alpha=lr * self.weight_decay_at_y * (1.0 - beta1))
                parameter.add_(z - parameter, alpha=1.0 - 1.0 / beta1)

                if update_type == "muon":
                    if "momentum_buffer" not in state:
                        state["momentum_buffer"] = torch.zeros_like(parameter)
                    update = _amuse_muon_update(
                        parameter.grad,
                        state["momentum_buffer"],
                        momentum=float(group["momentum"]),
                        steps=int(group.get("ns_steps", 5)),
                        aux_update_type=str(group.get("aux_update_type", "adamw")),
                    )
                    if group["weight_decay"]:
                        z.mul_(1.0 - lr * float(group["weight_decay"]))
                    z.add_(update.reshape_as(parameter), alpha=-lr)
                elif update_type == "adamw":
                    if "exp_avg_sq" not in state:
                        state["exp_avg_sq"] = torch.zeros_like(parameter)
                    beta2 = float(group.get("beta2", 0.999))
                    variance = state["exp_avg_sq"]
                    variance.mul_(beta2).addcmul_(
                        parameter.grad, parameter.grad, value=1.0 - beta2
                    )
                    denominator = variance.div(1.0 - beta2**t).sqrt_().add_(float(group.get("eps", 1e-10)))
                    update = parameter.grad / denominator
                    if group["weight_decay"]:
                        update = update.add(z, alpha=float(group["weight_decay"]))
                    z.add_(update, alpha=-lr)
                elif update_type == "sgd":
                    if group["weight_decay"]:
                        z.mul_(1.0 - lr * float(group["weight_decay"]))
                    z.add_(parameter.grad, alpha=-lr)
                else:  # pragma: no cover
                    raise ValueError(f"Unsupported AMUSE update type: {update_type}")

                parameter.add_(z - parameter, alpha=checkpoint)
                parameter.add_(z - parameter, alpha=1.0 - beta1)
            group["k"] = t
        return loss


@OPTIM_WRAPPERS.register_module()
class AmuseOptimWrapper(OptimWrapper):
    """Keep AMUSE in train mode for updates; validation hook calls ``eval``."""

    def __init__(self, *args: Any, **kwargs: Any) -> None:
        super().__init__(*args, **kwargs)
        self.optimizer.train()

    def step(self, **kwargs: Any) -> None:
        self.optimizer.train()
        super().step(**kwargs)


@OPTIM_WRAPPERS.register_module()
class AmpAmuseOptimWrapper(AmpOptimWrapper):
    """BF16/FP16 AMUSE wrapper with explicit schedule-free train mode."""

    def __init__(self, *args: Any, **kwargs: Any) -> None:
        super().__init__(*args, **kwargs)
        self.optimizer.train()

    def step(self, **kwargs: Any) -> None:
        self.optimizer.train()
        super().step(**kwargs)


__all__ = ["AmuseOptimizer", "AmuseOptimWrapper", "AmpAmuseOptimWrapper"]
