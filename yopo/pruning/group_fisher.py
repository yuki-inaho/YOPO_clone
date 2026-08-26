"""Native PyTorch Group Taylor/Fisher collection for explicit channel units.

The implementation follows the gate-gradient contract used by the reference
JAX pruning stack: for each example, all activation sites in one dependency
group contribute to the same virtual channel gate.  Site contributions are
summed before Taylor absolute values or Fisher squares are accumulated.
"""

from __future__ import annotations

from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from typing import Literal

import torch
from torch import Tensor, nn


@dataclass(frozen=True, slots=True)
class FFNGroupSpec:
    """One transformer FFN hidden width and its coupled parameter axes."""

    name: str
    hidden_channels: int
    sites: tuple[nn.Module, ...]
    first_weight: str
    first_bias: str | None
    second_weight: str
    channel_axis: int = -1


def _module_names(model: nn.Module) -> dict[int, str]:
    return {id(module): name for name, module in model.named_modules()}


def discover_transformer_ffn_groups(model: nn.Module) -> tuple[FFNGroupSpec, ...]:
    """Discover the safe Linear-hidden-Linear groups in YOPO transformers.

    The adapter deliberately validates YOPO/MMCV's current FFN layout instead
    of guessing through an arbitrary module graph.  An upstream layout change
    therefore fails before importance collection or checkpoint slicing.
    """

    names = _module_names(model)
    groups: list[FFNGroupSpec] = []
    for stack_name in ("encoder", "decoder"):
        stack = getattr(model, stack_name, None)
        layers = getattr(stack, "layers", None)
        if layers is None:
            raise TypeError(f"model.{stack_name} has no layer sequence")
        for index, layer in enumerate(layers):
            ffn = getattr(layer, "ffn", None)
            ffn_layers = getattr(ffn, "layers", None)
            if ffn_layers is None or len(ffn_layers) < 2:
                raise TypeError(
                    f"{stack_name}.layers.{index}.ffn must contain at least "
                    "two stages"
                )
            hidden_site = ffn_layers[0]
            first_linears = [
                module for module in hidden_site.modules()
                if isinstance(module, nn.Linear)
            ]
            if len(first_linears) != 1 or not isinstance(ffn_layers[1], nn.Linear):
                raise TypeError(
                    f"{stack_name}.layers.{index}.ffn has an unsupported layout"
                )
            first = first_linears[0]
            second = ffn_layers[1]
            if first.out_features != second.in_features:
                raise ValueError(
                    f"{stack_name}.layers.{index}.ffn hidden widths differ: "
                    f"{first.out_features} vs {second.in_features}"
                )
            first_name = names[id(first)]
            second_name = names[id(second)]
            groups.append(
                FFNGroupSpec(
                    name=f"{stack_name}.layers.{index}.ffn.hidden",
                    hidden_channels=first.out_features,
                    sites=(hidden_site,),
                    first_weight=f"{first_name}.weight",
                    first_bias=(f"{first_name}.bias" if first.bias is not None else None),
                    second_weight=f"{second_name}.weight",
                )
            )
    return tuple(groups)


class GroupGateImportanceCollector:
    """Collect per-example virtual-gate Taylor and Fisher importance.

    Call ``begin_batch`` before forward, backpropagate a scalar sum/mean loss,
    then call ``end_batch``.  Gradient checkpointing must be disabled during
    collection because recomputation changes hook invocation counts.
    """

    def __init__(self, groups: Sequence[FFNGroupSpec]) -> None:
        if not groups:
            raise ValueError("at least one channel group is required")
        if len({group.name for group in groups}) != len(groups):
            raise ValueError("channel group names must be unique")
        self._groups = {group.name: group for group in groups}
        self._pending: dict[str, list[Tensor]] | None = None
        self._taylor = {
            group.name: torch.zeros(group.hidden_channels, dtype=torch.float64)
            for group in groups
        }
        self._fisher = {
            group.name: torch.zeros(group.hidden_channels, dtype=torch.float64)
            for group in groups
        }
        self._sample_count = 0
        self._handles: list[torch.utils.hooks.RemovableHandle] = []

    @property
    def sample_count(self) -> int:
        return self._sample_count

    def install(self) -> None:
        if self._handles:
            raise RuntimeError("importance hooks are already installed")
        for group in self._groups.values():
            for site in group.sites:
                self._handles.append(
                    site.register_forward_hook(self._make_forward_hook(group))
                )

    def remove(self) -> None:
        for handle in self._handles:
            handle.remove()
        self._handles.clear()
        self._pending = None

    def __enter__(self) -> GroupGateImportanceCollector:
        self.install()
        return self

    def __exit__(self, *_exc_info) -> None:
        self.remove()

    def begin_batch(self) -> None:
        if not self._handles:
            raise RuntimeError("install hooks before beginning a batch")
        if self._pending is not None:
            raise RuntimeError("the previous calibration batch was not ended")
        self._pending = {name: [] for name in self._groups}

    def _make_forward_hook(self, group: FFNGroupSpec):
        def forward_hook(_module: nn.Module, _inputs, output) -> None:
            if self._pending is None:
                raise RuntimeError("call begin_batch before calibration forward")
            if not isinstance(output, Tensor):
                raise TypeError(f"importance site {group.name!r} did not return a Tensor")
            if not output.requires_grad:
                raise RuntimeError(
                    f"importance site {group.name!r} has no gradient; "
                    "do not collect under no_grad"
                )
            axis = group.channel_axis if group.channel_axis >= 0 \
                else output.ndim + group.channel_axis
            if axis <= 0 or axis >= output.ndim:
                raise ValueError(
                    f"channel axis {group.channel_axis} is invalid for "
                    f"{group.name!r} output {tuple(output.shape)}"
                )
            if output.shape[axis] != group.hidden_channels:
                raise ValueError(
                    f"importance site {group.name!r} has width {output.shape[axis]}, "
                    f"expected {group.hidden_channels}"
                )
            activation = output.detach()

            def gradient_hook(gradient: Tensor) -> None:
                reduction_axes = tuple(
                    dim for dim in range(gradient.ndim) if dim not in (0, axis)
                )
                product = activation.float() * gradient.float()
                contribution = (
                    product.sum(dim=reduction_axes) if reduction_axes else product
                )
                self._pending[group.name].append(contribution.detach().cpu())

            output.register_hook(gradient_hook)

        return forward_hook

    def end_batch(self) -> None:
        if self._pending is None:
            raise RuntimeError("call begin_batch before end_batch")
        batch_size: int | None = None
        for name, contributions in self._pending.items():
            if not contributions:
                raise RuntimeError(f"group {name!r} received no backward contribution")
            gate_gradient = torch.stack(contributions, dim=0).sum(dim=0)
            if gate_gradient.ndim != 2:
                raise ValueError(
                    f"group {name!r} gate gradient must be [batch, channel], "
                    f"got {tuple(gate_gradient.shape)}"
                )
            if gate_gradient.shape[1] != self._groups[name].hidden_channels:
                raise ValueError(f"group {name!r} gate-gradient width changed")
            current_batch = int(gate_gradient.shape[0])
            if batch_size is None:
                batch_size = current_batch
            elif batch_size != current_batch:
                raise ValueError("channel groups observed different batch sizes")
            self._taylor[name] += gate_gradient.abs().sum(dim=0, dtype=torch.float64)
            self._fisher[name] += 0.5 * gate_gradient.square().sum(
                dim=0, dtype=torch.float64
            )
        if batch_size is None or batch_size <= 0:
            raise ValueError("calibration batches must be non-empty")
        self._sample_count += batch_size
        self._pending = None

    def scores(
        self,
        estimator: Literal["taylor", "fisher"] = "fisher",
        *,
        costs: Mapping[str, Tensor] | None = None,
    ) -> dict[str, Tensor]:
        if self._sample_count == 0:
            raise RuntimeError("no completed calibration batch is available")
        source = self._fisher if estimator == "fisher" else self._taylor
        result: dict[str, Tensor] = {}
        for name, accumulated in source.items():
            score = (accumulated / self._sample_count).float()
            if costs is not None:
                if name not in costs:
                    raise KeyError(f"missing channel cost for group {name!r}")
                cost = torch.as_tensor(costs[name], dtype=torch.float32)
                if cost.shape != score.shape or not bool(torch.all(cost > 0)):
                    raise ValueError(f"invalid channel cost for group {name!r}")
                score = score / cost
            if not bool(torch.isfinite(score).all()):
                raise ValueError(f"importance for group {name!r} is not finite")
            result[name] = score
        return result


def select_top_channels(score: Tensor, remaining: int) -> tuple[int, ...]:
    """Select high-importance original indices with deterministic tie breaks."""

    values = torch.as_tensor(score, dtype=torch.float64).flatten()
    if remaining <= 0 or remaining > values.numel():
        raise ValueError(
            f"remaining must be in [1, {values.numel()}], got {remaining}"
        )
    if not bool(torch.isfinite(values).all()):
        raise ValueError("importance contains NaN or infinity")
    ranked = sorted(
        range(values.numel()),
        key=lambda index: (-float(values[index].item()), index),
    )
    return tuple(sorted(ranked[:remaining]))
