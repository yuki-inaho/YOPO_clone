"""AMUSE optimizer mode and checkpoint contracts."""

from copy import deepcopy

import torch

from yopo.engine.optimizers.amuse import AmuseOptimizer


def _one_update(parameter: torch.nn.Parameter, optimizer: AmuseOptimizer) -> None:
    optimizer.train()
    parameter.grad = torch.tensor([0.5, -0.25])
    optimizer.step()
    parameter.grad = None


def test_amuse_state_dict_restores_averaged_checkpoint_mode_exactly() -> None:
    parameter = torch.nn.Parameter(torch.tensor([1.0, -2.0]))
    optimizer = AmuseOptimizer([parameter], lr=1e-2, warmup_steps=2)
    _one_update(parameter, optimizer)
    _one_update(parameter, optimizer)
    _one_update(parameter, optimizer)
    training_weight = parameter.detach().clone()

    optimizer.eval()
    averaged_weight = parameter.detach().clone()
    assert not optimizer.train_mode
    assert not torch.equal(training_weight, averaged_weight)
    checkpoint = deepcopy(optimizer.state_dict())

    restored_parameter = torch.nn.Parameter(averaged_weight.clone())
    restored = AmuseOptimizer([restored_parameter], lr=1e-2, warmup_steps=2)
    restored.load_state_dict(checkpoint)
    assert not restored.train_mode

    restored.train()
    torch.testing.assert_close(restored_parameter, training_weight)


def test_amuse_state_dict_restores_training_checkpoint_without_mode_shift() -> None:
    parameter = torch.nn.Parameter(torch.tensor([1.0, -2.0]))
    optimizer = AmuseOptimizer([parameter], lr=1e-2, warmup_steps=2)
    _one_update(parameter, optimizer)
    _one_update(parameter, optimizer)
    _one_update(parameter, optimizer)
    checkpoint_weight = parameter.detach().clone()
    checkpoint = deepcopy(optimizer.state_dict())

    restored_parameter = torch.nn.Parameter(checkpoint_weight.clone())
    restored = AmuseOptimizer([restored_parameter], lr=1e-2, warmup_steps=2)
    restored.load_state_dict(checkpoint)
    assert restored.train_mode

    restored.train()
    torch.testing.assert_close(restored_parameter, checkpoint_weight)
