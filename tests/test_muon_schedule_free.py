import pytest
import torch


def test_custom_optimizer_classes_are_registered():
    pytest.importorskip('muon')
    pytest.importorskip('schedulefree')

    import yopo.engine.optimizers  # noqa: F401
    from yopo.registry import OPTIMIZERS

    assert OPTIMIZERS.get('AutoMuonWithAuxAdam') is not None
    assert OPTIMIZERS.get('AutoMuonScheduleFreeOptimizer') is not None
    assert OPTIMIZERS.get('AdamWScheduleFreeOptimizer') is not None


def test_auto_muon_with_aux_adam_updates_matrix_and_vector_params():
    pytest.importorskip('muon')

    from yopo.engine.optimizers import AutoMuonWithAuxAdam

    matrix_param = torch.nn.Parameter(torch.ones(2, 2))
    vector_param = torch.nn.Parameter(torch.ones(2))
    optimizer = AutoMuonWithAuxAdam(
        [matrix_param, vector_param],
        lr=0.01,
        weight_decay=0.0,
        adam_lr=0.001,
        adam_weight_decay=0.0)

    before_matrix = matrix_param.detach().clone()
    before_vector = vector_param.detach().clone()
    loss = matrix_param.square().sum() + vector_param.square().sum()
    loss.backward()
    optimizer.step()

    assert not torch.allclose(matrix_param, before_matrix)
    assert not torch.allclose(vector_param, before_vector)


def test_schedule_free_hook_switches_optimizer_modes():
    pytest.importorskip('muon')

    from yopo.engine.hooks import ScheduleFreeOptimizerHook
    from yopo.engine.optimizers import AutoMuonScheduleFreeOptimizer

    param = torch.nn.Parameter(torch.ones(2, 2))
    optimizer = AutoMuonScheduleFreeOptimizer([param])

    class OptimWrapper:
        pass

    class Runner:
        pass

    runner = Runner()
    runner.optim_wrapper = OptimWrapper()
    runner.optim_wrapper.optimizer = optimizer

    hook = ScheduleFreeOptimizerHook()
    hook.before_train(runner)
    assert all(group['train_mode'] for group in optimizer.param_groups)

    hook.before_val(runner)
    assert not any(group['train_mode'] for group in optimizer.param_groups)

    hook.after_val(runner)
    assert all(group['train_mode'] for group in optimizer.param_groups)


def test_adamw_schedule_free_registry_class_can_step():
    pytest.importorskip('schedulefree')

    from yopo.engine.optimizers import AdamWScheduleFreeOptimizer

    param = torch.nn.Parameter(torch.ones(2))
    optimizer = AdamWScheduleFreeOptimizer(
        [param], lr=0.001, weight_decay=0.0, warmup_steps=0)
    before = param.detach().clone()

    optimizer.train()
    loss = param.square().sum()
    loss.backward()
    optimizer.step()

    assert not torch.allclose(param, before)
