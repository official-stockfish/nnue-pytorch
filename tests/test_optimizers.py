import os
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import pytest
import torch
from torch.optim.lr_scheduler import StepLR

from model import NNUE, NNUELightningConfig
from model.optimizers import (
    AdamWConfig,
    AdamWWrapper,
    OptimizerConfig,
    RangerLiteWrapper,
    SafeOneCycleLR,
    ScheduleFreeWrapper,
)

ALL_OPTIMIZERS = ["rangerlite", "ranger21", "schedulefree", "adamw"]


def create_dummy_train_params():
    param1 = torch.nn.Parameter(torch.randn(2, 2))
    param2 = torch.nn.Parameter(torch.randn(2, 2))
    return [
        {"params": [param1], "lr": 1e-3, "weight_decay": 0.01},
        {"params": [param2], "lr": 5e-4, "weight_decay": 0.0},
    ]


class DummyPLModule:
    pass


@pytest.mark.parametrize("optimizer_name", ALL_OPTIMIZERS)
def test_optimizer_config_wrapper_instantiation(optimizer_name):
    config = OptimizerConfig(optimizer_name=optimizer_name)
    wrapper = config.get_optimizer_wrapper()
    assert wrapper is not None

    if optimizer_name in ("rangerlite", "ranger21"):
        assert isinstance(wrapper, RangerLiteWrapper)
        assert wrapper.legacy_mode == (optimizer_name == "ranger21")
    elif optimizer_name == "schedulefree":
        assert isinstance(wrapper, ScheduleFreeWrapper)
    elif optimizer_name == "adamw":
        assert isinstance(wrapper, AdamWWrapper)


@pytest.mark.parametrize("optimizer_name", ALL_OPTIMIZERS)
def test_wrapper_configure_optimizers(optimizer_name):
    config = OptimizerConfig(optimizer_name=optimizer_name)
    wrapper = config.get_optimizer_wrapper()
    train_params = create_dummy_train_params()

    res = wrapper.configure_optimizers(train_params)
    assert wrapper.optimizer is not None

    if optimizer_name == "schedulefree":
        # ScheduleFree returns optimizer directly
        assert res == wrapper.optimizer
    else:
        # RangerLite, Ranger21, and AdamW return ([optimizer], [scheduler])
        optimizers, schedulers = res
        assert len(optimizers) == 1
        assert optimizers[0] == wrapper.optimizer
        assert len(schedulers) == 1


@pytest.mark.parametrize("optimizer_name", ALL_OPTIMIZERS)
def test_wrapper_hooks_and_state_flips(optimizer_name):
    config = OptimizerConfig(optimizer_name=optimizer_name)
    wrapper = config.get_optimizer_wrapper()
    train_params = create_dummy_train_params()
    wrapper.configure_optimizers(train_params)

    pl_module = DummyPLModule()

    # Test state flips
    wrapper.switch_to_train(force=True)
    wrapper.switch_to_eval()

    # Test all lifecycle hooks
    wrapper.on_train_epoch_start(pl_module)
    wrapper.on_train_batch_start(pl_module, batch=None, batch_idx=0)
    wrapper.on_validation_epoch_start(pl_module)
    wrapper.on_test_epoch_start(pl_module)
    wrapper.on_train_epoch_end(pl_module)
    checkpoint = {}
    wrapper.on_save_checkpoint(pl_module, checkpoint)


@pytest.mark.parametrize("optimizer_name", ALL_OPTIMIZERS)
def test_nnue_integration_all_optimizers(optimizer_name):
    config = NNUELightningConfig()
    config.optimizer_config.optimizer_name = optimizer_name

    model = NNUE(config=config)
    res = model.configure_optimizers()
    assert res is not None

    # Test train / eval switches on NNUE
    model.train()
    model.eval()

    # Test NNUE hooks
    model.on_train_epoch_start()
    model.on_train_batch_start(batch=None, batch_idx=0)
    model.on_validation_epoch_start()
    model.on_test_epoch_start()
    model.on_train_epoch_end()
    model.on_validation_epoch_end()

    # Test state dict saving and loading with optimizer state
    state = model.state_dict()
    assert "optimizer_state_dict" in state

    # Create new model instance and load state dict
    new_model = NNUE(config=config)
    new_model.configure_optimizers()
    new_model.load_state_dict(state)


@pytest.mark.parametrize("optimizer_name", ["rangerlite", "ranger21", "adamw"])
def test_lr_scheduler_options(optimizer_name):
    # StepLR test (one_cycle_steps <= 0)
    step_config = OptimizerConfig(optimizer_name=optimizer_name, one_cycle_steps=0, gamma=0.95)
    wrapper = step_config.get_optimizer_wrapper()
    train_params = create_dummy_train_params()
    _, schedulers = wrapper.configure_optimizers(train_params)
    assert isinstance(schedulers[0], StepLR)
    assert schedulers[0].gamma == 0.95

    # OneCycleLR test (one_cycle_steps > 0)
    one_cycle_config = OptimizerConfig(optimizer_name=optimizer_name, one_cycle_steps=500)
    wrapper = one_cycle_config.get_optimizer_wrapper()
    _, schedulers = wrapper.configure_optimizers(train_params)
    assert isinstance(schedulers[0], dict)
    assert isinstance(schedulers[0]["scheduler"], SafeOneCycleLR)
    assert schedulers[0]["interval"] == "step"


def test_schedulefree_specific_options():
    sf_config = OptimizerConfig(optimizer_name="schedulefree", warmup_steps=500)
    wrapper = sf_config.get_optimizer_wrapper()
    train_params = create_dummy_train_params()
    wrapper.configure_optimizers(train_params)
    assert wrapper.warmup_steps == 500


def test_adamw_config_defaults_and_options():
    config = AdamWConfig()
    assert config.adamw_beta1 == 0.9
    assert config.adamw_beta2 == 0.999
    assert config.adamw_eps == 1.0e-8
    assert config.adamw_fused is True
    assert config.gamma == 0.992
    assert config.one_cycle_steps == 0

    adamw_config = OptimizerConfig(
        optimizer_name="adamw",
        adamw_beta1=0.85,
        adamw_beta2=0.98,
        adamw_eps=1e-7,
        adamw_fused=False,
    )
    wrapper = adamw_config.get_optimizer_wrapper()
    train_params = create_dummy_train_params()
    opts, _ = wrapper.configure_optimizers(train_params)
    adamw_opt = opts[0]

    assert isinstance(adamw_opt, torch.optim.AdamW)
    assert adamw_opt.defaults["betas"] == (0.85, 0.98)
    assert adamw_opt.defaults["eps"] == 1e-7


def test_ranger_stable_weight_decay_option():
    # Test default value (True)
    config_default = OptimizerConfig(optimizer_name="rangerlite")
    assert config_default.ranger_stable_weight_decay is True

    wrapper_default = config_default.get_optimizer_wrapper()
    param = torch.nn.Parameter(torch.tensor([[1.0, 2.0], [3.0, 4.0]]))
    train_params = [{"params": [param], "lr": 1e-3, "weight_decay": 0.01}]
    wrapper_default.configure_optimizers(train_params)
    assert wrapper_default.optimizer.use_stable_weight_decay is True

    # Test turning off stable weight decay (False)
    config_elementwise = OptimizerConfig(optimizer_name="rangerlite", ranger_stable_weight_decay=False)
    assert config_elementwise.ranger_stable_weight_decay is False

    wrapper_elementwise = config_elementwise.get_optimizer_wrapper()
    param_ew = torch.nn.Parameter(torch.tensor([[1.0, 2.0], [3.0, 4.0]]))
    param_ew.grad = torch.tensor([[0.1, 0.2], [0.3, 0.4]])
    train_params_ew = [{"params": [param_ew], "lr": 1e-3, "weight_decay": 0.01}]
    opts, _ = wrapper_elementwise.configure_optimizers(train_params_ew)
    opt = opts[0]

    assert opt.use_stable_weight_decay is False

    # Execute optimizer step with elementwise weight decay
    initial_p = param_ew.clone()
    opt.step()
    # Check that weights updated and changed from initial_p
    assert not torch.allclose(param_ew, initial_p)

