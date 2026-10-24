import os
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import pytest
import torch
from torch.optim.lr_scheduler import StepLR

from model import NNUE, NNUEConfig
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
    config = NNUEConfig()
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

    # Create new model instance with different weight decay and load state dict
    from copy import deepcopy

    new_cfg = deepcopy(config)
    new_cfg.optimizer_config.dense_weight_decay = 1.0e-6
    new_cfg.optimizer_config.dynamic_bias_weight_decay = 1.0e-6
    new_model = NNUE(config=new_cfg)
    new_model.configure_optimizers()
    new_model.load_state_dict(state)

    # Verify that new_model preserved its newly configured weight decays
    opt = new_model.optimizer_wrapper.optimizer
    assert any(g.get("weight_decay") == 1.0e-6 for g in opt.param_groups)


@pytest.mark.parametrize("optimizer_name", ["rangerlite", "ranger21", "adamw"])
def test_lr_scheduler_options(optimizer_name):
    # StepLR test (one_cycle_steps <= 0)
    step_config = OptimizerConfig(
        optimizer_name=optimizer_name, one_cycle_steps=0, gamma=0.95
    )
    wrapper = step_config.get_optimizer_wrapper()
    train_params = create_dummy_train_params()
    _, schedulers = wrapper.configure_optimizers(train_params)
    assert isinstance(schedulers[0], StepLR)
    assert schedulers[0].gamma == 0.95

    # OneCycleLR test (one_cycle_steps > 0)
    one_cycle_config = OptimizerConfig(
        optimizer_name=optimizer_name, one_cycle_steps=500
    )
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
    assert adamw_opt.defaults["weight_decay"] == 0.0

    # Test step() execution
    for p in adamw_opt.param_groups[0]["params"]:
        p.grad = torch.randn_like(p)
    adamw_opt.step()


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
    config_elementwise = OptimizerConfig(
        optimizer_name="rangerlite", ranger_stable_weight_decay=False
    )
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

    # Verify elementwise weight decay analytically with zero gradient
    p_zero_grad = torch.nn.Parameter(torch.ones(2, 2))
    p_zero_grad.grad = torch.zeros(2, 2)
    wrapper_ew = OptimizerConfig(
        optimizer_name="rangerlite", ranger_stable_weight_decay=False
    ).get_optimizer_wrapper()
    opts_ew, _ = wrapper_ew.configure_optimizers(
        [{"params": [p_zero_grad], "lr": 0.1, "weight_decay": 0.5}]
    )
    opts_ew[0].step()
    # Analytical: 1.0 * (1 - 0.1 * 0.5) = 0.95
    assert torch.allclose(p_zero_grad, torch.full((2, 2), 0.95))

    # Verify that stable weight decay and elementwise weight decay produce different results
    p_stable = torch.nn.Parameter(torch.tensor([[1.0, 2.0], [3.0, 4.0]]))
    p_stable.grad = torch.tensor([[0.1, 0.2], [0.3, 0.4]])
    wrapper_stable = OptimizerConfig(
        optimizer_name="rangerlite", ranger_stable_weight_decay=True
    ).get_optimizer_wrapper()
    opts_st, _ = wrapper_stable.configure_optimizers(
        [{"params": [p_stable], "lr": 1e-3, "weight_decay": 0.01}]
    )
    opts_st[0].step()
    assert not torch.allclose(param_ew, p_stable)


def test_virtual_ft_weight_decay_and_pruning():
    from model.nnue import _get_parameters

    # Test K32Q2^ which has virtual weights
    config_k32q2 = NNUEConfig(
        features="K32Q2^",
        optimizer_config=OptimizerConfig(
            ft_weight_decay=0.01,
            virtual_ft_weight_decay=0.05,
            dense_weight_decay=0.02,
        ),
    )
    model_k32q2 = NNUE(config=config_k32q2)
    real_ft = _get_parameters(
        [model_k32q2.model.input], get_biases=False, get_virtual=False
    )
    virtual_ft = _get_parameters(
        [model_k32q2.model.input], get_biases=False, get_virtual=True
    )
    assert len(real_ft) > 0
    assert len(virtual_ft) > 0

    res = model_k32q2.configure_optimizers()
    opt = res[0][0]
    # Check that virtual weights group exists with correct weight decay
    found_real = False
    found_virtual = False
    for group in opt.param_groups:
        assert len(group["params"]) > 0, "Empty parameter group was not pruned!"
        if group["weight_decay"] == 0.01:
            found_real = True
            assert group["params"] == real_ft
        elif group["weight_decay"] == 0.05:
            found_virtual = True
            assert group["params"] == virtual_ft
    assert found_real
    assert found_virtual

    # Test architecture without virtual weights (pruning empty group)
    config_no_virt = NNUEConfig(
        features="Full_Threats",
        optimizer_config=OptimizerConfig(
            ft_weight_decay=0.01,
            virtual_ft_weight_decay=0.05,
        ),
    )
    model_no_virt = NNUE(config=config_no_virt)
    assert (
        len(
            _get_parameters(
                [model_no_virt.model.input], get_biases=False, get_virtual=True
            )
        )
        == 0
    )
    res_no_virt = model_no_virt.configure_optimizers()
    opt_no_virt = res_no_virt[0][0]
    for group in opt_no_virt.param_groups:
        assert len(group["params"]) > 0
        assert group["weight_decay"] != 0.05  # No virtual group since it had no params


def test_weight_clipper_and_swa_export():
    from trainer.callbacks import ExplicitSWA, WeightClipper

    class DummyModel:
        def __init__(self):
            self.calls = []

        def clip_weights(self, include_input=False):
            self.calls.append(include_input)

    class DummyTrainer:
        def __init__(self, global_step=0, current_epoch=0, rank=0):
            self.global_step = global_step
            self.current_epoch = current_epoch
            self.rank = rank
            self.model = type("DummyWrapper", (), {"model": DummyModel()})()

    # Legacy clipping (interval <= 0)
    clipper_legacy = WeightClipper(input_clipping_interval=0)
    t = DummyTrainer(global_step=10)
    clipper_legacy.on_train_batch_end(t)
    assert t.model.model.calls == [False]
    clipper_legacy.on_train_epoch_end(t)
    assert t.model.model.calls == [False, True]

    # Interval clipping (interval = 5)
    clipper_interval = WeightClipper(input_clipping_interval=5)
    t = DummyTrainer(global_step=4)
    clipper_interval.on_train_batch_end(t)
    assert t.model.model.calls == [False]
    t.global_step = 5
    clipper_interval.on_train_batch_end(t)
    assert t.model.model.calls == [False, True]
    t.global_step = 10
    clipper_interval.on_train_batch_end(t)
    assert t.model.model.calls == [False, True, True]
    clipper_interval.on_train_epoch_end(t)
    assert t.model.model.calls == [False, True, True], (
        "Should not clip input on epoch end when interval > 0"
    )

    # ExplicitSWA clip on export
    swa = ExplicitSWA(swa_start_epoch=0, save_dir="/tmp")
    t_swa = DummyTrainer(current_epoch=0, rank=0)
    t_swa.save_checkpoint = lambda path: None
    t_swa.model.eval = lambda: None
    swa.swap_weights = lambda model, to_eval: None
    swa.on_train_end(t_swa)
    assert t_swa.model.model.calls == [True]


def test_top2moe_optimizer_param_groups():
    config = NNUEConfig(
        features="Full_Threats",
        optimizer_config=OptimizerConfig(
            ft_weight_decay=1e-4,
            dense_weight_decay=2e-4,
            dynamic_bias_weight_decay=3e-4,
        ),
    )
    nnue = NNUE(config=config)
    res = nnue.configure_optimizers()
    opt = res[0][0]

    all_model_params = {id(p) for p in nnue.model.parameters()}
    registered_params = set()

    found_ft = False
    found_dense = False
    found_dyn_bias = False

    for group in opt.param_groups:
        for p in group["params"]:
            assert id(p) not in registered_params, "Duplicate parameter in optimizer!"
            registered_params.add(id(p))

        if group["weight_decay"] == 1e-4:
            found_ft = True
        elif group["weight_decay"] == 2e-4:
            found_dense = True
            expected_dense = {
                id(nnue.model.layer_stacks.l1.adapter_down),
                id(nnue.model.layer_stacks.l1.adapter_up),
                id(nnue.model.layer_stacks.l2.adapter_down),
                id(nnue.model.layer_stacks.l2.adapter_up),
                id(nnue.model.layer_stacks.output.adapter_down),
                id(nnue.model.layer_stacks.output.adapter_up),
            }
            assert {id(p) for p in group["params"]} == expected_dense
        elif group["weight_decay"] == 3e-4:
            found_dyn_bias = True
            expected_dyn_bias = {
                id(nnue.model.router.bias_head_down.weight),
                id(nnue.model.router.bias_head_up.weight),
                id(nnue.model.router.bias_head_up.bias),
            }
            assert {id(p) for p in group["params"]} == expected_dyn_bias

    assert found_ft
    assert found_dense
    assert found_dyn_bias
    assert registered_params == all_model_params, (
        f"Missing {all_model_params - registered_params} params from optimizer!"
    )

    # Test backward compatibility alias
    cfg_alias = OptimizerConfig(factorized_weight_decay=4e-5)
    assert cfg_alias.dynamic_bias_weight_decay == 4e-5
