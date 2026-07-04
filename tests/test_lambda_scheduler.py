import pytest
import torch

from model.config import LossParams
from model.lambda_utils import LambdaController


def test_config_defaults():
    lp = LossParams()
    assert lp.lambda_config.lambda_ == 1.0
    assert lp.lambda_config.start_lambda == 1.0
    assert lp.lambda_config.end_lambda == 1.0
    assert lp.lambda_config.lambda_schedule_steps == -1
    assert lp.lambda_config.lambda_cycle_warmup_pct == 0.3
    assert lp.lambda_config.lambda_cycle_delta == 0.0
    assert lp.lambda_config.lambda_cycle_jitter is False

def test_linear_schedule():
    lp = LossParams()
    lp.lambda_config.start_lambda = 0.2
    lp.lambda_config.end_lambda = 0.8

    controller = LambdaController()

    # max_steps fallback (lambda_schedule_steps is -1)
    assert controller(lp, current_step=0, max_steps=100, is_training=False, scorenet=torch.zeros(1)) == 0.2
    assert controller(lp, current_step=50, max_steps=100, is_training=False, scorenet=torch.zeros(1)) == 0.5
    assert controller(lp, current_step=100, max_steps=100, is_training=False, scorenet=torch.zeros(1)) == 0.8
    assert controller(lp, current_step=150, max_steps=100, is_training=False, scorenet=torch.zeros(1)) == 0.8  # clamped ratio

def test_linear_schedule_with_cycle_steps():
    lp = LossParams()
    lp.lambda_config.start_lambda = 0.2
    lp.lambda_config.end_lambda = 0.8
    lp.lambda_config.lambda_schedule_steps = 50

    controller = LambdaController()

    # Linear schedule should follow the cycle duration (50 steps) instead of max_steps (100)
    assert controller(lp, current_step=0, max_steps=100, is_training=False, scorenet=torch.zeros(1)) == 0.2
    assert controller(lp, current_step=25, max_steps=100, is_training=False, scorenet=torch.zeros(1)) == 0.5
    assert controller(lp, current_step=50, max_steps=100, is_training=False, scorenet=torch.zeros(1)) == 0.8
    assert controller(lp, current_step=75, max_steps=100, is_training=False, scorenet=torch.zeros(1)) == 0.8

def test_cosine_cycle():
    lp = LossParams()
    lp.lambda_config.start_lambda = 0.5
    lp.lambda_config.end_lambda = 0.5
    lp.lambda_config.lambda_schedule_steps = 100
    lp.lambda_config.lambda_cycle_warmup_pct = 0.3
    lp.lambda_config.lambda_cycle_delta = 0.2

    controller = LambdaController()

    # S = 0: cos_val = 0.0 -> lambda = 0.5
    assert pytest.approx(controller(lp, current_step=0, max_steps=200, is_training=False, scorenet=torch.zeros(1))) == 0.5

    # S = 30 (peak of warmup): cos_val = 1.0 -> lambda = 0.5 + 0.2 = 0.7
    assert pytest.approx(controller(lp, current_step=30, max_steps=200, is_training=False, scorenet=torch.zeros(1))) == 0.7

    # S = 15 (halfway warmup): cos_val = 0.5 * (1 - cos(pi/2)) = 0.5 -> lambda = 0.5 + 0.2 * 0.5 = 0.6
    assert pytest.approx(controller(lp, current_step=15, max_steps=200, is_training=False, scorenet=torch.zeros(1))) == 0.6

    # S = 65 (halfway cooldown): cos_val = 0.5 * (1 + cos(pi * 35 / 70)) = 0.5 -> lambda = 0.5 + 0.2 * 0.5 = 0.6
    assert pytest.approx(controller(lp, current_step=65, max_steps=200, is_training=False, scorenet=torch.zeros(1))) == 0.6

    # S = 100 (end of cycle): cos_val = 0.0 -> lambda = 0.5
    assert pytest.approx(controller(lp, current_step=100, max_steps=200, is_training=False, scorenet=torch.zeros(1))) == 0.5

    # S = 150 (post cycle): keep last value -> lambda = 0.5
    assert pytest.approx(controller(lp, current_step=150, max_steps=200, is_training=False, scorenet=torch.zeros(1))) == 0.5

def test_jitter_scaling():
    lp = LossParams()
    lp.lambda_config.start_lambda = 0.5
    lp.lambda_config.end_lambda = 0.5
    lp.lambda_config.lambda_schedule_steps = 100
    lp.lambda_config.lambda_cycle_warmup_pct = 0.5
    lp.lambda_config.lambda_cycle_delta = 0.0
    lp.lambda_config.jitter_lambda_sample = 0.1
    lp.lambda_config.lambda_cycle_jitter = True

    controller = LambdaController()

    # With lambda_cycle_jitter = True:
    # At S = 0: cos_val = 0 -> sample jitter should be 0.0
    # We call with is_training = True to apply sample jitter
    scorenet = torch.zeros(1)
    val = controller(lp, current_step=0, max_steps=100, is_training=True, scorenet=scorenet)
    assert val == 0.5  # No jitter added because cos_val = 0

    # At S = 50: cos_val = 1 -> sample jitter should be fully added
    # Let's verify by checking that running with lambda_cycle_jitter=True at S=0 yields no variance,
    # but lambda_cycle_jitter=False at S=0 yields normal variance.

    lp.lambda_config.lambda_cycle_jitter = False
    vals_no_cycle = [controller(lp, current_step=0, max_steps=100, is_training=True, scorenet=scorenet).item() for _ in range(100)]
    assert any(v != 0.5 for v in vals_no_cycle)

    lp.lambda_config.lambda_cycle_jitter = True
    vals_cycle_zero = [controller(lp, current_step=0, max_steps=100, is_training=True, scorenet=scorenet).item() for _ in range(100)]
    assert all(v == 0.5 for v in vals_cycle_zero)
