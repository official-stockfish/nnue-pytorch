import math

import torch
from torch import Tensor, nn

from .config import LossParams


class LambdaController(nn.Module):
    """
    lambda_ = 0.0 - purely based on game results
    0.0 < lambda_ < 1.0 - interpolated score and result
    lambda_ = 1.0 - purely based on search scores
    """
    def __init__(self):
        super().__init__()

        # persistent=False ensures this is NOT saved in the standard model state_dict
        self.register_buffer("jitter_buffer", torch.zeros(1), persistent=False)


    def forward(self, loss_params: LossParams, current_step: int, max_steps: int, is_training: bool, scorenet: Tensor) -> Tensor | float:
        cfg = loss_params.lambda_config
        start_lambda = cfg.start_lambda if cfg.start_lambda is not None else cfg.lambda_
        end_lambda = cfg.end_lambda if cfg.end_lambda is not None else cfg.lambda_

        if cfg.lambda_schedule_steps >= 0:
            total_steps = cfg.lambda_schedule_steps
        else:
            total_steps = max_steps

        if total_steps > 0:
            ratio = min(1.0, current_step / total_steps)
        else:
            ratio = 1.0

        actual_lambda = start_lambda + (end_lambda - start_lambda) * ratio

        cos_val = 0.0
        if cfg.lambda_schedule_steps >= 0:
            if current_step >= cfg.lambda_schedule_steps:
                cos_val = 1.0 if cfg.lambda_cycle_warmup_pct == 1.0 else 0.0
            else:
                warmup_steps = cfg.lambda_cycle_warmup_pct * cfg.lambda_schedule_steps
                if current_step < warmup_steps:
                    if warmup_steps > 0:
                        cos_val = 0.5 * (1.0 - math.cos(math.pi * current_step / warmup_steps))
                    else:
                        cos_val = 1.0
                else:
                    cooldown_steps = cfg.lambda_schedule_steps - warmup_steps
                    if cooldown_steps > 0:
                        cos_val = 0.5 * (1.0 + math.cos(math.pi * (current_step - warmup_steps) / cooldown_steps))
                    else:
                        cos_val = 0.0

            actual_lambda = actual_lambda + cfg.lambda_cycle_delta * cos_val

        if is_training:
            if cfg.jitter_lambda_batch != 0.0:
                jitter_lambda_batch = cfg.jitter_lambda_batch * math.sqrt(1 - cfg.jitter_decay_lambda_batch ** 2)
                batch_jitter_delta = jitter_lambda_batch * torch.randn_like(self.jitter_buffer)
                self.jitter_buffer.mul_(cfg.jitter_decay_lambda_batch).add_(batch_jitter_delta)
                batch_jitter = self.jitter_buffer.expand_as(scorenet)
                if cfg.lambda_schedule_steps >= 0 and cfg.lambda_cycle_jitter:
                    batch_jitter = batch_jitter * cos_val
                actual_lambda = actual_lambda + batch_jitter

            if cfg.jitter_lambda_sample != 0.0:
                sample_jitter = torch.randn_like(scorenet) * cfg.jitter_lambda_sample
                if cfg.lambda_schedule_steps >= 0 and cfg.lambda_cycle_jitter:
                    sample_jitter = sample_jitter * cos_val
                actual_lambda = actual_lambda + sample_jitter
        else:
            eval_jitter_lambda = cfg.jitter_lambda_sample + cfg.jitter_lambda_batch
            if eval_jitter_lambda != 0.0:
                sample_jitter = torch.randn_like(scorenet) * eval_jitter_lambda
                if cfg.lambda_schedule_steps >= 0 and cfg.lambda_cycle_jitter:
                    sample_jitter = sample_jitter * cos_val
                actual_lambda = actual_lambda + sample_jitter

        if torch.is_tensor(actual_lambda):
            return actual_lambda.clamp(0.0, 1.0)
        return max(0.0, min(1.0, actual_lambda))


    def on_save_checkpoint(self, checkpoint):
        # Manually save the training-only buffer to the Lightning checkpoint
        checkpoint["jitter_buffer_value"] = self.jitter_buffer

    def on_load_checkpoint(self, pl_module, checkpoint, resuming: bool = False):
        if resuming and "jitter_buffer_value" in checkpoint:
            jitter_buffer_value = checkpoint["jitter_buffer_value"].to(
                device=self.jitter_buffer.device,
                dtype=self.jitter_buffer.dtype,
            )
            self.jitter_buffer.copy_(jitter_buffer_value)
