import math
import torch
from torch import Tensor, nn

class LambdaController(nn.Module):
    """
    lambda_ = 0.0 - purely based on game results
    0.0 < lambda_ < 1.0 - interpolated score and result
    lambda_ = 1.0 - purely based on search scores
    """
    def __init__(self, loss_params):
        super().__init__()
        self.loss_params = loss_params

        # persistent=False ensures this is NOT saved in the standard model state_dict
        self.register_buffer("jitter_buffer", torch.zeros(1), persistent=False)

    def forward(self, current_epoch: int, max_epoch: int, is_training: bool, scorenet: Tensor) -> Tensor | float:
        lp = self.loss_params

        actual_lambda = lp.start_lambda + (lp.end_lambda - lp.start_lambda) * (current_epoch / max_epoch)

        if is_training:
            if lp.jitter_lambda_batch != 0.0:
                jitter_lambda_batch = lp.jitter_lambda_batch * math.sqrt(1 - lp.jitter_decay_lambda_batch ** 2)
                batch_jitter_delta = jitter_lambda_batch * torch.randn_like(self.jitter_buffer)
                self.jitter_buffer.mul_(lp.jitter_decay_lambda_batch).add_(batch_jitter_delta)
                actual_lambda = actual_lambda + self.jitter_buffer.expand_as(scorenet)

            if lp.jitter_lambda_sample != 0.0:
                actual_lambda = actual_lambda + torch.randn_like(scorenet) * lp.jitter_lambda_sample
        else:
            eval_jitter_lambda = lp.jitter_lambda_sample + lp.jitter_lambda_batch
            if eval_jitter_lambda != 0.0:
                actual_lambda = actual_lambda + torch.randn_like(scorenet) * eval_jitter_lambda

        if torch.is_tensor(actual_lambda):
            return actual_lambda.clamp(0.0, 1.0)
        return max(0.0, min(1.0, actual_lambda))

    def on_save_checkpoint(self, checkpoint):
        # Manually save the training-only buffer to the Lightning checkpoint
        checkpoint["jitter_buffer_value"] = self.jitter_buffer

    def on_load_checkpoint(self, pl_module, checkpoint):
        trainer = pl_module.__dict__.get("_trainer", None)
        is_resuming = (
            trainer is not None and
            getattr(trainer, "ckpt_path", None) is not None
        )
        if is_resuming:
            if "jitter_buffer_value" in checkpoint:
                jitter_buffer_value = checkpoint["jitter_buffer_value"].to(
                    device=self.jitter_buffer.device,
                    dtype=self.jitter_buffer.dtype,
                )
                self.jitter_buffer.copy_(jitter_buffer_value)
