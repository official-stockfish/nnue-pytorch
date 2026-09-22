from dataclasses import dataclass
from typing import Any

import torch

from .lr_scheduler import LRSchedulerConfig, setup_lr_scheduler


@dataclass(kw_only=True, frozen=False)
class AdamWConfig(LRSchedulerConfig):
    adamw_beta1: float = 0.9
    """AdamW beta1 coefficient."""

    adamw_beta2: float = 0.999
    """AdamW beta2 coefficient."""

    adamw_eps: float = 1.0e-8
    """AdamW epsilon stability parameter."""

    adamw_fused: bool = True
    """Whether to use fused kernel for AdamW if supported by PyTorch and device."""


class AdamWWrapper:
    def __init__(self, config):
        self.config = config
        self.needs_train_flip = False
        self.optimizer = None

    def configure_optimizers(self, train_params):
        betas = (self.config.adamw_beta1, self.config.adamw_beta2)
        self.optimizer = torch.optim.AdamW(
            train_params,
            lr=self.config.lr,
            betas=betas,
            eps=self.config.adamw_eps,
            weight_decay=0.0,
            fused=self.config.adamw_fused,
        )

        scheduler = setup_lr_scheduler(self.optimizer, self.config)
        return [self.optimizer], [scheduler]

    def switch_to_train(self, force=False):
        pass

    def switch_to_eval(self):
        pass

    def on_train_epoch_start(self, pl_module: Any):
        pass

    def on_train_batch_start(self, pl_module: Any, batch, batch_idx):
        pass

    def on_validation_epoch_start(self, pl_module: Any):
        pass

    def on_test_epoch_start(self, pl_module: Any):
        pass

    def on_train_epoch_end(self, pl_module: Any):
        pass

    def on_save_checkpoint(self, pl_module: Any, checkpoint):
        pass
