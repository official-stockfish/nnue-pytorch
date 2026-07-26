from dataclasses import dataclass

import torch

from .lr_scheduler import LRSchedulerConfig, setup_lr_scheduler
from .ranger_lite import RangerLite


@dataclass(kw_only=True, frozen=False)
class RangerLiteConfig(LRSchedulerConfig):
    pnm_active: bool = True
    """Whether to activate Positive Negative Momentum."""

    pnm_momentum: float = 1.0
    """Positive Negative Momentum parameter. Value of 1.0 corresponds to ranger21 behaviour. Note: `pnm_momentum` was hardcoded to 1.0 in ranger21. The argument was unused."""

    lookahead_alpha: float = 0.5
    """Lookahead alpha parameter. Value of 0.5 corresponds to ranger21 behaviour."""

    lookahead_steps: int = 5
    """Lookahead steps parameter. Value of 5 corresponds to ranger21 behaviour."""


class RangerLiteWrapper:
    def __init__(
        self,
        config,
        legacy_mode,
    ):
        self.config = config
        self.pnm_active = config.pnm_active
        self.pnm_momentum = config.pnm_momentum
        self.lookahead_alpha = config.lookahead_alpha
        self.lookahead_steps = config.lookahead_steps
        self.legacy_mode = legacy_mode
        self.needs_train_flip = True

        self.optimizer = None

    def configure_optimizers(self, train_params):
        # train_params is expected to be a list of dicts: [{'params': ..., 'lr': ..., 'weight_decay': ...}]
        self.optimizer = RangerLite(
            train_params,
            # Global defaults acting as fallbacks if not defined in param groups
            lr=1.0,
            weight_decay=0.0,
            use_legacy_scoping_bug=self.legacy_mode,
            normloss_active=self.legacy_mode,
            pnm_activate=self.pnm_active,
            pnm_momentum=self.pnm_momentum,
            lookahead_blending_alpha=self.lookahead_alpha,
            lookahead_mergetime=self.lookahead_steps,
        )

        scheduler = setup_lr_scheduler(self.optimizer, train_params, self.config)
        return [self.optimizer], [scheduler]

    def switch_to_train(self, force=False):
        if (force or self.needs_train_flip) and not self.legacy_mode:
            self.optimizer.train()
            self.needs_train_flip = False

    def switch_to_eval(self):
        if not self.legacy_mode:
            self.optimizer.eval()
            self.needs_train_flip = True

    def on_train_epoch_start(self, pl_module: torch.nn.Module):
        self.switch_to_train(True)

    def on_train_batch_start(self, pl_module: torch.nn.Module, batch, batch_idx):
        self.switch_to_train()

    def on_validation_epoch_start(self, pl_module: torch.nn.Module):
        self.switch_to_eval()

    def on_test_epoch_start(self, pl_module: torch.nn.Module):
        self.switch_to_eval()

    def on_train_epoch_end(self, pl_module: torch.nn.Module):
        self.switch_to_eval()

    def on_save_checkpoint(self, pl_module: torch.nn.Module, checkpoint):
        self.switch_to_eval()
