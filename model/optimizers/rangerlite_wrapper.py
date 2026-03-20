import lightning as L
import torch

from dataclasses import dataclass
from .ranger_lite import RangerLite

@dataclass
class RangerLiteConfig:
    gamma: float = 0.992
    """Multiplicative factor applied to the learning rate after every epoch."""

    pnm_momentum: float = 1.0
    """Positive Negative Momentum parameter. Value of 1.0 corresponds to ranger21 behaviour."""

class RangerLiteWrapper:
    def __init__(
        self,
        config,
        legacy_mode,
    ):
        self.gamma = config.gamma
        self.pnm_momentum = config.pnm_momentum
        self.legacy_mode = legacy_mode
        self.needs_train_flip = True

    def configure_optimizers(self, train_params):
        # train_params is expected to be a list of dicts: [{'params': ..., 'lr': ..., 'weight_decay': ...}]
        self.optimizer = RangerLite(
            train_params,
            # Global defaults acting as fallbacks if not defined in param groups
            lr=1.0,
            weight_decay=0.0,
            use_legacy_scoping_bug=self.legacy_mode,
            normloss_active=self.legacy_mode,
            # pnm_momentum was hardcoded to 1.0 in ranger21. The argument was unused.
            pnm_momentum=self.pnm_momentum,
        )

        scheduler = torch.optim.lr_scheduler.StepLR(
            self.optimizer, step_size=1, gamma=self.gamma
        )

        print(
            f"[RangerLiteSetup] gamma={self.gamma} pnm_momentum={self.pnm_momentum}."
        )

        return [self.optimizer], [scheduler]

    def on_train_batch_start(self, pl_module: L.LightningModule, batch, batch_idx):
        if self.needs_train_flip and not self.legacy_mode:
            self.optimizer.restore_for_training()
            self.needs_train_flip = False

    def on_validation_epoch_start(self, pl_module: L.LightningModule):
        if not self.legacy_mode:
            self.optimizer.swap_for_inference()
            self.needs_train_flip = True

    def on_test_epoch_start(self, pl_module: L.LightningModule):
        if not self.legacy_mode:
            self.optimizer.swap_for_inference()
            self.needs_train_flip = True

    def on_train_epoch_end(self, pl_module: L.LightningModule):
        if not self.legacy_mode:
            self.optimizer.swap_for_inference()
            self.needs_train_flip = True

    def on_save_checkpoint(self, pl_module: L.LightningModule, checkpoint):
        if not self.legacy_mode:
            self.optimizer.swap_for_inference()
            self.needs_train_flip = True

    def on_train_epoch_start(self, pl_module: L.LightningModule):
        if not self.legacy_mode:
            self.optimizer.restore_for_training()
            self.needs_train_flip = False
