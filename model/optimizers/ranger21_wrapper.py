import lightning as L
import torch

from dataclasses import dataclass

from ..metal_support import MPS_AVAILABLE


@dataclass
class Ranger21Config:
    gamma: float = 0.992
    """Multiplicative factor applied to the learning rate after every epoch."""


class Ranger21Wrapper:
    def __init__(
        self,
        config,
        max_epoch,
        num_batches_per_epoch,
    ):
        self.max_epoch = max_epoch
        self.num_batches_per_epoch = num_batches_per_epoch
        self.gamma = config.gamma

    def configure_optimizers(self, train_params):
        if self.num_batches_per_epoch is None:
            raise RuntimeError(
                "[Ranger21Wrapper] Required parameter for training not set: num_batches_per_epoch"
            )

        use_metal = MPS_AVAILABLE and any(
            p.device.type == "mps"
            for group in train_params
            for p in group["params"]
        )

        if use_metal:
            from .fused_ranger21 import FusedRanger21

            optimizer = FusedRanger21(
                train_params,
                lr=1.0,
                betas=(0.9, 0.999),
                eps=1.0e-7,
            )
        else:
            from .ranger21_native import Ranger21Native

            optimizer = Ranger21Native(
                train_params,
                lr=1.0,
                betas=(0.9, 0.999),
                eps=1.0e-7,
            )

        scheduler = torch.optim.lr_scheduler.StepLR(
            optimizer, step_size=1, gamma=self.gamma
        )

        return [optimizer], [scheduler]

    # Ranger21 does not require train/eval flip hooks
    def on_train_epoch_start(self, pl_module: L.LightningModule):
        pass

    def on_train_epoch_end(self, pl_module: L.LightningModule):
        pass

    def on_validation_epoch_start(self, pl_module: L.LightningModule):
        pass

    def on_test_epoch_start(self, pl_module: L.LightningModule):
        pass

    def on_save_checkpoint(self, pl_module: L.LightningModule, checkpoint):
        pass

    def on_train_batch_start(self, pl_module: L.LightningModule, batch, batch_idx):
        pass
