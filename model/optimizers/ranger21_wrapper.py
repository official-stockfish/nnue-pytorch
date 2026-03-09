import lightning as L
import torch

from dataclasses import dataclass


try:
    import ranger21
    _ranger21_import_error = False
except ImportError:
    _ranger21_import_error = True

@dataclass
class Ranger21Config:
    gamma: float = 0.992
    """Multiplicative factor applied to the learning rate after every epoch."""

    num_batches_per_epoch: int = 100_000_000 // 16384
    """Number of batches per epoch, used by Ranger21 for scheduling."""

class Ranger21Wrapper:
    def __init__(
        self,
        config,
    ):
        if _ranger21_import_error:
            raise ImportError(
                "The required ranger21 library is not installed. "
            )
        self.max_epoch = config.max_epoch
        self.gamma = config.gamma
        self.num_batches_per_epoch = config.num_batches_per_epoch

    def configure_optimizers(self, train_params):
        optimizer = ranger21.Ranger21(
            train_params,
            lr=1.0,
            betas=(0.9, 0.999),
            eps=1.0e-7,
            using_gc=False,
            using_normgc=False,
            weight_decay=0.0,
            num_batches_per_epoch=self.num_batches_per_epoch,
            num_epochs=self.max_epoch,
            warmdown_active=False,
            use_warmup=False,
            use_adaptive_gradient_clipping=False,
            softplus=False,
            pnm_momentum_factor=0.0,
        )

        scheduler = torch.optim.lr_scheduler.StepLR(
            optimizer, step_size=1, gamma=self.gamma
        )

        return [optimizer], [scheduler]

    # Ranger21 does not require train/eval flip hooks
    def on_train_epoch_start(self, pl_module: L.LightningModule): pass
    def on_train_epoch_end(self, pl_module: L.LightningModule): pass
    def on_validation_epoch_start(self, pl_module: L.LightningModule): pass
    def on_test_epoch_start(self, pl_module: L.LightningModule): pass
    def on_save_checkpoint(self, pl_module: L.LightningModule, checkpoint): pass
    def on_train_batch_start(self, pl_module: L.LightningModule, batch, batch_idx): pass
