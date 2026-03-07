import warnings
import lightning as L

from dataclasses import dataclass

try:
    import schedulefree
    _schedulefree_import_error = False
except ImportError:
    _schedulefree_import_error = True

@dataclass
class ScheduleFreeConfig:
    warmup_steps: int = 10000
    """Number of steps to warmup schedulefree optimizer."""

class ScheduleFreeWrapper:
    def __init__(
        self,
        lr: float,
        warmup_steps: int,
        **kwargs,
    ):
        if _schedulefree_import_error:
            raise ImportError(
                "The required schedulefree library is not installed. "
            )

        self.lr = lr
        self.warmup_steps = warmup_steps
        self.needs_train_flip = False

        warning_parts = ["The following keyword arguments are unused and will be ignored:"]
        if kwargs:
            warning_parts.append(
                f"\n  - Unused Keyword Arguments: {', '.join(kwargs.keys())}"
            )
            warnings.warn("".join(warning_parts), UserWarning)

        print(
            f"Using schedule-free Adam with warmup_steps={warmup_steps}, lr={lr}."
        )

    def configure_optimizers(self, train_params):
        optimizer = schedulefree.AdamWScheduleFree(
            train_params,
            lr=self.lr,
            betas=(0.9, 0.999),
            eps=1.0e-7,
            warmup_steps=self.warmup_steps,
        )
        return optimizer

    def on_train_epoch_start(self, pl_module: L.LightningModule):
        pl_module.optimizers().optimizer.train()
        self.needs_train_flip = False

    def on_train_epoch_end(self, pl_module: L.LightningModule):
        pl_module.optimizers().optimizer.eval()
        self.needs_train_flip = True

    def on_validation_epoch_start(self, pl_module: L.LightningModule):
        pl_module.optimizers().optimizer.eval()
        self.needs_train_flip = True

    def on_test_epoch_start(self, pl_module: L.LightningModule):
        pl_module.optimizers().optimizer.eval()
        self.needs_train_flip = True

    def on_save_checkpoint(self, pl_module: L.LightningModule, checkpoint):
        pl_module.optimizers().optimizer.eval()
        self.needs_train_flip = True

    def on_train_batch_start(self, pl_module: L.LightningModule, batch, batch_idx):
        if self.needs_train_flip:
            pl_module.optimizers().optimizer.train()
            self.needs_train_flip = False
