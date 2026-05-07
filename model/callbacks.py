import lightning as L
import torch
from torch.optim.swa_utils import AveragedModel

from .lightning_module import NNUE


class WeightClippingCallback(L.Callback):
    def on_train_batch_start(
        self,
        trainer: L.Trainer,
        pl_module: L.LightningModule,
        batch,
        batch_idx: int,
    ) -> None:
        assert isinstance(pl_module, NNUE)
        pl_module.model.clip_weights()
        if batch_idx == 0:
            pl_module.model.clip_input_weights()

class ExplicitSWACallback(L.Callback):
    def __init__(self, swa_start_epoch: int = 0):
        super().__init__()
        self.swa_start_epoch = swa_start_epoch
        self.swa_model = None

    @torch.compiler.disable
    def on_train_start(self, trainer, pl_module):
        if not trainer.is_global_zero:
            return
        self.swa_model = AveragedModel(pl_module.model)

    @torch.compiler.disable
    def on_train_epoch_end(self, trainer, pl_module):
        if not trainer.is_global_zero or trainer.current_epoch < self.swa_start_epoch:
            return
        pl_module.eval()
        self.swa_model.update_parameters(pl_module.model)
        pl_module.train()

    def state_dict(self):
        # Prevent Lightning from saving the SWA callback's internal state
        # in the regular epoch checkpoints, avoiding redundant memory bloat.
        # Note that this prevents resuming SWA state from checkpoints,
        # but that's an acceptable tradeoff for now.
        return {}

    def load_state_dict(self, state_dict):
        pass

    def on_load_checkpoint(self, trainer, pl_module, checkpoint):
        checkpoint_epoch = checkpoint.get("epoch")
        if checkpoint_epoch is None:
            return

        if checkpoint_epoch >= self.swa_start_epoch:
            raise RuntimeError(
                f"Cannot resume training after SWA has started. "
                f"Checkpoint epoch {checkpoint_epoch} >= SWA start epoch {self.swa_start_epoch}"
            )
