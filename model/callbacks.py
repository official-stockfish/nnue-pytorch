import lightning as L
import os
import copy
import torch
from torch.optim.swa_utils import AveragedModel, update_bn

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

    def on_train_start(self, trainer, pl_module):
        if not trainer.is_global_zero:
            return
        self.swa_model = AveragedModel(pl_module.model)

    def on_train_epoch_end(self, trainer, pl_module):
        if not trainer.is_global_zero or trainer.current_epoch < self.swa_start_epoch:
            return
        optimizers = pl_module.optimizers()
        if hasattr(optimizers, 'eval') and callable(optimizers.eval):
            print("[ExplicitSWACallback] Switching optimizer to eval mode before SWA update.")
            optimizers.eval()
        optimizers.eval()
        self.swa_model.update_parameters(pl_module.model)
        if hasattr(optimizers, 'train') and callable(optimizers.train):
            print("[ExplicitSWACallback] Switching optimizer to train mode after SWA update.")
            optimizers.train()

    def state_dict(self):
        # Prevent Lightning from saving the SWA callback's internal state
        # in the regular epoch checkpoints, avoiding redundant memory bloat.
        return {}

    def load_state_dict(self, state_dict):
        pass
