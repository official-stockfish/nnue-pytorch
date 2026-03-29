import gc

import lightning as L

from .lightning_module import NNUE


class ManagedGCCallback(L.Callback):
    """Disable automatic GC during training to prevent multi-second stalls
    from bulk MPS tensor deallocation.  Collect between epochs instead."""

    def on_train_start(self, trainer, pl_module):
        gc.collect()
        gc.disable()
        gc.set_threshold(0, 0, 0)

    def on_train_epoch_end(self, trainer, pl_module):
        gc.collect()

    def on_train_end(self, trainer, pl_module):
        gc.set_threshold(700, 10, 10)
        gc.enable()


class WeightClippingCallback(L.Callback):
    def __init__(self, clip_every_n: int = 10):
        super().__init__()
        self.clip_every_n = clip_every_n

    def on_train_batch_start(
        self,
        trainer: L.Trainer,
        pl_module: L.LightningModule,
        batch,
        batch_idx: int,
    ) -> None:
        assert isinstance(pl_module, NNUE)
        if batch_idx % self.clip_every_n == 0:
            pl_module.model.clip_weights()
        if batch_idx == 0:
            pl_module.model.clip_input_weights()
