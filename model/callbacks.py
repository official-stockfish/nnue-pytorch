import lightning as L

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
            pl_module.model.clip_threat_weights()
