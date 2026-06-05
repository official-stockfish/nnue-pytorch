import lightning as L
import torch
import os
import copy
import shutil
from collections import deque

from torch.optim.swa_utils import AveragedModel

from .lightning_module import NNUE


class WeightClippingCallback(L.Callback):
    @torch.no_grad()
    @torch.compiler.disable
    def on_train_batch_end(
        self,
        trainer: L.Trainer,
        pl_module: L.LightningModule,
        outputs,
        batch,
        batch_idx: int,
    ) -> None:
        _, _ = trainer, batch  # Unused
        assert isinstance(pl_module, NNUE)
        pl_module.model.clip_weights(include_input=False)

    @torch.no_grad()
    @torch.compiler.disable
    def on_train_epoch_end(self, trainer, pl_module):
        _ = trainer # Unused
        assert isinstance(pl_module, NNUE)
        pl_module.model.clip_weights(include_input=True)

    @torch.no_grad()
    @torch.compiler.disable
    def on_save_checkpoint(self, trainer, pl_module, checkpoint):
        _, _ = trainer, checkpoint  # Unused
        assert isinstance(pl_module, NNUE)
        pl_module.model.clip_weights(include_input=True)


class ExplicitSWACallback(L.Callback):
    def __init__(self, swa_start_epoch: int, save_dir: str):
        super().__init__()
        self.swa_start_epoch = swa_start_epoch
        self.save_dir = save_dir

        self.swa_model = None
        self.to_eval = False

    def swap_weights(self, pl_module, to_eval):
        if self.swa_model is not None and self.to_eval != bool(to_eval):
            # Swap the model's weights with the SWA weights for evaluation
            tmp = copy.deepcopy(pl_module.model.state_dict())
            pl_module.model.load_state_dict(self.swa_model.module.state_dict())
            self.swa_model.module.load_state_dict(tmp)
            self.to_eval = not self.to_eval

    @torch.no_grad()
    @torch.compiler.disable
    def on_train_start(self, trainer, pl_module):
        if not trainer.is_global_zero:
            return
        self.swa_model = AveragedModel(pl_module.model)

    @torch.no_grad()
    @torch.compiler.disable
    def on_train_epoch_end(self, trainer, pl_module):
        if not trainer.is_global_zero or trainer.current_epoch < self.swa_start_epoch:
            return
        pl_module.eval()
        self.swa_model.update_parameters(pl_module.model)
        pl_module.train()

    @torch.no_grad()
    @torch.compiler.disable
    def state_dict(self):
        # Prevent Lightning from saving the SWA callback's internal state
        # in the regular epoch checkpoints, avoiding redundant memory bloat.
        # Note that this prevents resuming SWA state from checkpoints,
        # but that's an acceptable tradeoff for now.
        return {}

    @torch.no_grad()
    @torch.compiler.disable
    def load_state_dict(self, state_dict):
        _ = state_dict # Unused

    @torch.no_grad()
    @torch.compiler.disable
    def on_save_checkpoint(self, trainer, pl_module, checkpoint):
        _ = pl_module  # Unused
        if trainer.current_epoch >= self.swa_start_epoch:
            # Strip optimizer and lr_scheduler states from the checkpoint to prevent excessive memory usage,
            # since they are not needed for SWA evaluation and resuming is unsupported.
            # Thus they would be redundant with the main checkpoint.
            checkpoint.pop("optimizer_states", None)
            checkpoint.pop("lr_schedulers", None)

            if trainer.is_global_zero:
                print(f"[ExplicitSWACallback] Stripping optimizer and lr_scheduler states from checkpoint at epoch {trainer.current_epoch} to save memory.")

    @torch.no_grad()
    @torch.compiler.disable
    def on_load_checkpoint(self, trainer, pl_module, checkpoint):
        _, _ = trainer, pl_module  # Unused
        checkpoint_epoch = checkpoint.get("epoch")
        if checkpoint_epoch is None:
            return

        if checkpoint_epoch > self.swa_start_epoch:
            raise RuntimeError(
                f"Cannot resume training after SWA has started. "
                f"Checkpoint epoch {checkpoint_epoch} > SWA start epoch {self.swa_start_epoch}"
            )

    @torch.no_grad()
    @torch.compiler.disable
    def on_train_end(self, trainer, pl_module):
        if trainer.current_epoch < self.swa_start_epoch:
            return
        # Optimizer is assumed to perform pointer swaps on train() and eval(),
        # to be idempotent and to not overwrite weights on_save_checkpoint.
        # Note that resume from checkpoint after swa averaging has started is not supported.
        pl_module.eval()
        self.swap_weights(pl_module, to_eval=True)

        # NOTE: If BN is used, it has to be updated here. Be careful when using DDP.

        # Writes to last.ckpt to support pipelines build expecting last.ckpt to be the final checkpoint.
        # We rename last.ckpt to last_non_swa.ckpt to preserve the original for analysis purposes.

        swa_savepath = os.path.join(self.save_dir, "checkpoints", "last_swa.ckpt")
        trainer.save_checkpoint(swa_savepath)
        if trainer.is_global_zero:
            print(f"[ExplicitSWACallback] SWA model saved to {swa_savepath}")
        self.swap_weights(pl_module, to_eval=False)


class SimplePeriodicCheckpoint(L.Callback):
    def __init__(
        self,
        dirpath = None,
        every_n_epochs: int = 1,
        save_top_k: int = 1,
        swa_start_epoch: int = -1,
        save_last: bool = True
    ):
        super().__init__()
        self.dirpath = dirpath
        self.every_n_epochs = every_n_epochs
        self.keep_last_k = save_top_k
        self.swa_start_epoch = swa_start_epoch
        self.save_last = save_last
        self._saved_checkpoints = deque()

        if not self.save_last:
            assert self.every_n_epochs >= 1, "every_n_epochs must be at least 1 when save_last is False."

    def __resolve_ckpt_dir(self, trainer: L.Trainer) -> str:
        """
        COPIED FROM PyTorch Lightning's ModelCheckpoint callback.
        """
        if self.dirpath is not None:
            return self.dirpath

        if len(trainer.loggers) > 0:
            if trainer.loggers[0].save_dir is not None:
                save_dir = trainer.loggers[0].save_dir
            else:
                save_dir = trainer.default_root_dir
            name = trainer.loggers[0].name
            version = trainer.loggers[0].version
            version = version if isinstance(version, str) else f"version_{version}"
            ckpt_path = os.path.join(save_dir, str(name), version, "checkpoints")
        else:
            # if no loggers, use default_root_dir
            ckpt_path = os.path.join(trainer.default_root_dir, "checkpoints")

        return ckpt_path

    def on_train_start(self, trainer, pl_module):
        self.dirpath = self.__resolve_ckpt_dir(trainer)
        if trainer.is_global_zero:
            os.makedirs(self.dirpath, exist_ok=True)

    def _is_in_swa_phase(self, trainer) -> bool:
        return self.swa_start_epoch >= 0 and trainer.current_epoch >= self.swa_start_epoch

    def on_train_epoch_end(self, trainer, pl_module):
        if self._is_in_swa_phase(trainer):
            return

        # PyTorch Lightning epochs are 0-indexed.
        epoch = trainer.current_epoch
        step = trainer.global_step

        if (epoch + 1) % self.every_n_epochs == 0:
            self._save_and_rotate(trainer, epoch, step)

    def _save_and_rotate(self, trainer, epoch: int, step: int):
        assert self.dirpath is not None
        os.makedirs(self.dirpath, exist_ok=True)
        if self.keep_last_k > 0:
            filepath = os.path.join(self.dirpath, f"epoch={epoch}-step={step}.ckpt")
        else:
            filepath = os.path.join(self.dirpath, "last.ckpt")

        trainer.save_checkpoint(filepath)

        if self.keep_last_k > 0:
            self._saved_checkpoints.append(filepath)

            # File system operations must be restricted to global zero
            if trainer.is_global_zero and self.save_last:
                last_path = os.path.join(self.dirpath, "last.ckpt")
                shutil.copy2(filepath, last_path)

            while len(self._saved_checkpoints) > self.keep_last_k:
                oldest_ckpt = self._saved_checkpoints.popleft()
                if trainer.is_global_zero and os.path.exists(oldest_ckpt):
                    try:
                        os.remove(oldest_ckpt)
                    except OSError:
                        pass

    def on_train_end(self, trainer, pl_module):
        if not self.dirpath:
            return

        os.makedirs(self.dirpath, exist_ok=True)

        epoch = trainer.current_epoch
        step = trainer.global_step
        self._save_and_rotate(trainer, epoch, step)
