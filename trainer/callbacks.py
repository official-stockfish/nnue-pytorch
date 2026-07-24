"""Plain-Python training callbacks for SimpleTrainer."""

import copy
import math
import os
import re
import time
from datetime import timedelta

import torch
import torch.distributed as dist
from torch.optim.swa_utils import AveragedModel


class Callback:
    """Base callback interface. All hook methods receive the trainer."""

    def on_fit_start(self, trainer):
        pass

    def on_fit_end(self, trainer):
        pass

    def on_train_start(self, trainer):
        pass

    def on_train_end(self, trainer):
        pass

    def on_train_epoch_start(self, trainer):
        pass

    def on_train_epoch_end(self, trainer):
        pass

    def on_train_batch_start(self, trainer, batch, batch_idx):
        pass

    def on_train_batch_end(self, trainer, batch, batch_idx, outputs):
        pass

    def on_validation_start(self, trainer):
        pass

    def on_validation_end(self, trainer):
        pass

    def on_validation_epoch_start(self, trainer):
        pass

    def on_validation_epoch_end(self, trainer):
        pass

    def on_validation_batch_start(self, trainer, batch, batch_idx):
        pass

    def on_validation_batch_end(self, trainer, batch, batch_idx, outputs):
        pass

    def on_save_checkpoint(self, trainer, checkpoint):
        pass

    def on_load_checkpoint(self, trainer, checkpoint):
        pass


class WeightClipper(Callback):
    """Clip feature-transformer weights during training."""

    @torch.no_grad()
    @torch.compiler.disable
    def on_train_batch_end(self, trainer, batch=None, batch_idx=None, outputs=None):
        _, _, _ = batch, batch_idx, outputs  # Unused
        trainer.model.model.clip_weights(include_input=False)

    @torch.no_grad()
    @torch.compiler.disable
    def on_train_epoch_end(self, trainer):
        trainer.model.model.clip_weights(include_input=True)

    @torch.no_grad()
    @torch.compiler.disable
    def on_save_checkpoint(self, trainer, checkpoint=None):
        _ = checkpoint  # Unused
        trainer.model.model.clip_weights(include_input=True)


class ExplicitSWA(Callback):
    def __init__(self, swa_start_epoch: int, save_dir: str):
        super().__init__()
        self.swa_start_epoch = swa_start_epoch
        self.save_dir = save_dir

        self.swa_model = None
        self.to_eval = False

    def swap_weights(self, model, to_eval):
        if self.swa_model is not None and self.to_eval != bool(to_eval):
            # Swap the model's weights with the SWA weights for evaluation
            tmp = copy.deepcopy(model.model.state_dict())
            model.model.load_state_dict(self.swa_model.module.state_dict())
            self.swa_model.module.load_state_dict(tmp)
            self.to_eval = not self.to_eval

    @torch.no_grad()
    @torch.compiler.disable
    def on_train_start(self, trainer):
        if trainer.rank != 0:
            return
        self.swa_model = AveragedModel(trainer.model.model)

    @torch.no_grad()
    @torch.compiler.disable
    def on_train_epoch_end(self, trainer):
        if trainer.rank != 0 or trainer.current_epoch < self.swa_start_epoch:
            return
        trainer.model.eval()
        self.swa_model.update_parameters(trainer.model.model)
        trainer.model.train()

    @torch.no_grad()
    @torch.compiler.disable
    def state_dict(self):
        # Prevent the trainer from saving the SWA callback's internal state
        # in the regular epoch checkpoints, avoiding redundant memory bloat.
        # Note that this prevents resuming SWA state from checkpoints,
        # but that's an acceptable tradeoff for now.
        return {}

    @torch.no_grad()
    @torch.compiler.disable
    def load_state_dict(self, state_dict):
        _ = state_dict  # Unused

    @torch.no_grad()
    @torch.compiler.disable
    def on_save_checkpoint(self, trainer, checkpoint=None):
        if trainer.current_epoch >= self.swa_start_epoch:
            # Strip optimizer and lr_scheduler states from the checkpoint to
            # prevent excessive memory usage, since they are not needed for SWA
            # evaluation and resuming is unsupported. Thus they would be
            # redundant with the main checkpoint.
            checkpoint.pop("optimizer_states", None)
            checkpoint.pop("lr_schedulers", None)
            state_dict = checkpoint.get("state_dict")
            if isinstance(state_dict, dict):
                state_dict.pop("optimizer_state_dict", None)

            if trainer.rank == 0:
                print(
                    f"[ExplicitSWA] Stripping optimizer and lr_scheduler states "
                    f"from checkpoint at epoch {trainer.current_epoch} to save memory."
                )

    @torch.no_grad()
    @torch.compiler.disable
    def on_load_checkpoint(self, trainer, checkpoint=None):
        _ = trainer  # Unused
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
    def on_train_end(self, trainer):
        if trainer.current_epoch < self.swa_start_epoch:
            return

        # Optimizer is assumed to perform pointer swaps on train() and eval(),
        # to be idempotent and to not overwrite weights on_save_checkpoint.
        # Note that resume from checkpoint after swa averaging has started is
        # not supported.
        trainer.model.eval()
        self.swap_weights(trainer.model, to_eval=True)

        # NOTE: If BN is used, it has to be updated here. Be careful when using
        # DDP.

        # Writes to last_swa.ckpt to support pipelines expecting last.ckpt to
        # be the final checkpoint.
        swa_savepath = os.path.join(self.save_dir, "checkpoints", "last_swa.ckpt")
        trainer.save_checkpoint(swa_savepath)
        if trainer.rank == 0:
            print(f"[ExplicitSWA] SWA model saved to {swa_savepath}")
        self.swap_weights(trainer.model, to_eval=False)


class TimeLimit(Callback):
    def __init__(self, max_time: str):
        super().__init__()
        parts = list(map(int, max_time.strip().split(":")))
        if len(parts) != 4:
            raise ValueError("max_time must be in format 'DD:HH:MM:SS'")
        days, hours, minutes, seconds = parts
        self.max_duration = timedelta(
            days=days, hours=hours, minutes=minutes, seconds=seconds
        ).total_seconds()
        self.start_time = None

    def on_fit_start(self, trainer):
        self.start_time = time.time()

    def on_train_epoch_end(self, trainer):
        elapsed = time.time() - self.start_time
        if elapsed >= self.max_duration:
            trainer.should_stop = True
            print(
                f"[TimeLimit] Time limit reached ({elapsed:.1f}s), stopping after checkpoint."
            )


class TerminateOnNaN(Callback):
    """Stop training if a non-finite train or validation loss is detected."""

    def __init__(self):
        super().__init__()
        self.nan_detected = False

    @staticmethod
    def _is_finite(value):
        if value is None:
            return True
        if isinstance(value, torch.Tensor):
            return bool(torch.isfinite(value).all().item())
        return bool(math.isfinite(value))

    def _check_and_stop(self, trainer, loss, phase: str):
        if self._is_finite(loss):
            return
        local_stop = True
        if trainer.world_size > 1 and dist.is_available() and dist.is_initialized():
            stop_tensor = torch.tensor(
                1.0 if local_stop else 0.0,
                device=trainer.device,
            )
            dist.all_reduce(stop_tensor, op=dist.ReduceOp.MAX)
            local_stop = stop_tensor.item() > 0.5
        if local_stop:
            self.nan_detected = True
            rank_msg = f" [Rank {trainer.rank}]" if trainer.world_size > 1 else ""
            print(
                f"\n[TerminateOnNaN]{rank_msg} NaN/Inf detected in {phase} loss. Aborting training...",
                flush=True,
            )
            trainer.should_stop = True

    def on_train_batch_end(self, trainer, batch=None, batch_idx=None, outputs=None):
        _ = batch, batch_idx
        loss = outputs.get("loss") if isinstance(outputs, dict) else outputs
        self._check_and_stop(trainer, loss, "train")

    def on_validation_epoch_end(self, trainer):
        val_loss = trainer.callback_metrics.get("val_loss_epoch")
        self._check_and_stop(trainer, val_loss, "validation")


class CheckpointManager(Callback):
    def __init__(
        self,
        save_last: bool = True,
        every_n_epochs: int = 20,
        save_top_k: int = -1,
        dirpath: str = None,
    ):
        super().__init__()
        self.save_last = save_last
        self.every_n_epochs = every_n_epochs
        self.save_top_k = save_top_k
        self.dirpath = dirpath
        self._saved_checkpoints = []

    def _get_dirpath(self, trainer):
        if self.dirpath is not None:
            return self.dirpath
        return os.path.join(
            trainer.default_root_dir,
            "lightning_logs",
            f"version_{trainer.logger.version}",
            "checkpoints",
        )

    def on_train_epoch_end(self, trainer):
        if trainer.rank != 0:
            return

        dirpath = self._get_dirpath(trainer)
        os.makedirs(dirpath, exist_ok=True)

        should_save_periodic = (
            self.every_n_epochs is not None
            and self.every_n_epochs > 0
            and (trainer.current_epoch + 1) % self.every_n_epochs == 0
        )

        saved = []
        if should_save_periodic:
            periodic_path = os.path.join(
                dirpath,
                f"epoch={trainer.current_epoch}-step={trainer.global_step}.ckpt",
            )
            trainer.save_checkpoint(periodic_path)
            saved.append(periodic_path)

        if self.save_last:
            last_path = os.path.join(dirpath, "last.ckpt")
            trainer.save_checkpoint(last_path)
            saved.append(last_path)

        self._saved_checkpoints.extend(saved)
        self._enforce_save_top_k()

    def _enforce_save_top_k(self):
        if self.save_top_k < 0:
            return

        periodic = [p for p in self._saved_checkpoints if os.path.basename(p) != "last.ckpt"]
        if len(periodic) <= self.save_top_k:
            return

        # Sort by epoch number embedded in the filename.
        def _epoch_key(path):
            match = re.search(r"epoch=(\d+)", os.path.basename(path))
            return int(match.group(1)) if match else -1

        periodic.sort(key=_epoch_key)
        to_remove = periodic[: len(periodic) - self.save_top_k]
        for path in to_remove:
            if os.path.exists(path):
                os.remove(path)
            self._saved_checkpoints.remove(path)

    def on_train_end(self, trainer):
        if trainer.rank != 0:
            return

        dirpath = self._get_dirpath(trainer)
        os.makedirs(dirpath, exist_ok=True)

        last_path = os.path.join(dirpath, "last.ckpt")
        if not os.path.exists(last_path):
            trainer.save_checkpoint(last_path)


class SimpleLineLogger(Callback):
    def __init__(
        self,
        refresh_rate=None,
        train_metric_step="train_loss",
        train_metric_epoch="train_loss_epoch",
        val_metric="val_loss_epoch",
    ):
        super().__init__()
        self.train_metric_step = train_metric_step
        self.train_metric_epoch = train_metric_epoch
        self.val_metric = val_metric

        self.refresh_rate = refresh_rate

        # Train tracking
        self.train_start_time = None
        self.train_last_time = None
        self.train_last_step = 0

        # Val tracking
        self.val_start_time = None
        self.val_last_time = None
        self.val_last_step = 0

    def _format_time(self, seconds):
        m, s = divmod(int(seconds), 60)
        h, m = divmod(m, 60)
        return f"{h:02d}:{m:02d}:{s:02d}" if h else f"{m:02d}:{s:02d}"

    def _get_refresh_rate(self, trainer):
        if self.refresh_rate is not None:
            return self.refresh_rate
        return trainer.log_every_n_steps

    # ==========================================
    # TRAINING LOOP
    # ==========================================
    @torch.compiler.disable
    def on_train_epoch_start(self, trainer):
        if trainer.rank == 0:
            self.train_start_time = time.time()
            print("-" * 60)

    @torch.compiler.disable
    def on_train_batch_end(self, trainer, batch=None, batch_idx=None, outputs=None):
        _ = batch, outputs  # Unused
        if trainer.rank != 0:
            return

        current_step = batch_idx + 1
        total_batches = trainer.num_training_batches

        if (
            current_step % self._get_refresh_rate(trainer) == 0
            or current_step == total_batches
        ):
            now = time.time()
            elapsed_total = now - self.train_start_time
            rate = current_step / elapsed_total if elapsed_total > 0 else 0

            remaining = (total_batches - current_step) / rate if rate > 0 else 0
            loss_val = trainer.callback_metrics.get(
                self.train_metric_step, float("nan")
            )

            print(
                f"Epoch {trainer.current_epoch:>2} (Train): "
                f"{current_step / total_batches:>4.0%}| "
                f"{current_step:>5}/{total_batches:<5} "
                f"[{self._format_time(elapsed_total)}<{self._format_time(remaining)}, "
                f"{rate:>6.2f}it/s, "
                f"{self.train_metric_step}={loss_val:.5f}, "
                f"v_num={trainer.logger.version}]",
                flush=True,
            )

            self.train_last_time = now
            self.train_last_step = current_step

    @torch.compiler.disable
    def on_train_epoch_end(self, trainer):
        if trainer.rank != 0 or trainer.sanity_checking:
            return

        train_loss = trainer.callback_metrics.get(self.train_metric_epoch, float("nan"))
        print(
            f"Epoch {trainer.current_epoch:>2} (Train): "
            f"[{self.train_metric_epoch}={train_loss:.5f}]",
            flush=True,
        )

    # ==========================================
    # VALIDATION LOOP
    # ==========================================
    @torch.compiler.disable
    def on_validation_epoch_start(self, trainer):
        if trainer.rank == 0 and not trainer.sanity_checking:
            self.val_start_time = time.time()

    @torch.compiler.disable
    def on_validation_batch_end(self, trainer, batch=None, batch_idx=None, outputs=None):
        _ = batch, outputs  # Unused
        if trainer.rank != 0 or trainer.sanity_checking:
            return

        current_step = batch_idx + 1
        val_batches = trainer.num_val_batches
        if isinstance(val_batches, int):
            total_batches = val_batches
        else:
            total_batches = sum(val_batches)

        if (
            current_step % self._get_refresh_rate(trainer) == 0
            or current_step == total_batches
        ):
            now = time.time()
            elapsed_total = now - self.val_start_time

            rate = current_step / elapsed_total if elapsed_total > 0 else 0
            remaining = (total_batches - current_step) / rate if rate > 0 else 0

            print(
                f"Epoch {trainer.current_epoch:>2} (Val)  : "
                f"{current_step / total_batches:>4.0%}| "
                f"{current_step:>5}/{total_batches:<5} "
                f"[{self._format_time(elapsed_total)}<{self._format_time(remaining)}, "
                f"{rate:>6.2f}it/s]",
                flush=True,
            )

    @torch.compiler.disable
    def on_validation_epoch_end(self, trainer):
        if trainer.rank != 0 or trainer.sanity_checking:
            return

        val_loss = trainer.callback_metrics.get(self.val_metric, float("nan"))
        print(
            f"Epoch {trainer.current_epoch:>2} (Val): "
            f"[{self.val_metric}={val_loss:.5f}]",
            flush=True,
        )
