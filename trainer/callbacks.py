"""Plain-Python training callbacks for SimpleTrainer."""

import copy
import math
import os
import re
import time
from datetime import timedelta

import torch
import torch.distributed as dist


def _tensor_to_bytes(t: torch.Tensor) -> bytes:
    """Fast uint8 tensor -> bytes conversion.

    bytes(tensor) iterates one Python scalar tensor per element (~1 s
    for the 2^20-register HLL state); numpy().tobytes() is a zero-copy
    view (sub-millisecond).
    """
    return t.detach().cpu().numpy().tobytes()


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

    def __init__(self, input_clipping_interval: int = 0):
        super().__init__()
        self.input_clipping_interval = input_clipping_interval

    @torch.no_grad()
    @torch.compiler.disable
    def on_train_batch_end(self, trainer, batch=None, batch_idx=None, outputs=None):
        _, _, _ = batch, batch_idx, outputs  # Unused
        if (
            self.input_clipping_interval > 0
            and trainer.global_step % self.input_clipping_interval == 0
        ):
            trainer.model.model.clip_weights(include_input=True)
        else:
            trainer.model.model.clip_weights(include_input=False)

    @torch.no_grad()
    @torch.compiler.disable
    def on_train_epoch_end(self, trainer):
        if self.input_clipping_interval <= 0:
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
        trainer.model.model.clip_weights(include_input=True)

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
    """Stop training if a non-finite train or validation loss is detected.

    The NaN check is deferred to the same cadence as logging
    (``log_every_n_steps``) to avoid a per-step GPU→CPU sync.
    Between checks, a GPU-side flag tensor accumulates
    ``torch.isfinite(loss).all()`` via ``logical_and``.
    """

    def __init__(self):
        super().__init__()
        self.nan_detected = False
        self._finite_flag = None  # GPU tensor, initialized lazily

    def _ensure_flag(self, device):
        if self._finite_flag is None:
            self._finite_flag = torch.tensor(True, device=device, dtype=torch.bool)

    @torch.compiler.disable
    def on_train_batch_end(self, trainer, batch=None, batch_idx=None, outputs=None):
        _ = batch, batch_idx
        loss = outputs.get("loss") if isinstance(outputs, dict) else outputs
        if loss is None:
            return

        # Accumulate finiteness on GPU without syncing.
        self._ensure_flag(trainer.device)
        if isinstance(loss, torch.Tensor):
            self._finite_flag &= torch.isfinite(loss).all()
        else:
            self._finite_flag &= torch.tensor(
                math.isfinite(loss), device=trainer.device, dtype=torch.bool
            )

        # Only sync to CPU when logging is due or at the last batch.
        current_step = (batch_idx or 0) + 1
        if (
            current_step % trainer.log_every_n_steps == 0
            or current_step == trainer.num_training_batches
        ):
            # Every rank must enter the collective, even if only one rank
            # saw a non-finite loss. Otherwise the failing rank deadlocks.
            if trainer.world_size > 1 and dist.is_available() and dist.is_initialized():
                dist.all_reduce(self._finite_flag, op=dist.ReduceOp.MIN)
            if not self._finite_flag.item():
                self._check_and_stop(trainer, loss, "train")
            # Reset for the next window.
            self._finite_flag.fill_(True)

    def _check_and_stop(self, trainer, loss, phase: str):
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

    def on_validation_epoch_end(self, trainer):
        val_loss = trainer.callback_metrics.get("val_loss_epoch")
        if val_loss is not None:
            if isinstance(val_loss, torch.Tensor):
                if not torch.isfinite(val_loss).all().item():
                    self._check_and_stop(trainer, val_loss, "validation")
            elif not math.isfinite(val_loss):
                self._check_and_stop(trainer, val_loss, "validation")


class CheckpointManager(Callback):
    def __init__(
        self,
        save_last: bool = True,
        every_n_epochs: int = 20,
        save_top_k: int = -1,
        dirpath: str | None = None,
    ):
        super().__init__()
        self.save_last = save_last
        self.every_n_epochs = every_n_epochs
        self.save_top_k = save_top_k
        self.dirpath = dirpath
        self._saved_checkpoints = []
        self._last_save_epoch: int | None = None

    def _get_dirpath(self, trainer):
        if self.dirpath is not None:
            return self.dirpath
        return os.path.join(
            trainer.default_root_dir,
            "training_logs",
            f"version_{trainer.logger.version}",
            "checkpoints",
        )

    def on_train_epoch_end(self, trainer):
        dirpath = self._get_dirpath(trainer)
        if trainer.rank == 0:
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
            self._last_save_epoch = trainer.current_epoch

        if trainer.rank == 0:
            self._saved_checkpoints.extend(saved)
            self._enforce_save_top_k()

    def _enforce_save_top_k(self):
        if self.save_top_k < 0:
            return

        periodic = [
            p for p in self._saved_checkpoints if os.path.basename(p) != "last.ckpt"
        ]
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
        dirpath = self._get_dirpath(trainer)
        if trainer.rank == 0:
            os.makedirs(dirpath, exist_ok=True)

        last_path = os.path.join(dirpath, "last.ckpt")
        if self.save_last and self._last_save_epoch != trainer.current_epoch:
            trainer.save_checkpoint(last_path)
        # Wait for asynchronous writes so post-training code (e.g. the
        # SWA checkpoint renames in train.py) sees complete files.
        if trainer.rank == 0:
            trainer.flush_checkpoints()


class UniquePositionLogger(Callback):
    """Tracks approximate unique positions, total positions (after
    filtering), and preskip positions (before filtering) seen by the
    data loader, using HyperLogLog.

    - on_train_epoch_end: one synchronization per epoch —
      all_reduce(MAX) of 1 MiB HLL registers for accurate global
      unique, all_gather of per-rank totals and preskips.  Results
      are logged to CSV/TensorBoard and printed on the epoch summary
      line.  The merged state is stashed for on_save_checkpoint.
    - Checkpoint: stores the global merged HLL state, per-rank
      totals, and per-rank preskips for race-free restore on restart.
    """

    HLL_REGISTER_COUNT = 1 << 20  # 2^20 = 1,048,576
    HLL_HEADER_SIZE = 16
    HLL_STATE_SIZE = HLL_HEADER_SIZE + HLL_REGISTER_COUNT

    def __init__(self, train_dataset, rank=0, world_size=1):
        super().__init__()
        self.dataset = train_dataset
        self.rank = rank
        self.world_size = world_size
        self._ddp = dist.is_available() and dist.is_initialized() and world_size > 1
        # Stash for on_save_checkpoint.
        self._merged_hll_state = None
        self._merged_per_rank_totals = None
        self._merged_per_rank_preskips = None

    def _get_local_stats(self):
        """Return (preskip, total, unique) from the data loader."""
        try:
            return self.dataset.get_unique_stats()
        except Exception:  # noqa: BLE001
            return 0, 0, 0

    @torch.compiler.disable
    def on_train_batch_end(self, trainer, batch=None, batch_idx=None, outputs=None):
        pass

    @torch.compiler.disable
    def on_train_epoch_end(self, trainer):
        # One synchronization point per epoch.  All ranks participate
        # (no early return) to avoid deadlock.  This runs before
        # CheckpointManager.on_train_epoch_end (which calls
        # trainer.save_checkpoint → on_save_checkpoint), so the stash
        # is up to date when the checkpoint is written.
        from data_loader.stream import hll_count_from_state

        local_preskip, local_total, _local_unique = self._get_local_stats()
        hll_bytes = self.dataset.get_hll_state()

        if not hll_bytes or len(hll_bytes) < self.HLL_STATE_SIZE:
            if self._ddp:
                print(
                    f"  [Warning] Rank {self.rank}: empty HLL state at epoch end, "
                    f"skipping global merge.",
                    flush=True,
                )
            self._merged_hll_state = None
            self._merged_per_rank_totals = None
            self._merged_per_rank_preskips = None
            return

        # Extract registers (skip 16-byte header)
        registers = bytearray(hll_bytes[self.HLL_HEADER_SIZE : self.HLL_STATE_SIZE])

        if self._ddp:
            reg_tensor = torch.frombuffer(
                bytearray(registers), dtype=torch.uint8
            ).clone()
            reg_tensor = reg_tensor.cuda() if torch.cuda.is_available() else reg_tensor
            dist.all_reduce(reg_tensor, op=dist.ReduceOp.MAX)
            merged_regs = _tensor_to_bytes(reg_tensor)

            global_unique = hll_count_from_state(
                bytes(hll_bytes[: self.HLL_HEADER_SIZE]) + merged_regs
            )

            val_tensor = torch.tensor(
                [local_total],
                dtype=torch.int64,
                device="cuda" if torch.cuda.is_available() else "cpu",
            )
            gathered = [torch.zeros_like(val_tensor) for _ in range(self.world_size)]
            dist.all_gather(gathered, val_tensor)
            per_rank_totals = [int(t.item()) for t in gathered]
            global_total = sum(per_rank_totals)

            preskip_tensor = torch.tensor(
                [local_preskip],
                dtype=torch.int64,
                device="cuda" if torch.cuda.is_available() else "cpu",
            )
            gathered_preskip = [
                torch.zeros_like(preskip_tensor) for _ in range(self.world_size)
            ]
            dist.all_gather(gathered_preskip, preskip_tensor)
            per_rank_preskips = [int(t.item()) for t in gathered_preskip]
            global_preskip = sum(per_rank_preskips)
        else:
            global_unique = hll_count_from_state(hll_bytes)
            global_total = (
                local_total * self.world_size if self.world_size > 1 else local_total
            )
            per_rank_totals = (
                [local_total]
                if self.world_size == 1
                else [local_total] * self.world_size
            )
            global_preskip = (
                local_preskip * self.world_size
                if self.world_size > 1
                else local_preskip
            )
            per_rank_preskips = (
                [local_preskip]
                if self.world_size == 1
                else [local_preskip] * self.world_size
            )
            merged_regs = bytes(registers)

        # Stash for on_save_checkpoint (non-collective).
        self._merged_hll_state = bytes(hll_bytes[: self.HLL_HEADER_SIZE]) + merged_regs
        self._merged_per_rank_totals = per_rank_totals
        self._merged_per_rank_preskips = per_rank_preskips

        # Set callback_metrics so SimpleLineLogger can print them.
        trainer.callback_metrics["unique"] = float(global_unique)
        trainer.callback_metrics["total"] = float(global_total)
        trainer.callback_metrics["preskip"] = float(global_preskip)

        trainer.callback_metrics["unique_global"] = float(global_unique)
        trainer.callback_metrics["total_global"] = float(global_total)
        trainer.callback_metrics["preskip_global"] = float(global_preskip)
        trainer._log_metrics(
            {
                "unique_global": global_unique,
                "total_global": global_total,
                "preskip_global": global_preskip,
            },
            step=trainer.global_step,
        )

    @torch.compiler.disable
    def on_save_checkpoint(self, trainer, checkpoint):
        # No collectives — reads stash populated by on_train_batch_end.
        if self._merged_hll_state is None or self._merged_per_rank_totals is None:
            return

        if self.rank == 0:
            checkpoint["dataloader_hll"] = self._merged_hll_state
            checkpoint["dataloader_total_per_rank"] = self._merged_per_rank_totals
            checkpoint["dataloader_preskip_per_rank"] = self._merged_per_rank_preskips

    @torch.compiler.disable
    def on_load_checkpoint(self, trainer, checkpoint):
        hll_bytes = checkpoint.get("dataloader_hll")
        per_rank_totals = checkpoint.get("dataloader_total_per_rank")
        per_rank_preskips = checkpoint.get("dataloader_preskip_per_rank")
        if not hll_bytes or not per_rank_totals:
            return

        # The HLL registers are always the global merged state (same on
        # all ranks).  The per-rank totals and preskips must each sum
        # to their respective global totals.  When the number of ranks
        # at restart differs from the checkpoint, redistribute evenly.
        global_total = sum(per_rank_totals)
        if self.world_size == len(per_rank_totals):
            initial_total = per_rank_totals[self.rank]
        else:
            base = global_total // self.world_size
            remainder = global_total % self.world_size
            initial_total = base + (1 if self.rank < remainder else 0)

        if per_rank_preskips:
            global_preskip = sum(per_rank_preskips)
            if self.world_size == len(per_rank_preskips):
                initial_preskip = per_rank_preskips[self.rank]
            else:
                base = global_preskip // self.world_size
                remainder = global_preskip % self.world_size
                initial_preskip = base + (1 if self.rank < remainder else 0)
        else:
            # Backward compatibility: old checkpoints without preskip.
            initial_preskip = 0

        try:
            self.dataset.set_initial_hll(hll_bytes, initial_total, initial_preskip)
        except Exception as e:  # noqa: BLE001
            print(f"  [Warning] Failed to restore HLL state: {e}", flush=True)


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
                f"lr={trainer.callback_metrics['lr']:.7f}, "
                f"lambda={trainer.callback_metrics['lambda']:.5f}, "
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
        unique_val = trainer.callback_metrics.get("unique", None)
        total_val = trainer.callback_metrics.get("total", None)
        preskip_val = trainer.callback_metrics.get("preskip", None)

        unique_str = f", unique={unique_val:.0f}" if unique_val is not None else ""
        total_str = f", total={total_val:.0f}" if total_val is not None else ""
        preskip_str = f", preskip={preskip_val:.0f}" if preskip_val is not None else ""

        print(
            f"Epoch {trainer.current_epoch:>2} (Train): "
            f"[{self.train_metric_epoch}={train_loss:.5f}"
            f"{unique_str}{total_str}{preskip_str}]",
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
    def on_validation_batch_end(
        self, trainer, batch=None, batch_idx=None, outputs=None
    ):
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
            f"[{self.val_metric}={val_loss:.7f}]",
            flush=True,
        )


def _available_cpus_per_rank() -> int:
    try:
        cpus = len(os.sched_getaffinity(0))
    except (AttributeError, OSError):
        cpus = os.cpu_count() or 1
    local_world_size = int(os.environ.get("LOCAL_WORLD_SIZE", "1"))
    return max(1, cpus // max(1, local_world_size))


class DataLoaderBottleneckMonitor(Callback):
    """Reports, once per training epoch, whether the data loader limits
    training speed and whether more loader workers would help.

    Reads the timing counters of the training FixedNumBatchesDataset:

    - wait: fraction of the epoch the training step was blocked waiting
      for a batch. GPU work already queued can still run during the wait,
      so this bounds, rather than equals, the GPU idle time.
    - idle: fraction of the epoch the prefetch thread had a batch ready but
      the queue was full, i.e. spare loader capacity.
    - C++ fetch / transport: fraction of the epoch the prefetch thread spent
      waiting for the C++ decoder workers vs. moving the decoded batch to the
      device itself. Only the first gets faster with more workers.

    Verdicts (``wait`` above ``wait_threshold`` counts as starved):

    - starved, little spare capacity, mostly transport: the prefetch thread's
      own transport work is the bottleneck; more workers will not help.
    - starved, little spare capacity, mostly C++ fetch: decoder throughput is
      the bottleneck, raise ``--num-workers`` (if CPUs are left to use).
    - starved, but plenty of spare capacity: production is bursty and the
      prefetch queue is too shallow to smooth it, raise
      ``--data-loader-queue-size``.
    - not starved, but almost no spare capacity: the loader is just keeping
      up; a few more workers buy headroom.
    - otherwise: not loader bound (the GPU step is the limit).

    In DDP every rank's loader is measured and the worst rank decides (one
    slow rank stalls all of them at the gradient all-reduce).
    """

    def __init__(
        self,
        train_dataset,
        num_workers: int,
        wait_threshold: float = 0.05,
        idle_threshold: float = 0.10,
    ):
        super().__init__()
        self.dataset = train_dataset
        self.num_workers = num_workers
        self.wait_threshold = wait_threshold
        self.idle_threshold = idle_threshold
        self.cpus = _available_cpus_per_rank()
        self._start_stats = None
        self._start_time = None
        self.last_report: dict | None = None

    def _stats(self):
        get = getattr(self.dataset, "get_loader_stats", None)
        return get() if get is not None else None

    @torch.compiler.disable
    def on_train_epoch_start(self, trainer):
        self._start_stats = self._stats()
        self._start_time = time.perf_counter()

    def _verdict(
        self, wait: float, idle: float, native: float, transport: float
    ) -> str:
        """Classify one epoch. ``native`` and ``transport`` split the prefetch
        thread's busy time: waiting for the C++ decoder workers (scales with
        --num-workers) vs. moving the finished batch to the device on the
        prefetch thread itself (does not)."""
        transport_bound = transport > native
        if wait >= self.wait_threshold:
            if idle >= self.idle_threshold:
                return (
                    "BURSTY: the loader has spare capacity overall but the queue still ran dry; "
                    "raise --data-loader-queue-size"
                )
            if transport_bound:
                return (
                    "TRANSPORT BOUND: the single prefetch thread's batch transport to the device "
                    "is the limit, not the C++ decoders; more --num-workers will not help"
                )
            if self.num_workers < self.cpus:
                return (
                    f"DATA-LOADER BOUND: raise --num-workers (now {self.num_workers}, "
                    f"{self.cpus} CPUs available)"
                )
            return (
                f"DATA-LOADER BOUND with all {self.cpus} CPUs in use ({self.num_workers} workers): "
                "loader is CPU or I/O limited; more workers will not help"
            )
        if idle < self.idle_threshold:
            if transport_bound:
                return "NEAR CAPACITY: batch transport on the prefetch thread is only just keeping up"
            return (
                "NEAR CAPACITY: the loader is only just keeping up; "
                "a few more --num-workers would add headroom"
            )
        return "not loader bound (GPU step is the limit)"

    @torch.compiler.disable
    def on_train_epoch_end(self, trainer):
        end = self._stats()
        if end is None or self._start_stats is None:
            return
        wall = time.perf_counter() - self._start_time
        start = self._start_stats
        delta = {k: end[k] - start[k] for k in end}
        batches = delta["consumer_batches"]
        if wall <= 0 or batches <= 0:
            return

        native = min(1.0, delta["native_fetch_s"] / wall)
        transport = min(
            1.0, max(0.0, delta["producer_fetch_s"] - delta["native_fetch_s"]) / wall
        )
        wait = min(1.0, delta["consumer_wait_s"] / wall)
        idle = min(1.0, delta["producer_put_wait_s"] / wall)
        empty = delta["consumer_empty_gets"] / batches

        if dist.is_available() and dist.is_initialized() and trainer.world_size > 1:
            device = (
                trainer.device if dist.get_backend() == "nccl" else torch.device("cpu")
            )
            # MAX over every "worse" direction: the worst rank decides.
            t = torch.tensor(
                [wait, -idle, empty, native, transport],
                dtype=torch.float64,
                device=device,
            )
            dist.all_reduce(t, op=dist.ReduceOp.MAX)
            wait, idle, empty, native, transport = t.tolist()
            idle = -idle

        capacity = getattr(self.dataset, "queue_capacity", 0) or 0
        fill = delta["consumer_fill_sum"] / batches
        verdict = self._verdict(wait, idle, native, transport)
        self.last_report = {
            "wait": wait,
            "idle": idle,
            "native": native,
            "transport": transport,
            "empty": empty,
            "fill": fill,
            "verdict": verdict,
        }

        trainer.callback_metrics["loader_wait_frac"] = wait
        trainer.callback_metrics["loader_idle_frac"] = idle
        trainer._log_metrics(
            {
                "loader_wait_frac": wait,
                "loader_idle_frac": idle,
                "loader_native_frac": native,
                "loader_transport_frac": transport,
            },
            step=trainer.global_step,
        )

        if trainer.rank == 0:
            startup = ""
            if start["startup_wait_s"] == 0 and end["startup_wait_s"] > 0:
                startup = f", first batch after {end['startup_wait_s']:.1f}s"
            print(
                f"Epoch {trainer.current_epoch:>2} (Loader): "
                f"step waited for data {wait:.1%} of the epoch, "
                f"loader idle (queue full) {idle:.1%}, "
                f"in C++ fetch {native:.1%}, in transport {transport:.1%}, "
                f"queue empty at {empty:.0%} of fetches, "
                f"avg fill {fill:.1f}/{capacity}{startup} -> {verdict}",
                flush=True,
            )
