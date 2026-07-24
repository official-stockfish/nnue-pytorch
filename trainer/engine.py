"""Lightning-free training engine for NNUE.

`SimpleTrainer` mirrors the parts of `lightning.Trainer` used by the project,
while keeping `model.NNUE` as a plain `torch.nn.Module`.
"""

from __future__ import annotations

import os
from collections.abc import Iterable
from typing import Any

import torch
import torch.distributed as dist
from torch import nn
from torch.nn.parallel import DistributedDataParallel as DDP


def init_distributed(
    backend: str | None = None,
    local_rank: int | None = None,
    device_type: str | None = None,
):
    """Initialize the distributed process group if multi-process training.

    Returns ``(rank, world_size, local_rank)``.  For single-process runs the
    returned values are ``(0, 1, 0)``.
    """
    if local_rank is None:
        local_rank = int(os.environ.get("LOCAL_RANK", "0"))

    if dist.is_available() and dist.is_initialized():
        rank = dist.get_rank()
        world_size = dist.get_world_size()
    else:
        rank = int(os.environ.get("RANK", "0"))
        world_size = int(os.environ.get("WORLD_SIZE", "1"))

        if world_size > 1 and dist.is_available():
            if backend is None:
                if device_type == "cpu":
                    backend = "gloo"
                else:
                    backend = "nccl" if torch.cuda.is_available() else "gloo"
            dist.init_process_group(backend, init_method="env://")

    if torch.cuda.is_available() and device_type != "cpu":
        torch.cuda.set_device(local_rank)

    return rank, world_size, local_rank


def is_one_cycle(scheduler: Any) -> bool:
    """Return True if *scheduler* should be stepped per optimizer step."""
    if isinstance(scheduler, dict):
        scheduler = scheduler.get("scheduler", scheduler)
    if scheduler is None:
        return False
    if isinstance(scheduler, torch.optim.lr_scheduler.OneCycleLR):
        return True
    # The project wraps OneCycleLR in SafeOneCycleLR.
    name = scheduler.__class__.__name__
    return "OneCycleLR" in name


def _unwrap_module(module: nn.Module) -> nn.Module:
    """Strip DDP and torch.compile() wrappers until the original module."""
    while isinstance(module, DDP):
        module = module.module
    while hasattr(module, "_orig_mod"):
        module = module._orig_mod
    return module


class SimpleTrainer:
    """Minimal replacement for ``lightning.Trainer``.

    The trainer owns the loop, callbacks, logging, checkpointing and basic
    distributed-data-parallel handling.
    """

    def __init__(
        self,
        model: nn.Module,
        optimizer: torch.optim.Optimizer,
        schedulers: Iterable[Any] | None,
        max_epochs: int,
        check_val_every_n_epoch: int,
        gradient_clip_val: float,
        log_every_n_steps: int,
        default_root_dir: str | None,
        callbacks: Iterable[Any] | None,
        logger: Any | None,
        device: str | torch.device,
        rank: int,
        world_size: int,
        local_rank: int,
    ):
        self.model = model
        self.optimizer = optimizer
        self.max_epochs = max_epochs
        self.check_val_every_n_epoch = check_val_every_n_epoch
        self.gradient_clip_val = gradient_clip_val
        self.log_every_n_steps = log_every_n_steps
        self.default_root_dir = default_root_dir or "."
        self.callbacks: list[Any] = list(callbacks) if callbacks is not None else []

        # Accept a single logger or a list.  Expose the first logger as
        # ``self.logger`` (mirroring PyTorch Lightning) and keep the full list
        # as ``self.loggers`` so metrics can be forwarded to every logger.
        if isinstance(logger, (list, tuple)):
            self.loggers: list[Any] = list(logger)
            self.logger = self.loggers[0] if self.loggers else None
        else:
            self.loggers = [logger] if logger is not None else []
            self.logger = logger

        self.device = torch.device(device)
        self.rank = rank
        self.world_size = world_size
        self.local_rank = local_rank

        # Parity aliases used by callbacks and future train.py code.
        self.is_global_zero = self.rank == 0
        self.sanity_checking: bool = False
        self.should_stop: bool = False

        self.current_epoch: int = 0
        self.global_step: int = 0
        self.num_training_batches: int | None = None
        self.num_val_batches: int | None = None
        self.callback_metrics: dict = {}

        # Normalise scheduler list: the project returns either a scheduler or a
        # dict of the form {"scheduler": scheduler, "interval": "step"}.
        self._schedulers: list[Any] = []
        for sch in schedulers or []:
            if isinstance(sch, dict):
                self._schedulers.append(sch.get("scheduler", sch))
            else:
                self._schedulers.append(sch)

        # Move model to the target device and, in multi-process settings, wrap
        # the inner NNUEModel with DDP.  Gradients all-reduce through the model
        # forward pass; the loss is computed outside so callbacks can keep the
        # plain NNUE object as ``self.model``.
        self.model.to(self.device)
        self._ddp_model: nn.Module | None = None
        if self.world_size > 1:
            inner_model = _unwrap_module(self.model).model
            inner_model.to(self.device)
            kwargs = {
                "gradient_as_bucket_view": True,
                "static_graph": True,
                "find_unused_parameters": False,
                "broadcast_buffers": False,
                "bucket_cap_mb": 50,
            }
            if self.device.type == "cuda":
                kwargs["device_ids"] = [self.local_rank]
                kwargs["output_device"] = self.local_rank
            self._ddp_model = DDP(inner_model, **kwargs)

    # ------------------------------------------------------------------
    # Helpers
    # ------------------------------------------------------------------
    def _call_callbacks(self, hook: str, *args, **kwargs):
        for cb in self.callbacks:
            method = getattr(cb, hook, None)
            if method is not None:
                method(*args, **kwargs)

    def _move_batch_to_device(self, batch):
        return tuple(
            t.to(self.device, non_blocking=True)
            if isinstance(t, torch.Tensor)
            else t
            for t in batch
        )

    def _log_metrics(self, metrics: dict, step: int | None = None):
        if self.rank != 0:
            return
        if not self.loggers:
            return
        for logger in self.loggers:
            logger.log_metrics(metrics, step=step)

    def _training_step(self, batch):
        if self._ddp_model is not None:
            # DDP wraps the inner NNUEModel; run its forward to get scorenet,
            # then compute the loss on the plain NNUE object.
            us, them, white_indices, black_indices, _outcome, _score, piece_count = batch
            scorenet = self._ddp_model(
                us,
                them,
                white_indices,
                black_indices,
                piece_count,
                self.model.config.use_fake_act_quantization,
                self.model.config.use_fake_weight_quantization,
            )
            loss = _unwrap_module(self.model).compute_loss_with_scorenet(
                scorenet, batch, self.current_epoch
            )
            _unwrap_module(self.model).loss_metrics["train_loss_epoch"].update(loss.detach())
            outputs = {"loss": loss, "train_loss": loss.detach()}
        else:
            outputs = self.model.train_step(batch, self.current_epoch, self.global_step)
        return outputs

    # ------------------------------------------------------------------
    # Checkpointing
    # ------------------------------------------------------------------
    def save_checkpoint(self, path: str):
        """Save a checkpoint.

        The checkpoint contains the epoch, global step, model state (which
        already includes optimizer/lambda state from ``NNUE.state_dict``), and
        scheduler states.  The file is written only on rank 0, but the save
        hooks are run on every rank so that in-flight state (e.g. weight
        clipping, optimizer eval mode) stays consistent across processes.
        """
        unwrapped = _unwrap_module(self.model)
        checkpoint = {
            "epoch": self.current_epoch,
            "global_step": self.global_step,
            "state_dict": unwrapped.state_dict(),
            "lr_schedulers": [sch.state_dict() for sch in self._schedulers],
        }

        # Model-level hook (e.g. lambda/optimizer state and eval-mode swaps).
        if hasattr(unwrapped, "on_save_checkpoint"):
            unwrapped.on_save_checkpoint(checkpoint)

        # Callback-level hooks.
        self._call_callbacks("on_save_checkpoint", self, checkpoint)

        if self.rank == 0:
            parent = os.path.dirname(path)
            if parent:
                os.makedirs(parent, exist_ok=True)
            torch.save(checkpoint, path)

    def load_checkpoint(self, path: str):
        """Load a checkpoint and return the raw dictionary."""
        checkpoint = torch.load(path, map_location=self.device, weights_only=False)

        unwrapped = _unwrap_module(self.model)
        unwrapped.load_state_dict(checkpoint["state_dict"])
        self.current_epoch = checkpoint.get("epoch", 0)
        self.global_step = checkpoint.get("global_step", 0)

        # Restore scheduler states if they were saved.
        scheduler_states = checkpoint.get("lr_schedulers", [])
        for sch, state in zip(self._schedulers, scheduler_states):
            sch.load_state_dict(state)

        # Model-level resume hook (e.g. lambda jitter buffer).
        if hasattr(unwrapped, "on_load_checkpoint"):
            unwrapped.on_load_checkpoint(checkpoint, resuming=True)

        self._call_callbacks("on_load_checkpoint", self, checkpoint)
        return checkpoint

    # ------------------------------------------------------------------
    # Training loop
    # ------------------------------------------------------------------
    def fit(self, train_loader, val_loader=None):
        self.num_training_batches = len(train_loader) if hasattr(train_loader, "__len__") else None

        self._call_callbacks("on_fit_start", self)
        self._call_callbacks("on_train_start", self)

        for epoch in range(self.current_epoch, self.max_epochs):
            self.current_epoch = epoch
            self._train_epoch(train_loader)

            if val_loader is not None and (self.current_epoch + 1) % self.check_val_every_n_epoch == 0:
                self.validate(val_loader)

            if self.should_stop:
                break

        self._call_callbacks("on_train_end", self)
        self._call_callbacks("on_fit_end", self)

    def _train_epoch(self, train_loader):
        self.model.train()

        if hasattr(self.model, "on_train_epoch_start"):
            self.model.on_train_epoch_start()
        self._call_callbacks("on_train_epoch_start", self)

        for batch_idx, batch in enumerate(train_loader):
            batch = self._move_batch_to_device(batch)

            if hasattr(self.model, "on_train_batch_start"):
                self.model.on_train_batch_start(batch, batch_idx)
            self._call_callbacks("on_train_batch_start", self, batch, batch_idx)

            outputs = self._training_step(batch)
            loss = outputs["loss"]

            self.optimizer.zero_grad(set_to_none=True)
            loss.backward()

            if self.gradient_clip_val is not None and self.gradient_clip_val > 0:
                torch.nn.utils.clip_grad_norm_(
                    self.model.parameters(), self.gradient_clip_val
                )

            self.optimizer.step()

            # Step OneCycleLR-style schedulers every optimizer step.
            for sch in self._schedulers:
                if is_one_cycle(sch):
                    sch.step()

            self.global_step += 1

            self.callback_metrics["train_loss"] = float(outputs["train_loss"])

            if (
                (batch_idx + 1) % self.log_every_n_steps == 0
                or (batch_idx + 1) == self.num_training_batches
            ):
                self._log_metrics(
                    {"train_loss": self.callback_metrics["train_loss"]},
                    step=self.global_step,
                )

            self._call_callbacks("on_train_batch_end", self, batch, batch_idx, outputs)

        # Compute and log epoch-level training loss before the model resets it.
        train_loss_epoch = self.model.loss_metrics["train_loss_epoch"].compute().item()
        self.callback_metrics["train_loss_epoch"] = train_loss_epoch
        self._log_metrics(
            {"train_loss_epoch": train_loss_epoch, "epoch": self.current_epoch},
            step=self.global_step,
        )

        if hasattr(self.model, "on_train_epoch_end"):
            self.model.on_train_epoch_end()
        self._call_callbacks("on_train_epoch_end", self)

        # Step epoch-interval schedulers (e.g. StepLR) once per epoch.
        for sch in self._schedulers:
            if not is_one_cycle(sch):
                sch.step()

    # ------------------------------------------------------------------
    # Validation loop
    # ------------------------------------------------------------------
    def validate(self, val_loader):
        if val_loader is None:
            return

        self.num_val_batches = len(val_loader) if hasattr(val_loader, "__len__") else None

        self._call_callbacks("on_validation_start", self)
        self.model.eval()

        if hasattr(self.model, "on_validation_epoch_start"):
            self.model.on_validation_epoch_start()
        self._call_callbacks("on_validation_epoch_start", self)

        total_val_loss = 0.0
        num_val_batches = 0

        with torch.no_grad():
            for batch_idx, batch in enumerate(val_loader):
                batch = self._move_batch_to_device(batch)

                self._call_callbacks("on_validation_batch_start", self, batch, batch_idx)

                outputs = self.model.val_step(batch, self.current_epoch, self.global_step)
                total_val_loss += float(outputs["val_loss"])
                num_val_batches += 1

                self._call_callbacks("on_validation_batch_end", self, batch, batch_idx, outputs)

        mean_val_loss = (
            total_val_loss / num_val_batches if num_val_batches > 0 else float("nan")
        )

        if self.world_size > 1:
            mean_tensor = torch.tensor(mean_val_loss, device=self.device)
            dist.all_reduce(mean_tensor, op=dist.ReduceOp.SUM)
            mean_tensor /= self.world_size
            mean_val_loss = mean_tensor.item()

        self.callback_metrics["val_loss_epoch"] = mean_val_loss
        self._log_metrics(
            {"val_loss_epoch": mean_val_loss, "epoch": self.current_epoch},
            step=self.global_step,
        )

        if hasattr(self.model, "on_validation_epoch_end"):
            self.model.on_validation_epoch_end()
        self._call_callbacks("on_validation_epoch_end", self)
        self._call_callbacks("on_validation_end", self)

        return {"val_loss_epoch": mean_val_loss}
