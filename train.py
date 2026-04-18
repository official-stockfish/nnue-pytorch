import time
import warnings
import os
import sys
from datetime import timedelta

import lightning as L
import torch
from torch import set_num_threads as t_set_num_threads
from torch.utils.data import DataLoader
from lightning.pytorch import loggers as pl_loggers
from lightning.pytorch.callbacks import Callback, ModelCheckpoint

import data_loader
import model as M
import tyro

from config import TrainingConfig

warnings.filterwarnings("ignore", ".*does not have many workers.*")


class TimeLimitAfterCheckpoint(Callback):
    def __init__(self, max_time: str):
        parts = list(map(int, max_time.strip().split(":")))
        if len(parts) != 4:
            raise ValueError("max_time must be in format 'DD:HH:MM:SS'")
        days, hours, minutes, seconds = parts
        self.max_duration = timedelta(
            days=days, hours=hours, minutes=minutes, seconds=seconds
        ).total_seconds()
        self.start_time = None

    def on_fit_start(self, trainer, pl_module):
        self.start_time = time.time()

    def on_train_epoch_end(self, trainer, pl_module):
        elapsed = time.time() - self.start_time
        if elapsed >= self.max_duration:
            trainer.should_stop = True
            print(
                f"[TimeLimit] Time limit reached ({elapsed:.1f}s), stopping after checkpoint."
            )

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
    def on_train_epoch_start(self, trainer, pl_module):
        if trainer.global_rank == 0:
            self.train_start_time = time.time()
            print("-"*60)

    @torch.compiler.disable
    def on_train_batch_end(self, trainer, pl_module, outputs, batch, batch_idx):
        if trainer.global_rank != 0:
            return

        current_step = batch_idx + 1
        total_batches = trainer.num_training_batches

        if current_step % self._get_refresh_rate(trainer) == 0 or current_step == total_batches:
            now = time.time()
            elapsed_total = now - self.train_start_time
            rate = current_step / elapsed_total if elapsed_total > 0 else 0

            remaining = (total_batches - current_step) / rate if rate > 0 else 0
            loss_val = trainer.callback_metrics.get(self.train_metric_step, float('nan'))

            print(
                f"Epoch {trainer.current_epoch:>2} (Train): "
                f"{current_step / total_batches:>4.0%}| "
                f"{current_step:>5}/{total_batches:<5} "
                f"[{self._format_time(elapsed_total)}<{self._format_time(remaining)}, "
                f"{rate:>6.2f}it/s, "
                f"{self.train_metric_step}={loss_val:.5f}, ",
                f"v_num={trainer.logger.version}]",
                flush=True,
            )

            self.train_last_time = now
            self.train_last_step = current_step

    @torch.compiler.disable
    def on_train_epoch_end(self, trainer, pl_module):
        if trainer.global_rank != 0 or trainer.sanity_checking:
            return

        pl_module._log_epoch_end(self.train_metric_epoch)
        train_loss = trainer.callback_metrics.get(self.train_metric_epoch, float('nan'))
        print(
            f"Epoch {trainer.current_epoch:>2} (Train): "
            f"[{self.train_metric_epoch}={train_loss:.5f}]",
            flush=True
        )

    # ==========================================
    # VALIDATION LOOP
    # ==========================================
    @torch.compiler.disable
    def on_validation_epoch_start(self, trainer, pl_module):
        if trainer.global_rank == 0 and not trainer.sanity_checking:
            self.val_start_time = time.time()

    @torch.compiler.disable
    def on_validation_batch_end(self, trainer, pl_module, outputs, batch, batch_idx, dataloader_idx=0):
        if trainer.global_rank != 0 or trainer.sanity_checking:
            return

        current_step = batch_idx + 1
        val_batches = trainer.num_val_batches
        if isinstance(val_batches, int):
            total_batches = val_batches
        else:
            total_batches = sum(val_batches)

        if current_step % self._get_refresh_rate(trainer) == 0 or current_step == total_batches:
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
    def on_validation_epoch_end(self, trainer, pl_module):
        if trainer.global_rank != 0 or trainer.sanity_checking:
            return

        pl_module._log_epoch_end(self.val_metric)
        val_loss = trainer.callback_metrics.get(self.val_metric, float('nan'))
        print(
            f"Epoch {trainer.current_epoch:>2} (Val): "
            f"[{self.val_metric}={val_loss:.5f}]",
            flush=True
        )


def make_data_loaders(
    train_filenames,
    val_filenames,
    feature_name: str,
    num_workers,
    batch_size,
    config: data_loader.DataloaderSkipConfig,
    epoch_size,
    val_size,
    pin_memory,
    queue_size_limit,
):
    # Epoch and validation sizes are arbitrary
    features_name = feature_name
    train_infinite = data_loader.SparseBatchDataset(
        features_name,
        train_filenames,
        batch_size,
        num_workers=num_workers,
        config=config,
    )
    # num_workers has to be 0 for sparse, and 1 for dense
    # it currently cannot work in parallel mode but it shouldn't need to
    train = DataLoader(
        data_loader.FixedNumBatchesDataset(
            train_infinite,
            (epoch_size + batch_size - 1) // batch_size,
            pin_memory=pin_memory,
            queue_size_limit=queue_size_limit,
        ),
        batch_size=None,
        batch_sampler=None,
        num_workers=0,
    )
    if val_size <= 0:
        val = None
    elif val_filenames is None:
        val = DataLoader(
            data_loader.FixedNumBatchesDataset(
                train_infinite,
                (val_size + batch_size - 1) // batch_size,
                pin_memory=pin_memory,
                queue_size_limit=queue_size_limit,
            ),
            batch_size=None,
            batch_sampler=None,
            num_workers=0,
        )
    else:
        val_infinite = data_loader.SparseBatchDataset(
            features_name,
            val_filenames,
            batch_size,
            config=config,
        )
        val = DataLoader(
            data_loader.FixedNumBatchesDataset(
                val_infinite,
                (val_size + batch_size - 1) // batch_size,
                pin_memory=pin_memory,
                queue_size_limit=queue_size_limit,
            ),
            batch_size=None,
            batch_sampler=None,
            num_workers=0,
        )
    return train, val


def is_master_process():
    # torchrun sets 'RANK'. If not set, we assume it's a single-process run (Rank 0).
    return int(os.environ.get("RANK", 0)) == 0


def main():
    args = tyro.cli(TrainingConfig)
    actual_threads, actual_workers = args.threads, args.num_workers

    for dataset in args.datasets:
        if not os.path.exists(dataset):
            raise Exception("{0} does not exist".format(dataset))

    for val_dataset in args.validation_datasets:
        if not os.path.exists(val_dataset):
            raise Exception("{0} does not exist".format(val_dataset))

    train_datasets = args.datasets
    val_datasets = None

    if len(args.validation_datasets) > 0:
        val_datasets = args.validation_datasets

    loss_params = args.nnue_lightning_config.loss_params
    if (loss_params.start_lambda is not None) != (loss_params.end_lambda is not None):
        raise Exception(
            "Either both or none of start_lambda and end_lambda must be specified."
        )

    loss_params.start_lambda = (
        loss_params.start_lambda
        if loss_params.start_lambda is not None
        else loss_params.lambda_
    )
    loss_params.end_lambda = (
        loss_params.end_lambda
        if loss_params.end_lambda is not None
        else loss_params.lambda_
    )

    global_batch_size_requested = args.batch_size

    accelerator = args.accelerator
    if accelerator == "auto":
        if torch.cuda.is_available():
            accelerator = "cuda"
        elif torch.backends.mps.is_available():
            accelerator = "mps"
        else:
            accelerator = "cpu"

    if args.compile_backend == "cudagraphs" and accelerator != "cuda":
        raise ValueError(
            f"--compile-backend=cudagraphs requires accelerator='cuda', "
            f"got accelerator='{accelerator}'. Use --compile-backend=inductor instead."
        )

    # temporarily default to using only device 0 if user didn't specify --gpus
    # doing this so that batch size is consistent since if we rely on "auto" behavior
    # we don't know at this point in the code what the world size is.
    # TODO: refactor initialization so that we can support default behavior of "auto" with proper batch sizing
    if accelerator == "cuda":
        if args.gpus:
            try:
                devices = [int(x) for x in args.gpus.rstrip(",").split(",") if x]
            except ValueError:
                print(
                    f"Invalid --gpus argument: '{args.gpus}'. "
                    "Expected a comma separated list of ints, e.g. 0,1",
                    file=sys.stderr,
                )
                return
        else:
            devices = [0]
        n_devices = len(devices)
        if n_devices == 0:
            print(
                f"Invalid --gpus argument: '{args.gpus}'. "
                "Expected a comma separated list of ints, e.g. 0,1",
                file=sys.stderr,
            )
            return
    else:
        if args.gpus:
            print(
                f"Warning: --gpus is ignored for accelerator='{accelerator}'",
                file=sys.stderr,
            )
        devices = 1
        n_devices = 1
    if global_batch_size_requested % n_devices != 0:
        msg = (
            f"--batch-size {global_batch_size_requested} must be divisible by "
            f"number of devices ({n_devices}) for accelerator='{accelerator}'."
        )
        if accelerator == "cuda":
            msg += f" Got --gpus={args.gpus or '0'}."
        raise ValueError(msg)
    per_gpu_batch_size = global_batch_size_requested // n_devices
    feature_name = args.nnue_lightning_config.features

    max_epoch = args.max_epochs or 800
    if args.resume_from_model is None:
        nnue = M.NNUE(
            config=args.nnue_lightning_config,
            max_epoch=max_epoch,
            num_batches_per_epoch=args.num_batches_per_epoch,
            param_index=args.dataloader_config.param_index,
            quantize_config=M.QuantizationConfig(),
        )
    else:
        assert os.path.exists(args.resume_from_model)
        try:
            nnue = torch.load(
                args.resume_from_model, weights_only=False, map_location="cpu"
            )
        except ModuleNotFoundError as e:
            raise RuntimeError(
                f"Could not load checkpoint: {e}. The model to be resumed was probably saved with a different version of the code."
            )
        # we can set the following here just like that because when resuming
        # from .pt the optimizer is only created after the training is started
        nnue.max_epoch = max_epoch
        nnue.num_batches_per_epoch = args.num_batches_per_epoch
        nnue.config = args.nnue_lightning_config
        nnue.param_index = args.dataloader_config.param_index

    input_feature_name = nnue.model.input_feature_name

    L.seed_everything(args.seed)

    logdir = args.default_root_dir if args.default_root_dir else "logs/"
    tb_logger = pl_loggers.TensorBoardLogger(logdir)
    csv_logger = pl_loggers.CSVLogger(logdir, version=tb_logger.version)
    loggers = [tb_logger, csv_logger]

    if is_master_process():
        print(
            f"batch_size(global)={global_batch_size_requested} | n_devices={n_devices} | batch_size(per_gpu)={per_gpu_batch_size}"
        )
        print("Loss parameters:")
        print(loss_params)
        print("Feature set: {}".format(feature_name))
        print("Num inputs: {}".format(nnue.model.input.NUM_INPUTS))

        print("Training with: {}".format(train_datasets))
        print("Validating with: {}".format(val_datasets))
        print("Seed {}".format(args.seed))
        print(args.dataloader_config)
        print("Using log dir {}".format(tb_logger.log_dir))
        print(f"Using {actual_workers} workers for C++ data loader.")
        if actual_threads > 0:
            print("Set torch num_threads to {} threads.".format(actual_threads))
        else:
            print("Using default torch num_threads setting.")
        print("", flush=True)

    checkpoint_callback = ModelCheckpoint(
        save_last=args.save_last_network,
        every_n_epochs=args.network_save_period,
        save_top_k=-1,
    )

    # Since we compile the entire lightning module we have quite a few graph breaks
    torch._dynamo.config.cache_size_limit = 32
    nnue = torch.compile(nnue, backend=args.compile_backend)
    # PL hack, undo slurm cluster detection which is broken for us. 'force interactive mode'
    # see lightning/fabric/plugins/environments/slurm.py near line 110
    os.environ["SLURM_JOB_NAME"] = "bash"

    train, val = make_data_loaders(
        train_datasets,
        val_datasets,
        input_feature_name,
        actual_workers,
        per_gpu_batch_size,
        args.dataloader_config,
        args.epoch_size,
        args.validation_size,
        pin_memory=args.pin_memory,
        queue_size_limit=args.data_loader_queue_size,
    )

    refresh_rate = max(1, (args.num_batches_per_epoch + 4) // 5)
    trainer = L.Trainer(
        default_root_dir=logdir,
        max_epochs=args.max_epochs,
        accelerator=accelerator,
        strategy="ddp" if n_devices > 1 else "auto",
        devices=devices,
        logger=loggers,
        callbacks=[
            checkpoint_callback,
            SimpleLineLogger(refresh_rate=refresh_rate),
            TimeLimitAfterCheckpoint(args.max_time),
            M.WeightClippingCallback(),
        ],
        log_every_n_steps=refresh_rate,
        enable_progress_bar=False,
        enable_checkpointing=True,
        benchmark=True,
        num_sanity_val_steps=0 if val is None else 4,
    )

    if actual_threads > 0:
        t_set_num_threads(actual_threads)

    if args.resume_from_checkpoint:
        trainer.fit(nnue, train, val, ckpt_path=args.resume_from_checkpoint)
    else:
        trainer.fit(nnue, train, val)

    if trainer.is_global_zero:
        with open(os.path.join(logdir, "training_finished"), "w"):
            pass


if __name__ == "__main__":
    main()
    if sys.platform == "win32":
        os._exit(0)
