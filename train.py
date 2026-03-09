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
from lightning.pytorch.callbacks import TQDMProgressBar, Callback, ModelCheckpoint

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
    val_infinite = data_loader.SparseBatchDataset(
        features_name,
        val_filenames,
        batch_size,
        config=config,
    )
    # num_workers has to be 0 for sparse, and 1 for dense
    # it currently cannot work in parallel mode but it shouldn't need to
    train = DataLoader(
        data_loader.FixedNumBatchesDataset(
            train_infinite, (epoch_size + batch_size - 1) // batch_size,
            pin_memory=pin_memory,
            queue_size_limit=queue_size_limit,
        ),
        batch_size=None,
        batch_sampler=None,
        num_workers=0,
    )
    val = (
        None
        if val_size == 0
        else DataLoader(
            data_loader.FixedNumBatchesDataset(
                val_infinite, (val_size + batch_size - 1) // batch_size,
                pin_memory=pin_memory,
                queue_size_limit=queue_size_limit,
            ),
            batch_size=None,
            batch_sampler=None,
            num_workers=0,
        )
    )
    return train, val


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
    val_datasets = train_datasets

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
    if global_batch_size_requested <= 0:
        global_batch_size_requested = 16384
    # temporarily default to using only device 0 if user didn't specify --gpus
    # doing this so that batch size is consistent since if we rely on "auto" behavior
    # we don't know at this point in the code what the world size is.
    # TODO: refactor initialization so that we can support default behavior of "auto" with proper batch sizing
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
    if global_batch_size_requested % n_devices != 0:
        raise ValueError(
            f"--batch-size {global_batch_size_requested} must be divisible by number of gpus ({n_devices}). "
            f"Got --gpus={args.gpus or '0'}"
        )
    per_gpu_batch_size = global_batch_size_requested // n_devices
    print(
        f"batch_size(global)={global_batch_size_requested} | n_devices={n_devices} | batch_size(per_gpu)={per_gpu_batch_size}",
        flush=True,
    )

    feature_name = args.nnue_lightning_config.features

    print("Loss parameters:")
    print(loss_params)

    num_batches_per_epoch=max(
                1, args.epoch_size // global_batch_size_requested
            )

    max_epoch = args.max_epochs or 800
    if args.resume_from_model is None:
        nnue = M.NNUE(
            config=args.nnue_lightning_config,
            max_epoch=max_epoch,
            param_index=args.dataloader_config.param_index,
            quantize_config=M.QuantizationConfig(),
        )
    else:
        assert os.path.exists(args.resume_from_model)
        try:
            nnue = torch.load(args.resume_from_model, weights_only=False, map_location="cpu")
        except ModuleNotFoundError as e:
            raise RuntimeError(
                f"Could not load checkpoint: {e}. The model to be resumed was probably saved with a different version of the code."
            )
        nnue.loss_params = loss_params
        nnue.max_epoch = max_epoch
        # we can set the following here just like that because when resuming
        # from .pt the optimizer is only created after the training is started
        nnue.optimizer_config = args.optimizer_config
        nnue.param_index = args.dataloader_config.param_index

    input_feature_name = nnue.model.input_feature_name
    print("Feature set: {}".format(feature_name))
    print("Num inputs: {}".format(nnue.model.input.NUM_INPUTS))

    print("Training with: {}".format(train_datasets))
    print("Validating with: {}".format(val_datasets))

    L.seed_everything(args.seed)
    print("Seed {}".format(args.seed))

    print(args.dataloader_config)

    logdir = args.default_root_dir if args.default_root_dir else "logs/"

    tb_logger = pl_loggers.TensorBoardLogger(logdir)

    print("Using log dir {}".format(tb_logger.log_dir), flush=True)

    checkpoint_callback = ModelCheckpoint(
        save_last=args.save_last_network,
        every_n_epochs=args.network_save_period,
        save_top_k=-1,
    )

    nnue = torch.compile(nnue, backend=args.compile_backend)
    # PL hack, undo slurm cluster detection which is broken for us. 'force interactive mode'
    # see lightning/fabric/plugins/environments/slurm.py near line 110
    os.environ["SLURM_JOB_NAME"] = "bash"

    refresh_rate = max(1, (num_batches_per_epoch + 4) // 5)
    trainer = L.Trainer(
        default_root_dir=logdir,
        max_epochs=args.max_epochs,
        accelerator="cuda",
        strategy="ddp" if len(devices) > 1 else "auto",
        devices=devices,
        logger=tb_logger,
        callbacks=[
            checkpoint_callback,
            TQDMProgressBar(refresh_rate=refresh_rate),
            TimeLimitAfterCheckpoint(args.max_time),
            M.WeightClippingCallback(),
        ],
        enable_progress_bar=True,
        enable_checkpointing=True,
        benchmark=True,
        num_sanity_val_steps=0,
    )

    if actual_threads > 0:
        print("Set torch num_threads to {} threads.".format(actual_threads))
        t_set_num_threads(actual_threads)
    else:
        print("Using default torch num_threads setting.", flush=True)
    print(f"Using {actual_workers} workers for C++ data loader.", flush=True)
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
