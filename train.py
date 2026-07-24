import time
import warnings
import os
import sys
import random
import numpy as np
from datetime import timedelta

import torch
import torch.distributed as dist
from torch import set_num_threads as t_set_num_threads
from torch.utils.data import DataLoader

import data_loader
import model as M
import tyro

from config import TrainingConfig
from data_loader.config import DataloaderDDPConfig
from trainer.engine import init_distributed, SimpleTrainer
from trainer.callbacks import (
    CheckpointManager,
    ExplicitSWA,
    SimpleLineLogger,
    TerminateOnNaN,
    TimeLimit,
    WeightClipper,
)
from trainer.loggers import CSVLogger, TensorBoardLogger

warnings.filterwarnings("ignore", ".*does not have many workers.*")


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
    prefetch_device=None,
    rank=0,
    world_size=1,
):
    # Epoch and validation sizes are global; shard across DDP ranks.
    features_name = feature_name
    effective_batch_size = batch_size * world_size
    train_num_batches = max(1, epoch_size // effective_batch_size)
    train_infinite = data_loader.SparseBatchDataset(
        features_name,
        train_filenames,
        batch_size,
        num_workers=num_workers,
        config=config,
        ddp_config=DataloaderDDPConfig(rank=rank, world_size=world_size),
    )
    # num_workers has to be 0 for sparse, and 1 for dense
    # it currently cannot work in parallel mode but it shouldn't need to
    train = DataLoader(
        data_loader.FixedNumBatchesDataset(
            train_infinite,
            train_num_batches,
            pin_memory=pin_memory,
            queue_size_limit=queue_size_limit,
            device=prefetch_device,
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
                max(1, val_size // effective_batch_size),
                pin_memory=pin_memory,
                queue_size_limit=queue_size_limit,
                device=prefetch_device,
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
            ddp_config=DataloaderDDPConfig(rank=rank, world_size=world_size),
        )
        val = DataLoader(
            data_loader.FixedNumBatchesDataset(
                val_infinite,
                max(1, val_size // effective_batch_size),
                pin_memory=pin_memory,
                queue_size_limit=queue_size_limit,
                device=prefetch_device,
            ),
            batch_size=None,
            batch_sampler=None,
            num_workers=0,
        )
    return train, val


def is_master_process():
    # torchrun sets 'RANK'. If not set, we assume it's a single-process run (Rank 0).
    return int(os.environ.get("RANK", 0)) == 0


def _normalize_optimizer_and_schedulers(optimizer_and_schedulers):
    """Normalize configure_optimizers() return value to (optimizer, schedulers_list)."""
    if isinstance(optimizer_and_schedulers, tuple):
        optimizer, schedulers = optimizer_and_schedulers
    else:
        optimizer = optimizer_and_schedulers
        schedulers = []

    if isinstance(optimizer, list):
        if len(optimizer) != 1:
            raise ValueError(
                f"Expected a single optimizer, got {len(optimizer)} optimizers."
            )
        optimizer = optimizer[0]

    if schedulers is None:
        schedulers = []
    elif not isinstance(schedulers, list):
        schedulers = [schedulers]

    normalized_schedulers = []
    for sch in schedulers:
        if isinstance(sch, dict):
            sch = sch.get("scheduler", sch)
        normalized_schedulers.append(sch)

    return optimizer, normalized_schedulers


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

    rank, world_size, local_rank = init_distributed(
        args.process_group_backend, device_type=accelerator
    )

    # temporarily default to using only device 0 if user didn't specify --gpus
    # doing this so that batch size is consistent since if we rely on "auto" behavior
    # we don't know at this point in the code what the world size is.
    # TODO: refactor initialization so that we can support default behavior of "auto" with proper batch sizing
    if world_size > 1:
        n_devices = world_size
    elif accelerator == "cuda":
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
        )
    else:
        assert os.path.exists(args.resume_from_model)
        try:
            nnue = torch.load(
                args.resume_from_model, weights_only=False, map_location="cpu"
            )
            nnue.train()
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

    torch.manual_seed(args.seed)
    random.seed(args.seed)
    np.random.seed(args.seed)
    torch.backends.cudnn.benchmark = True

    logdir = args.default_root_dir if args.default_root_dir else "logs/"
    tb_logger = TensorBoardLogger(logdir)
    csv_logger = CSVLogger(logdir, version=tb_logger.version)
    loggers = [tb_logger, csv_logger]

    if is_master_process():
        print(
            f"batch_size(global)={global_batch_size_requested} | n_devices={n_devices} | batch_size(per_gpu)={per_gpu_batch_size}"
        )
        print("Loss parameters:")
        print(args.nnue_lightning_config.loss_params)
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

    if accelerator == "mps":
        # On MPS, torch.compile is currently unstable
        if is_master_process():
            print("Disabling torch.compile for accelerator='mps'.")
    else:
        # Compile only the inner NNUEModel; the outer NNUE shell handles loss,
        # lambda scheduling and checkpointing in plain Python.
        torch._dynamo.config.cache_size_limit = 64
        nnue.model = torch.compile(nnue.model, backend=args.compile_backend)

    if accelerator == "cuda":
        device = torch.device(f"cuda:{local_rank}")
    elif accelerator == "mps":
        device = torch.device("mps")
    else:
        device = torch.device("cpu")

    train, val = make_data_loaders(
        train_datasets,
        val_datasets,
        input_feature_name,
        actual_workers,
        per_gpu_batch_size,
        args.dataloader_config,
        args.epoch_size,
        args.validation_size,
        pin_memory=args.pin_memory and accelerator == "cuda",
        queue_size_limit=args.data_loader_queue_size,
        prefetch_device=device if accelerator == "cuda" else None,
        rank=rank,
        world_size=world_size,
    )

    optimizer_and_schedulers = nnue.configure_optimizers()
    optimizer, schedulers = _normalize_optimizer_and_schedulers(optimizer_and_schedulers)

    refresh_rate = max(1, (args.num_batches_per_epoch + 4) // 5)
    nan_callback = TerminateOnNaN()
    trainer_callbacks = [
        CheckpointManager(
            save_last=args.save_last_network,
            every_n_epochs=args.network_save_period,
            save_top_k=args.save_top_k,
            dirpath=os.path.join(tb_logger.log_dir, "checkpoints"),
        ),
        SimpleLineLogger(refresh_rate=refresh_rate),
        TimeLimit(args.max_time),
        WeightClipper(),
        nan_callback,
    ]
    if 0 <= args.swa_start_epoch < args.max_epochs:
        swa_callback = ExplicitSWA(args.swa_start_epoch, tb_logger.log_dir)
        trainer_callbacks.append(swa_callback)

    trainer = SimpleTrainer(
        model=nnue,
        optimizer=optimizer,
        schedulers=schedulers,
        max_epochs=args.max_epochs,
        check_val_every_n_epoch=args.check_val_every_n_epoch,
        gradient_clip_val=2.0,
        log_every_n_steps=refresh_rate,
        default_root_dir=logdir,
        callbacks=trainer_callbacks,
        logger=loggers,
        device=device,
        rank=rank,
        world_size=world_size,
        local_rank=local_rank,
    )

    if actual_threads > 0:
        t_set_num_threads(actual_threads)

    if args.resume_from_checkpoint:
        trainer.load_checkpoint(args.resume_from_checkpoint)

    trainer.fit(train, val)

    aborted_due_to_nan = nan_callback.nan_detected

    if 0 <= args.swa_start_epoch < args.max_epochs and not aborted_due_to_nan:
        nnue.eval()
        swa_callback.swap_weights(nnue, to_eval=True)
        if val is not None:
            trainer.validate(val)
        else:
            trainer.validate(train)
        swa_callback.swap_weights(nnue, to_eval=False)

    if rank == 0:
        last_savepath = os.path.join(tb_logger.log_dir, "checkpoints", "last.ckpt")
        swa_savepath = os.path.join(tb_logger.log_dir, "checkpoints", "last_swa.ckpt")
        non_swa_path = os.path.join(tb_logger.log_dir, "checkpoints", "last_non_swa.ckpt")
        if os.path.exists(swa_savepath):
            if os.path.exists(last_savepath):
                print(f"Renaming existing checkpoint at {last_savepath} to {non_swa_path} to preserve original model.")
                os.rename(last_savepath, non_swa_path)
            os.rename(swa_savepath, last_savepath)

        with open(os.path.join(logdir, "training_finished"), "w"):
            pass

    if dist.is_initialized():
        dist.destroy_process_group()

    if aborted_due_to_nan:
        print(
            "\n[TerminateOnNaN] Exiting with non-zero status code due to NaN/Inf detection.",
            flush=True,
        )
        sys.exit(1)


if __name__ == "__main__":
    main()
    if sys.platform == "win32":
        os._exit(0)
