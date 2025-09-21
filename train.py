import argparse
import model as M
import data_loader
import lightning as L
import features
import os
import sys
import torch
from torch import set_num_threads as t_set_num_threads
from lightning.pytorch import loggers as pl_loggers
from lightning.pytorch.callbacks import TQDMProgressBar, Callback, ModelCheckpoint
from torch.utils.data import DataLoader
import time
from datetime import timedelta
from features.feature_set import FeatureSet

import warnings

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

    def on_validation_end(self, trainer, pl_module):
        elapsed = time.time() - self.start_time
        if elapsed >= self.max_duration:
            trainer.should_stop = True
            print(
                f"[TimeLimit] Time limit reached ({elapsed:.1f}s), stopping after checkpoint."
            )


def make_data_loaders(
    train_filenames,
    val_filenames,
    feature_set: FeatureSet,
    num_workers,
    batch_size,
    config: data_loader.DataloaderSkipConfig,
    epoch_size,
    val_size,
):
    # Epoch and validation sizes are arbitrary
    features_name = feature_set.name
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
            train_infinite, (epoch_size + batch_size - 1) // batch_size
        ),
        batch_size=None,
        batch_sampler=None,
    )
    val = DataLoader(
        data_loader.FixedNumBatchesDataset(
            val_infinite, (val_size + batch_size - 1) // batch_size
        ),
        batch_size=None,
        batch_sampler=None,
    )
    return train, val


def str2bool(v):
    if isinstance(v, bool):
        return v
    if v.lower() in ("yes", "true", "t", "y", "1"):
        return True
    elif v.lower() in ("no", "false", "f", "n", "0"):
        return False
    else:
        raise argparse.ArgumentTypeError("Boolean value expected.")


def flatten_once(lst):
    return sum(lst, [])


def main():
    parser = argparse.ArgumentParser(description="Trains the network.")
    parser.add_argument(
        "datasets",
        action="append",
        nargs="+",
        help="Training datasets (.binpack). Interleaved at chunk level if multiple specified. Same data is used for training and validation if not validation data is specified.",
    )
    parser.add_argument(
        "--default_root_dir",
        type=str,
        default=None,
        dest="default_root_dir",
        help="Default root directory for logs and checkpoints. Default: None (use current directory).",
    )
    parser.add_argument(
        "--gpus",
        type=str,
        default=None,
        dest="gpus",
        help="List of gpus to use, e.g. 0,1,2,3 for 4 gpus. Default: None (use all available gpus).",
    )
    parser.add_argument(
        "--max_epochs",
        default=800,
        type=int,
        dest="max_epochs",
        help="Maximum number of epochs to train for. Default 800.",
    )
    parser.add_argument(
        "--max_time",
        default="30:00:00:00",
        type=str,
        dest="max_time",
        help="The maximum time to train for. A string in the format DD:HH:MM:SS (Default 30:00:00:00).",
    )
    parser.add_argument(
        "--validation-data",
        type=str,
        action="append",
        nargs="+",
        dest="validation_datasets",
        help="Validation data to use for validation instead of the training data.",
    )
    parser.add_argument(
        "--lambda",
        default=1.0,
        type=float,
        dest="lambda_",
        help="lambda=1.0 = train on evaluations, lambda=0.0 = train on game results, interpolates between (default=1.0).",
    )
    parser.add_argument(
        "--start-lambda",
        default=None,
        type=float,
        dest="start_lambda",
        help="lambda to use at first epoch.",
    )
    parser.add_argument(
        "--end-lambda",
        default=None,
        type=float,
        dest="end_lambda",
        help="lambda to use at last epoch.",
    )
    parser.add_argument(
        "--qp-asymmetry",
        default=0.0,
        type=float,
        dest="qp_asymmetry",
        help="Adjust to loss for those if q (prediction) > p (reference) (default=0.0)",
    )
    parser.add_argument(
        "--pow-exp",
        default=2.5,
        type=float,
        dest="pow_exp",
        help="exponent of the power law used for the mean error (default=2.5)",
    )
    parser.add_argument(
        "--in-offset",
        default=270,
        type=float,
        dest="in_offset",
        help="offset for conversion to win on input (default=270.0)",
    )
    parser.add_argument(
        "--out-offset",
        default=270,
        type=float,
        dest="out_offset",
        help="offset for conversion to win on input (default=270.0)",
    )
    parser.add_argument(
        "--in-scaling",
        default=340,
        type=float,
        dest="in_scaling",
        help="scaling for conversion to win on input (default=340.0)",
    )
    parser.add_argument(
        "--out-scaling",
        default=380,
        type=float,
        dest="out_scaling",
        help="scaling for conversion to win on input (default=380.0)",
    )
    parser.add_argument(
        "--gamma",
        default=0.992,
        type=float,
        dest="gamma",
        help="Multiplicative factor applied to the learning rate after every epoch.",
    )
    parser.add_argument(
        "--lr", default=8.75e-4, type=float, dest="lr", help="Initial learning rate."
    )
    parser.add_argument(
        "--num-workers",
        default=1,
        type=int,
        dest="num_workers",
        help="Number of worker threads to use for data loading. Currently only works well for binpack.",
    )
    parser.add_argument(
        "--batch-size",
        default=-1,
        type=int,
        dest="batch_size",
        help="Number of positions per batch / per iteration. Default on GPU = 8192 on CPU = 128.",
    )
    parser.add_argument(
        "--threads",
        default=-1,
        type=int,
        dest="threads",
        help="Number of torch threads to use. Default automatic (cores) .",
    )
    parser.add_argument(
        "--compile-backend",
        default="inductor",
        choices=["inductor", "cudagraphs"],
        type=str,
        dest="compile_backend",
        help="Which backend to use for torch.compile. inductor works well with larger nets, cudagraphs with smaller nets",
    )
    parser.add_argument(
        "--seed", default=42, type=int, dest="seed", help="torch seed to use."
    )
    parser.add_argument(
        "--smart-fen-skipping",
        action="store_true",
        dest="smart_fen_skipping_deprecated",
        help="If enabled positions that are bad training targets will be skipped during loading. Default: True, kept for backwards compatibility. This option is ignored",
    )
    parser.add_argument(
        "--no-smart-fen-skipping",
        action="store_true",
        dest="no_smart_fen_skipping",
        help="If used then no smart fen skipping will be done. By default smart fen skipping is done.",
    )
    parser.add_argument(
        "--no-wld-fen-skipping",
        action="store_true",
        dest="no_wld_fen_skipping",
        help="If used then no wld fen skipping will be done. By default wld fen skipping is done.",
    )
    parser.add_argument(
        "--random-fen-skipping",
        default=3,
        type=int,
        dest="random_fen_skipping",
        help="skip fens randomly on average random_fen_skipping before using one.",
    )
    parser.add_argument(
        "--resume-from-model",
        dest="resume_from_model",
        help="Initializes training using the weights from the given .pt model",
    )
    parser.add_argument(
        "--resume-from-checkpoint",
        dest="resume_from_checkpoint",
        help="Initializes training using a given .ckpt model",
    )
    parser.add_argument(
        "--network-save-period",
        type=int,
        default=20,
        dest="network_save_period",
        help="Number of epochs between network snapshots. None to disable.",
    )
    parser.add_argument(
        "--save-last-network",
        type=str2bool,
        default=True,
        dest="save_last_network",
        help="Whether to always save the last produced network.",
    )
    parser.add_argument(
        "--epoch-size",
        type=int,
        default=100000000,
        dest="epoch_size",
        help="Number of positions per epoch.",
    )
    parser.add_argument(
        "--validation-size",
        type=int,
        default=1000000,
        dest="validation_size",
        help="Number of positions per validation step.",
    )
    parser.add_argument(
        "--param-index",
        type=int,
        default=0,
        dest="param_index",
        help="Indexing for parameter scans.",
    )
    parser.add_argument(
        "--early-fen-skipping",
        type=int,
        default=-1,
        dest="early_fen_skipping",
        help="Skip n plies from the start.",
    )
    parser.add_argument(
        "--simple-eval-skipping",
        type=int,
        default=-1,
        dest="simple_eval_skipping",
        help="Skip positions that have abs(simple_eval(pos)) < n",
    )
    features.add_argparse_args(parser)
    args = parser.parse_args()

    args.datasets = flatten_once(args.datasets)
    if args.validation_datasets:
        args.validation_datasets = flatten_once(args.validation_datasets)
    else:
        args.validation_datasets = []

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

    if (args.start_lambda is not None) != (args.end_lambda is not None):
        raise Exception(
            "Either both or none of start_lambda and end_lambda must be specified."
        )

    batch_size = args.batch_size
    if batch_size <= 0:
        batch_size = 16384
    print("Using batch size {}".format(batch_size))

    feature_set = features.get_feature_set_from_name(args.features)

    loss_params = M.LossParams(
        in_offset=args.in_offset,
        in_scaling=args.in_scaling,
        out_offset=args.out_offset,
        out_scaling=args.out_scaling,
        start_lambda=args.start_lambda or args.lambda_,
        end_lambda=args.end_lambda or args.lambda_,
        pow_exp=args.pow_exp,
        qp_asymmetry=args.qp_asymmetry,
    )
    print("Loss parameters:")
    print(loss_params)

    max_epoch = args.max_epochs or 800
    if args.resume_from_model is None:
        nnue = M.NNUE(
            feature_set=feature_set,
            loss_params=loss_params,
            max_epoch=max_epoch,
            batch_size=batch_size,
            num_batches_per_epoch=args.epoch_size / batch_size,
            gamma=args.gamma,
            lr=args.lr,
            param_index=args.param_index,
        )
    else:
        nnue = torch.load(args.resume_from_model, weights_only=False)
        nnue.model.set_feature_set(feature_set)
        nnue.loss_params = loss_params
        nnue.max_epoch = max_epoch
        nnue.num_batches_per_epoch = args.epoch_size / batch_size
        # we can set the following here just like that because when resuming
        # from .pt the optimizer is only created after the training is started
        nnue.gamma = args.gamma
        nnue.lr = args.lr
        nnue.param_index = args.param_index

    print("Feature set: {}".format(feature_set.name))
    print("Num real features: {}".format(feature_set.num_real_features))
    print("Num virtual features: {}".format(feature_set.num_virtual_features))
    print("Num features: {}".format(feature_set.num_features))

    print("Training with: {}".format(train_datasets))
    print("Validating with: {}".format(val_datasets))

    L.seed_everything(args.seed)
    print("Seed {}".format(args.seed))

    print("Smart fen skipping: {}".format(not args.no_smart_fen_skipping))
    print("WLD fen skipping: {}".format(not args.no_wld_fen_skipping))
    print("Random fen skipping: {}".format(args.random_fen_skipping))
    print("Skip early plies: {}".format(args.early_fen_skipping))
    print("Skip simple eval : {}".format(args.simple_eval_skipping))
    print("Param index: {}".format(args.param_index))

    if args.threads > 0:
        print("limiting torch to {} threads.".format(args.threads))
        t_set_num_threads(args.threads)

    logdir = args.default_root_dir if args.default_root_dir else "logs/"
    print("Using log dir {}".format(logdir), flush=True)

    tb_logger = pl_loggers.TensorBoardLogger(logdir)
    checkpoint_callback = ModelCheckpoint(
        save_last=args.save_last_network,
        every_n_epochs=args.network_save_period,
        save_top_k=-1,
    )

    trainer = L.Trainer(
        default_root_dir=logdir,
        max_epochs=args.max_epochs,
        accelerator="cuda",
        devices=[int(x) for x in args.gpus.rstrip(",").split(",") if x]
        if args.gpus
        else "auto",
        logger=tb_logger,
        callbacks=[
            checkpoint_callback,
            TQDMProgressBar(refresh_rate=300),
            TimeLimitAfterCheckpoint(args.max_time),
        ],
        enable_progress_bar=True,
        enable_checkpointing=True,
        benchmark=True,
    )

    print("Using C++ data loader")
    train, val = make_data_loaders(
        train_datasets,
        val_datasets,
        feature_set,
        args.num_workers,
        batch_size,
        data_loader.DataloaderSkipConfig(
            filtered=not args.no_smart_fen_skipping,
            random_fen_skipping=args.random_fen_skipping,
            wld_filtered=not args.no_wld_fen_skipping,
            early_fen_skipping=args.early_fen_skipping,
            simple_eval_skipping=args.simple_eval_skipping,
            param_index=args.param_index,
        ),
        args.epoch_size,
        args.validation_size,
    )

    if args.resume_from_checkpoint:
        trainer.fit(nnue, train, val, ckpt_path=args.resume_from_checkpoint)
    else:
        trainer.fit(nnue, train, val)

    with open(os.path.join(logdir, "training_finished"), "w"):
        pass


if __name__ == "__main__":
    main()
    if sys.platform == "win32":
        os.system(f'wmic process where processid="{os.getpid()}" call terminate >nul')
