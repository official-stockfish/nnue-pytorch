from dataclasses import dataclass, field
from typing import Optional, Tuple, Literal
import tyro
from tyro.conf import (
    OmitArgPrefixes,
    UseAppendAction,
    FlagConversionOff,
    Positional,
)

from data_loader.config import DataloaderSkipConfig
from model.config import NNUELightningConfig


@dataclass(kw_only=True)
class TrainingConfig:
    datasets: Positional[Tuple[str, ...]] = ()
    """Training datasets (.binpack). Interleaved at chunk level if multiple specified. Same data is used for training and validation if no validation data is specified."""

    validation_datasets: UseAppendAction[Tuple[str, ...]] = ()
    """Validation data to use for validation instead of the training data."""

    validation_size: int = 0
    """Number of positions in validation epoch (<= 0 disables validation)."""

    check_val_every_n_epoch: int = 1
    """Number of epochs between validation (has to be >= 1)."""

    default_root_dir: Optional[str] = None
    """Default root directory for logs and checkpoints. Default: None (use current directory)."""

    gpus: Optional[str] = None
    """List of gpus to use, e.g. 0,1,2,3 for 4 gpus. Only used when accelerator="cuda"."""

    pin_memory: bool = True
    """Whether to use pin memory in the data pipeline. Should generally be left on unless you encounter issues with too much RAM usage."""

    data_loader_queue_size: int = 16
    """Size of the prefetching queue. Should be conservative if pin_memory is active."""

    max_epochs: int = 800
    """Maximum number of epochs to train for."""

    swa_start_epoch: int = -1
    """Start epoch for swa. Negative number to disable."""

    max_time: str = "30:00:00:00"
    """The maximum time to train for. A string in the format DD:HH:MM:SS (Default 30:00:00:00)."""

    num_workers: int = 1
    """Number of worker threads to use for data loading. Currently only works well for binpack."""

    batch_size: int = 16384
    """Number of positions per batch / per iteration."""

    threads: int = -1
    """Number of torch threads to use. Default automatic (cores)."""

    accelerator: Literal["auto", "cuda", "mps", "cpu"] = "auto"
    """Hardware accelerator. 'auto' picks cuda > mps > cpu in order of availability."""

    compile_backend: Literal["inductor", "cudagraphs"] = "inductor"
    """Which backend to use for torch.compile. inductor works well with larger nets, cudagraphs with smaller nets."""

    seed: int = 42
    """Torch seed to use."""

    resume_from_model: Optional[str] = None
    """Initializes training using the weights from the given .pt model."""

    resume_from_checkpoint: Optional[str] = None
    """Initializes training using a given .ckpt model."""

    network_save_period: int = 20
    """Number of epochs between network snapshots. None to disable."""

    save_last_network: FlagConversionOff[bool] = True
    """Whether to always save the last produced network."""

    save_top_k: int = -1
    """Number of networks to save as a history"""

    epoch_size: int = 100_000_000
    """Number of positions per epoch."""

    dataloader_config: OmitArgPrefixes[DataloaderSkipConfig] = field(
        default_factory=DataloaderSkipConfig
    )

    nnue_lightning_config: OmitArgPrefixes[NNUELightningConfig] = field(
        default_factory=NNUELightningConfig
    )

    @property
    def num_batches_per_epoch(self) -> int:
        """Calculates batches per epoch based on validated batch size."""
        return max(1, self.epoch_size // self.batch_size)

    def __post_init__(self):
        if not self.datasets:
            raise ValueError("Argument `datasets` is required.")
        if self.max_epochs <= 0 or self.epoch_size <= 0 or self.batch_size <= 0:
            raise ValueError(
                "Arguments `max_epochs`, `epoch_size` and `batch_size` must be positive."
            )
        if self.check_val_every_n_epoch < 1:
            raise ValueError(
                "check_val_every_n_epoch has to be >= 1, "
                f"got {self.check_val_every_n_epoch}."
            )


if __name__ == "__main__":
    config = tyro.cli(TrainingConfig)
    print(config)
