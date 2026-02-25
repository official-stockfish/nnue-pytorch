from dataclasses import dataclass
from typing import Optional, Tuple, Literal
import tyro
from tyro.conf import UseAppendAction, FlagConversionOff

from data_loader.config import DataloaderSkipConfig
from model.config import LossParams, ModelConfig
from model.modules.features import FeatureConfig


@dataclass
class TrainingConfig(LossParams, DataloaderSkipConfig, FeatureConfig, ModelConfig):
    datasets: UseAppendAction[Tuple[str, ...]] = ()
    """Training datasets (.binpack). Interleaved at chunk level if multiple specified. Same data is used for training and validation if no validation data is specified."""

    default_root_dir: Optional[str] = None
    """Default root directory for logs and checkpoints. Default: None (use current directory)."""

    gpus: Optional[str] = None
    """List of gpus to use, e.g. 0,1,2,3 for 4 gpus. Default: None (use all available gpus)."""

    max_epochs: int = 800
    """Maximum number of epochs to train for. Default 800."""

    max_time: str = "30:00:00:00"
    """The maximum time to train for. A string in the format DD:HH:MM:SS (Default 30:00:00:00)."""

    validation_datasets: UseAppendAction[Tuple[str, ...]] = ()
    """Validation data to use for validation instead of the training data."""

    gamma: float = 0.992
    """Multiplicative factor applied to the learning rate after every epoch."""

    lr: float = 8.75e-4
    """Initial learning rate."""

    num_workers: int = 1
    """Number of worker threads to use for data loading. Currently only works well for binpack."""

    batch_size: int = -1
    """Number of positions per batch / per iteration. Default on GPU = 8192 on CPU = 128."""

    threads: int = -1
    """Number of torch threads to use. Default automatic (cores)."""

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

    epoch_size: int = 100_000_000
    """Number of positions per epoch."""

    validation_size: int = 0
    """Number of positions per validation step."""


if __name__ == "__main__":
    config = tyro.cli(TrainingConfig)