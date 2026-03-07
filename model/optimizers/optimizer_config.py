from dataclasses import dataclass
from typing import Literal

from .ranger21_wrapper import Ranger21Config
from .schedulefree_wrapper import ScheduleFreeConfig

@dataclass
class OptimizerConfig(Ranger21Config, ScheduleFreeConfig):
    optimizer_name: Literal["schedulefree", "ranger21"] = "ranger21"
    """Which optimizer to use. """

    ft_weight_decay: float = 0.0
    """Weight decay to apply to the feature transformer parameters.."""

    dense_weight_decay: float = 0.0
    """Weight decay to apply to the dense layer parameters."""

    lr: float = 8.75e-4
    """Initial learning rate. Only used for schedulefree."""
