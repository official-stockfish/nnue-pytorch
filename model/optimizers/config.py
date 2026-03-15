from dataclasses import dataclass
from typing import Literal

from .ranger21_wrapper import Ranger21Config, Ranger21Wrapper
from .schedulefree_wrapper import ScheduleFreeConfig, ScheduleFreeWrapper

@dataclass(kw_only=True)
class OptimizerConfig(Ranger21Config, ScheduleFreeConfig):
    optimizer_name: Literal["schedulefree", "ranger21"] = "ranger21"
    """Which optimizer to use. """

    ft_weight_decay: float = 0.0
    """Weight decay to apply to the feature transformer parameters."""

    dense_weight_decay: float = 0.0
    """Weight decay to apply to the dense layer parameters."""

    lr: float = 8.75e-4
    """Initial learning rate."""

    def get_optimizer_wrapper(self, max_epoch, num_batches_per_epoch):
        optimizer_name = self.optimizer_name.lower().strip()
        if optimizer_name == "schedulefree":
            wrapper = ScheduleFreeWrapper(self)
        elif optimizer_name == "ranger21":
            wrapper = Ranger21Wrapper(self, max_epoch, num_batches_per_epoch)
        else:
            raise ValueError(f"Unknown optimizer_name: '{optimizer_name}'. Expected 'schedulefree' or 'ranger21'.")

        if self.dense_weight_decay > 0.0 or self.ft_weight_decay > 0.0:
            print(f"Using weight decay - ft_weight_decay: {self.ft_weight_decay}, dense_weight_decay: {self.dense_weight_decay}")
        return wrapper
