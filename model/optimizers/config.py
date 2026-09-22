from dataclasses import dataclass
from typing import Literal

from .adamw_wrapper import AdamWConfig, AdamWWrapper
from .rangerlite_wrapper import RangerLiteConfig, RangerLiteWrapper
from .schedulefree_wrapper import ScheduleFreeConfig, ScheduleFreeWrapper

OPTIMIZER_WRAPPERS = {
    "schedulefree": ScheduleFreeWrapper,
    "ranger21": lambda cfg: RangerLiteWrapper(cfg, legacy_mode=True),
    "rangerlite": lambda cfg: RangerLiteWrapper(cfg, legacy_mode=False),
    "adamw": AdamWWrapper,
}


@dataclass(kw_only=True)
class OptimizerConfig(RangerLiteConfig, ScheduleFreeConfig, AdamWConfig):
    optimizer_name: Literal["schedulefree", "ranger21", "rangerlite", "adamw"] = "rangerlite"
    """Which optimizer to use. Note that ranger21 is a specific configuration of rangerlite emulating ranger21 behaviour with legacy_mode=True."""

    ft_weight_decay: float = 0.0
    """Weight decay to apply to the feature transformer parameters."""

    dense_weight_decay: float = 0.0
    """Weight decay to apply to the dense layer parameters."""

    factorized_weight_decay: float = 0.0
    """Weight decay to apply to the factorized dense layer parameters."""

    lr: float = 8.75e-4
    """Initial learning rate."""

    def get_optimizer_wrapper(self):
        optimizer_name = self.optimizer_name.lower().strip()
        wrapper_factory = OPTIMIZER_WRAPPERS.get(optimizer_name)
        if wrapper_factory is None:
            valid_names = ", ".join(repr(name) for name in OPTIMIZER_WRAPPERS)
            raise ValueError(
                f"Unknown optimizer_name: '{optimizer_name}'. Expected one of {valid_names}."
            )
        wrapper = wrapper_factory(self)

        info_str = f"[OptimizerConfig] Using {optimizer_name} optimizer with lr: {self.lr}"
        if self.dense_weight_decay > 0.0 or self.ft_weight_decay > 0.0:
            info_str += f" and ft_weight_decay: {self.ft_weight_decay}, dense_weight_decay: {self.dense_weight_decay}"
        print(info_str + ".")
        return wrapper
