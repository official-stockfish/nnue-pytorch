from dataclasses import dataclass
from typing import Any

import torch


@dataclass(kw_only=True, frozen=False)
class LRSchedulerConfig:
    gamma: float = 0.992
    """Multiplicative factor applied to the learning rate after every epoch."""

    one_cycle_steps: int = 0
    """Number of steps for the One Cycle LR scheduler. If set to a positive value, One Cycle LR scheduler will be used. If set to 0 or a negative value, StepLR with step_size=1 will be used."""

    one_cycle_warmup_pct: float = 0.2
    """Fraction of the cycle to spend increasing the learning rate in the One Cycle LR scheduler."""

    one_cycle_start_div: float = 25
    """Initial lr div factor when using One Cycle LR scheduler."""

    one_cycle_final_div: float = 50
    """Final lr div factor when using One Cycle LR scheduler."""


class SafeOneCycleLR(torch.optim.lr_scheduler.OneCycleLR):
    def step(self, epoch=None):
        if self.last_epoch < self.total_steps - 1:
            super().step(epoch)


def setup_lr_scheduler(
    optimizer: torch.optim.Optimizer,
    train_params: list[dict[str, Any]],
    config: LRSchedulerConfig,
) -> torch.optim.lr_scheduler.StepLR | dict[str, Any]:
    if config.one_cycle_steps <= 0:
        scheduler = torch.optim.lr_scheduler.StepLR(
            optimizer, step_size=1, gamma=config.gamma
        )
    else:
        LRs = [group["lr"] for group in train_params]
        one_cycle_scheduler = SafeOneCycleLR(
            optimizer,
            max_lr=LRs,
            total_steps=config.one_cycle_steps,
            div_factor=config.one_cycle_start_div,
            final_div_factor=config.one_cycle_final_div,
            pct_start=config.one_cycle_warmup_pct,
            cycle_momentum=False,
        )
        scheduler = {"scheduler": one_cycle_scheduler, "interval": "step"}

    return scheduler
