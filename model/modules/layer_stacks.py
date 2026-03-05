from typing import Generator

import torch
from torch import nn

from ..config import ModelConfig
from .stacked_linear import FactorizedStackedLinear, StackedLinear


class LayerStacks(nn.Module):
    def __init__(self, count: int, config: ModelConfig):
        super().__init__()

        self.count = count
        self.L1 = config.L1
        self.L2 = config.L2
        self.L3 = config.L3

        # Factorizer only for the first layer because later
        # there's a non-linearity and factorization breaks.
        # This is by design. The weights in the further layers should be
        # able to diverge a lot.
        self.l1 = FactorizedStackedLinear(2 * self.L1 // 2, self.L2 + 1, count)
        self.l2 = StackedLinear(self.L2 * 2, self.L3, count)
        self.output = StackedLinear(self.L3, 1, count)

        with torch.no_grad():
            self.output.linear.bias.zero_()

    def forward(self, x: torch.Tensor, ls_indices: torch.Tensor):
        l1c_ = self.l1(x, ls_indices)
        l1x_, l1x_out = l1c_.split(self.L2, dim=1)
        # multiply sqr crelu result by (127/128) to match quantized version
        l1x_ = torch.clamp(
            torch.cat([torch.pow(l1x_, 2.0) * (127 / 128), l1x_], dim=1), 0.0, 1.0
        )

        l2c_ = self.l2(l1x_, ls_indices)
        l2x_ = torch.clamp(l2c_, 0.0, 1.0)

        l3c_ = self.output(l2x_, ls_indices)
        l3x_ = l3c_ + l1x_out

        return l3x_

    @torch.no_grad()
    def get_coalesced_layer_stacks(
        self,
    ) -> Generator[tuple[nn.Linear, nn.Linear, nn.Linear], None, None]:
        # During training the buckets are represented by a single, wider, layer.
        # This representation needs to be transformed into individual layers
        # for the serializer, because the buckets are interpreted as separate layers.
        for i in range(self.count):
            yield self.l1.at_index(i), self.l2.at_index(i), self.output.at_index(i)

    @torch.no_grad()
    def coalesce_layer_stacks_inplace(self) -> None:
        self.l1.coalesce_weights()
