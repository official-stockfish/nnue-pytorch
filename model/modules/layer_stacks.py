from typing import Generator

import torch
from torch import nn

from .stacked_linear import FactorizedStackedLinear, StackedLinear
from .config import LayerStacksConfig
from ..quantize import QuantizationManager

class LayerStacks(nn.Module):
    def __init__(self, count: int, config: LayerStacksConfig, quantization: QuantizationManager):
        super().__init__()

        self.count = count
        self.L1 = config.L1
        self.L2 = config.L2
        self.L3 = config.L3
        self.quantization = quantization

        self.l1 = StackedLinear(2 * self.L1 // 2, self.L2 + 1, count)
        self.l2 = StackedLinear(self.L2 * 2, self.L3, count)
        self.output = StackedLinear(self.L3, 1, count)

        with torch.no_grad():
            self.output.linear.bias.zero_()

    def forward(self, x: torch.Tensor, routing_data: torch.Tensor):
        # x: (Batch, in_features)

        # l1c_: (Batch, Count, L2 + 1)
        l1c_ = self.l1(x)

        # Split on the feature dimension (dim=2)
        l1x_, l1x_out = l1c_.split([self.L2, 1], dim=2)

        sqr_l1x = torch.pow(l1x_, 2.0) * self.quantization.sqcrele_correction_factor
        l1x_ = torch.clamp(
            torch.cat([sqr_l1x, l1x_], dim=2),
            0.0,
            self.quantization.max_hidden_activation,
        )

        # l2c_: (Batch, Count, L3)
        l2c_ = self.l2(l1x_)
        l2x_ = torch.clamp(l2c_, 0.0, self.quantization.max_hidden_activation)

        # l3c_: (Batch, Count, 1)
        l3c_ = self.output(l2x_)

        # l3x_: (Batch, Count, 1)
        l3x_ = l3c_ + l1x_out

        # Final Routing Selection / Mixing
        if routing_data.dtype in (torch.int32, torch.int64):
            # Hard inference select: routing_data is (B,)
            indices = routing_data.view(-1, 1, 1)
            selected = l3x_.gather(1, indices)
            return selected.squeeze(1) # Returns (B, 1)
        else:
            # Soft training mix: routing_data is (B, Count)
            weights = routing_data.unsqueeze(-1) # (B, Count, 1)
            return (l3x_ * weights).sum(dim=1) # Returns (B, 1)

    @torch.no_grad()
    def get_coalesced_layer_stacks(
        self,
    ) -> Generator[tuple[nn.Linear, nn.Linear, nn.Linear], None, None]:
        for i in range(self.count):
            yield self.l1.at_index(i), self.l2.at_index(i), self.output.at_index(i)

    @torch.no_grad()
    def coalesce_layer_stacks_inplace(self) -> None:
        pass
