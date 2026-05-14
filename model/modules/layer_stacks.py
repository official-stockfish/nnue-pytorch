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

        # Factorizer only for the first layer because later
        # there's a non-linearity and factorization breaks.
        # This is by design. The weights in the further layers should be
        # able to diverge a lot.
        self.l1 = FactorizedStackedLinear(2 * self.L1 // 2, self.L2 + 1, count)
        self.l2 = StackedLinear(self.L2 * 2, self.L3, count)
        self.output = StackedLinear(self.L3, 1, count)

        with torch.no_grad():
            self.output.linear.bias.zero_()

    def forward(
        self, x: torch.Tensor,
        ls_indices: torch.Tensor,
        fake_quantize_acts: bool=False,
    ):
        l1c_ = self.l1(x, ls_indices)
        if fake_quantize_acts:
            l1c_ = self.quantization.fake_quantize_ls_act(l1c_)
        l1x_, l1x_out = l1c_.split(self.L2, dim=1)

        l1_sqr = torch.pow(l1x_, 2.0)
        # multiply sqr crelu result by scale correction to match quantized version
        l1_sqr = l1_sqr * (self.quantization.sqr_crelu_correction_factor)
        if fake_quantize_acts:
            l1_sqr = self.quantization.fake_quantize_ls_act(l1_sqr)

        l1x_ = self.quantization.clip_ls_act(
            torch.cat([l1_sqr, l1x_], dim=1)
        )

        l2c_ = self.l2(l1x_, ls_indices)
        if fake_quantize_acts:
            l2c_ = self.quantization.fake_quantize_ls_act(l2c_)
        l2x_ = self.quantization.clip_ls_act(l2c_)

        l3c_ = self.output(l2x_, ls_indices)
        if fake_quantize_acts:
            l3c_ = self.quantization.fake_quantize_ls_act(l3c_)
            l1x_out = self.quantization.fake_quantize_skip_act(l1x_out)

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
