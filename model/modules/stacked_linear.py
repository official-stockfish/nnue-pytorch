import torch
from torch import nn
import torch.nn.functional as F

def apply_stacked_linear(
    x: torch.Tensor,
    weight: torch.Tensor,
    bias: torch.Tensor,
    count: int,
    in_features: int,
    out_features: int,
) -> torch.Tensor:
    """
    Applies the stacked linear transformation for both shared (2D) and independent (3D) inputs.
    """
    if x.dim() == 2:
        # 1. First layer case: Shared input (B, in_features)
        stacked_output = F.linear(x, weight, bias)
        return stacked_output.view(-1, count, out_features)

    elif x.dim() == 3:
        # 2. Subsequent layers case: Independent inputs per expert (B, count, in_features)
        w = weight.view(count, out_features, in_features)
        b = bias.view(count, out_features)

        # This applies each expert's weights strictly to its own data.
        return torch.einsum('bci,coi->bco', x, w) + b

    else:
        raise ValueError("apply_stacked_linear expects 2D or 3D input")


class StackedLinear(nn.Module):
    def __init__(self, in_features: int, out_features: int, count: int):
        super().__init__()

        self.in_features = in_features
        self.out_features = out_features
        self.count = count
        self.linear = nn.Linear(in_features, out_features * count)

        self._init_uniformly()

    @torch.no_grad()
    def _init_uniformly(self) -> None:
        init_weight = self.linear.weight[0 : self.out_features, :]
        init_bias = self.linear.bias[0 : self.out_features]

        self.linear.weight.copy_(init_weight.repeat(self.count, 1))
        self.linear.bias.copy_(init_bias.repeat(self.count))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return apply_stacked_linear(
            x,
            self.linear.weight,
            self.linear.bias,
            self.count,
            self.in_features,
            self.out_features,
        )

    @torch.no_grad()
    def at_index(self, index: int) -> nn.Linear:
        layer = nn.Linear(self.in_features, self.out_features)

        begin = index * self.out_features
        end = (index + 1) * self.out_features

        layer.weight.copy_(self.linear.weight[begin:end, :])
        layer.bias.copy_(self.linear.bias[begin:end])

        return layer


class FactorizedStackedLinear(StackedLinear):
    def __init__(self, in_features: int, out_features: int, count: int):
        super().__init__(in_features, out_features, count)

        self.factorized_linear = nn.Linear(in_features, out_features)

        with torch.no_grad():
            self.factorized_linear.weight.zero_()
            self.factorized_linear.bias.zero_()

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        merged_weight = self.linear.weight + self.factorized_linear.weight.repeat(
            self.count, 1
        )
        merged_bias = self.linear.bias + self.factorized_linear.bias.repeat(self.count)

        return apply_stacked_linear(
            x,
            merged_weight,
            merged_bias,
            self.count,
            self.in_features,
            self.out_features,
        )

    @torch.no_grad()
    def at_index(self, index: int) -> nn.Linear:
        layer = super().at_index(index)

        layer.weight.add_(self.factorized_linear.weight)
        layer.bias.add_(self.factorized_linear.bias)

        return layer

    @torch.no_grad()
    def coalesce_weights(self) -> None:
        for i in range(self.count):
            begin = i * self.out_features
            end = (i + 1) * self.out_features

            self.linear.weight[begin:end, :].add_(self.factorized_linear.weight)
            self.linear.bias[begin:end].add_(self.factorized_linear.bias)

        self.factorized_linear.weight.zero_()
        self.factorized_linear.bias.zero_()
